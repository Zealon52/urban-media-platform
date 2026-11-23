"""Owns the expensive D1-D5 objects and gets them loaded without the rep
watching a spinner.

THE PROBLEM THIS SOLVES. `Orchestrator.__init__` reads nine CSVs (audience_cube
21 MB, pricing_grain_d3 24 MB, pricing_grain_d3_encoded 26 MB, win_loss_table
12 MB) and builds D3's rate card, win model and relationship table. Worse, the
FIRST date-aware quote lazily triggers `ridership_trend.load_tables()`, which
parses `ridership_actuals.csv` -- 129 MB -- and then fits one OLS per
(corridor_id, time_block_id). Left alone, all of that lands inside the first
plan run, as a one-to-two minute stall in the middle of a demo.

THE FIX. Two observations make it disappear:

  * Screens 01-02 (intake, interpretation) only need `cities.csv` and
    `dim_slot.csv` -- a few KB -- because `parse_brief(text, cities, dim_slot)`
    takes nothing else. So those screens can be fully interactive immediately.
  * The expensive lazy state (`_event_tables`, `_ridership_tables`,
    `_ridership_models`, `_screen_corridor_map`) is WINDOW-INDEPENDENT and
    cached on the engine instance. Only `_ridership_forecast_for_window` is
    per-campaign, and that part is cheap.

So a background thread builds the Orchestrator and fires one throwaway
date-aware quote to populate all of it, while the rep is reading and correcting
the brief. By the time they press Run, the costly work is already done.

Nothing here touches D1-D5. It imports and calls them.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app import config

config.load_dotenv_if_present()


# --------------------------------------------------------------------------- #
# Warmup state -- plain module globals + a lock, deliberately NOT st.cache_*.
#
# Streamlit's cache decorators expect to run on a script thread with a
# ScriptRunContext; calling them from a worker thread emits warnings and is not
# a supported path. A module-level singleton is thread-safe here because the
# lock guards every mutation and the worker never touches a Streamlit API.
# --------------------------------------------------------------------------- #

@dataclass
class WarmupState:
    stage: str = "cold"          # cold|loading|warming|ready|failed
    detail: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    traceback: str = ""
    orchestrator: Any = None
    solver: str = "unknown"      # cp_sat | greedy
    stage_timings: Dict[str, float] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.stage == "ready"

    @property
    def failed(self) -> bool:
        return self.stage == "failed"

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.monotonic()
        return end - self.started_at


_LOCK = threading.Lock()
_STATE = WarmupState()
_THREAD: Optional[threading.Thread] = None


def _set(**kw) -> None:
    with _LOCK:
        for k, v in kw.items():
            setattr(_STATE, k, v)


def warmup_state() -> WarmupState:
    """A snapshot. Callers must not mutate the returned object."""
    with _LOCK:
        return WarmupState(
            stage=_STATE.stage, detail=_STATE.detail, started_at=_STATE.started_at,
            finished_at=_STATE.finished_at, error=_STATE.error,
            traceback=_STATE.traceback, orchestrator=_STATE.orchestrator,
            solver=_STATE.solver, stage_timings=dict(_STATE.stage_timings),
        )


def ortools_available() -> bool:
    try:
        import ortools  # noqa: F401
        from ortools.sat.python import cp_model  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Light tables -- available immediately, no warmup required
# --------------------------------------------------------------------------- #

_LIGHT: Dict[str, Any] = {}
_LIGHT_LOCK = threading.Lock()


def light_tables() -> Dict[str, Any]:
    """cities.csv + dim_slot.csv, the only two tables `parse_brief` needs.

    Kept separate from the heavy load so screens 01-02 never wait on it.
    """
    with _LIGHT_LOCK:
        if _LIGHT:
            return _LIGHT
        import pandas as pd
        _LIGHT["cities"] = pd.read_csv(config.DATA_DIR / "cities.csv")
        _LIGHT["dim_slot"] = pd.read_csv(config.DATA_DIR / "dim_slot.csv")
        # screens.csv is 600 KB -- cheap enough to sit alongside the other two,
        # and city inference needs it to answer "where is the inventory the brief
        # is asking for actually located".
        _LIGHT["screens"] = pd.read_csv(
            config.DATA_DIR / "screens.csv",
            usecols=lambda c: c in {"screen_id", "city_id", "screen_type",
                                    "position", "screen_size"})
        return _LIGHT


def lost_leads_sample(n: int = 200):
    """A slice of real lost leads for the intake screen's replay selector.

    Reads only the columns the selector renders, so this stays fast even though
    lost_leads.csv is 400 KB of 15 columns.
    """
    import pandas as pd
    with _LIGHT_LOCK:
        if "lost_leads" in _LIGHT:
            df = _LIGHT["lost_leads"]
        else:
            df = pd.read_csv(config.DATA_DIR / "lost_leads.csv")
            _LIGHT["lost_leads"] = df
    return df.head(n)


# --------------------------------------------------------------------------- #
# The heavy warmup
# --------------------------------------------------------------------------- #

def _warm() -> None:
    t0 = time.monotonic()
    timings: Dict[str, float] = {}
    try:
        _set(stage="loading", detail="importing D1-D5 modules", started_at=t0)

        t = time.monotonic()
        from build_d5_orchestrator import Orchestrator
        timings["import"] = time.monotonic() - t

        _set(detail="reading screens, locations, POI, audience cube, "
                    "pricing grain; building D3 rate card + win model")
        t = time.monotonic()
        orch = Orchestrator()
        timings["orchestrator_init"] = time.monotonic() - t
        _set(orchestrator=orch, stage="warming",
             detail="priming event overlap + ridership trend models "
                    "(parses ridership_actuals.csv once, then fits one OLS per "
                    "corridor x time block)")

        # One throwaway date-aware quote. The window is irrelevant: what this
        # buys is the WINDOW-INDEPENDENT state -- _event_tables,
        # _ridership_tables, _ridership_models, _screen_corridor_map -- all of
        # which the engine caches on itself. Without this, the first real plan
        # run pays for it while the rep waits.
        t = time.monotonic()
        try:
            import pandas as pd
            engine = orch.engine
            # _cell is D3's (screen_id, time_block_id) MultiIndex. Taking a key
            # from it guarantees quote() will not raise KeyError on a probe.
            probe_key = engine._cell.index[0]
            start = pd.Timestamp.today().normalize()
            end = start + pd.Timedelta(days=7)
            engine.quote(probe_key[0], int(probe_key[1]),
                         campaign_start=start.strftime("%Y-%m-%d"),
                         campaign_end=end.strftime("%Y-%m-%d"))
        except Exception as exc:  # noqa: BLE001
            # A failed probe is not fatal -- it only means the first real quote
            # pays the lazy load instead. Say so rather than failing the boot.
            _set(detail=f"warmup probe skipped ({type(exc).__name__}: {exc}); "
                        f"the first plan run will load these tables instead")
        timings["lazy_prime"] = time.monotonic() - t

        _set(stage="ready", detail="", finished_at=time.monotonic(),
             solver="cp_sat" if ortools_available() else "greedy",
             stage_timings=timings)

    except Exception as exc:  # noqa: BLE001
        _set(stage="failed", error=f"{type(exc).__name__}: {exc}",
             traceback=traceback.format_exc(), finished_at=time.monotonic(),
             stage_timings=timings)


def start_warmup() -> None:
    """Idempotent, non-blocking. Safe to call on every Streamlit rerun."""
    global _THREAD
    with _LOCK:
        if _STATE.stage in {"loading", "warming", "ready"}:
            return
        if _THREAD is not None and _THREAD.is_alive():
            return
        _STATE.stage = "loading"
        _STATE.started_at = time.monotonic()
    _THREAD = threading.Thread(target=_warm, name="d6-warmup", daemon=True)
    _THREAD.start()


def get_orchestrator(block: bool = True, timeout: float = 900.0):
    """The Orchestrator, warming it first if needed.

    `block=True` waits, which is what the Run button wants: by then the warmup
    has usually been running since the app opened, so the wait is short or zero.
    """
    start_warmup()
    if not block:
        return warmup_state().orchestrator

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = warmup_state()
        if st.ready:
            return st.orchestrator
        if st.failed:
            raise RuntimeError(f"engine warmup failed: {st.error}\n{st.traceback}")
        time.sleep(0.25)
    raise TimeoutError(f"engine warmup did not finish within {timeout:.0f}s")


def reset_for_tests() -> None:
    global _THREAD, _STATE
    with _LOCK:
        _STATE = WarmupState()
    _THREAD = None

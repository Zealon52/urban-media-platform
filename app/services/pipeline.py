"""The one place the app invokes the engines.

D5's `Orchestrator.plan()` does the orchestration -- city funnel, daypart
resolution, D2 relevance, D4 candidate cells with D3 event-adjusted pricing,
affordability, shortlist sizing, the at-budget solve, the stretch solve, the
budget verdict and the ordered refusal diagnosis. This module does NOT
re-implement any of that. It:

  1. calls `plan()` once per revision,
  2. gathers the two things `plan()` does not return but the UI needs -- D2's
     scored shortlist (screen 04) and per-line D3 price bands (screen 05),
  3. runs the second day-type solve when the brief asked for a weekday/weekend
     split, which D2 detects and D5 does not consume,
  4. serialises the result for SQLite.

HOW A REP'S "DROP THIS SCREEN" REACHES THE OPTIMISER WITHOUT EDITING D5.
`plan()` scores relevance via `score_screens(brief, self.profiles, ...)` --
reading the orchestrator's own `profiles` attribute. So restricting the
candidate set is a matter of narrowing that input frame before the call and
restoring it after, which is configuring D5's inputs rather than changing its
behaviour. A pristine copy is kept so repeated revisions never compound.

`StructuredBrief.exclude` cannot express this: it holds
(screen_type, position) CATEGORY pairs, so it can drop "all bus-rear screens"
but never "this one screen" -- which is exactly what a rep unticking a row
means.
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from app import config


# --------------------------------------------------------------------------- #
# result containers
# --------------------------------------------------------------------------- #

@dataclass
class DayTypeSolve:
    """One day-type's solve, used for the weekday/weekend split report."""
    day_type: str
    spend: float
    reach: float
    impressions: float
    n_lines: int
    lines: pd.DataFrame


@dataclass
class RunResult:
    plan: Any                                  # build_d5_orchestrator.Plan
    scored: pd.DataFrame                       # D2 ranked shortlist
    cells: pd.DataFrame                        # D4 cells for the SELECTED screens
    quotes: Dict[Tuple[str, int], Any]         # (screen_id, block) -> D3 PriceQuote
    metrics: Dict[str, Any]
    duration_ms: int
    timings: Dict[str, float] = field(default_factory=dict)
    split: Optional[Dict[str, DayTypeSolve]] = None
    split_note: str = ""
    excluded_screen_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def lines(self) -> pd.DataFrame:
        return self.plan.lines

    @property
    def presentable(self) -> bool:
        return bool(self.plan.presentable)


# --------------------------------------------------------------------------- #
# candidate narrowing
# --------------------------------------------------------------------------- #

_PRISTINE_ATTR = "_d6_pristine_profiles"


def _pristine_profiles(orch) -> pd.DataFrame:
    """The orchestrator's original profiles frame, cached on the instance the
    first time we narrow it. Without this, two successive revisions would each
    narrow an already-narrowed frame and the candidate pool would shrink
    monotonically regardless of what the rep actually asked for."""
    if not hasattr(orch, _PRISTINE_ATTR):
        setattr(orch, _PRISTINE_ATTR, orch.profiles.copy())
    return getattr(orch, _PRISTINE_ATTR)


class _NarrowedProfiles:
    """Context manager: restrict `orch.profiles`, then always restore it."""

    def __init__(self, orch, excluded: Iterable[str]):
        self.orch = orch
        self.excluded = {str(s) for s in (excluded or []) if str(s).strip()}
        self.pristine = _pristine_profiles(orch)

    def __enter__(self):
        if self.excluded:
            keep = ~self.pristine["screen_id"].astype(str).isin(self.excluded)
            self.orch.profiles = self.pristine[keep].reset_index(drop=True)
        else:
            self.orch.profiles = self.pristine
        return self.orch

    def __exit__(self, *exc):
        self.orch.profiles = self.pristine
        return False


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def run(orch, *, brief, brief_text: str, campaign_start: Optional[str],
        campaign_end: Optional[str], day_type: str = "weekday",
        allow_unevidenced: bool = False, force_greedy: bool = False,
        excluded_screen_ids: Optional[Sequence[str]] = None,
        want_split: Optional[bool] = None,
        progress=None) -> RunResult:
    """One full revision. `progress(label)` is an optional UI callback."""
    from build_d2_relevance_scorer import score_screens
    from build_d5_orchestrator import Orchestrator  # noqa: F401 -- type clarity

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    excluded = [str(s) for s in (excluded_screen_ids or [])]
    timings: Dict[str, float] = {}
    notes: List[str] = []
    t_run = time.monotonic()

    # ---- 1. D5 orchestrates the whole plan ---------------------------------- #
    say("D5 orchestrating: relevance, pricing, capacity, allocation")
    t = time.monotonic()
    with _NarrowedProfiles(orch, excluded):
        plan = orch.plan(
            brief, brief_text=brief_text, day_type=day_type,
            allow_unevidenced=allow_unevidenced,
            campaign_start=campaign_start, campaign_end=campaign_end,
            force_greedy=force_greedy,
            time_limit_s=config.SOLVE_TIME_LIMIT_S,
        )
        timings["d5_plan"] = time.monotonic() - t

        # ---- 2. D2's ranked shortlist, for screen 04 ----------------------- #
        # plan() computes this internally and does not return it. Re-running
        # score_screens is deterministic and pure pandas (no LLM, no solver),
        # so the frame here is identical to the one the plan used -- and it must
        # run inside the same narrowing, or the table on screen 04 would list
        # screens the plan was not allowed to consider.
        say("collecting D2 relevance detail")
        t = time.monotonic()
        try:
            scored, _stats = score_screens(brief, orch.profiles, orch.cube,
                                           orch.screens, orch.locations,
                                           orch.poi, orch.cities)
        except Exception as exc:  # noqa: BLE001
            scored = pd.DataFrame()
            notes.append(f"D2 detail unavailable for display ({type(exc).__name__}: {exc}); "
                         f"the plan itself is unaffected")
        timings["d2_detail"] = time.monotonic() - t

    lines = plan.lines
    say("pricing the selected lines")

    # ---- 3. D4 cells + D3 bands for the SELECTED screens only -------------- #
    # Rebuilding cells for just the chosen screens gives the impressions figure
    # the optimiser's OWN lambda (people, dwell, visibility, event bump) rather
    # than a second formula invented here that could disagree with it.
    t = time.monotonic()
    cells = pd.DataFrame()
    quotes: Dict[Tuple[str, int], Any] = {}
    if len(lines):
        cells = _cells_for_selected(orch, plan, brief, campaign_start,
                                     campaign_end, day_type, notes)
        quotes = _quote_selected(orch, lines, campaign_start, campaign_end, notes)
    timings["pricing_detail"] = time.monotonic() - t

    # ---- 4. metrics --------------------------------------------------------- #
    from app.services import metrics as metrics_mod
    met = metrics_mod.compute(plan=plan, cells=cells, quotes=quotes,
                              scored=scored, brief=brief)

    # ---- 5. the weekday/weekend split D5 does not do ----------------------- #
    split: Optional[Dict[str, DayTypeSolve]] = None
    split_note = ""
    if want_split is None:
        want_split = (getattr(brief, "day_type_weighting", "") == "report_split")
    if want_split and len(lines):
        say("solving the other day type for the weekday/weekend split")
        t = time.monotonic()
        split, split_note = _day_type_split(
            orch, brief=brief, brief_text=brief_text, primary=plan,
            primary_day_type=day_type, campaign_start=campaign_start,
            campaign_end=campaign_end, allow_unevidenced=allow_unevidenced,
            force_greedy=force_greedy, excluded=excluded, notes=notes)
        timings["day_type_split"] = time.monotonic() - t

    duration_ms = int((time.monotonic() - t_run) * 1000)
    return RunResult(plan=plan, scored=scored, cells=cells, quotes=quotes,
                     metrics=met, duration_ms=duration_ms, timings=timings,
                     split=split, split_note=split_note,
                     excluded_screen_ids=excluded, notes=notes)


def _cells_for_selected(orch, plan, brief, campaign_start, campaign_end,
                        day_type, notes: List[str]) -> pd.DataFrame:
    from build_d4_allocator import CampaignRequest, build_candidate_cells
    lines = plan.lines
    screen_ids = sorted({str(s) for s in lines["screen_id"]})
    blocks = sorted({int(b) for b in lines["time_block_id"]}) if "time_block_id" in lines else None
    try:
        req = CampaignRequest(
            client_id=None, budget=float(brief.budget), budget_variance_pct=0.0,
            duration_days=int(brief.duration_days), candidate_screen_ids=screen_ids,
            preferred_blocks=blocks or None, day_type=day_type,
            campaign_start=campaign_start, campaign_end=campaign_end,
        )
        return build_candidate_cells(req, orch.d4_tables, orch.engine)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"could not rebuild D4 cells for the selected lines "
                     f"({type(exc).__name__}: {exc}); impressions fall back to a "
                     f"reach-only figure")
        return pd.DataFrame()


def _quote_selected(orch, lines: pd.DataFrame, campaign_start, campaign_end,
                    notes: List[str]) -> Dict[Tuple[str, int], Any]:
    """One D3 quote per allocated line -- floor / target / cap / recommended
    plus the component ledger and any event / ridership band shift.

    Only the selected lines are quoted (typically 5-30), so this is cheap; the
    expensive window-independent tables were already primed at warmup.
    """
    out: Dict[Tuple[str, int], Any] = {}
    failed = 0
    for _, r in lines.iterrows():
        sid = str(r.get("screen_id", ""))
        blk = int(r.get("time_block_id", 0) or 0)
        if (sid, blk) in out:
            continue
        try:
            out[(sid, blk)] = orch.engine.quote(
                sid, blk, client_id=None,
                campaign_start=campaign_start, campaign_end=campaign_end)
        except Exception:  # noqa: BLE001 -- a missing grid cell is not fatal
            failed += 1
    if failed:
        notes.append(f"{failed} allocated line(s) had no D3 rate-card cell, so no "
                     f"price band is shown for them")
    return out


def _day_type_split(orch, *, brief, brief_text, primary, primary_day_type,
                    campaign_start, campaign_end, allow_unevidenced,
                    force_greedy, excluded, notes) -> Tuple[Optional[dict], str]:
    """Solve the OTHER day type so the split can be reported separately.

    WHY THIS EXISTS. D2 detects `day_type_weighting == "report_split"` (brief 3
    asks for "the weekend-versus-weekday impression split shown separately"),
    but D5's `plan()` only branches on "weekend_heavy" -- so the requirement is
    parsed and then dropped. D1-D5 are frozen, so the app closes the gap by
    running the second solve itself through the same public entry point.

    Both solves are reported as separate delivery patterns, NOT summed: they are
    alternative allocations of the same budget, and adding them would double-
    count the money.
    """
    other = "weekend" if primary_day_type == "weekday" else "weekday"
    try:
        with _NarrowedProfiles(orch, excluded):
            alt = orch.plan(
                brief, brief_text=brief_text, day_type=other,
                allow_unevidenced=allow_unevidenced,
                campaign_start=campaign_start, campaign_end=campaign_end,
                force_greedy=force_greedy,
                time_limit_s=config.SOLVE_TIME_LIMIT_S,
            )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"the {other} solve for the split failed "
                     f"({type(exc).__name__}: {exc}); reporting the "
                     f"{primary_day_type} pattern only")
        return None, ""

    from app.services import metrics as metrics_mod

    def _pack(p, dt: str) -> DayTypeSolve:
        # Rebuild D4's cells for THIS day type's own selection. Passing
        # cells=None silently fell back to the reach x slots lower bound, so the
        # split table reported a different impressions basis from the headline
        # figure -- 3.72M against 8.14M for the identical weekday plan. Two
        # disagreeing impressions formulas in one document is worse than none.
        c = (_cells_for_selected(orch, p, brief, campaign_start, campaign_end, dt,
                                 notes) if len(p.lines) else pd.DataFrame())
        imp = metrics_mod.impressions_from_lines(
            p.lines, cells=c, duration_days=int(brief.duration_days))
        return DayTypeSolve(day_type=dt, spend=float(p.spend), reach=float(p.reach),
                            impressions=float(imp), n_lines=int(len(p.lines)),
                            lines=p.lines)

    split = {primary_day_type: _pack(primary, primary_day_type),
             other: _pack(alt, other)}
    note = (f"The brief asks for the weekday/weekend split to be shown separately. "
            f"Each column below is a COMPLETE alternative plan for the same "
            f"{float(brief.budget):,.0f} budget solved against that day type's own "
            f"audience -- they are alternatives, not additive, so the two spends "
            f"must not be summed.")
    return split, note


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #

def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def plan_to_dict(result: RunResult) -> dict:
    """JSON-safe snapshot of a run, for `runs.plan_json`.

    Rich enough to rebuild every screen and every export without re-solving,
    which is what makes the run history in the sidebar replayable.
    """
    p = result.plan
    b = p.brief
    quotes = [{
        "screen_id": k[0], "time_block_id": k[1],
        "floor": q.floor, "target": q.target, "cap": q.cap,
        "recommended": q.recommended, "basis": q.rate_card_basis,
        "p_win": q.p_win_at_recommended, "expected_revenue": q.expected_revenue,
        "event_adjustment_pct": q.event_adjustment_pct, "event_names": q.event_names,
        "ridership_trend_pct": q.ridership_trend_pct,
        "ridership_trend_note": q.ridership_trend_note,
        "components": q.components, "warnings": q.warnings,
    } for k, q in result.quotes.items()]

    return {
        "brief": _jsonable(b),
        "day_type": p.day_type,
        "industry": _jsonable(p.industry),
        "cohort": _jsonable(p.cohort),
        "blocks": _jsonable(p.blocks),
        "sizing": _jsonable(p.sizing),
        "funnel": {name: n for name, n in p.funnel.rows()},
        "verdict": _jsonable(p.verdict),
        "refusals": [_jsonable(r) for r in p.refusals],
        "warnings": list(p.warnings),
        "solver_used": p.at_budget.get("solver_used"),
        "n_candidate_cells": p.at_budget.get("n_candidate_cells"),
        "lines": _jsonable(p.lines),
        "stretch": {
            "spend": float(p.stretch["spend"]) if p.stretch else None,
            "reach": float(p.stretch["deduped_reach"]) if p.stretch else None,
            "n_selected": int(p.stretch.get("n_selected", 0)) if p.stretch else 0,
            "overrun_pct": p.stretch_overrun_pct,
            "worth_offering": bool(p.stretch_worth_offering),
            "gain": list(p.stretch_gain),
        } if p.stretch else None,
        "quotes": quotes,
        "metrics": _jsonable(result.metrics),
        "split": ({dt: {"day_type": s.day_type, "spend": s.spend, "reach": s.reach,
                         "impressions": s.impressions, "n_lines": s.n_lines}
                   for dt, s in result.split.items()} if result.split else None),
        "split_note": result.split_note,
        "excluded_screen_ids": result.excluded_screen_ids,
        "timings": result.timings,
        "notes": result.notes,
        "render": p.render(),
    }

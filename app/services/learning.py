"""Cross-session learning: what earlier campaigns taught us, applied to this one.

The ADAPT pillar asks the agent to learn from the rep's notes and feedback. This
module is that memory. Four properties make it safe enough to leave on:

  1. EVIDENCE, NOT COEFFICIENTS. `learned_signals` stores one row per feedback
     event. Weights are decayed and aggregated at READ time, so the raw evidence
     stays inspectable and a bad rule can be changed without rewriting history.

  2. IT AGES. Weight decays on a 45-day halflife, reusing the same shape as
     `pricing_engine_d3.recency_weight` so the app's decay agrees with D3's
     rather than inventing a second notion of "recent". The problem statement
     calls this out directly: older unconverted signals are weaker signals.

  3. IT NEEDS CORROBORATION. Below `LEARNING_MIN_EVIDENCE` independent events a
     signal is recorded and displayed but NOT applied -- the same reasoning as
     D5's MIN_COHORT, where a 3-client median is noise wearing a decimal point.
     The gate counts EVENTS; decay scales the resulting magnitude. See
     `Adjustments._shift` for why conflating the two was a real bug.

  4. IT IS CAPPED AND VISIBLE. Price moves at most +/-10%, relevance at most
     +/-0.08, and every applied adjustment carries the sentence explaining it.
     A rep can switch it off in the sidebar and see the plan without it.

WHAT IT DOES NOT DO. It never writes to D1-D5 and never changes the optimiser's
cost basis. The solver always allocates against D3's own recommended price; the
learned shift moves the price the app RECOMMENDS THE REP QUOTE, which is a
commercial judgement layered on top of the model, and the UI says so. Learning a
price down and then also spending that lower price as if it were capacity would
be double-counting a single signal.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from app import config
from app.db import store


def recency_weight(age_days: float,
                   halflife_days: float = config.LEARNING_HALFLIFE_DAYS) -> float:
    """Same exponential form D3 uses for lead recency."""
    if age_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / max(halflife_days, 1e-9)))


#: Which loss reasons teach what, and at which scope. A reason absent here is
#: still recorded for the audit trail but deliberately teaches nothing --
#: `client_went_quiet` and `creative_rejected` say nothing about inventory or
#: price, and pretending otherwise would poison the signal.
SIGNAL_RULES: Dict[str, Dict[str, Any]] = {
    "price_too_high":       {"scopes": ["zone", "screen_type"], "kind": "price", "dir": -1.0},
    "budget_mismatch":      {"scopes": ["zone"],                "kind": "price", "dir": -1.0},
    "went_with_competitor": {"scopes": ["zone"],                "kind": "price", "dir": -1.0},
    "targeting_mismatch":   {"scopes": ["screen_type", "zone"], "kind": "score", "dir": -1.0},
    "geography_mismatch":   {"scopes": ["zone"],                "kind": "score", "dir": -1.0},
    "inventory_unavailable": {"scopes": ["screen"],             "kind": "score", "dir": -1.0},
}

#: How much one fully-weighted event moves things, before the cap.
PRICE_STEP = 0.025      # 2.5% per corroborating event
SCORE_STEP = 0.02       # 0.02 relevance points per corroborating event


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #

def screen_attributes(scored: Optional[pd.DataFrame],
                      screens: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, str]]:
    """screen_id -> {zone, screen_type}. Built from D2's scored frame, which
    already carries both, so no extra table read is needed."""
    out: Dict[str, Dict[str, str]] = {}
    if scored is None or not len(scored):
        return out
    cols = scored.columns
    for _, r in scored.iterrows():
        sid = str(r.get("screen_id", ""))
        if not sid:
            continue
        out[sid] = {
            "zone": str(r.get("zone_id", "")) if "zone_id" in cols else "",
            "screen_type": str(r.get("screen_type", "")) if "screen_type" in cols else "",
        }
    return out


def record(*, campaign_id: int, loss_reasons: Sequence[str],
           affected_screen_ids: Sequence[str],
           attrs: Dict[str, Dict[str, str]],
           source_feedback_id: Optional[int] = None,
           note: str = "") -> List[int]:
    """Write one signal row per (reason x scope x affected key).

    `affected_screen_ids` is what the feedback was about: the dropped screens
    for an inventory objection, or the whole allocation for a price objection
    (a rep saying "too expensive" is talking about the package they were shown).
    """
    written: List[int] = []
    for reason in {str(r) for r in loss_reasons}:
        rule = SIGNAL_RULES.get(reason)
        if not rule:
            continue
        for scope in rule["scopes"]:
            keys = set()
            for sid in affected_screen_ids:
                a = attrs.get(str(sid), {})
                if scope == "screen":
                    keys.add(str(sid))
                elif scope == "zone" and a.get("zone"):
                    keys.add(a["zone"])
                elif scope == "screen_type" and a.get("screen_type"):
                    keys.add(a["screen_type"])
            for key in keys:
                written.append(store.add_learned_signal(
                    scope=scope, scope_key=key, signal=reason,
                    direction=float(rule["dir"]),
                    source_feedback_id=source_feedback_id,
                    campaign_id=campaign_id, note=note))
    return written


# --------------------------------------------------------------------------- #
# reading / applying
# --------------------------------------------------------------------------- #

@dataclass
class Adjustments:
    enabled: bool = True
    #: (scope, key, kind) -> {"weight": decayed, "events": n, "signals": {...}}
    agg: Dict[Tuple[str, str, str], Dict[str, Any]] = field(default_factory=dict)
    attrs: Dict[str, Dict[str, str]] = field(default_factory=dict)
    total_events: int = 0

    # -- lookups ---------------------------------------------------------- #
    def _keys_for(self, screen_id: str) -> List[Tuple[str, str]]:
        a = self.attrs.get(str(screen_id), {})
        out = [("screen", str(screen_id))]
        if a.get("zone"):
            out.append(("zone", a["zone"]))
        if a.get("screen_type"):
            out.append(("screen_type", a["screen_type"]))
        return out

    def _shift(self, screen_id: str, kind: str, step: float,
               cap: float) -> Tuple[float, List[str]]:
        """Gate on the RAW event count, scale by the DECAYED weight.

        These are two different questions and conflating them was a real bug:
        decay only ever pushes weight below the event count, so testing
        `weight >= MIN_EVIDENCE` was satisfiable only at literally zero age --
        three events one day old weigh 3 * 0.5**(1/45) = 2.954 and would never
        clear a threshold of 3 again. Corroboration is "how many independent
        events", which does not decay; recency belongs in the magnitude, where
        an ageing signal fades toward zero on its own.
        """
        if not self.enabled:
            return 0.0, []
        total = 0.0
        why: List[str] = []
        for scope, key in self._keys_for(screen_id):
            rec = self.agg.get((scope, key, kind))
            if not rec:
                continue
            if rec["events"] < config.LEARNING_MIN_EVIDENCE:
                continue
            delta = rec["direction"] * step * rec["weight"]
            total += delta
            sigs = ", ".join(sorted(rec["signals"]))
            why.append(f"{scope} {key}: {rec['events']} past event(s) "
                       f"({sigs}), decayed weight {rec['weight']:.1f}")
        clamped = max(-cap, min(cap, total))
        if why and abs(clamped) < abs(total) - 1e-12:
            why.append(f"capped at {cap:+.0%}" if kind == "price"
                       else f"capped at {cap:+.3f}")
        return clamped, why

    def price_multiplier(self, screen_id: str) -> Tuple[float, List[str]]:
        """Multiplier on the price the app recommends quoting. <1 means past
        feedback says this inventory is priced above what buyers accept."""
        shift, why = self._shift(screen_id, "price", PRICE_STEP,
                                 config.LEARNING_MAX_PRICE_SHIFT)
        return 1.0 + shift, why

    def score_delta(self, screen_id: str) -> Tuple[float, List[str]]:
        return self._shift(screen_id, "score", SCORE_STEP,
                           config.LEARNING_MAX_SCORE_SHIFT)

    # -- reporting -------------------------------------------------------- #
    def active_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for (scope, key, kind), rec in sorted(
                self.agg.items(), key=lambda kv: -kv[1]["weight"]):
            applied = rec["events"] >= config.LEARNING_MIN_EVIDENCE
            step = PRICE_STEP if kind == "price" else SCORE_STEP
            cap = (config.LEARNING_MAX_PRICE_SHIFT if kind == "price"
                   else config.LEARNING_MAX_SCORE_SHIFT)
            raw = rec["direction"] * step * rec["weight"]
            rows.append({
                "scope": scope, "key": key, "kind": kind,
                "events": rec["events"], "weight": round(rec["weight"], 2),
                "signals": ", ".join(sorted(rec["signals"])),
                "effect": max(-cap, min(cap, raw)) if applied else 0.0,
                "applied": applied,
                "status": "applied" if applied else
                          f"below the {config.LEARNING_MIN_EVIDENCE}-event threshold",
            })
        return rows

    @property
    def n_applied(self) -> int:
        return sum(1 for r in self.active_rows() if r["applied"])

    def summary(self) -> str:
        if not self.enabled:
            return "Cross-session learning is switched off for this run."
        if not self.agg:
            return ("No prior feedback recorded yet, so nothing is being adjusted. "
                    "Signals accumulate as reps submit feedback.")
        n = self.n_applied
        if n == 0:
            return (f"{self.total_events} prior feedback event(s) recorded across "
                    f"{len(self.agg)} scope(s), none yet at the "
                    f"{config.LEARNING_MIN_EVIDENCE}-event threshold, so nothing is "
                    f"being adjusted.")
        return (f"{n} of {len(self.agg)} learned scope(s) are above the "
                f"{config.LEARNING_MIN_EVIDENCE}-event evidence threshold and are "
                f"adjusting this plan, from {self.total_events} prior feedback "
                f"event(s).")


def load(*, campaign_id: Optional[int], scored: Optional[pd.DataFrame],
         enabled: bool = True) -> Adjustments:
    """Aggregate every prior signal into decayed weights.

    The current campaign is excluded: feeding a rep's own just-submitted comment
    back to them as independent historical evidence would double-count it.
    """
    adj = Adjustments(enabled=enabled, attrs=screen_attributes(scored))
    if not enabled:
        return adj

    rows = store.learned_signals(exclude_campaign_id=campaign_id)
    adj.total_events = len(rows)
    now = datetime.now(timezone.utc)
    buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for r in rows:
        rule = SIGNAL_RULES.get(r["signal"])
        if not rule:
            continue
        kind = rule["kind"]
        try:
            created = datetime.fromisoformat(r["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = max((now - created).total_seconds() / 86400.0, 0.0)
        except (TypeError, ValueError):
            age_days = 0.0
        w = recency_weight(age_days)
        k = (r["scope"], r["scope_key"], kind)
        b = buckets.setdefault(k, {"weight": 0.0, "events": 0, "signals": set(),
                                   "direction": float(r["direction"])})
        b["weight"] += w
        b["events"] += 1
        b["signals"].add(r["signal"])

    adj.agg = buckets
    return adj


def apply_to_quotes(lines: pd.DataFrame, quotes: Dict, adj: Adjustments,
                    duration_days: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per-line quoted price after learning, as a display/RFP layer.

    Returns a frame with the D3 price, the learned multiplier, the adjusted
    price CLAMPED INTO D3's OWN BAND, and the reason. Clamping matters: learning
    is a commercial nudge inside a modelled range, not a licence to quote below
    a floor that was derived from real comparable bookings.
    """
    out_rows: List[Dict[str, Any]] = []
    total_d3 = 0.0
    total_adj = 0.0
    n_moved = 0

    slots_col = ("slots_booked_per_day" if "slots_booked_per_day" in lines.columns
                 else "slots")
    for _, r in lines.iterrows():
        sid = str(r.get("screen_id", ""))
        blk = int(r.get("time_block_id", 0) or 0)
        q = quotes.get((sid, blk))
        base = float(r.get("price", 0.0) or 0.0)
        units = float(r.get(slots_col, 0) or 0) * float(duration_days)
        mult, why = adj.price_multiplier(sid)
        adjusted = base * mult
        clamped_note = ""
        if q is not None:
            lo, hi = float(q.floor), float(q.cap)
            if adjusted < lo:
                adjusted, clamped_note = lo, f"clamped up to D3 floor {lo:,.2f}"
            elif adjusted > hi:
                adjusted, clamped_note = hi, f"clamped down to D3 cap {hi:,.2f}"
        if abs(adjusted - base) > 1e-9:
            n_moved += 1
        total_d3 += base * units
        total_adj += adjusted * units
        out_rows.append({
            "screen_id": sid, "time_block_id": blk,
            "d3_price": base, "learned_multiplier": mult,
            "quoted_price": adjusted, "units": units,
            "d3_cost": base * units, "quoted_cost": adjusted * units,
            "reason": "; ".join(why) + (f" ({clamped_note})" if clamped_note else ""),
        })

    return pd.DataFrame(out_rows), {
        "d3_total": total_d3, "quoted_total": total_adj,
        "delta": total_adj - total_d3,
        "delta_pct": ((total_adj / total_d3 - 1.0) if total_d3 > 0 else 0.0),
        "n_lines_moved": n_moved,
    }

"""Rep-facing metrics, derived only from columns the engines actually produced.

IMPRESSIONS DESERVE A NOTE, because the problem statement calls out the
non-linearity explicitly ("ads with higher number of slots per minute have
higher chances of being noticed... account for the non-linearity in impressions
vs boards vs pricing").

D4 models a 6-slot loop of 10s each, so one full cycle is 60s. A viewer present
for `dwell_seconds` sees `dwell/60` cycles, and in each cycle sees each slot
they own once. Owning k slots therefore gives

    lambda = k * (dwell_seconds / 60) * visibility        <- cells.cycles_per_slot * k
    reach  = people * (1 - exp(-lambda))                  <- concave, saturating
    impressions = people * lambda                         <- linear in slots

So reach and impressions are DIFFERENT curves over the same buy, and the ratio
between them is average frequency. This module never invents a second
impressions formula: it reads `people` and `cycles_per_slot` back out of D4's
own cells frame, so the figure shown agrees with what the optimiser maximised.

When the cells frame is unavailable, impressions degrade to a stated
lower-bound (reach x slots) and `impressions_basis` says so, rather than
silently reporting a number of unknown provenance.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd

SLOTS_PER_BLOCK = 6          # mirrors build_d4_allocator.SLOTS_PER_BLOCK


def _slots_col(lines: pd.DataFrame) -> str:
    return "slots_booked_per_day" if "slots_booked_per_day" in lines.columns else "slots"


def _f(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def line_costs(lines: pd.DataFrame, duration_days: int) -> pd.Series:
    """Per-line window cost. Uses D4's `cost` column when present, else the
    same arithmetic D5's render() uses: price x slots x days."""
    if "cost" in lines.columns:
        return lines["cost"].astype(float)
    sc = _slots_col(lines)
    return lines["price"].astype(float) * lines[sc].astype(float) * float(duration_days)


def impressions_from_lines(lines: pd.DataFrame, cells: Optional[pd.DataFrame],
                           duration_days: int) -> float:
    """Total impressions over the flight. See the module docstring."""
    if lines is None or not len(lines):
        return 0.0
    sc = _slots_col(lines)

    if cells is not None and len(cells) and "cycles_per_slot" in cells.columns:
        key = ["screen_id", "time_block_id"]
        c = cells[key + ["people", "cycles_per_slot"]].drop_duplicates(subset=key)
        m = lines[key + [sc]].merge(c, on=key, how="left")
        lam = m[sc].astype(float) * m["cycles_per_slot"].astype(float).fillna(0.0)
        per_day = (m["people"].astype(float).fillna(0.0) * lam).sum()
        if per_day > 0:
            return float(per_day * duration_days)

    # Fallback: reach x slots is a genuine lower bound on impressions (every
    # reached person sees the ad at least once per slot owned), and it is
    # labelled as such by the caller.
    if "people_reach" in lines.columns:
        per_day = (lines["people_reach"].astype(float)
                   * lines[sc].astype(float)).sum()
        return float(per_day * duration_days)
    return 0.0


def impressions_basis(cells: Optional[pd.DataFrame]) -> str:
    """Client-facing, so it names the METHOD rather than the module that
    implements it -- this string is rendered in the generated proposal."""
    if cells is not None and len(cells) and "cycles_per_slot" in cells.columns:
        return ("share-of-voice model: audience x slots owned x (dwell time / 60s "
                "rotation) x visibility, per day, across the flight")
    return ("lower bound only (reach x slots x days) -- per-screen dwell and "
            "visibility were unavailable, so no share-of-voice weighting was "
            "applied")


def package_price_range(lines: pd.DataFrame, quotes: Dict, duration_days: int) -> Dict[str, Any]:
    """The quotable range for the WHOLE package.

    A per-screen band is what D3 produces; a rep negotiates one number for the
    deal. Summing each line's floor / target / cap across the allocation gives
    the package band, and the recommended total is what D4 actually costed the
    plan at.
    """
    if not len(lines) or not quotes:
        return {"floor": 0.0, "target": 0.0, "cap": 0.0, "recommended": 0.0,
                "n_priced": 0, "n_lines": int(len(lines)), "coverage": 0.0}

    sc = _slots_col(lines)
    tot = {"floor": 0.0, "target": 0.0, "cap": 0.0, "recommended": 0.0}
    n_priced = 0
    p_win_num = 0.0
    p_win_den = 0.0
    exp_rev = 0.0

    for _, r in lines.iterrows():
        sid, blk = str(r.get("screen_id", "")), int(r.get("time_block_id", 0) or 0)
        q = quotes.get((sid, blk))
        units = _f(r.get(sc), 0.0) * float(duration_days)
        if q is None:
            # Still count its real cost so `recommended` matches the plan spend.
            tot["recommended"] += _f(r.get("price")) * units
            continue
        n_priced += 1
        tot["floor"] += _f(q.floor) * units
        tot["target"] += _f(q.target) * units
        tot["cap"] += _f(q.cap) * units
        line_cost = _f(r.get("price"), _f(q.recommended)) * units
        tot["recommended"] += line_cost
        p_win_num += _f(q.p_win_at_recommended) * line_cost
        p_win_den += line_cost
        exp_rev += line_cost * _f(q.p_win_at_recommended)

    tot["n_priced"] = n_priced
    tot["n_lines"] = int(len(lines))
    tot["coverage"] = n_priced / max(len(lines), 1)
    tot["weighted_p_win"] = (p_win_num / p_win_den) if p_win_den > 0 else 0.0
    tot["expected_revenue"] = exp_rev
    return tot


def compute(*, plan, cells: Optional[pd.DataFrame], quotes: Dict,
            scored: Optional[pd.DataFrame], brief) -> Dict[str, Any]:
    """The full metric block rendered on screens 05-06 and exported."""
    lines = plan.lines
    days = int(brief.duration_days or 0)
    budget = _f(brief.budget)
    spend = _f(plan.spend)
    reach = _f(plan.reach)

    imps = impressions_from_lines(lines, cells, days)
    band = package_price_range(lines, quotes, days)

    out: Dict[str, Any] = {
        # --- delivery ---
        "reach_per_day": reach,
        "impressions_total": imps,
        "impressions_per_day": (imps / days) if days else 0.0,
        "impressions_basis": impressions_basis(cells),
        "avg_frequency_per_day": ((imps / days) / reach) if (days and reach > 0) else 0.0,
        "n_lines": int(len(lines)),
        "n_screens": int(lines["screen_id"].nunique()) if len(lines) else 0,
        "n_pools": int(lines["pool_id"].nunique()) if ("pool_id" in lines.columns and len(lines)) else 0,
        "flight_days": days,
        "day_type": plan.day_type,
        "objective": getattr(brief, "optimise_for", "reach"),

        # --- commercial ---
        "budget": budget,
        "spend": spend,
        "budget_utilisation": (spend / budget) if budget > 0 else 0.0,
        "unspent": max(budget - spend, 0.0),
        "package_floor": band["floor"],
        "package_target": band["target"],
        "package_cap": band["cap"],
        "package_recommended": band["recommended"],
        "band_coverage": band["coverage"],
        "weighted_p_win": band.get("weighted_p_win", 0.0),
        "expected_revenue": band.get("expected_revenue", 0.0),
        "effective_cpm": (spend / imps * 1000.0) if imps > 0 else 0.0,
        "cost_per_1k_daily_reach": (spend / (reach / 1000.0)) if reach > 0 else 0.0,

        # --- solver / trust ---
        "solver_used": plan.at_budget.get("solver_used", "?"),
        "n_candidate_cells": int(plan.at_budget.get("n_candidate_cells", 0) or 0),
        "presentable": bool(plan.presentable),
        "verdict_state": plan.verdict.state,
        "verdict_reason": plan.verdict.reason,
        "n_refusals": len(plan.refusals),
        "n_warnings": len(plan.warnings),
    }

    # --- capacity utilisation ---
    if len(lines):
        sc = _slots_col(lines)
        bought = _f(lines[sc].astype(float).sum())
        out["slots_bought_per_day"] = bought
        if cells is not None and len(cells) and "remaining_capacity" in cells.columns:
            key = ["screen_id", "time_block_id"]
            cap = cells[key + ["remaining_capacity"]].drop_duplicates(subset=key)
            m = lines[key].merge(cap, on=key, how="left")
            avail = _f(m["remaining_capacity"].astype(float).sum())
            out["slots_available_per_day"] = avail
            out["capacity_utilisation"] = (bought / avail) if avail > 0 else 0.0
        else:
            out["slots_available_per_day"] = float(len(lines) * SLOTS_PER_BLOCK)
            out["capacity_utilisation"] = bought / max(len(lines) * SLOTS_PER_BLOCK, 1)

    # --- stretch option ---
    ds, dv, per1k = plan.stretch_gain
    out["stretch_extra_spend"] = _f(ds)
    out["stretch_extra_reach"] = _f(dv)
    out["stretch_reach_per_1k"] = _f(per1k)
    out["stretch_overrun_pct"] = _f(plan.stretch_overrun_pct)
    out["stretch_worth_offering"] = bool(plan.stretch_worth_offering)
    out["cohort_tolerance_pct"] = _f(plan.cohort.variance_pct)
    out["stretch_inside_tolerance"] = (
        out["stretch_overrun_pct"] <= out["cohort_tolerance_pct"] + 1e-9)

    # --- recommendation quality, from D2 + D1 ---
    if scored is not None and len(scored) and len(lines):
        sel = set(lines["screen_id"].astype(str))
        s = scored[scored["screen_id"].astype(str).isin(sel)]
        if len(s):
            out["relevance_mean"] = _f(s["relevance_score"].mean())
            out["relevance_min"] = _f(s["relevance_score"].min())
            out["relevance_max"] = _f(s["relevance_score"].max())
            if "confidence" in s.columns:
                out["audience_confidence_mean"] = _f(s["confidence"].mean())
            if "has_booking_history" in s.columns:
                costs = line_costs(lines, days)
                # Materialise to real Python bools before comparing. A pandas
                # nullable-boolean scalar is never the `True` SINGLETON, so an
                # `is not True` test here reported EVERY screen as cold-start --
                # which contradicted D2's own per-screen reason text and D3's
                # rate_card_basis inside the same generated proposal.
                hist = dict(zip(
                    s["screen_id"].astype(str),
                    s["has_booking_history"].fillna(False).astype(bool).tolist()))
                cold = sum(c for sid, c in zip(lines["screen_id"].astype(str), costs)
                           if not hist.get(sid, False))
                out["cold_start_spend"] = _f(cold)
                out["cold_start_spend_share"] = (cold / spend) if spend > 0 else 0.0
                out["n_cold_start_screens"] = int(
                    sum(1 for sid in sel if not hist.get(sid, False)))
        out["n_scored_candidates"] = int(len(scored))

    # --- how much of the buy was PRICED from peers rather than own history ---
    #
    # This is a DIFFERENT question from the cold-start figure above, and
    # conflating them produced a document that contradicted itself. D1's
    # `has_booking_history` is SCREEN-level: has this screen ever been booked.
    # D3's `rate_card_basis` is CELL-level: did this screen x time_block have
    # enough history for a credible rate, or was the band borrowed from peers.
    # A screen with bookings in the morning block can still need peer pricing
    # for the evening one, so 0% cold-start alongside 14 peer-priced lines is
    # not an error -- but only if the proposal says which is which.
    if len(lines) and quotes:
        costs = line_costs(lines, days)
        peer_spend = 0.0
        peer_lines = 0
        for (sid, blk), c in zip(
                zip(lines["screen_id"].astype(str),
                    lines["time_block_id"].astype(int)), costs):
            q = quotes.get((sid, blk))
            basis = str(getattr(q, "rate_card_basis", "") or "")
            if "cold_start" in basis or "peer_fallback" in basis:
                peer_spend += float(c)
                peer_lines += 1
        out["peer_priced_lines"] = peer_lines
        out["peer_priced_spend"] = _f(peer_spend)
        out["peer_priced_spend_share"] = (peer_spend / spend) if spend > 0 else 0.0

    # --- co-location / shared-audience exposure ---
    # The brief warns that boards on the same route or location are not
    # independent reach. D4 handles it via pool saturation; this surfaces HOW
    # MUCH of the buy is exposed to that adjustment.
    if len(lines) and "pool_id" in lines.columns:
        per_pool = lines.groupby("pool_id", observed=True).size()
        shared = int(per_pool[per_pool > 1].sum())
        out["lines_sharing_a_pool"] = shared
        out["pool_share_of_lines"] = shared / max(len(lines), 1)

    return out


def money(x: Optional[float], dp: int = 0) -> str:
    if x is None:
        return "--"
    return f"${_f(x):,.{dp}f}"


def compact(x: Optional[float]) -> str:
    """2.4M / 688k / 412 -- for reach and impression tiles."""
    v = _f(x)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:,.0f}"


def pct(x: Optional[float], dp: int = 1) -> str:
    return f"{_f(x) * 100:.{dp}f}%"

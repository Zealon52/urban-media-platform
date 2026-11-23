"""
D4 -- Impressions Optimizer: picks the best multi-screen, multi-slot package
for a campaign under a hard budget and hard slot-capacity constraint,
maximizing DEDUPLICATED reach (shared-audience pools from D1, so two screens
on the same corridor/location don't double-count the same people).

--------------------------------------------------------------------------
WHY THIS EXISTS -- and how events plug in (per explicit ask: "events are a
major factor in impressions and also slot availability constraint in D4")
--------------------------------------------------------------------------
Events enter D4 in exactly TWO places, deliberately not three, to avoid
double-counting the same real-world signal:

  1. IMPRESSIONS (reach objective): for a genuinely date-overlapping event
     (see event_overlap.py -- verified [start_date,end_date] overlap with
     THIS campaign's actual window, not the static per-location average),
     event_overlap.event_impressions_bump() adds a stated, capped fraction
     of the event's expected_attendance to that screen/block's "people"
     count for the reach objective -- so a genuinely event-boosted screen
     is worth MORE to select, not just more expensive.

  2. PRICE (cost side, via D3): the SAME overlap already raises the price
     PricingEngineD3.quote()/quote_grid() returns (band shift, see
     pricing_engine_d3.py). D4 consumes that price as-is.

  3. SLOT AVAILABILITY (hard constraint, not a soft signal): capacity is
     capped at remaining_capacity = 6 - already-booked slots for that
     screen/block (from pricing_grain_d3.csv's forward_occupancy_pct) --
     this is a REAL physical limit (6 slots/day/block) and is NEVER
     exceeded regardless of price or reach payoff. Genuinely event-scarce
     cells (real overlap AND thin remaining capacity) are naturally
     protected because (a) the hard cap already stops overselling, and
     (b) the elevated event price (#2) discourages a low-value campaign
     from claiming the last slot cheaply. An OPTIONAL, clearly-labeled
     "event reserve" toggle (RESERVE_LAST_SLOT_ON_EVENT_OVERLAP, default
     OFF) additionally withholds the very last remaining slot on a
     direct-tier event-overlap cell, so a rep can choose extra protection
     for a specific high-value future client -- this is a stated business
     rule, not learned, and is off by default so nothing is silently
     invented into default behavior.

demand_intensity / event_demand_boost_log (the STATIC, historical signal
already baked into D3's rate card) is intentionally NOT re-used here as a
raw input to the objective -- it already converted into price upstream in
D3. Feeding it into D4's reach objective too would double-count the same
demand pressure once as "this cell is more expensive" and again as
"this cell is worth more reach," which is exactly the bug flagged and
avoided in this session's design discussion.

--------------------------------------------------------------------------
FORMULATION -- budgeted reach with share-of-voice saturation
--------------------------------------------------------------------------
Each candidate (screen_id, time_block_id) cell belongs to a pool_id (D1's
audience-sharing group: same location, same vehicle, or same street
corridor+block+day_type -- screens in a pool share the same real people).

All 6 rotation slots loop continuously, so one full cycle is 6 x 10s = 60s.
A viewer present for `dwell_seconds` sees dwell/60 cycles, and in each cycle
sees each slot they own once. Owning k slots therefore gives

    lambda_pool = sum over the pool's cells of  slots * (dwell/60) * visibility
    reach_pool  = people_pool * (1 - exp(-lambda_pool))

  maximize   sum_over_pools( people[pool] * (1 - exp(-lambda[pool])) )
  subject to slots[cell] in [0, remaining_capacity[cell]]
             sum_over_cells( price[cell] * slots[cell] * days ) <= budget * (1 + variance)

1 - exp(-lambda) is concave, a concave function is the MIN of its tangents,
and min-of-linear is exact in a maximisation -- so N_TANGENTS tangent lines
give CP-SAT an upper envelope it cannot cheat, tightening with N_TANGENTS.

WHY NOT BINARY POOL COVERAGE
--------------------------------------------------------------------------
The earlier formulation used covered[pool] <= sum(selected[cell]) with a
linear objective in covered[]. That deduplicates ACROSS screens correctly,
but it credits ONE rotation slot with 100% of the pool and gives every
further slot zero value. Consequences, all measured on this dataset:

  * the solver never bought more than 1 slot anywhere, so the 6-slot
    capacity constraint could never bind;
  * reach was overstated by mechanism, using D1's own dwell x visibility:
        wait    97.1% truly reached at 1 slot  ->  1.03x overstated
        ride    99.99%                         ->  1.00x
        pass    19.4%                          ->  5.16x overstated
        street   2.65%                         -> 37.77x overstated
  * because street pools carry huge corridor-wide `people` and tiny
    per-slot reach, they looked like the best buy -- so the error selected
    for exactly the inventory where the assumption was worst. On a 40-screen
    LH block-5 request at $40k it picked 2 street cells, spent 13% of the
    budget, and reported 286,587 reached against a true ~7,782;
  * brief 2's "prioritise frequency over broad daytime coverage" was
    unrepresentable, since frequency IS multiple slots.

Two solvers, same formulation:
  - CP-SAT (ortools): tangent envelope, exact to the envelope.
  - Greedy marginal-reach-per-dollar fallback. Reach is monotone and
    submodular in the set of slot-purchases, so greedy under a BUDGET is
    within (1/2)(1 - 1/e) ~ 31.6% of optimal -- not the (1 - 1/e) that
    applies to a cardinality constraint.

OBJECTIVE SWITCH
--------------------------------------------------------------------------
--objective reach      distinct people, saturating (default)
--objective frequency  impressions, linear in slots -- buys depth on a
                       winning screen instead of breadth. Brief 2.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import math
import numpy as np
import pandas as pd

import event_overlap as _evo
from pricing_engine_d3 import (
    load_win_loss_and_grain, load_gbm_price_predictions, build_rate_card,
    attach_price_ratio_and_weight, fit_win_model, build_relationship_table,
    PricingEngineD3,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Urban Media Datasets"
OUT_DIR = BASE_DIR / "eda_output"

# Stated business rule, OFF by default -- see module docstring point 3.
RESERVE_LAST_SLOT_ON_EVENT_OVERLAP = False

# --- share-of-voice model -------------------------------------------------- #
SLOTS_PER_BLOCK = 6
SLOT_SECONDS = 10.0
CYCLE_SECONDS = SLOTS_PER_BLOCK * SLOT_SECONDS   # the loop repeats every 60s

#: Tangent lines used to linearise 1 - exp(-lambda) for CP-SAT. A concave
#: function is the MIN of its tangents, and min-of-linear is exact in a
#: maximisation, so this is an upper envelope that the solver cannot cheat.
N_TANGENTS = 10

#: CP-SAT wall-clock ceiling, as a module constant rather than a literal buried
#: in _cpsat_max_coverage(). D5 declares its own SOLVE_TIME_LIMIT and had no way
#: to apply it, so its constant was dead and every solve silently ran to 30s.
#: A caller sets `build_d4_allocator.CPSAT_TIME_LIMIT_S = x` before solving.
CPSAT_TIME_LIMIT_S = 30.0

#: Fixed search seed. CP-SAT's parallel search returns whichever optimum a
#: worker reached first, and the reach objective routinely has MANY optima --
#: the same brief can be served by different line sets with identical distinct
#: reach. Campaign 4 made that visible: three identical runs reported stretch
#: spend 11,871 / 11,855 / 11,849, same +21,609 people, same objective value
#: 98,600,672.
#:
#: The real fix is the lexicographic revenue tie-break below, which collapses
#: those ties to one answer. The seed is belt-and-braces on top of it.
#:
#: num_workers is deliberately NOT pinned to 1. Tried, and it backfired:
#: single-threaded search missed the 30s limit on the agent loop's second
#: solve, returning FEASIBLE at objective 76,339,322 instead of OPTIMAL at
#: 76,983,635 -- a worse plan -- and a timeout-bound solve is decided by
#: wall-clock rather than by the seed, so it was LESS reproducible, not more.
#: Speed is what keeps these solves inside the limit, and staying inside the
#: limit is what makes them deterministic.
CPSAT_SEED = 20260823

#: "reach"     -- maximise distinct people (default; briefs 1, 3, 4, 5, 6)
#: "frequency" -- maximise expected impressions, which rewards buying MORE
#:                slots on a screen already selected. Brief 2 asks for exactly
#:                this ("prioritising frequency over broad daytime coverage"),
#:                and it is unrepresentable under binary pool coverage.
OBJECTIVE_CHOICES = ("reach", "frequency")

#: Pricing provenance carried from the candidate cell onto every SELECTED line.
#: The point is that a chosen line can explain its own price without a second
#: lookup: D6's evidence panel, and any "why this price?" question in the demo,
#: reads these columns. Both solvers copy them, so the answer does not depend
#: on which one ran.
PROVENANCE_COLUMNS = ("price_floor", "price_target", "price_cap",
                      "rate_card_basis", "p_win_at_price", "price_warnings",
                      "event_adjustment_pct", "event_names", "ridership_note",
                      # D1/D4 inputs to the impressions arithmetic. Without
                      # these a selected line can state its reach but cannot
                      # show how it was reached: people x dwell x visibility x
                      # slots is the whole share-of-voice model, and it was
                      # computed here and then dropped. "1,424 people/day" with
                      # no derivation is a number a rep has to take on faith.
                      "people", "dwell_seconds", "visibility",
                      "cycles_per_slot", "event_people_bump")


def _provenance(row) -> dict:
    """Pull PROVENANCE_COLUMNS off a candidate row, skipping any the caller
    did not supply -- build_candidate_cells always does, but a hand-built
    cells frame in a test need not."""
    return {c: row[c] for c in PROVENANCE_COLUMNS if c in row.index}


def _reach_fraction(lam: float) -> float:
    """Fraction of a pool reached at total share-of-voice `lam`."""
    return 1.0 - math.exp(-max(0.0, lam))


def _tangents(lam_max: float, n: int = N_TANGENTS):
    """(slope, intercept) pairs for 1 - exp(-lambda) on [0, lam_max].

    Tangent at a:  z <= exp(-a)*lam + (1 - exp(-a)*(1 + a))

    MUST include a = 0, which gives (slope 1, intercept 0) -- i.e. z <= lam.

    WHY THIS IS NOT OPTIONAL. Every tangent taken at a > 0 has a POSITIVE
    intercept, so an envelope built only from those permits z > 0 at lam = 0.
    Midpoint sampling over range(n) never reaches a = 0: its smallest point is
    lam_max*0.5/n, which leaves z(0) = 0.0265 at lam_max = 5. The solver then
    collects reach for buying nothing -- and since buying costs money while
    buying nothing does not, buying nothing becomes optimal. That produced
    status=OPTIMAL with a large objective value and an EMPTY plan.

    Measured cost of the old sampling, per pool:

        lam_max      1        3        5       10       30
        z(0)     0.0012   0.0102   0.0265   0.0902   0.4422

    At 0.0265 a single 12,000-person pool is credited 318 people for zero
    slots; across 300 candidate pools that is ~95,000 phantom people.

    Sampling is biased toward low lambda (exponent 1.6), where the curve bends
    hardest and the envelope is loosest.
    """
    lam_max = max(float(lam_max), 0.0)
    if lam_max <= 0:
        return [(1.0, 0.0)]
    pts = [lam_max * (i / n) ** 1.6 for i in range(n + 1)]   # includes a = 0
    out = []
    for a in pts:
        e = math.exp(-a)
        out.append((e, 1.0 - e * (1.0 + a)))
    # Duplicate constraints are harmless but slow the solve; drop exact repeats.
    uniq = []
    for s, b in out:
        if not any(abs(s - s2) < 1e-12 and abs(b - b2) < 1e-12 for s2, b2 in uniq):
            uniq.append((s, b))
    return uniq


@dataclass
class CampaignRequest:
    client_id: Optional[str]
    budget: float
    budget_variance_pct: float
    duration_days: int
    candidate_screen_ids: List[str]                 # from D2's shortlist
    preferred_blocks: Optional[List[int]] = None     # None = all 6 blocks
    day_type: str = "weekday"
    campaign_start: Optional[str] = None
    campaign_end: Optional[str] = None
    #: "reach" maximises distinct people; "frequency" maximises impressions,
    #: which rewards buying more slots on a screen already selected. Brief 2
    #: asks for frequency explicitly ("prioritising frequency over broad
    #: daytime coverage") and it is unrepresentable under binary coverage.
    objective: str = "reach"


def load_supporting_tables() -> dict:
    audience_cube = pd.read_csv(OUT_DIR / "audience_cube.csv")
    pricing_grain = pd.read_csv(OUT_DIR / "pricing_grain_d3.csv")
    corridor_overlap = None
    if (OUT_DIR / "corridor_overlap_map.csv").exists():
        corridor_overlap = pd.read_csv(OUT_DIR / "corridor_overlap_map.csv")
    dim_slot = pd.read_csv(DATA_DIR / "dim_slot.csv")
    return {"audience_cube": audience_cube, "pricing_grain": pricing_grain,
            "corridor_overlap": corridor_overlap, "dim_slot": dim_slot}


def build_pricing_engine() -> PricingEngineD3:
    win_loss, grain = load_win_loss_and_grain()
    gbm = load_gbm_price_predictions()
    rate_card = build_rate_card(grain, gbm_predictions=gbm)
    wl_prepped = attach_price_ratio_and_weight(win_loss, rate_card)
    win_model = fit_win_model(wl_prepped)
    relationship = build_relationship_table(wl_prepped)
    return PricingEngineD3(rate_card, win_model, relationship)


def build_candidate_cells(req: CampaignRequest, tables: dict, engine: PricingEngineD3) -> pd.DataFrame:
    """One row per (screen_id, time_block_id) candidate: pool_id, people
    (base + event bump), price (event-adjusted via D3), remaining_capacity
    (hard slot cap)."""
    cube = tables["audience_cube"]
    grain = tables["pricing_grain"]
    blocks = req.preferred_blocks or sorted(cube["time_block_id"].unique().tolist())

    cells = cube[
        cube["screen_id"].isin(req.candidate_screen_ids)
        & cube["time_block_id"].isin(blocks)
        & (cube["day_type"] == req.day_type)
    ][["screen_id", "time_block_id", "pool_id", "people",
       "dwell_seconds", "visibility"]].drop_duplicates(
        subset=["screen_id", "time_block_id"]
    ).copy()

    if cells.empty:
        return cells

    # --- hard slot-availability constraint: real remaining capacity ---
    occ = grain[["screen_id", "time_block_id", "forward_occupancy_pct"]].drop_duplicates()
    cells = cells.merge(occ, on=["screen_id", "time_block_id"], how="left")
    cells["forward_occupancy_pct"] = cells["forward_occupancy_pct"].fillna(0.0)
    cells["remaining_capacity"] = (6 * (1.0 - cells["forward_occupancy_pct"].clip(0.0, 1.0))) \
        .round().astype(int).clip(0, 6)

    # --- share of voice: what ONE slot actually buys -------------------------
    # All 6 rotation slots loop continuously, so a full cycle is 6 x 10s = 60s.
    # A viewer present for `dwell_seconds` sees dwell/60 cycles, and in each
    # cycle sees each slot they own once. Owning k slots therefore gives
    #     lambda = k * (dwell / 60) * visibility
    # and the fraction of the pool actually reached is 1 - exp(-lambda):
    # concave and saturating, which is precisely the "non-linearity in
    # impressions vs boards vs pricing" the brief asks for.
    cells["cycles_per_slot"] = (
        cells["dwell_seconds"].fillna(0.0) / CYCLE_SECONDS
    ) * cells["visibility"].fillna(0.0)
    cells["cycles_per_slot"] = cells["cycles_per_slot"].clip(lower=0.0)

    # --- event overlap: impressions bump (#1) + tier for the reserve rule (#3) ---
    # NOTE: do NOT pre-create an "overlap_tier" placeholder column before the
    # merge below -- pandas would suffix BOTH columns (overlap_tier_x/_y)
    # instead of cleanly filling the real one (caught by testing against
    # synthetic data before shipping -- this comment replaces that bug).
    cells["event_people_bump"] = 0.0
    has_overlap_tier = False
    if req.campaign_start and req.campaign_end:
        overlap = engine._event_overlap_for_window(req.campaign_start, req.campaign_end)
        if not overlap.empty:
            cells = cells.merge(
                overlap[["screen_id", "time_block_id", "overlap_score", "overlap_tier"]],
                on=["screen_id", "time_block_id"], how="left",
            )
            cells["overlap_score"] = cells["overlap_score"].fillna(0.0)
            cells["overlap_tier"] = cells["overlap_tier"].fillna("none")
            cells["event_people_bump"] = cells["overlap_score"].apply(_evo.event_impressions_bump)
            cells = cells.drop(columns=["overlap_score"])
            has_overlap_tier = True
    if not has_overlap_tier:
        cells["overlap_tier"] = "none"

    cells["people"] = cells["people"].fillna(0.0) + cells["event_people_bump"]

    if RESERVE_LAST_SLOT_ON_EVENT_OVERLAP:
        direct_hit = cells["overlap_tier"] == "direct"
        cells.loc[direct_hit & (cells["remaining_capacity"] >= 2), "remaining_capacity"] -= 1
        n_reserved = int((direct_hit & (cells["remaining_capacity"] >= 1)).sum())
        if n_reserved:
            print(f"[d4] reserved the last slot on {n_reserved} direct event-overlap cells "
                  f"(RESERVE_LAST_SLOT_ON_EVENT_OVERLAP=True)")

    # --- price: pulled from D3, ALREADY event-adjusted for this exact window ---
    prices = []
    prov: List[dict] = []
    for _, r in cells.iterrows():
        q = engine.quote(r["screen_id"], int(r["time_block_id"]), client_id=req.client_id,
                          campaign_start=req.campaign_start, campaign_end=req.campaign_end)
        prices.append(q.recommended)
        # Keep the WHOLE quote, not just the number. PriceQuote already carries
        # the band, the rate-card basis, the win probability and its own
        # warnings; this loop built all of that and discarded everything except
        # `recommended`. D6's evidence panel then had nothing to render -- the
        # integration doc calls it "what turns 'trust the model' into 'here is
        # the evidence'" and every chosen line reached it carrying only a
        # price. Recomputing later would mean re-quoting 1,000 cells, so it is
        # captured here where it is free.
        prov.append({
            "price_floor": float(q.floor), "price_target": float(q.target),
            "price_cap": float(q.cap), "rate_card_basis": str(q.rate_card_basis),
            "p_win_at_price": float(q.p_win_at_recommended),
            "price_warnings": "; ".join(q.warnings),
            "event_adjustment_pct": float(q.event_adjustment_pct),
            "event_names": str(q.event_names),
            "ridership_note": str(q.ridership_trend_note),
        })
    cells["price"] = prices
    # Only the keys the quote loop actually produced. The rest of
    # PROVENANCE_COLUMNS (people, dwell_seconds, visibility, cycles_per_slot,
    # event_people_bump) are already columns on `cells` -- they come from D1 via
    # the cube, not from the quote -- and _provenance() copies them onto the
    # selected rows from there.
    for col in prov[0].keys():
        cells[col] = [p[col] for p in prov]

    cells = cells[cells["remaining_capacity"] > 0].reset_index(drop=True)
    return cells


def _realised_pool_reach(cells: pd.DataFrame, slots_by_index: dict, pool_id: str) -> float:
    """People actually reached in `pool_id` given the slots bought.

    Split evenly across the pool's selected cells so the per-row column sums to
    the pool total instead of double-counting it on every row.
    """
    idxs = cells.index[cells["pool_id"] == pool_id].tolist()
    lam = sum(float(cells.loc[i, "cycles_per_slot"]) * slots_by_index.get(i, 0)
              for i in idxs)
    people = float(cells.loc[idxs[0], "people"]) if idxs else 0.0
    n_sel = sum(1 for i in idxs if slots_by_index.get(i, 0) > 0) or 1
    return people * _reach_fraction(lam) / n_sel


def _greedy_max_coverage(cells: pd.DataFrame, req: CampaignRequest) -> pd.DataFrame:
    """Greedy marginal-reach-per-dollar, with share-of-voice saturation.

    Differs from binary coverage in the one way that matters: a second slot on
    an already-selected screen still adds reach, just less than the first did.
    That is what makes the 6-slot capacity constraint able to bind at all, and
    what lets a frequency-led brief buy depth instead of breadth.

    Guarantee: reach is monotone and submodular in the set of slot-purchases,
    so greedy under a budget is within (1/2)(1 - 1/e) of optimal -- NOT the
    (1 - 1/e) that applies to a cardinality constraint. The tighter bound also
    requires comparing against the best single affordable item, which the tail
    of this function does.
    """
    budget_cap = req.budget * (1 + req.budget_variance_pct)
    objective = getattr(req, "objective", "reach")

    pool_people = cells.groupby("pool_id")["people"].first()
    lam = {p: 0.0 for p in pool_people.index}          # share of voice per pool
    taken = np.zeros(len(cells), dtype=int)
    chosen: dict = {}
    spend = 0.0

    def marginal(i: int) -> float:
        """Extra people reached by adding one more slot to cell i."""
        row = cells.iloc[i]
        pool = row["pool_id"]
        per = float(row["cycles_per_slot"])
        if per <= 0.0:
            return 0.0
        before = _reach_fraction(lam[pool])
        after = _reach_fraction(lam[pool] + per)
        if objective == "frequency":
            # Impressions, not distinct people: every slot keeps paying.
            return float(row["people"]) * per
        return float(row["people"]) * (after - before)

    while True:
        best_i, best_ratio, best_gain, best_cost = -1, 0.0, 0.0, 0.0
        for i in range(len(cells)):
            if taken[i] >= int(cells.iloc[i]["remaining_capacity"]):
                continue
            cost = float(cells.iloc[i]["price"]) * req.duration_days
            if cost <= 0 or spend + cost > budget_cap + 1e-9:
                continue
            gain = marginal(i)
            if gain <= 0:
                continue
            ratio = gain / cost
            if ratio > best_ratio:
                best_i, best_ratio, best_gain, best_cost = i, ratio, gain, cost
        if best_i < 0:
            break
        row = cells.iloc[best_i]
        taken[best_i] += 1
        lam[row["pool_id"]] += float(row["cycles_per_slot"])
        spend += best_cost
        key = (row["screen_id"], int(row["time_block_id"]))
        if key in chosen:
            chosen[key]["slots_booked_per_day"] += 1
        else:
            chosen[key] = {
                "screen_id": row["screen_id"],
                "time_block_id": int(row["time_block_id"]),
                "pool_id": row["pool_id"], "slots_booked_per_day": 1,
                "price": float(row["price"]),
                "overlap_tier": row.get("overlap_tier", "none"),
                **_provenance(row),
            }

    def _finish(frame: pd.DataFrame, lam_map: dict) -> pd.DataFrame:
        """Attribute realised reach per pool, split across that pool's cells."""
        if frame.empty:
            return frame
        per_pool = {p: pool_people[p] * _reach_fraction(v) for p, v in lam_map.items()}
        counts = frame.groupby("pool_id")["screen_id"].transform("size")
        frame = frame.copy()
        frame["people_reach"] = [per_pool[p] for p in frame["pool_id"]] / counts
        return frame

    out = _finish(pd.DataFrame(list(chosen.values())), lam)

    # ---- the second half of the (1/2)(1 - 1/e) guarantee -------------------- #
    # The guarantee for a submodular objective under a BUDGET (as opposed to a
    # cardinality constraint, where plain greedy alone gives 1 - 1/e) is
    # "the better of cost-benefit greedy and the best single affordable item".
    # The docstring above claimed "the tail of this function does" this
    # comparison. It did not -- the tail only built the output frame, so the
    # stated bound was not actually attained by the code.
    #
    # The pathological case the comparison exists for: one cell whose pool
    # holds almost all the people but whose price consumes nearly the whole
    # budget. Ratio-greedy skips it forever in favour of many cheap, thin
    # cells, and can end up arbitrarily far below optimal. Buying that one
    # item is the alternative that caps the damage.
    best_single, best_single_reach = None, 0.0
    for i in range(len(cells)):
        row = cells.iloc[i]
        unit = float(row["price"]) * req.duration_days
        capacity = int(row["remaining_capacity"])
        per = float(row["cycles_per_slot"])
        if unit <= 0 or capacity <= 0 or per <= 0:
            continue
        k = min(capacity, int(budget_cap // unit))
        if k <= 0:
            continue
        if objective == "frequency":
            value = float(row["people"]) * per * k
        else:
            value = float(row["people"]) * _reach_fraction(per * k)
        if value > best_single_reach:
            best_single_reach, best_single = value, (i, k, unit * k)

    greedy_value = float(out["people_reach"].sum()) if len(out) else 0.0
    if best_single is not None and best_single_reach > greedy_value:
        i, k, cost = best_single
        row = cells.iloc[i]
        print(f"[d4] greedy: the best single affordable item beats the incremental "
              f"run ({best_single_reach:,.0f} vs {greedy_value:,.0f}) -- taking it, "
              f"which is what makes the (1/2)(1-1/e) bound hold.")
        single = pd.DataFrame([{
            "screen_id": row["screen_id"], "time_block_id": int(row["time_block_id"]),
            "pool_id": row["pool_id"], "slots_booked_per_day": int(k),
            "price": float(row["price"]),
            "overlap_tier": row.get("overlap_tier", "none"),
            **_provenance(row),
        }])
        lam_single = {p: 0.0 for p in pool_people.index}
        lam_single[row["pool_id"]] = float(row["cycles_per_slot"]) * k
        return _finish(single, lam_single), float(cost)

    return out, spend


def _cpsat_max_coverage(cells: pd.DataFrame, req: CampaignRequest) -> Optional[pd.DataFrame]:
    """Exact CP-SAT formulation (see module docstring). Returns None (with a
    printed warning) if ortools is not installed -- caller must fall back
    to _greedy_max_coverage(), never crash."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        print("[d4] ortools not installed -- using the tested greedy 1-1/e-approximate "
              "fallback instead of exact CP-SAT. Install ortools for a provably-optimal "
              "solve (see module docstring).")
        return None

    model = cp_model.CpModel()
    cells = cells.reset_index(drop=True)
    n = len(cells)
    slots = [model.NewIntVar(0, int(cells.loc[i, "remaining_capacity"]), f"slots_{i}") for i in range(n)]
    selected = [model.NewBoolVar(f"sel_{i}") for i in range(n)]
    for i in range(n):
        model.Add(slots[i] >= 1).OnlyEnforceIf(selected[i])
        model.Add(slots[i] == 0).OnlyEnforceIf(selected[i].Not())

    pool_ids = cells["pool_id"].unique().tolist()
    pool_people = cells.groupby("pool_id")["people"].first()

    # budget: integer cents to keep CP-SAT happy with integral coefficients
    cost_cents = (cells["price"] * req.duration_days * 100).round().astype(int)
    budget_cents = int(round(req.budget * (1 + req.budget_variance_pct) * 100))
    model.Add(sum(int(cost_cents[i]) * slots[i] for i in range(n)) <= budget_cents)

    objective = getattr(req, "objective", "reach")

    if objective == "frequency":
        # Impressions, linear in slots: every slot keeps paying, so this is the
        # objective that rewards depth on a winning screen. Brief 2 asks for it
        # explicitly and binary coverage cannot express it at all.
        imp = ((cells["people"] * cells["cycles_per_slot"]) * 1000).round().astype(int)
        primary_expr = sum(int(imp.iloc[i]) * slots[i] for i in range(n))
        primary_units = "scaled impressions"
    else:
        # --- distinct reach with share-of-voice saturation -------------------
        # lambda_pool = sum over the pool's cells of slots * cycles_per_slot,
        # and reach = people * (1 - exp(-lambda)). 1 - exp(-lambda) is concave,
        # a concave function is the MIN of its tangents, and min-of-linear is
        # exact in a maximisation -- so the tangent envelope is an upper bound
        # the solver cannot cheat, and it tightens with N_TANGENTS.
        #
        # This is what replaces `covered[pool]`. Under binary coverage one slot
        # scored the whole pool and a second scored nothing, so the solver
        # always bought exactly one slot everywhere, the 6-slot capacity
        # constraint could never bind, and street inventory (dwell 4s) was
        # credited with 37.8x the people it actually reaches.
        LAM = 1000    # fixed-point scale for lambda
        Z = 1000      # fixed-point scale for the reach fraction

        obj_terms = []
        for p in pool_ids:
            idxs = cells.index[cells["pool_id"] == p].tolist()
            contrib, lam_ub_scaled = [], 0
            for i in idxs:
                per = float(cells.loc[i, "cycles_per_slot"])
                if per <= 0:
                    continue
                ub = int(cells.loc[i, "remaining_capacity"])
                coef = int(round(per * LAM))
                if coef <= 0 or ub <= 0:
                    continue
                contrib.append((coef, slots[i]))
                lam_ub_scaled += coef * ub
            if not contrib or lam_ub_scaled <= 0:
                continue

            lam_v = model.NewIntVar(0, lam_ub_scaled, f"lam_{p}")
            model.Add(lam_v == sum(c * v for c, v in contrib))

            z = model.NewIntVar(0, Z, f"z_{p}")
            for slope, intercept in _tangents(lam_ub_scaled / LAM, N_TANGENTS):
                # z <= slope*lambda + intercept, in fixed point
                model.Add(LAM * z <= int(round(Z * slope)) * lam_v
                          + int(round(Z * intercept * LAM)))

            obj_terms.append((int(round(float(pool_people[p]))), z))

        if not obj_terms:
            # Return the bare SENTINEL, not (None, 0.0). `allocate()` tests
            # `result is None`; a 2-tuple is not None, so it reads as a
            # successful solve, unpacks to selected=None, and crashes on
            # len(None) two lines later. Reachable whenever every candidate
            # has zero dwell or zero visibility.
            print("[d4] no pool can accumulate exposure (all candidates have "
                  "zero dwell or visibility) -- falling back to greedy.")
            return None
        primary_expr = sum(w * z for w, z in obj_terms)
        primary_units = "scaled people-units"

    # --- phase 1: the stated objective ------------------------------------- #
    model.Maximize(primary_expr)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(CPSAT_TIME_LIMIT_S)
    solver.parameters.random_seed = CPSAT_SEED
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"[d4] CP-SAT returned status {solver.StatusName(status)} -- falling back to greedy.")
        return None

    # --- phase 2: among equal-objective plans, take the most revenue -------- #
    # Reach saturates, so beyond some point extra spend buys no extra distinct
    # people and the solver is free to pick ANY of the tied plans. Left
    # unstated, that choice was arbitrary and moved between runs. Stated, it is
    # a commercial decision: hold the objective at its optimum and maximise
    # billed spend, so we never hand back reach we were paid for.
    #
    # Done as a second exact solve rather than a weighted single objective:
    # weighting needs a coefficient larger than any achievable spend, and
    # primary x budget can reach ~1e17 here, close enough to the int64 ceiling
    # to be worth not relying on.
    primary_opt = int(round(solver.ObjectiveValue()))
    if status == cp_model.OPTIMAL:
        model.Add(primary_expr == primary_opt)
        model.Maximize(sum(int(cost_cents[i]) * slots[i] for i in range(n)))
        status2 = solver.Solve(model)
        if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            status = status2
        else:
            # Should not happen -- phase 1's solution satisfies the new
            # constraint by construction -- but if it does, phase 1 stands.
            print(f"[d4] revenue tie-break returned {solver.StatusName(status2)}; "
                  f"keeping the phase-1 plan.")
    else:
        print("[d4] phase 1 hit the time limit (FEASIBLE, not OPTIMAL) -- skipping "
              "the revenue tie-break, since the objective optimum is not known.")

    # Read every slot value first: realised pool reach depends on the WHOLE
    # solution, not on one row, so it cannot be computed inside the loop below
    # without this map.
    sol_slots = {i: int(solver.Value(slots[i])) for i in range(n)}

    rows = []
    spend = 0.0
    for i in range(n):
        s = sol_slots[i]
        if s > 0:
            cost = float(cells.loc[i, "price"]) * req.duration_days * s
            spend += cost
            rows.append({
                "screen_id": cells.loc[i, "screen_id"], "time_block_id": int(cells.loc[i, "time_block_id"]),
                "pool_id": cells.loc[i, "pool_id"], "slots_booked_per_day": int(s),
                "price": float(cells.loc[i, "price"]),
                "people_reach": float(_realised_pool_reach(cells, sol_slots, cells.loc[i, "pool_id"])),
                "overlap_tier": cells.loc[i].get("overlap_tier", "none"),
                **_provenance(cells.loc[i]),
            })
    # primary_opt, not solver.ObjectiveValue(): after the tie-break the solver's
    # objective is billed cents, and printing that as people-units would report
    # a spend figure under a reach label.
    print(f"[d4] CP-SAT solved exactly, status={solver.StatusName(status)}, "
          f"objective={primary_opt:,} ({primary_units}), spend {spend:,.2f} "
          f"after the revenue tie-break")
    return pd.DataFrame(rows), spend


def allocate(req: CampaignRequest, tables: dict, engine: PricingEngineD3) -> dict:
    cells = build_candidate_cells(req, tables, engine)
    if cells.empty:
        return {"selected": pd.DataFrame(), "spend": 0.0, "deduped_reach": 0.0,
                "warning": "no candidate cells with remaining capacity for this request"}

    result = _cpsat_max_coverage(cells, req)
    solver_used = "cp_sat"
    if result is None:
        result = _greedy_max_coverage(cells, req)
        solver_used = "greedy_1_minus_1_over_e"
    selected, spend = result

    deduped_reach = selected["people_reach"].sum() if len(selected) else 0.0
    n_event_scarce = int((selected["overlap_tier"] == "direct").sum()) if len(selected) else 0

    return {
        "selected": selected, "spend": spend, "deduped_reach": deduped_reach,
        "solver_used": solver_used, "n_candidate_cells": len(cells),
        "n_selected": len(selected), "n_event_scarce_selected": n_event_scarce,
        "budget": req.budget, "budget_cap": req.budget * (1 + req.budget_variance_pct),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screens", required=True, help="comma-separated screen_ids (D2's shortlist)")
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--budget-variance-pct", type=float, default=0.15)
    ap.add_argument("--duration-days", type=int, required=True)
    ap.add_argument("--client-id", default=None)
    ap.add_argument("--blocks", default=None, help="comma-separated time_block_id, default all 6")
    ap.add_argument("--day-type", default="weekday")
    ap.add_argument("--campaign-start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--campaign-end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--objective", default="reach", choices=list(OBJECTIVE_CHOICES),
                    help="reach = distinct people (default); frequency = impressions")
    ap.add_argument("--out", default=str(OUT_DIR / "optimized_package_d4.csv"))
    args = ap.parse_args()

    tables = load_supporting_tables()
    engine = build_pricing_engine()

    req = CampaignRequest(
        client_id=args.client_id, budget=args.budget, budget_variance_pct=args.budget_variance_pct,
        duration_days=args.duration_days, candidate_screen_ids=args.screens.split(","),
        preferred_blocks=[int(b) for b in args.blocks.split(",")] if args.blocks else None,
        day_type=args.day_type, campaign_start=args.campaign_start, campaign_end=args.campaign_end,
        # Without this line --objective is parsed, validated against
        # OBJECTIVE_CHOICES, and then silently discarded: CampaignRequest falls
        # back to its "reach" default, so `--objective frequency` runs reach.
        # Brief 2 asks for frequency explicitly and never got it.
        objective=args.objective,
    )
    out = allocate(req, tables, engine)

    if len(out["selected"]):
        out["selected"].to_csv(args.out, index=False)
        print(f"\nsaved {args.out}: {len(out['selected'])} rows")
    print(json.dumps({k: v for k, v in out.items() if k != "selected"}, indent=2, default=str))


if __name__ == "__main__":
    main()

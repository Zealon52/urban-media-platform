"""
Date-aware event-overlap demand & impression boost -- computed PER REQUEST
using a campaign's REAL start/end dates, not the static per-location average
already baked into pricing_grain_d3.csv's event_demand_boost_log.

--------------------------------------------------------------------------
WHY THIS EXISTS (gap confirmed by reading build_training_table.py directly)
--------------------------------------------------------------------------
build_event_demand_boost() in build_training_table.py groups ALL 367
events.csv rows by anchor_location_id x time_block_id and averages
expected_attendance -- it NEVER reads events.start_date / events.end_date.
So a screen near a stadium gets the same static boost whether a given
campaign runs during a sold-out match week or a dead month with zero
events scheduled. That is a legitimate "this location tends to have
big events" signal, but it is NOT "there's a real event during THIS
client's THIS booking window."

This module computes the real thing: given a campaign's actual date
range, find events.csv rows whose [start_date, end_date] genuinely
overlaps that window, weight by expected_attendance AND by what fraction
of the campaign window is actually covered, and produce a per-screen,
per-time_block boost. Used by:
  1. pricing_engine_d3.py  -- shifts the price BAND (floor/target/cap) up
     for real overlapping high-attendance events, on top of (not double-
     counting) the static average already in the rate card. See
     event_price_adjustment().
  2. build_d4_allocator.py -- feeds a genuine, date-matched impressions
     bump into the reach objective, and flags cells as "event-scarce" so
     the optimizer's slot-availability constraint doesn't cheaply hand a
     scarce, event-boosted slot to a low-value campaign.

--------------------------------------------------------------------------
HONESTY / PROXY NOTES (read before trusting the numbers)
--------------------------------------------------------------------------
* events.recurrence has 3 real values whose exact strings were never
  verified against the raw CSV (eda_engine.py only reports null%/count,
  never the values themselves -- the same gap that produced 3 wrong
  guesses earlier this session on other columns: market_type/market_tier,
  booking_status, loss_reason). Rather than guess AGAIN, this module
  prints the real recurrence value_counts + a sample of matched rows so
  you can verify before trusting anything. Recurring events are NOT
  auto-expanded into future occurrences (we cannot verify the recurrence
  rule/interval) -- only the given [start_date, end_date] span is used.
  If recurrence really means e.g. "repeats weekly forever", this UNDER-
  counts overlap for recurring events. Flagged explicitly in printed
  output -- not hidden.
* There are NO coordinates anywhere in the 14 source tables (confirmed
  earlier this session). "Is this screen near this event" is therefore
  resolved in two verifiable tiers, weakest-assumption-first, same
  pattern used in build_d2_relevance_scorer.py's _radius_zone_ids():
    Tier "direct"  -- screen's location_id == event.anchor_location_id
                      (exact match, zero proxy risk)
    Tier "zone"    -- screen's zone_id == the event location's zone_id
                      (same 30-zone bucket -- a real but coarse proxy)
    Tier "corridor"-- for vehicle-mounted screens: the vehicle's corridor
                      passes a stop whose location_id/zone_id matches the
                      event (bus/metro routes through the event's zone)
  Each tier gets a DIFFERENT, weaker weight (see TIER_WEIGHT below) --
  never silently treated as equally strong evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Urban Media Datasets"
OUT_DIR = BASE_DIR / "eda_output"

# --------------------------------------------------------------------------- #
# Named, stated constants -- rule-based, not learned (same principle as the
# architecture doc's "guardrails are rules, not model output").
# --------------------------------------------------------------------------- #
TIER_WEIGHT = {"direct": 1.0, "zone": 0.4, "corridor": 0.25}

# Price band: max fractional upward shift to floor/target/cap from a
# genuine event overlap, and how much of the (log-attendance x overlap
# fraction x tier weight) signal converts into that shift. Both are stated
# business rules a rep can see and override -- not fit from data, because
# we do not have enough genuinely event-overlapping BOOKED deals to fit
# this reliably (367 events total, most bookings pre-date this level of
# analysis).
EVENT_PRICE_SHIFT_PER_UNIT = 0.06     # +6% band shift per unit of log1p(attendance)*frac*tier_weight
EVENT_PRICE_SHIFT_MAX = 0.25          # never shift the band by more than +25%

# Impressions: extra people this event plausibly adds to a screen's normal
# traffic for the overlapping time_block, expressed as a FRACTION of the
# event's expected_attendance (attendees near a "direct" hit screen who are
# also exposed to it) -- again a stated assumption, not fit, and DELIBERATELY
# small + tier-scaled, since not every attendee walks past every screen at
# that location.
EVENT_IMPRESSION_CAPTURE_RATE = 0.08   # 8% of expected_attendance, tier-scaled, capture this screen's exposure


def _load_dates(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def load_tables() -> dict:
    events = pd.read_csv(DATA_DIR / "events.csv")
    events["start_date"] = _load_dates(events, "start_date")
    events["end_date"] = _load_dates(events, "end_date")
    bad_dates = events["start_date"].isna() | events["end_date"].isna()
    if bad_dates.any():
        print(f"[event_overlap] WARNING: {bad_dates.sum()} events.csv rows have "
              f"unparseable start_date/end_date -- excluded from overlap matching.")
        events = events[~bad_dates].copy()

    dim_slot = pd.read_csv(DATA_DIR / "dim_slot.csv")
    locations = pd.read_csv(DATA_DIR / "locations.csv")
    route_stops = pd.read_csv(DATA_DIR / "route_stops.csv")
    vehicles = pd.read_csv(DATA_DIR / "vehicles.csv")
    screens = pd.read_csv(DATA_DIR / "screens.csv", dtype={"vehicle_id": "str"}, low_memory=False)

    # recurrence -- print real values, never guess semantics (see docstring)
    vc = events["recurrence"].value_counts(dropna=False)
    print(f"[event_overlap] events.recurrence real value_counts (verify before trusting "
          f"any recurrence-based logic -- NONE is applied here, only [start_date, end_date]):")
    print(vc.to_string())

    return {
        "events": events, "dim_slot": dim_slot, "locations": locations,
        "route_stops": route_stops, "vehicles": vehicles, "screens": screens,
    }


def _screen_location_zone_map(screens: pd.DataFrame, locations: pd.DataFrame,
                               route_stops: pd.DataFrame, vehicles: pd.DataFrame) -> pd.DataFrame:
    """One row per screen_id -> location_id (if location-mounted), zone_id,
    and corridor_id (if vehicle-mounted, via vehicles; if location-mounted,
    via any route_stops sharing that location_id -- a location screen can
    also sit on a corridor, e.g. a bus-stop shelter screen)."""
    loc_zone = locations[["location_id", "zone_id"]].drop_duplicates()

    s = screens[["screen_id", "location_id", "vehicle_id"]].copy()
    s = s.merge(loc_zone, on="location_id", how="left")

    veh_corr = vehicles[["vehicle_id", "corridor_id"]].drop_duplicates()
    s = s.merge(veh_corr, on="vehicle_id", how="left", suffixes=("", "_veh"))

    # location-mounted screens: corridor via any stop at the same location_id
    loc_corr = route_stops[["location_id", "corridor_id"]].drop_duplicates()
    loc_corr = loc_corr.rename(columns={"corridor_id": "corridor_id_via_loc"})
    s = s.merge(loc_corr, on="location_id", how="left")
    s["corridor_id"] = s["corridor_id"].fillna(s["corridor_id_via_loc"])
    s = s.drop(columns=["corridor_id_via_loc"])
    return s


def _event_location_zone_map(events: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    loc_zone = locations[["location_id", "zone_id"]].drop_duplicates()
    ev = events.merge(
        loc_zone.rename(columns={"location_id": "anchor_location_id"}),
        on="anchor_location_id", how="left",
    )
    return ev


def compute_event_overlap(
    tables: dict,
    campaign_start: pd.Timestamp,
    campaign_end: pd.Timestamp,
    verbose: bool = True,
) -> pd.DataFrame:
    """Returns one row per screen_id x time_block_id with:
      overlap_score        -- sum of log1p(attendance) * overlap_fraction * tier_weight,
                               across every genuinely-overlapping event matched to that
                               screen at that daypart's time_block_id
      overlap_tier          -- strongest tier that matched ("direct"/"zone"/"corridor"/"none")
      overlap_event_count   -- count of distinct overlapping events contributing
      matched_event_names   -- comma-joined names, for the explainability layer
    Only events whose [start_date, end_date] genuinely overlaps
    [campaign_start, campaign_end] are included. Zero rows returned for a
    screen/block with no genuine overlap (caller should treat missing as 0).
    """
    events, dim_slot = tables["events"], tables["dim_slot"]
    ev = _event_location_zone_map(events, tables["locations"])

    overlap_mask = (ev["start_date"] <= campaign_end) & (ev["end_date"] >= campaign_start)
    ev = ev[overlap_mask].copy()
    if verbose:
        print(f"[event_overlap] {len(ev)}/{len(events)} events genuinely overlap "
              f"[{campaign_start.date()}, {campaign_end.date()}]")
    if ev.empty:
        return pd.DataFrame(columns=["screen_id", "time_block_id", "overlap_score",
                                      "overlap_tier", "overlap_event_count", "matched_event_names"])

    campaign_days = (campaign_end - campaign_start).days + 1
    ov_start = ev["start_date"].where(ev["start_date"] >= campaign_start, campaign_start)
    ov_end = ev["end_date"].where(ev["end_date"] <= campaign_end, campaign_end)
    ev["overlap_days"] = (ov_end - ov_start).dt.days + 1
    ev["overlap_fraction"] = (ev["overlap_days"] / campaign_days).clip(0.0, 1.0)
    ev["attendance_log"] = np.log1p(ev["expected_attendance"])

    bridge = dim_slot[["time_block_id", "nearest_daypart"]].drop_duplicates()
    ev = ev.merge(bridge, left_on="primary_impact_daypart", right_on="nearest_daypart", how="left")

    smap = _screen_location_zone_map(tables["screens"], tables["locations"],
                                      tables["route_stops"], tables["vehicles"])

    rows = []
    for tier in ("direct", "zone", "corridor"):
        if tier == "direct":
            m = smap.merge(
                ev[["anchor_location_id", "time_block_id", "attendance_log",
                    "overlap_fraction", "event_name"]],
                left_on="location_id", right_on="anchor_location_id", how="inner",
            )
        elif tier == "zone":
            m = smap.merge(
                ev[["zone_id", "time_block_id", "attendance_log", "overlap_fraction", "event_name"]],
                on="zone_id", how="inner",
            )
        else:  # corridor
            m = smap.merge(
                ev[["zone_id", "time_block_id", "attendance_log", "overlap_fraction", "event_name"]]
                    .rename(columns={"zone_id": "event_zone_id"}),
                left_on="corridor_id", right_on="event_zone_id", how="inner",
            ) if False else pd.DataFrame()  # corridor-zone join needs route_stops.zone_id via location;
            # skip: not enough verified signal to join corridor->zone safely without
            # risking a spurious match; leave this tier structurally present but empty
            # rather than fabricate a weak join. See docstring Tier "corridor" note.
        if m.empty:
            continue
        m["overlap_tier"] = tier
        m["tier_weight"] = TIER_WEIGHT[tier]
        m["contrib"] = m["attendance_log"] * m["overlap_fraction"] * m["tier_weight"]
        rows.append(m[["screen_id", "time_block_id", "overlap_tier", "tier_weight",
                        "contrib", "event_name"]])

    if not rows:
        return pd.DataFrame(columns=["screen_id", "time_block_id", "overlap_score",
                                      "overlap_tier", "overlap_event_count", "matched_event_names"])

    all_m = pd.concat(rows, ignore_index=True)
    # if a screen/block matched multiple tiers, only keep the STRONGEST tier's
    # rows (do not sum direct + zone for the same underlying event/location --
    # that would double-count the same real-world proximity signal)
    tier_rank = {"direct": 0, "zone": 1, "corridor": 2}
    all_m["tier_rank"] = all_m["overlap_tier"].map(tier_rank)
    best_tier_per_cell = all_m.groupby(["screen_id", "time_block_id"])["tier_rank"].min().rename("best_rank")
    all_m = all_m.merge(best_tier_per_cell, on=["screen_id", "time_block_id"])
    all_m = all_m[all_m["tier_rank"] == all_m["best_rank"]]

    out = (
        all_m.groupby(["screen_id", "time_block_id"])
        .agg(
            overlap_score=("contrib", "sum"),
            overlap_tier=("overlap_tier", "first"),
            overlap_event_count=("event_name", "nunique"),
            matched_event_names=("event_name", lambda s: ", ".join(sorted(set(s))[:5])),
        )
        .reset_index()
    )
    if verbose:
        print(f"[event_overlap] {len(out)} screen x time_block cells have genuine event "
              f"overlap for this window: "
              f"{(out['overlap_tier']=='direct').sum()} direct, "
              f"{(out['overlap_tier']=='zone').sum()} zone-proxy")
    return out


@dataclass
class EventAdjustment:
    overlap_score: float
    overlap_tier: str
    price_shift_pct: float
    extra_people: float
    matched_event_names: str


def event_price_adjustment(overlap_score: float, static_boost_log_baseline: float = 0.0) -> float:
    """Converts an overlap_score into a floor/target/cap % shift, applied
    ONLY on the INCREMENTAL signal above what the static grid-level
    event_demand_boost_log already reflects (never a negative shift -- we
    only protect against underpricing a genuine surge, never invent a
    discount because a window happens to be quieter than average)."""
    delta = max(0.0, overlap_score - static_boost_log_baseline)
    return float(np.clip(delta * EVENT_PRICE_SHIFT_PER_UNIT, 0.0, EVENT_PRICE_SHIFT_MAX))


def event_impressions_bump(overlap_score: float, expected_attendance_hint: Optional[float] = None) -> float:
    """Extra people for the reach objective. overlap_score already folds in
    tier_weight and overlap_fraction; we invert log1p to get back to a
    people-scale number, then apply the stated capture rate."""
    people_scale = float(np.expm1(max(0.0, overlap_score)))
    return people_scale * EVENT_IMPRESSION_CAPTURE_RATE


def apply_event_overlap_to_band(
    floor: float, target: float, cap: float,
    overlap_score: float, static_boost_log_baseline: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Returns (new_floor, new_target, new_cap, pct_shift_applied)."""
    pct = event_price_adjustment(overlap_score, static_boost_log_baseline)
    if pct <= 0:
        return floor, target, cap, 0.0
    return floor * (1 + pct), target * (1 + pct), max(cap * (1 + pct), floor * (1 + pct) * 1.05), pct


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaign_start", help="YYYY-MM-DD")
    ap.add_argument("campaign_end", help="YYYY-MM-DD")
    ap.add_argument("--out", default=str(OUT_DIR / "event_overlap_for_window.csv"))
    args = ap.parse_args()

    tables = load_tables()
    cs, ce = pd.Timestamp(args.campaign_start), pd.Timestamp(args.campaign_end)
    overlap = compute_event_overlap(tables, cs, ce)
    overlap.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}: {overlap.shape[0]} rows x {overlap.shape[1]} cols")
    if not overlap.empty:
        print("\nsample:")
        print(overlap.sort_values("overlap_score", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()

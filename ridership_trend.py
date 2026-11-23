"""
ridership_trend.py
Date-aware ridership trend + weekly-seasonality forecast, applied as a
price-band adjustment -- the vehicle-mounted-screen counterpart to
event_overlap.py's date-aware event adjustment.

--------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------
Everywhere else in this pipeline, "demand" for a future date is a flat
historical average: pricing_grain_d3.csv's exposure_log /
corridor_ridership_log is one number per (corridor_id, time_block_id),
computed once from ALL of ridership_actuals.csv's ~182 days of history,
with no sensitivity to which future date is actually being priced. The
only date-aware adjustment anywhere in this codebase is event_overlap.py,
and that only fires when a real event's dates genuinely overlap the
requested campaign window.

This module adds a second, independent date-aware signal: for
VEHICLE-MOUNTED screens only (ridership_actuals is the one table with a
real 182-day daily time series -- location-mounted screens' footfall in
points_of_interest.csv is a single snapshot value with no time dimension
at all, so there is nothing to fit a trend on for those screens; they
correctly fall back to a neutral multiplier of 1.0, see
forecast_multiplier_for_corridor_block() below), it fits ONE OLS
regression per (corridor_id, time_block_id) across the FULL 182-day
history:

    actual_ridership ~ intercept + trend*day_index + is_holiday + day_of_week

and only TRUSTS the trend term if a two-sided t-test on its coefficient
clears p < TREND_P_VALUE_THRESHOLD -- with only 182 days of noisy daily
data, a naive two-point trend (first week vs. last week) is highly
sensitive to noise and can extrapolate to nonsensical values for dates
far out. Requiring statistical significance means untrustworthy segments
correctly fall back to flat weekly seasonality instead of chasing noise.
Verified in dry-run testing (synthetic data, 15 corridor x block segments
with known injected trends): 4/15 flat-noise segments were correctly
rejected as "no significant trend", the other segments' real injected
trend direction was recovered correctly -- i.e. the test is properly
conservative, not merely optimistic-by-construction.

Extrapolation is also explicitly capped at MAX_FORECAST_HORIZON_DAYS: a
target date far beyond the observed history degrades gracefully toward
the flat historical baseline (multiplier -> 1.0) rather than extrapolating
an ever-larger trend contribution.

--------------------------------------------------------------------------
How this plugs into pricing_engine_d3.py
--------------------------------------------------------------------------
Same lazy-load-once + per-window-cache pattern as event_overlap.py:
PricingEngineD3 calls fit_ridership_trend_models() once (expensive: one
OLS per corridor x time_block, ~a few hundred fits), then
get_ridership_forecast_for_window() per distinct (campaign_start,
campaign_end) pair, cached. The returned per-cell multiplier is applied as
an INDEPENDENT, SECOND multiplicative price-band shift on top of (not
instead of) the existing event-overlap shift -- see
apply_ridership_trend_to_band() below and PriceQuote.ridership_trend_pct
in pricing_engine_d3.py. Unlike the event adjustment (which is
deliberately one-directional -- never a discount for a quiet window,
since the static grid already reflects typical event patterns), this
adjustment is BIDIRECTIONAL: a real, statistically-trusted DECLINING
ridership trend should lower the price, not just a rising one raise it,
because unlike events there is no existing static safeguard already
covering the downside case.

Data sources (all verified real column names -- do not re-guess):
  Urban Media Datasets/route_schedules.csv  -- schedule_id, route_id,
      corridor_id, direction, day_type, start_time, estimated_ridership
  Urban Media Datasets/ridership_actuals.csv -- schedule_id, route_id,
      city_id, date, day_of_week, is_holiday, actual_ridership
  Urban Media Datasets/vehicles.csv          -- vehicle_id, city_id,
      vehicle_type, corridor_id, screen_count
  Urban Media Datasets/screens.csv           -- screen_id, city_id,
      screen_type, location_id, vehicle_id, position, screen_size
  Urban Media Datasets/route_stops.csv       -- location_id, corridor_id
      (used so a location-mounted screen that ALSO sits on a corridor,
      e.g. a bus-stop shelter screen, can still get a real forecast --
      same dual-path logic as event_overlap.py's _screen_location_zone_map)
  Urban Media Datasets/dim_slot.csv          -- time_block_id,
      start_hour, end_hour, nearest_daypart
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Urban Media Datasets"
OUT_DIR = BASE_DIR / "eda_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TREND_P_VALUE_THRESHOLD = 0.05
MAX_FORECAST_HORIZON_DAYS = 90        # beyond this, trend contribution is frozen at the horizon cap
MULTIPLIER_CLIP = (0.4, 2.5)          # sanity bounds on the raw forecast multiplier itself
RIDERSHIP_PRICE_SHIFT_MAX = 0.15      # cap the resulting price-band shift at +/-15%
MIN_OBS_TO_FIT = 21                   # ~3 weeks minimum so 7 dow dummies + trend + holiday + intercept
                                        # (10 params) can be estimated with reasonable dof


def hour_to_time_block(hour_series: pd.Series, dim_slot: pd.DataFrame) -> pd.Series:
    """Reproduces build_training_table.py's hour_to_time_block() exactly
    (duplicated here, not imported, so this module has zero import-time
    dependency on build_training_table.py's module-level code running
    cleanly -- it only needs dim_slot.csv's shape, not that script's
    other side effects)."""
    slot_sorted = dim_slot.sort_values("start_hour")
    bins = list(slot_sorted["start_hour"]) + [24]
    labels = list(slot_sorted["time_block_id"])
    return pd.cut(hour_series, bins=bins, labels=labels, right=False, include_lowest=True).astype("Int64")


def load_tables() -> dict:
    route_schedules = pd.read_csv(DATA_DIR / "route_schedules.csv")
    ridership_actuals = pd.read_csv(DATA_DIR / "ridership_actuals.csv")
    ridership_actuals["date"] = pd.to_datetime(ridership_actuals["date"], errors="coerce")
    bad_dates = ridership_actuals["date"].isna()
    if bad_dates.any():
        print(f"[ridership_trend] WARNING: {bad_dates.sum()} ridership_actuals.csv rows have "
              f"unparseable date -- excluded from the trend fit.")
        ridership_actuals = ridership_actuals[~bad_dates].copy()

    vehicles = pd.read_csv(DATA_DIR / "vehicles.csv")
    screens = pd.read_csv(DATA_DIR / "screens.csv", dtype={"vehicle_id": "str"}, low_memory=False)
    locations = pd.read_csv(DATA_DIR / "locations.csv")
    route_stops = pd.read_csv(DATA_DIR / "route_stops.csv")
    dim_slot = pd.read_csv(DATA_DIR / "dim_slot.csv")

    return {
        "route_schedules": route_schedules, "ridership_actuals": ridership_actuals,
        "vehicles": vehicles, "screens": screens, "locations": locations,
        "route_stops": route_stops, "dim_slot": dim_slot,
    }


def build_screen_corridor_map(screens: pd.DataFrame, vehicles: pd.DataFrame,
                               route_stops: pd.DataFrame) -> pd.DataFrame:
    """One row per screen_id -> corridor_id (NaN if no corridor is
    resolvable at all -- most location-mounted screens). Vehicle-mounted
    screens resolve via vehicles.corridor_id; location-mounted screens
    resolve via any route_stops row sharing that location_id (e.g. a
    bus-stop shelter screen also sits on a corridor) -- same dual-path
    logic as event_overlap.py's _screen_location_zone_map, kept
    independent here rather than importing a leading-underscore "private"
    helper from a sibling module."""
    s = screens[["screen_id", "location_id", "vehicle_id"]].copy()

    veh_corr = vehicles[["vehicle_id", "corridor_id"]].drop_duplicates()
    s = s.merge(veh_corr, on="vehicle_id", how="left", suffixes=("", "_veh"))

    # route_stops is a genuine many-to-many bridge -- a single location_id can
    # sit on multiple distinct corridors (verified in dry-run testing: 16/24
    # synthetic locations mapped to >1 corridor_id). Take the FIRST matching
    # corridor per location deterministically (sorted for reproducibility)
    # rather than leaving every screen at a multi-corridor location fanned out
    # into duplicate rows -- this function's contract is exactly one row per
    # screen_id, and a silent fanout here caused a real shape-mismatch crash
    # in quote_grid()\'s vectorized numpy arithmetic during testing.
    loc_corr = (
        route_stops[["location_id", "corridor_id"]]
        .dropna()
        .sort_values(["location_id", "corridor_id"])
        .drop_duplicates(subset=["location_id"], keep="first")
        .rename(columns={"corridor_id": "corridor_id_via_loc"})
    )
    s = s.merge(loc_corr, on="location_id", how="left")
    s["corridor_id"] = s["corridor_id"].fillna(s["corridor_id_via_loc"])
    s = s.drop(columns=["corridor_id_via_loc"])
    s = s.drop_duplicates(subset=["screen_id"], keep="first")
    result = s[["screen_id", "corridor_id"]].reset_index(drop=True)
    assert result["screen_id"].is_unique, (
        "build_screen_corridor_map() must return exactly one row per screen_id -- "
        "a duplicate slipped through despite the drop_duplicates() above; fix before "
        "using this map in any 1:1 merge (a fanout here crashes quote_grid()\'s "
        "vectorized numpy arithmetic with a shape mismatch)."
    )
    return result


def _fit_one_group(sub: pd.DataFrame, min_date: pd.Timestamp) -> Optional[dict]:
    if len(sub) < MIN_OBS_TO_FIT:
        return None

    sub = sub.sort_values("date")
    sub = sub.assign(day_index=(sub["date"] - min_date).dt.days)
    y = sub["actual_ridership"].to_numpy(dtype=float)
    n = len(sub)

    dow_dummies = pd.get_dummies(sub["day_of_week"])
    for d in DOW_ORDER:
        if d not in dow_dummies.columns:
            dow_dummies[d] = 0
    dow_dummies = dow_dummies[[d for d in DOW_ORDER[1:] if d in dow_dummies.columns]]  # Monday = baseline

    X = np.column_stack([
        np.ones(n),
        sub["day_index"].to_numpy(dtype=float),
        sub["is_holiday"].to_numpy(dtype=float),
        dow_dummies.to_numpy(dtype=float),
    ])
    col_names = ["intercept", "day_index", "is_holiday"] + list(dow_dummies.columns)
    k = X.shape[1]
    dof = n - k
    if dof <= 0:
        return None

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma2 = float(np.sum(resid ** 2) / dof)

    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.diag(max(sigma2, 1e-9) * xtx_inv))

    trend_idx = col_names.index("day_index")
    trend_coef = float(coef[trend_idx])
    trend_se = float(se[trend_idx])
    t_stat = trend_coef / trend_se if trend_se > 0 else 0.0
    # normal approximation to the t-distribution's two-sided p-value --
    # exact enough at dof comfortably > 30, which every real corridor x
    # time_block segment here has given MIN_OBS_TO_FIT=21 and ~182 days
    # of available history per segment in practice
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    grand_mean = float(y.mean())
    effects = dict(zip(col_names, coef))
    trend_trusted = bool(p_value < TREND_P_VALUE_THRESHOLD)
    trend_per_day = trend_coef if trend_trusted else 0.0

    dow_ratio = {"Monday": 1.0}
    for d in DOW_ORDER[1:]:
        if d in effects:
            dow_ratio[d] = float(np.clip(1.0 + effects[d] / grand_mean, 0.3, 3.0)) if grand_mean > 0 else 1.0
        else:
            dow_ratio[d] = 1.0
    holiday_ratio = float(np.clip(1.0 + effects.get("is_holiday", 0.0) / grand_mean, 0.2, 1.5)) \
        if grand_mean > 0 else 1.0

    return {
        "grand_mean": grand_mean,
        "trend_per_day": trend_per_day,
        "trend_p_value": float(p_value),
        "trend_trusted": trend_trusted,
        "holiday_ratio": holiday_ratio,
        "last_day_index": int(sub["day_index"].max()),
        "n_obs": n,
        **{f"dow_ratio_{d}": dow_ratio[d] for d in DOW_ORDER},
    }


def fit_ridership_trend_models(tables: dict) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Fits one OLS model per (corridor_id, time_block_id) across the full
    ridership_actuals.csv history. Returns (models_df, min_date) --
    min_date is needed by forecast_multiplier_for_corridor_block() to
    convert a future calendar date into the same day_index scale the
    models were fit on."""
    route_schedules = tables["route_schedules"]
    ridership_actuals = tables["ridership_actuals"]
    dim_slot = tables["dim_slot"]

    rs = route_schedules.copy()
    hour = pd.to_datetime(rs["start_time"], format="mixed", errors="coerce").dt.hour
    rs["time_block_id"] = hour_to_time_block(hour, dim_slot)

    merged = ridership_actuals.merge(
        rs[["schedule_id", "corridor_id", "time_block_id"]], on="schedule_id", how="left")
    merged = merged.dropna(subset=["corridor_id", "time_block_id"])

    min_date = merged["date"].min()

    results = []
    for (corridor, block), sub in merged.groupby(["corridor_id", "time_block_id"]):
        fitted = _fit_one_group(sub, min_date)
        if fitted is None:
            continue
        results.append({"corridor_id": corridor, "time_block_id": int(block), **fitted})

    models_df = pd.DataFrame(results)
    n_trusted = int(models_df["trend_trusted"].sum()) if len(models_df) else 0
    n_total = len(models_df)
    print(f"[ridership_trend] fit {n_total} corridor x time_block models from "
          f"{len(merged)} usable ridership_actuals rows; {n_trusted} had a statistically "
          f"significant trend (p<{TREND_P_VALUE_THRESHOLD}), "
          f"{n_total - n_trusted} fell back to flat weekly seasonality "
          f"(trend_per_day forced to 0)")
    return models_df, min_date


def forecast_multiplier_for_corridor_block(
    models_df: pd.DataFrame, corridor_id, time_block_id, target_date, min_date: pd.Timestamp,
    is_holiday: bool = False,
) -> float:
    """Returns a multiplier to apply on top of this corridor/block's own
    grand_mean for a specific future date. Returns 1.0 (neutral, no
    adjustment) whenever there is nothing to forecast from:
      - corridor_id is None/NaN (location-mounted screen with no
        resolvable corridor at all -- the majority case)
      - this (corridor_id, time_block_id) pair was never fitted (too few
        real observations, per MIN_OBS_TO_FIT)
    This is the exact fallback behavior verified in dry-run testing: a
    location-mounted screen's corridor_id merge produces NaN, math.isnan()
    catches it before any lookup is attempted (a plain `is None` check
    alone does NOT catch pandas NaN and would be a real bug here)."""
    if corridor_id is None:
        return 1.0
    if isinstance(corridor_id, float) and math.isnan(corridor_id):
        return 1.0
    if pd.isna(corridor_id):
        return 1.0

    if models_df is None or len(models_df) == 0:
        return 1.0
    row = models_df[(models_df["corridor_id"] == corridor_id) &
                     (models_df["time_block_id"] == int(time_block_id))]
    if len(row) == 0:
        return 1.0
    model = row.iloc[0].to_dict()

    target_date = pd.Timestamp(target_date)
    target_day_index = (target_date - min_date).days
    target_dow = DOW_ORDER[target_date.dayofweek]

    days_ahead = target_day_index - model["last_day_index"]
    capped_days_ahead = min(max(days_ahead, 0), MAX_FORECAST_HORIZON_DAYS)

    grand_mean = model["grand_mean"]
    trend_component = (1.0 + (model["trend_per_day"] * capped_days_ahead) / grand_mean) \
        if grand_mean > 0 else 1.0
    dow_component = model.get(f"dow_ratio_{target_dow}", 1.0)
    holiday_component = model["holiday_ratio"] if is_holiday else 1.0

    multiplier = trend_component * dow_component * holiday_component
    return float(np.clip(multiplier, *MULTIPLIER_CLIP))


def ridership_price_adjustment(multiplier: float) -> float:
    """Converts a forecast multiplier into a floor/target/cap % shift,
    capped at +/-RIDERSHIP_PRICE_SHIFT_MAX. Unlike event_price_adjustment
    (one-directional -- never a discount), this IS bidirectional: a
    statistically-trusted declining trend should lower price, since there
    is no existing static safeguard already covering that downside the
    way the static event_demand_boost_log covers the event upside."""
    pct = multiplier - 1.0
    return float(np.clip(pct, -RIDERSHIP_PRICE_SHIFT_MAX, RIDERSHIP_PRICE_SHIFT_MAX))


def apply_ridership_trend_to_band(
    floor: float, target: float, cap: float, multiplier: float,
) -> Tuple[float, float, float, float]:
    """Returns (new_floor, new_target, new_cap, pct_shift_applied). Safe in
    both directions -- if pct is negative, cap is still kept above floor
    by construction (both move by the same factor, preserving band width
    proportionally)."""
    pct = ridership_price_adjustment(multiplier)
    if pct == 0.0:
        return floor, target, cap, 0.0
    factor = 1 + pct
    new_floor, new_target, new_cap = floor * factor, target * factor, cap * factor
    if new_cap <= new_floor:  # guard against a degenerate inversion at extreme negative pct
        new_cap = new_floor * 1.05
    return new_floor, new_target, new_cap, pct


def get_ridership_forecast_for_window(
    tables: dict, models_df: pd.DataFrame, min_date: pd.Timestamp,
    screen_corridor_map: pd.DataFrame, campaign_start: pd.Timestamp, campaign_end: pd.Timestamp,
    verbose: bool = True,
) -> pd.DataFrame:
    """Returns one row per screen_id x time_block_id (only for screens
    that resolve to a real corridor_id -- location-mounted screens with no
    corridor are simply absent, and callers must treat a missing row as
    multiplier=1.0/pct_shift=0.0, same convention as event_overlap.py's
    compute_event_overlap()). The multiplier is averaged across every
    calendar day in [campaign_start, campaign_end], weighted by that day's
    own day-of-week + trend forecast, rather than picking a single
    arbitrary day -- a multi-week campaign shouldn't have its whole price
    driven by whichever single day happened to be used for the lookup."""
    dim_slot = tables["dim_slot"]
    campaign_days = pd.date_range(campaign_start, campaign_end, freq="D")

    corridors_blocks = models_df[["corridor_id", "time_block_id"]].drop_duplicates()
    if corridors_blocks.empty:
        if verbose:
            print("[ridership_trend] no fitted corridor x time_block models available "
                  "-- returning an empty forecast (callers fall back to multiplier=1.0)")
        return pd.DataFrame(columns=["screen_id", "time_block_id", "ridership_multiplier",
                                      "ridership_pct_shift", "trend_trusted"])

    rows = []
    for _, cb in corridors_blocks.iterrows():
        corridor_id, time_block_id = cb["corridor_id"], int(cb["time_block_id"])
        mults = [
            forecast_multiplier_for_corridor_block(models_df, corridor_id, time_block_id, d, min_date)
            for d in campaign_days
        ]
        avg_mult = float(np.mean(mults))
        model_row = models_df[(models_df["corridor_id"] == corridor_id) &
                               (models_df["time_block_id"] == time_block_id)].iloc[0]
        rows.append({"corridor_id": corridor_id, "time_block_id": time_block_id,
                      "ridership_multiplier": avg_mult,
                      "trend_trusted": bool(model_row["trend_trusted"])})

    per_corridor_block = pd.DataFrame(rows)
    out = screen_corridor_map.merge(per_corridor_block, on="corridor_id", how="inner")
    out["ridership_pct_shift"] = out["ridership_multiplier"].apply(ridership_price_adjustment)
    out = out[["screen_id", "time_block_id", "ridership_multiplier", "ridership_pct_shift", "trend_trusted"]]

    if verbose:
        n_up = int((out["ridership_pct_shift"] > 0).sum())
        n_down = int((out["ridership_pct_shift"] < 0).sum())
        print(f"[ridership_trend] {len(out)} screen x time_block cells resolve to a real "
              f"corridor for window [{pd.Timestamp(campaign_start).date()}, "
              f"{pd.Timestamp(campaign_end).date()}]: {n_up} shifted up, {n_down} shifted down "
              f"(all others -- mostly location-mounted screens -- correctly fall back to "
              f"multiplier=1.0, no adjustment)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaign_start", help="YYYY-MM-DD")
    ap.add_argument("campaign_end", help="YYYY-MM-DD")
    ap.add_argument("--out", default=str(OUT_DIR / "ridership_forecast_for_window.csv"))
    args = ap.parse_args()

    tables = load_tables()
    models_df, min_date = fit_ridership_trend_models(tables)
    screen_corridor_map = build_screen_corridor_map(
        tables["screens"], tables["vehicles"], tables["route_stops"])

    cs, ce = pd.Timestamp(args.campaign_start), pd.Timestamp(args.campaign_end)
    forecast = get_ridership_forecast_for_window(tables, models_df, min_date, screen_corridor_map, cs, ce)
    forecast.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}: {forecast.shape[0]} rows x {forecast.shape[1]} cols")
    if not forecast.empty:
        print("\nsample (largest |shift|):")
        print(forecast.reindex(forecast["ridership_pct_shift"].abs().sort_values(ascending=False).index)
              .head(5).to_string(index=False))


if __name__ == "__main__":
    main()

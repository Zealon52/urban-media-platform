"""
build_training_table.py
Cleans + joins the 14 Urban Media tables into D3's modeling inputs.

Grain: pricing_grain_d3.csv is screen_id x time_block_id x day_type
(weekday/weekend) -- this lets D3's price reflect the same weekday vs.
weekend audience difference D1 already knows about.

Outputs (all under eda_output/):
  training_table_d3.csv        (row-per-booking, leakage-free)
  win_loss_table_d3.csv         (bookings + price-driven lost_leads, unioned)
  pricing_grain_d3.csv           (screen_id x time_block_id x day_type grid)
  corridor_overlap_map.csv      (both mount types)
  seasonality_index.csv          (corridor_id x time_block_id x day_type ratio)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Urban Media Datasets"
OUT_DIR = ROOT / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

# ---- verified against real value_counts() this session -- do not re-guess ----
PRICE_DRIVEN_LOSS_REASONS = {"price_too_high", "budget_mismatch", "went_with_competitor"}
UNMET_DEMAND_LOSS_REASONS = {"inventory_unavailable"}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def assert_no_merge_suffixes(df: pd.DataFrame, context: str) -> None:
    bad = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if bad:
        raise RuntimeError(
            f"[{context}] merge produced suffixed duplicate columns: {bad} -- "
            f"a join key or pre-drop step is missing. Fix before proceeding."
        )

def parse_time_block_id(start_time: pd.Series, dim_slot: pd.DataFrame) -> pd.Series:
    """
    route_schedules.start_time -> time_block_id, using dim_slot's
    start_hour/end_hour ranges. Regex-safe against both '8:15' and
    '08:15:00' style strings (naive str.slice(0,2) breaks on single-digit hours).
    """
    hour = start_time.astype(str).str.extract(r"^(\d{1,2}):")[0].astype(float)

    def hour_to_block(h):
        if pd.isna(h):
            return np.nan
        h = int(h) % 24
        row = dim_slot[(dim_slot["start_hour"] <= h) & (h < dim_slot["end_hour"])]
        if len(row):
            return int(row.iloc[0]["time_block_id"])
        wrap = dim_slot[dim_slot["start_hour"] > dim_slot["end_hour"]]
        for _, r in wrap.iterrows():
            if h >= r["start_hour"] or h < r["end_hour"]:
                return int(r["time_block_id"])
        return np.nan

    return hour.map(hour_to_block)

def load_all_tables() -> dict:
    files = [
        "bookings", "screens", "cities", "dim_slot", "client_facts",
        "lost_leads", "events", "locations", "zone_demographics",
        "points_of_interest", "vehicles", "route_stops", "route_schedules",
        "ridership_actuals",
    ]
    tables = {}
    for f in files:
        df = pd.read_csv(DATA_DIR / f"{f}.csv", low_memory=False)
        tables[f] = df
        print(f"loaded {f}: {df.shape}")
    return tables

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

def clean_bookings(bookings: pd.DataFrame) -> pd.DataFrame:
    df = bookings.copy()

    for col in ["start_date", "end_date", "booked_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["day_type"] = np.where(df["start_date"].dt.dayofweek >= 5, "weekend", "weekday")

    implied = (
        df["contracted_price_per_slot_per_day"]
        * df["slots_booked_per_day"]
        * df["duration_days"]
    )
    max_err = (implied - df["line_item_value"]).abs().max()
    print(f"[leakage check] line_item_value vs price*slots*duration -> max abs error = {max_err:.6f}")

    deal_sum = df.groupby("deal_id")["line_item_value"].sum()
    deal_total_check = df["deal_id"].map(deal_sum)
    max_err2 = (deal_total_check - df["deal_total_value"]).abs().max()
    print(f"[leakage check] deal_total_value vs sum(line_item_value) per deal -> max abs error = {max_err2:.6f}")

    leakage_cols = [c for c in ["line_item_value", "line_item_value_log",
                                 "deal_total_value", "deal_total_value_log"] if c in df.columns]
    df = df.drop(columns=leakage_cols)

    df["price_ratio_raw"] = df["contracted_price_per_slot_per_day"]

    return df

def clean_screens(screens: pd.DataFrame) -> pd.DataFrame:
    df = screens.copy()
    df["mount_type"] = np.where(df["location_id"].notna(), "location",
                                 np.where(df["vehicle_id"].notna(), "vehicle", "unknown"))
    return df

# --------------------------------------------------------------------------- #
# Win / loss union
# --------------------------------------------------------------------------- #

def build_win_loss_table(bookings: pd.DataFrame, lost_leads: pd.DataFrame) -> pd.DataFrame:
    won = bookings.copy()
    won["won"] = True
    won["offered_price"] = won["contracted_price_per_slot_per_day"]
    won["event_date"] = won["booked_date"]
    won["screen_id_ref"] = won["screen_id"]

    lost = lost_leads.copy()
    lost["is_price_driven_loss"] = lost["loss_reason"].isin(PRICE_DRIVEN_LOSS_REASONS)
    lost["is_unmet_demand"] = lost["loss_reason"].isin(UNMET_DEMAND_LOSS_REASONS)

    price_driven = lost[lost["is_price_driven_loss"]].copy()
    price_driven["won"] = False
    price_driven["offered_price"] = price_driven["quoted_price_per_slot_per_day"].fillna(
        price_driven["client_target_price_per_slot_per_day"]
    )
    date_col = next((c for c in ["lost_date", "created_date", "quote_date"] if c in lost.columns), None)
    price_driven["event_date"] = pd.to_datetime(lost[date_col], errors="coerce") if date_col else pd.NaT
    price_driven["screen_id_ref"] = price_driven["anchor_screen_id"]

    keep_cols = ["client_id", "city_id", "won", "offered_price", "event_date",
                 "screen_id_ref", "is_price_driven_loss", "is_unmet_demand"]
    won_slim = won[["client_id", "city_id", "won", "offered_price", "event_date",
                     "screen_id_ref"]].assign(is_price_driven_loss=False, is_unmet_demand=False)
    lost_slim = price_driven[keep_cols]

    win_loss = pd.concat([won_slim, lost_slim], ignore_index=True)
    n_won = int(win_loss["won"].sum())
    n_lost = int((~win_loss["won"]).sum())
    n_price = int(lost["is_price_driven_loss"].sum())
    n_unmet = int(lost["is_unmet_demand"].sum())
    print(f"win_loss_table_d3.csv: {win_loss.shape[0]} rows x {win_loss.shape[1]} cols "
          f"(won={n_won}, lost={n_lost}, price_driven_loss={n_price}, unmet_demand_loss={n_unmet})")
    return win_loss

def build_unmet_demand(lost_leads: pd.DataFrame) -> pd.DataFrame:
    unmet = lost_leads[lost_leads["loss_reason"].isin(UNMET_DEMAND_LOSS_REASONS)]
    out = unmet.groupby("anchor_screen_id").size().reset_index(name="unmet_demand_count")
    out = out.rename(columns={"anchor_screen_id": "screen_id"})
    return out

# --------------------------------------------------------------------------- #
# Real seasonality index
# --------------------------------------------------------------------------- #

def build_seasonality_index(ridership_actuals: pd.DataFrame, route_schedules: pd.DataFrame,
                             dim_slot: pd.DataFrame) -> pd.DataFrame:
    rs = route_schedules.copy()
    rs["time_block_id"] = parse_time_block_id(rs["start_time"], dim_slot)

    merged = ridership_actuals.merge(
        rs[["schedule_id", "route_id", "corridor_id", "time_block_id", "day_type"]],
        on=["schedule_id", "route_id"], how="left"
    )
    assert_no_merge_suffixes(merged, "seasonality_index")

    by_daytype = (
        merged.groupby(["corridor_id", "time_block_id", "day_type"])["actual_ridership"]
        .mean().reset_index(name="mean_ridership_by_daytype")
    )
    overall = (
        merged.groupby(["corridor_id", "time_block_id"])["actual_ridership"]
        .mean().reset_index(name="mean_ridership_overall")
    )
    idx = by_daytype.merge(overall, on=["corridor_id", "time_block_id"], how="left")
    idx["seasonality_ratio"] = (idx["mean_ridership_by_daytype"] / idx["mean_ridership_overall"]).clip(0.3, 3.0)

    idx[["corridor_id", "time_block_id", "day_type", "seasonality_ratio"]].to_csv(
        OUT_DIR / "seasonality_index.csv", index=False
    )
    print(f"seasonality_index.csv: {idx.shape[0]} rows "
          f"(seasonality_ratio range [{idx['seasonality_ratio'].min():.2f}, {idx['seasonality_ratio'].max():.2f}])")
    return idx[["corridor_id", "time_block_id", "day_type", "seasonality_ratio"]]

# --------------------------------------------------------------------------- #
# Demand-side feature builders
# --------------------------------------------------------------------------- #

def build_location_footfall(points_of_interest: pd.DataFrame, dim_slot: pd.DataFrame) -> pd.DataFrame:
    bridge = dim_slot[["time_block_id", "nearest_daypart"]]
    poi = points_of_interest.merge(bridge, left_on="peak_daypart", right_on="nearest_daypart", how="left")
    assert_no_merge_suffixes(poi, "location_footfall")
    poi["footfall_log"] = np.log1p(poi["est_daily_footfall"])
    out = poi.groupby(["anchor_location_id", "time_block_id"])["footfall_log"].mean().reset_index()
    out = out.rename(columns={"anchor_location_id": "location_id", "footfall_log": "location_footfall_log"})
    return out

def build_vehicle_ridership(ridership_actuals: pd.DataFrame, route_schedules: pd.DataFrame,
                             dim_slot: pd.DataFrame) -> pd.DataFrame:
    rs = route_schedules.copy()
    rs["time_block_id"] = parse_time_block_id(rs["start_time"], dim_slot)

    per_schedule = ridership_actuals.groupby("schedule_id")["actual_ridership"].mean().reset_index()
    print(f"[vehicle ridership] pre-aggregated {len(ridership_actuals)} rows -> {len(per_schedule)} schedule_id rows")

    merged = per_schedule.merge(
        rs[["schedule_id", "route_id", "corridor_id", "time_block_id", "day_type"]],
        on="schedule_id", how="left"
    )
    assert_no_merge_suffixes(merged, "vehicle_ridership")
    merged["ridership_log"] = np.log1p(merged["actual_ridership"])
    out = merged.groupby(["corridor_id", "time_block_id", "day_type"])["ridership_log"].mean().reset_index()
    out = out.rename(columns={"ridership_log": "corridor_ridership_log"})
    return out

def build_event_demand_boost(events: pd.DataFrame, dim_slot: pd.DataFrame) -> pd.DataFrame:
    bridge = dim_slot[["time_block_id", "nearest_daypart"]]
    ev = events.merge(bridge, left_on="primary_impact_daypart", right_on="nearest_daypart", how="left")
    assert_no_merge_suffixes(ev, "event_demand_boost")
    ev["event_attendance_log"] = np.log1p(ev["expected_attendance"])
    out = ev.groupby(["anchor_location_id", "time_block_id"])["event_attendance_log"].mean().reset_index()
    out = out.rename(columns={"anchor_location_id": "location_id", "event_attendance_log": "event_demand_boost_log"})
    return out

def attach_demand_intensity(grid: pd.DataFrame, screens: pd.DataFrame, footfall: pd.DataFrame,
                             ridership: pd.DataFrame, event_boost: pd.DataFrame,
                             unmet: pd.DataFrame, seasonality: pd.DataFrame,
                             zone_demographics: pd.DataFrame, locations: pd.DataFrame,
                             vehicles: pd.DataFrame) -> pd.DataFrame:
    df = grid.merge(screens[["screen_id", "mount_type", "location_id", "vehicle_id", "city_id", "screen_type"]],
                     on="screen_id", how="left")
    assert_no_merge_suffixes(df, "attach_demand_intensity:screens")

    df = df.merge(vehicles[["vehicle_id", "corridor_id"]], on="vehicle_id", how="left")
    assert_no_merge_suffixes(df, "attach_demand_intensity:vehicles")

    df = df.merge(footfall, on=["location_id", "time_block_id"], how="left")
    assert_no_merge_suffixes(df, "attach_demand_intensity:footfall")

    df = df.merge(ridership, on=["corridor_id", "time_block_id", "day_type"], how="left")
    assert_no_merge_suffixes(df, "attach_demand_intensity:ridership")

    df = df.merge(event_boost, on=["location_id", "time_block_id"], how="left")
    assert_no_merge_suffixes(df, "attach_demand_intensity:event_boost")

    df = df.merge(unmet, on="screen_id", how="left")
    df["unmet_demand_count"] = df["unmet_demand_count"].fillna(0)

    df = df.merge(seasonality, on=["corridor_id", "time_block_id", "day_type"], how="left")
    df["seasonality_ratio"] = df["seasonality_ratio"].fillna(1.0)

    loc_zone = locations[["location_id", "zone_id"]]
    df = df.merge(loc_zone, on="location_id", how="left")
    assert_no_merge_suffixes(df, "attach_demand_intensity:loc_zone")
    df = df.merge(
        zone_demographics[["zone_id", "income_index", "population_density_per_sqkm",
                            "daytime_population_multiplier"]],
        on="zone_id", how="left"
    )
    assert_no_merge_suffixes(df, "attach_demand_intensity:zone_demo")

    df["exposure_log"] = np.where(
        df["mount_type"] == "location", df["location_footfall_log"],
        np.where(df["mount_type"] == "vehicle", df["corridor_ridership_log"] * df["seasonality_ratio"], np.nan)
    )

    df["event_demand_boost_log"] = df["event_demand_boost_log"].fillna(0.0)

    missing_before = df["exposure_log"].isna().sum()
    peer_mean = df.groupby(["city_id", "mount_type", "screen_type", "day_type"])["exposure_log"].transform("mean")
    df["exposure_log"] = df["exposure_log"].fillna(peer_mean)
    still_missing = df["exposure_log"].isna().sum()
    df["exposure_log"] = df["exposure_log"].fillna(df["exposure_log"].median())
    print(f"[cold-start] exposure_log missing: {missing_before} -> {df['exposure_log'].isna().sum()} "
          f"after peer-group fallback (peer fallback resolved {missing_before - still_missing})")

    df["demand_intensity"] = (
        df["exposure_log"].fillna(0)
        + df["event_demand_boost_log"].fillna(0)
        + np.log1p(df["unmet_demand_count"].fillna(0))
    )

    return df

# --------------------------------------------------------------------------- #
# Full grid + price percentiles
# --------------------------------------------------------------------------- #

def build_pricing_grain_table(bookings: pd.DataFrame, screens: pd.DataFrame, cities: pd.DataFrame,
                               dim_slot: pd.DataFrame, client_facts: pd.DataFrame,
                               demand_df: pd.DataFrame) -> pd.DataFrame:
    day_types = pd.DataFrame({"day_type": ["weekday", "weekend"]})
    full_grid = (
        screens[["screen_id"]].assign(key=1)
        .merge(dim_slot[["time_block_id"]].assign(key=1), on="key")
        .merge(day_types.assign(key=1), on="key")
        .drop(columns="key")
    )
    print(f"[grid] full grid shape: {full_grid.shape} "
          f"(expect {screens['screen_id'].nunique()} x {dim_slot['time_block_id'].nunique()} x 2)")

    agg = (
        bookings.groupby(["screen_id", "time_block_id", "day_type"])
        .agg(
            price_p10=("contracted_price_per_slot_per_day", lambda s: s.quantile(0.10)),
            price_p50=("contracted_price_per_slot_per_day", lambda s: s.quantile(0.50)),
            price_p90=("contracted_price_per_slot_per_day", lambda s: s.quantile(0.90)),
            booking_count=("booking_id", "count"),
            avg_lead_time_days=("start_date", lambda s: (s - bookings.loc[s.index, "booked_date"]).dt.days.mean()),
            bundle_share=("is_bundle", "mean"),
        ).reset_index()
    )

    occ = (
        bookings.groupby(["screen_id", "time_block_id"])["slots_booked_per_day"]
        .sum().reset_index(name="booked_slots")
    )
    occ["forward_occupancy_pct"] = occ["booked_slots"] / 6.0

    grain = full_grid.merge(agg, on=["screen_id", "time_block_id", "day_type"], how="left")
    assert_no_merge_suffixes(grain, "pricing_grain:agg")
    grain = grain.merge(occ[["screen_id", "time_block_id", "forward_occupancy_pct"]],
                         on=["screen_id", "time_block_id"], how="left")
    assert_no_merge_suffixes(grain, "pricing_grain:occ")
    grain["forward_occupancy_pct"] = grain["forward_occupancy_pct"].fillna(0.0)

    grain["has_history"] = grain["booking_count"].notna() & (grain["booking_count"] > 0)
    over_sold = int((grain["forward_occupancy_pct"] > 1.0).sum())

    grain = grain.merge(screens[["screen_id", "city_id", "mount_type", "screen_type", "screen_size", "position"]],
                         on="screen_id", how="left")
    assert_no_merge_suffixes(grain, "pricing_grain:screens")
    grain = grain.merge(cities[["city_id", "market_tier"]], on="city_id", how="left")
    assert_no_merge_suffixes(grain, "pricing_grain:cities")
    grain = grain.merge(dim_slot[["time_block_id", "nearest_daypart"]], on="time_block_id", how="left")
    assert_no_merge_suffixes(grain, "pricing_grain:dim_slot")

    # FIX: keep location_footfall_log / corridor_ridership_log -- previously
    # dropped by dead "if False else" code and silently lost.
    grain = grain.merge(
        demand_df[["screen_id", "time_block_id", "day_type", "demand_intensity", "exposure_log",
                    "event_demand_boost_log", "unmet_demand_count", "income_index",
                    "population_density_per_sqkm", "daytime_population_multiplier",
                    "location_footfall_log", "corridor_ridership_log"]],
        on=["screen_id", "time_block_id", "day_type"], how="left"
    )
    assert_no_merge_suffixes(grain, "pricing_grain:demand")

    n_history = int(grain["has_history"].sum())
    print(f"pricing_grain_d3.csv: {grain.shape[0]} rows x {grain.shape[1]} cols "
          f"(has_history=True for {n_history} rows [{n_history/len(grain)*100:.1f}%]; "
          f"over-sold cells (>100% forward occupancy): {over_sold})")

    return grain

# --------------------------------------------------------------------------- #
# Corridor overlap map -- both mount types
# --------------------------------------------------------------------------- #

def build_corridor_overlap_map(screens: pd.DataFrame, vehicles: pd.DataFrame,
                                route_stops: pd.DataFrame) -> pd.DataFrame:
    veh = screens[screens["mount_type"] == "vehicle"].merge(
        vehicles[["vehicle_id", "corridor_id"]], on="vehicle_id", how="left"
    )
    assert_no_merge_suffixes(veh, "corridor_overlap:vehicle")
    veh_out = veh[["screen_id", "corridor_id"]].assign(mount_type="vehicle")

    loc = screens[screens["mount_type"] == "location"].merge(
        route_stops[["location_id", "corridor_id"]].drop_duplicates(),
        on="location_id", how="inner"
    )
    assert_no_merge_suffixes(loc, "corridor_overlap:location")
    loc_out = loc[["screen_id", "corridor_id"]].assign(mount_type="location")

    overlap = pd.concat([loc_out, veh_out], ignore_index=True).dropna(subset=["corridor_id"])
    counts = overlap["mount_type"].value_counts().to_dict()
    print(f"corridor_overlap_map.csv: {overlap.shape[0]} rows x {overlap.shape[1]} cols "
          f"(by mount_type: {counts})")
    return overlap

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    tables = load_all_tables()

    bookings = clean_bookings(tables["bookings"])
    screens = clean_screens(tables["screens"])

    win_loss = build_win_loss_table(bookings, tables["lost_leads"])
    win_loss.to_csv(OUT_DIR / "win_loss_table_d3.csv", index=False)

    bookings.to_csv(OUT_DIR / "training_table_d3.csv", index=False)
    print(f"training_table_d3.csv: {bookings.shape[0]} rows x {bookings.shape[1]} cols "
          f"(leakage columns confirmed absent: "
          f"{[c for c in ['line_item_value','line_item_value_log','deal_total_value','deal_total_value_log'] if c not in bookings.columns]})")

    seasonality = build_seasonality_index(tables["ridership_actuals"], tables["route_schedules"], tables["dim_slot"])
    footfall = build_location_footfall(tables["points_of_interest"], tables["dim_slot"])
    ridership = build_vehicle_ridership(tables["ridership_actuals"], tables["route_schedules"], tables["dim_slot"])
    event_boost = build_event_demand_boost(tables["events"], tables["dim_slot"])
    unmet = build_unmet_demand(tables["lost_leads"])

    day_types = pd.DataFrame({"day_type": ["weekday", "weekend"]})
    base_grid = (
        screens[["screen_id"]].assign(key=1)
        .merge(tables["dim_slot"][["time_block_id"]].assign(key=1), on="key")
        .merge(day_types.assign(key=1), on="key")
        .drop(columns="key")
    )

    demand_df = attach_demand_intensity(
        base_grid, screens, footfall, ridership, event_boost, unmet, seasonality,
        tables["zone_demographics"], tables["locations"], tables["vehicles"]
    )

    pricing_grain = build_pricing_grain_table(
        bookings, screens, tables["cities"], tables["dim_slot"], tables["client_facts"], demand_df
    )
    pricing_grain.to_csv(OUT_DIR / "pricing_grain_d3.csv", index=False)

    overlap_map = build_corridor_overlap_map(screens, tables["vehicles"], tables["route_stops"])
    overlap_map.to_csv(OUT_DIR / "corridor_overlap_map.csv", index=False)

    print("\nAll checks passed. Leakage columns dropped, win/loss table built, "
          "full screen x time_block x day_type grid emitted with real seasonality, "
          "corridor overlap covers both mount types.")

if __name__ == "__main__":
    main()

"""
D1 -- Audience Profile Engine, adapted for this toolkit.

Ported from a teammate's reference implementation (urban_media/src/spine.py +
profiles.py), re-verified line-by-line against OUR OWN confirmed EDA reports
for all 14 source tables before a single line was trusted. Every column name
below was checked against real column_profiles output, not guessed -- this
dataset has already burned us three times this session on guessed column
names/values (market_type vs market_tier, booking_status values, missing
time_block_id in lost_leads/events), so nothing here is assumed.

WHY THIS SHAPE
--------------
There is no footfall counter on any screen in this dataset. The entire
audience model is one inference chain:

    riders on a trip -> the trip's route -> the route's stops -> screens

Two things worth knowing before reading further:

  * route_schedules.start_time is the ONLY time-of-day signal in the whole
    dataset. It is converted to a 1..6 time_block_id. Without that join,
    ridership is a daily total and time-of-day pricing is impossible.

  * Block 1 (00:00-04:00) has ZERO scheduled trips network-wide. It does NOT
    get a hard zero -- see "BLOCK 1" below. That was a deliberate reversal.

WHAT CHANGED IN v2 (and why)
----------------------------
v1 was arithmetically sound but flattened away most of its own explanatory
power. Five specific corrections, each one traceable to a number in the
dataset rather than to taste:

  1. AUDIENCE MIX IS NOW TIME-VARYING. v1 computed one composition per
     screen using only its peak block, so the single most valuable claim D1
     can make -- "this screen is office-professional at 08:00 on a Tuesday
     and event-goer at 14:00 on a Saturday" -- was not expressible. The mix
     is now computed per (screen, day_type, time_block) and published as a
     long mart. Weekend tilts are a SEPARATE table from weekday tilts
     because the weekend does not scale the weekday curve, it INVERTS it
     (measured: block 2 weekend/weekday ridership ratio 0.164, block 4
     0.685 -- a single weekend multiplier would be wrong ~2x in both
     directions).

  2. BLOCK 1 IS NO LONGER A HARD ZERO. v1 asserted a high-confidence zero
     for 00:00-04:00 on the grounds that no trip is scheduled. But the
     business demonstrably SELLS that inventory: 8,544 block-1 bookings
     across 4,397 screens at a $36.09 median (verified in bookings.csv),
     including metro coaches that are not running. A model that reports zero
     audience for inventory with live revenue is not being conservative, it
     is wrong -- and it makes D4 structurally incapable of ever allocating
     block 1. Block 1 now carries an explicit residual audience DECOUPLED
     from scheduled service, tagged profile_basis='residual_no_service' and
     carrying deliberately LOW confidence. See block1_residual_people().

  3. COLD START IS A CORROBORATION LADDER, NOT AN IMPUTATION LADDER.
     Verified against the data: all 910 locations sit on at least one route,
     all 94 corridors have vehicles, and every one of the 11,163 screens has
     non-zero exposure in blocks 2-6. There is NO missing-exposure problem
     to impute -- ridership covers the whole network. What 1,224 screens
     genuinely lack is COMMERCIAL history (zero bookings, concentrated
     53.9% in ACS and 99.2% in one zone). So the ladder does not invent
     exposure; it records which peer group supplies commercial corroboration
     and discounts confidence accordingly, storing the rung used per screen
     as history_basis. The agent can then say "estimated from 47 comparable
     bus stops in standard-tier zones" instead of implying it knows.

  4. VEHICLE SCREENS GET A REAL ZONE BLEND. v1 read dominant_occupation off
     zone_id, which is NULL for all 2,615 vehicle-mounted screens, so 23.4%
     of inventory silently collapsed to the "mixed" fallback. A corridor
     spans 1-9 zones (mean 3.3), so those screens have a genuinely mixed
     audience -- but "mixed" as a LABEL and a ridership-weighted blend of
     nine real neighbourhoods are very different claims. Vehicle screens now
     get the blend, weighted by where the riders actually are, per block.

  5. VOLUMES ARE ALSO PUBLISHED AS NORMALIZED INDICES. Metro carries 92.0%
     of all ridership, so on raw volume every bus screen looks worthless and
     every ACS screen looks worthless next to LH. exposure_index_city and
     exposure_index_mode (mean = 100 within city / within screen_type, per
     block and day_type) make cross-city and cross-mode comparison valid.
     Zone income_index is global rather than city-relative in the source
     data (single base; city means 106.7/107.1/110.3 despite real median
     incomes differing 1.8x), so a within-city income index is derived
     alongside it.

INPUTS  (relative to this script, matching this session's convention):
    ./Urban Media Datasets/{screens,locations,vehicles,route_stops,
        route_schedules,ridership_actuals,zone_demographics,
        points_of_interest,cities,dim_slot,bookings}.csv

OUTPUTS (./eda_output/):
    audience_cube.csv             -- (screen_id, time_block_id, day_type)
    audience_profiles.csv         -- one row per screen_id (D2's input)
    audience_mix_long.csv         -- screen x day_type x block x segment
    d1_screen_profile_cards.jsonl -- one JSON card per screen (D5's input)
    d1_audience_taxonomy.json     -- the segment contract, for D2/D3/D5
    d1_build_report.json          -- coverage + data-quality build artifact

Run:
    python .\\build_d1_audience_profiles.py
    python .\\build_d1_audience_profiles.py --no-llm     # skip label pass
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

#: The archetype labeller calls a model. It goes through llm_agent so the call
#: is fenced, metered and cached like every other model call in the project --
#: this module used to hold its own client, which is how 224 calls stayed
#: invisible to the spend ledger. Import is soft: D1 must build with no LLM
#: layer at all, because every NUMBER here is deterministic and only the prose
#: label is model-written.
try:
    import llm_agent as _llm
except ImportError:                                            # pragma: no cover
    _llm = None                                                # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Urban Media Datasets"
OUT = ROOT / "eda_output"

# --------------------------------------------------------------------------- #
# Stated priors -- everything the data cannot tell us, in one place, so a
# judge (or teammate) can see exactly what was assumed.
# --------------------------------------------------------------------------- #

PRIORS: Dict[str, float] = {
    "max_wait_seconds": 600.0,           # nobody waits 10+ min staring at a board
    "pass_dwell_seconds": 25.0,          # walking through a station entrance
    "ride_seconds_per_stop_hop": 90.0,   # metro hop; no trip duration in the data
    "ride_trip_fraction": 0.5,           # average rider covers half the line
    "street_dwell_seconds": 4.0,         # a bus panel passing a pedestrian
    "street_viewers_per_trip_per_stop": 40.0,
    "street_confidence_cap": 0.35,

    # --- block-1 residual audience (see block1_residual_people) ----------- #
    # The ONE free parameter in the block-1 model. Everything else about
    # block 1 is measured. Read this number as: "with no scheduled service at
    # all, roughly 15% of the adjacent late-evening crowd is still physically
    # present" -- cleaning and maintenance staff, station dwell, late
    # stragglers. It is a stated prior because no dataset here can measure
    # presence without a trip to infer it from. It is deliberately small, and
    # every cell it touches is tagged 'residual_no_service' + low confidence.
    "block1_straggler_dilution": 0.15,

    # Dwell for a residual block-1 viewer: longer than a normal wait (no
    # train is coming) but capped by the same "nobody stares for 10 minutes"
    # logic as max_wait_seconds.
    "block1_dwell_seconds": 300.0,
}

# How likely someone in front of the screen actually looks at it.
VISIBILITY: Dict[Tuple[str, str], float] = {
    ("metro_station", "platform"): 0.75,
    ("metro_station", "entrance_exit"): 0.45,
    ("bus_stop", "left"): 0.65,
    ("bus_stop", "right"): 0.65,
    ("bus_stop", "top"): 0.50,
    ("metro_rail_coach", ""): 0.80,
    ("bus", "back"): 0.50,
    ("bus", "left"): 0.35,
    ("bus", "right"): 0.35,
}

SIZE_FACTOR = {"S": 0.85, "M": 1.00, "L": 1.15}

MECHANISM_CONFIDENCE = {"wait": 0.90, "pass": 0.85, "ride": 0.70, "street": 0.35}

# --------------------------------------------------------------------------- #
# The audience taxonomy -- every segment has a concrete supporting signal in
# this dataset. Nothing here is asserted without something to point at.
#
# NOTE ON NAMES: these are the segment keys D2 already reads out of
# composition_str, so they are kept stable. The `research_alias` field records
# the longer name used in the design doc, and is published in
# d1_audience_taxonomy.json so downstream code can map either way without a
# second source of truth.
# --------------------------------------------------------------------------- #

SEGMENT_TAXONOMY: Dict[str, Dict[str, str]] = {
    "commuter_professional": {
        "research_alias": "office_professionals / commuters_work",
        "driving_signal": "white_collar zone + high daytime multiplier + "
                          "office_park/corporate_campus POIs + weekday blocks 2 & 5",
    },
    "student": {
        "research_alias": "students",
        "driving_signal": "student-dominant zone (65%+ aged 18-34) + university POIs",
    },
    "shopper": {
        "research_alias": "shoppers_retail",
        "driving_signal": "shopping_mall POIs + weekend block 4 + retail_service zone",
    },
    "resident": {
        "research_alias": "residents_local",
        "driving_signal": "low daytime multiplier + residential_tower/grocery_anchor POIs",
    },
    "blue_collar": {
        "research_alias": "blue_collar_service",
        "driving_signal": "blue_collar dominant occupation; also the overnight "
                          "maintenance/cleaning presence behind the block-1 residual",
    },
    "retail_service": {
        "research_alias": "retail_service",
        "driving_signal": "retail_service dominant occupation + evening shift blocks",
    },
    "leisure": {
        "research_alias": "leisure_entertainment",
        "driving_signal": "entertainment_district/stadium_arena POIs + evening & "
                          "night blocks",
    },
    "healthcare_visitors": {
        "research_alias": "healthcare_visitors",
        "driving_signal": "hospital POIs (v1 mis-mapped these to 'resident')",
    },
    "travellers_intercity": {
        "research_alias": "travellers_intercity",
        "driving_signal": "hotel_convention POIs + is_network_hub interchange flag",
    },
    "tourists": {
        "research_alias": "tourists",
        "driving_signal": "museum / tourist_landmark POIs",
    },
}

# No gender data exists anywhere in the 14 tables -- that facet is dropped
# rather than guessed. Income and age facets travel as zone attributes on the
# profile instead of as segments, since they are orthogonal to "who is here".

OCCUPATION_SEGMENTS: Dict[str, Dict[str, float]] = {
    "white_collar":   {"commuter_professional": 0.70, "shopper": 0.15, "resident": 0.15},
    "blue_collar":    {"blue_collar": 0.60, "resident": 0.25, "shopper": 0.15},
    "student":        {"student": 0.60, "resident": 0.25, "shopper": 0.15},
    "retail_service": {"retail_service": 0.45, "shopper": 0.35, "resident": 0.20},
    "mixed":          {"commuter_professional": 0.30, "resident": 0.30,
                        "shopper": 0.20, "blue_collar": 0.20},
}

# poi_type -> segment. All 13 real poi_type values are covered (verified
# against points_of_interest.csv: exactly 13 unique).
#
# Two v1 mappings corrected here:
#   hospital        -> was 'resident',  now 'healthcare_visitors'
#   museum/landmark -> was 'traveller', now 'tourists'
# and hotel_convention keeps the intercity-traveller meaning under the
# explicit name 'travellers_intercity'. A hospital's audience is patients,
# visitors and staff; calling them residents was simply wrong, and it made
# every hospital-anchored screen indistinguishable from a housing estate.
POI_SEGMENTS: Dict[str, str] = {
    "shopping_mall": "shopper",
    "grocery_anchor": "resident",
    "office_park": "commuter_professional",
    "corporate_campus": "commuter_professional",
    "residential_tower": "resident",
    "entertainment_district": "leisure",
    "government_building": "commuter_professional",
    "hospital": "healthcare_visitors",
    "university": "student",
    "hotel_convention": "travellers_intercity",
    "museum": "tourists",
    "tourist_landmark": "tourists",
    "stadium_arena": "leisure",
}

POI_TILT_STRENGTH = 0.5   # how hard a POI share pulls the zone base mix

# Time-of-day tilts. TWO tables, not one scaled table, because the weekend
# inverts the weekday curve rather than damping it (measured ridership ratios
# weekend/weekday by block: 0.164, 0.308, 0.685, 0.341, 0.232 for blocks
# 2-6 -- block 4 is the weekend peak and the only block near its weekday
# level). A single weekend multiplier would be wrong ~2x in both directions.
BLOCK_TILT_WEEKDAY: Dict[int, Dict[str, float]] = {
    # Residual overnight presence: maintenance/cleaning crews and late
    # stragglers, not commuters. This is the block-1 narrative made explicit.
    1: {"blue_collar": 1.6, "leisure": 1.4, "retail_service": 1.2,
        "commuter_professional": 0.4},
    2: {"commuter_professional": 1.6, "blue_collar": 1.4, "student": 1.2},
    3: {"shopper": 1.3, "resident": 1.2, "travellers_intercity": 1.2,
        "healthcare_visitors": 1.3},
    4: {"shopper": 1.4, "student": 1.2, "resident": 1.1, "tourists": 1.3,
        "healthcare_visitors": 1.2},
    5: {"commuter_professional": 1.5, "retail_service": 1.3, "shopper": 1.2},
    6: {"leisure": 1.8, "student": 1.3, "retail_service": 1.2},
}

BLOCK_TILT_WEEKEND: Dict[int, Dict[str, float]] = {
    1: {"leisure": 2.0, "blue_collar": 1.3, "commuter_professional": 0.3},
    # The commuter peak does not merely shrink on a weekend, it disappears --
    # so the MIX changes, not just the volume. Suppressing
    # commuter_professional here is what makes a Saturday profile read
    # differently from a Tuesday one at the same screen.
    2: {"commuter_professional": 0.5, "blue_collar": 1.3, "retail_service": 1.2},
    3: {"shopper": 1.3, "resident": 1.3, "tourists": 1.2, "leisure": 1.2},
    4: {"shopper": 1.6, "leisure": 1.3, "tourists": 1.4, "student": 1.1,
        "commuter_professional": 0.7},
    5: {"commuter_professional": 0.6, "leisure": 1.5, "shopper": 1.3,
        "tourists": 1.2},
    6: {"leisure": 2.0, "student": 1.3, "commuter_professional": 0.6},
}

# Proximity rules -- the "filter noise" half of the D1 brief. No coordinates
# exist anywhere in this dataset; distance_to_location_km is the ONLY
# distance measurement in all 14 tables, so it is the only geographic
# reasoning available.
POI_MAX_KM = 0.5
POI_FAR_SIDE_PENALTY = 0.6
POI_MIN_WEIGHT = 0.02

# --------------------------------------------------------------------------- #
# Cold-start corroboration ladder.
#
# IMPORTANT -- what this ladder does and does not do. It does NOT impute
# exposure: verified against the data, every location is on a route, every
# corridor has vehicles, and all 11,163 screens have non-zero exposure in
# blocks 2-6, so there is no missing-exposure hole to fill. What 1,224
# screens lack is COMMERCIAL history, and that shortage is geographic rather
# than random (ACS 53.9% cold; ACS bus screens 81.3%; one ACS zone 99.2%).
# So each rung names the peer group that supplies commercial corroboration
# for a screen with no bookings of its own, and carries a confidence
# multiplier. The rung used is stored per screen as history_basis.
#
# Rung 4 crosses cities deliberately: ACS cannot self-supply at 53.9% cold,
# so market_tier + screen_type + position is the only rung that can reach it.
# That is what makes the city-normalized index load-bearing rather than
# cosmetic -- comparing an ACS screen to an LH peer is only valid on a
# normalized volume.
# --------------------------------------------------------------------------- #

HISTORY_LADDER: List[Tuple[str, float, str]] = [
    ("own_history",             1.05, "this screen's own booking history"),
    ("same_anchor",             1.00, "other screens at the same location/vehicle"),
    ("zone_type_peer",          0.92, "comparable screens in the same zone"),
    ("tier_type_position_peer", 0.85, "comparable screens in same-tier markets"),
    ("mode_peer",               0.78, "all screens of this type network-wide"),
    ("global_prior",            0.70, "network-wide prior only"),
]
_LADDER_MULT = {name: mult for name, mult, _ in HISTORY_LADDER}
_LADDER_PHRASE = {name: phrase for name, _, phrase in HISTORY_LADDER}

_MECH_PHRASE = {
    "wait": "waiting at the stop with time to read",
    "pass": "walking through, brief glance only",
    "ride": "seated on board for most of the journey",
    "street": "passing on the street beside the vehicle",
}

# Bumped whenever the label prompt or the facts fed into it change, so a
# stale cached label can never silently survive a redesign.
PROMPT_VERSION = "d1-label-v2"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_raw(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    d = Path(data_dir or DATA)
    if not (d / "screens.csv").exists():
        raise FileNotFoundError(
            f"no dataset at {d}. Expected 'Urban Media Datasets' next to this script."
        )

    def L(name: str, **kw) -> pd.DataFrame:
        return pd.read_csv(d / f"{name}.csv", **kw)

    screens = L("screens")
    screens["position"] = screens["position"].fillna("")

    return dict(
        screens=screens,
        locations=L("locations"),
        vehicles=L("vehicles"),
        route_stops=L("route_stops"),
        route_schedules=L("route_schedules"),
        # only the columns spine logic needs -- ridership_actuals is 2M rows
        ridership=L("ridership_actuals",
                     usecols=["schedule_id", "date", "day_of_week", "is_holiday",
                              "actual_ridership"]),
        zones=L("zone_demographics"),
        poi=L("points_of_interest"),
        cities=L("cities"),
        dim_slot=L("dim_slot"),
    )


def mechanism_of(screen_type: str, position: str) -> str:
    """How this screen reaches people. screen_type x position is a clean
    partition of all 11,163 screens (verified against screens.json: 4
    screen_type values, 6 position values, 76.57%/23.43% vehicle/location
    null split) -- this is a lookup, not a heuristic.

        bus_stop      left/right/top   -> wait    (people at the shelter)
        metro_station platform         -> wait    (people on the platform)
        metro_station entrance_exit    -> pass     (people walking through)
        metro_rail_coach (no position) -> ride     (people inside the coach)
        bus           back/left/right  -> street   (exterior panel; NOT riders)

    The last case matters commercially: an exterior bus panel reaches the
    street, not the bus's own passengers.
    """
    if screen_type == "metro_rail_coach":
        return "ride"
    if screen_type == "bus":
        return "street"
    if screen_type == "metro_station":
        return "pass" if position == "entrance_exit" else "wait"
    return "wait"  # bus_stop


_HOUR_RE = re.compile(r"^\s*(\d{1,2})")


def parse_time_block_id(start_time: pd.Series) -> pd.Series:
    """Extract hour from start_time robustly, then bucket into 1..6.

    The reference implementation used start_time.str.slice(0,2).astype(int),
    which silently breaks if any time lacks a leading zero (e.g. "8:15" would
    slice to "8:" and fail int()). This uses a regex that tolerates both
    "8:15" and "08:15:00" -- same result on well-formed 2-digit input, safe
    on the single-digit edge case the naive slice would choke on.

    Timezone note: route_schedules.start_time is a local wall-clock departure
    time for its own city, and dim_slot's blocks are local wall-clock bands
    (start_hour 0,4,8,12,16,20). Bucketing the raw hour is therefore correct
    and needs NO tz conversion. The tz hazard flagged in the design doc
    applies to any future path that derives a block from a UTC TIMESTAMP --
    if one is ever added, localize per city first (three timezones:
    America/New_York, America/Denver, America/Chicago).
    """
    hours = start_time.astype(str).str.extract(_HOUR_RE)[0].astype(float)
    block = (hours.fillna(0) // 4 + 1).clip(1, 6).astype(int)
    return block


# --------------------------------------------------------------------------- #
# step 1 -- riders per route per block per day type
# --------------------------------------------------------------------------- #

def route_block_riders(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean riders per DAY on each (route, block, day_type)."""
    sch = raw["route_schedules"].copy()
    sch["time_block_id"] = parse_time_block_id(sch["start_time"])

    r = raw["ridership"].merge(
        sch[["schedule_id", "route_id", "time_block_id", "day_type"]],
        on="schedule_id", how="inner", validate="many_to_one")
    if len(r) != len(raw["ridership"]):
        lost = len(raw["ridership"]) - len(r)
        print(f"  [warn] {lost:,} ridership rows failed to join route_schedules "
              f"(dropped, not fatal)")

    per_day = (r.groupby(["route_id", "time_block_id", "day_type", "date"],
                          observed=True)
                 .actual_ridership.sum().reset_index())
    out = (per_day.groupby(["route_id", "time_block_id", "day_type"], observed=True)
                  .actual_ridership.mean().reset_index()
                  .rename(columns={"actual_ridership": "riders"}))
    return out


def route_block_trips(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Scheduled trips per (route, block, day_type) -- drives headway."""
    sch = raw["route_schedules"].copy()
    sch["time_block_id"] = parse_time_block_id(sch["start_time"])
    return (sch.groupby(["route_id", "time_block_id", "day_type"], observed=True)
               .size().reset_index(name="trips"))


def _wait_seconds(trips_in_block: pd.Series) -> pd.Series:
    """Mean wait for ONE route = its headway / 2. A 4h block is 14,400s.
    Applied per route -- a station served by 6 routes does not give riders a
    sixth of the wait; each rider waits for their own route."""
    headway = 14400.0 / trips_in_block.clip(lower=1)
    return (headway / 2.0).clip(upper=PRIORS["max_wait_seconds"])


# --------------------------------------------------------------------------- #
# step 2 -- allocate route riders onto stops
# --------------------------------------------------------------------------- #

def stop_share(route_stops: pd.DataFrame) -> pd.DataFrame:
    """Fraction of a route's riders present AT each stop (boarding/alighting).

    No origin-destination data exists, so this uses the simplest profile with
    no free parameters: boardings taper linearly, alightings build linearly.
    Through-riders are deliberately excluded (they're inside the vehicle --
    counted by the 'ride' mechanism instead).

    This is also what keeps the brief's "boards on the same route share an
    audience" honest for station screens: a route's riders are DIVIDED across
    its stops rather than replicated at each one, so two stations on one line
    cannot both claim the same bodies.
    """
    rs = route_stops[["route_id", "location_id", "stop_sequence", "num_stops"]].copy()
    n = rs["num_stops"].astype(float)
    k = rs["stop_sequence"].astype(float)
    denom = (n * (n - 1) / 2.0).replace(0.0, np.nan)
    rs["board_share"] = ((n - k).clip(lower=0) / denom).fillna(0.0)
    rs["alight_share"] = ((k - 1).clip(lower=0) / denom).fillna(0.0)
    rs["footfall_share"] = rs["board_share"] + rs["alight_share"]
    return rs


def location_block_people(raw: Dict[str, pd.DataFrame], rbr: pd.DataFrame,
                           rbt: pd.DataFrame) -> pd.DataFrame:
    """People present at each (location, block, day_type) per day, and the
    passenger-weighted mean wait. A location sits on ~2.68 routes on average
    (verified: 2,436 route_stops rows / 910 locations)."""
    share = stop_share(raw["route_stops"])
    m = share.merge(rbr, on="route_id", how="inner")
    m = m.merge(rbt, on=["route_id", "time_block_id", "day_type"], how="left")
    m["people"] = m["riders"] * m["footfall_share"]
    m["wait"] = _wait_seconds(m["trips"].fillna(0))
    m["wait_x_people"] = m["wait"] * m["people"]

    g = (m.groupby(["location_id", "time_block_id", "day_type"], observed=True)
          .agg(people=("people", "sum"),
               wait_x_people=("wait_x_people", "sum"),
               trips=("trips", "sum"),
               n_routes=("route_id", "nunique"))
          .reset_index())
    g["wait_seconds"] = np.where(g["people"] > 0,
                                  g["wait_x_people"] / g["people"], 0.0)
    return g.drop(columns="wait_x_people")


def vehicle_block_people(raw: Dict[str, pd.DataFrame], rbr: pd.DataFrame) -> pd.DataFrame:
    """Riders aboard one vehicle per (block, day_type). Corridor riders split
    across the vehicles assigned to that corridor."""
    r2c = raw["route_stops"][["route_id", "corridor_id"]].drop_duplicates()
    cor = (rbr.merge(r2c, on="route_id", how="inner")
              .groupby(["corridor_id", "time_block_id", "day_type"], observed=True)
              .riders.sum().reset_index())
    fleet = raw["vehicles"].groupby("corridor_id").vehicle_id.nunique().rename("n_veh")
    cor = cor.merge(fleet, on="corridor_id", how="left")
    cor["n_veh"] = cor["n_veh"].fillna(1).clip(lower=1)
    cor["people_per_vehicle"] = cor["riders"] / cor["n_veh"]

    stops = (raw["route_stops"].groupby("corridor_id").num_stops.mean()
             .rename("corridor_stops"))
    return cor.merge(stops, on="corridor_id", how="left")


def corridor_street_people(raw: Dict[str, pd.DataFrame], rbt: pd.DataFrame,
                            rbr: pd.DataFrame) -> pd.DataFrame:
    """Street-side audience for exterior bus panels -- the weakest number
    here. No traffic/pedestrian counts exist anywhere in this dataset, so
    this is an explicit proxy: trips x stops x viewers-per-stop x zone index.
    Confidence for street cells is capped low (see confidence_for)."""
    zi = raw["zones"].copy()
    zi["dens"] = zi["population_density_per_sqkm"] * zi["daytime_population_multiplier"]
    med = zi["dens"].median()
    loc_z = raw["locations"][["location_id", "zone_id"]].merge(
        zi[["zone_id", "dens"]], on="zone_id", how="left")
    route_idx = (raw["route_stops"][["route_id", "location_id", "num_stops"]]
                 .merge(loc_z, on="location_id", how="left")
                 .groupby("route_id")
                 .agg(zone_index=("dens", "mean"), num_stops=("num_stops", "first"))
                 .reset_index())
    route_idx["zone_index"] = (route_idx["zone_index"] / med).fillna(1.0)

    m = rbt.merge(route_idx, on="route_id", how="left")
    m = m.merge(rbr[["route_id", "time_block_id", "day_type"]].assign(has_riders=1),
                on=["route_id", "time_block_id", "day_type"], how="left")
    m = m[m["has_riders"] == 1]
    m["people"] = (m["trips"] * m["num_stops"]
                   * PRIORS["street_viewers_per_trip_per_stop"] * m["zone_index"])

    r2c = raw["route_stops"][["route_id", "corridor_id"]].drop_duplicates()
    return (m.merge(r2c, on="route_id", how="inner")
             .groupby(["corridor_id", "time_block_id", "day_type"], observed=True)
             .people.sum().reset_index()
             .rename(columns={"people": "street_people"}))


# --------------------------------------------------------------------------- #
# block 1 -- the residual, no-service audience
# --------------------------------------------------------------------------- #

def observed_block_price_ratios(bookings_path: Path) -> Dict[int, float]:
    """Median contracted price per slot per day by block, as a ratio of the
    peak block. This is the market's own valuation of each block, and it is
    the only evidence available about block 1's worth -- ridership cannot
    speak, because no trip is scheduled 00:00-04:00.

    Measured on the real file: block 1 = $36.09 (0.421 of peak), block 6 =
    $36.40 (0.425), peak = block 5 at $85.67. The business prices the two
    night blocks essentially identically.
    """
    b = pd.read_csv(bookings_path,
                    usecols=["time_block_id", "contracted_price_per_slot_per_day"])
    med = b.groupby("time_block_id")["contracted_price_per_slot_per_day"].median()
    peak = float(med.max())
    return {int(k): float(v) / peak for k, v in med.items()}


def block1_residual_people(block6_people: pd.Series,
                            price_ratios: Dict[int, float]) -> pd.Series:
    """Block 1's audience, derived from block 6 rather than from ridership.

    THE REVERSAL THIS IMPLEMENTS. It is tempting to call 00:00-04:00 a
    high-confidence zero: no trip is scheduled, so nobody is there. But the
    business sells that inventory for real money -- 8,544 bookings across
    4,397 screens at a $36.09 median, 2,976 of them on metro coaches that
    are not running. A zero makes D4 structurally unable to allocate block 1
    and makes D3's rate card unexplainable. "No scheduled service" and "no
    audience" are different claims and this function keeps them apart.

    The estimate has exactly two factors, one measured and one stated:

      measured  price_ratio[1] / price_ratio[6] = 0.421 / 0.425 = 0.991
                The market values a block-1 slot the same as a block-6 slot.
                Block 6 (20:00-24:00) is the adjacent night block and DOES
                have scheduled service, which makes it the natural anchor.

      stated    block1_straggler_dilution = 0.15
                Price parity cannot mean audience parity: block 6 runs trains
                until 23:00 and block 1 runs none. So the anchor is diluted
                to the share of that late crowd plausibly still present with
                no service at all -- maintenance and cleaning crews, station
                dwell, late stragglers. This is the single free parameter in
                the block-1 model and it is deliberately conservative.

    Net effect ~14.9% of each screen's own block-6 audience. Every cell
    produced here is tagged profile_basis='residual_no_service' and carries
    the low confidence that a stated prior deserves -- so the agent can sell
    block 1 while saying plainly that the number is an estimate, not a
    measurement.
    """
    r1 = price_ratios.get(1, 0.42)
    r6 = price_ratios.get(6, 0.42)
    anchor = (r1 / r6) if r6 > 0 else 1.0
    return block6_people * anchor * PRIORS["block1_straggler_dilution"]


# --------------------------------------------------------------------------- #
# step 3 -- the audience cube
# --------------------------------------------------------------------------- #

def _normalized_index(cube: pd.DataFrame, by: List[str], out_col: str) -> pd.Series:
    """Express `people` as an index with mean 100 inside each `by` group.

    Metro carries 92.0% of all ridership (338.6M vs 29.4M) across fewer
    corridors, and LH dwarfs ACS, so raw volume makes every bus screen and
    every ACS screen look worthless. Normalizing within city and within mode
    is what makes the cross-city rung of the cold-start ladder valid at all.
    Zero-audience cells stay 0 rather than being dragged to the group mean.
    """
    grp = cube.groupby(by, observed=True)["people"].transform("mean")
    idx = np.where(grp > 0, cube["people"] / grp * 100.0, 0.0)
    return pd.Series(idx, index=cube.index, name=out_col)


def build_audience_cube(raw: Dict[str, pd.DataFrame],
                         price_ratios: Dict[int, float]) -> pd.DataFrame:
    """One row per (screen, block, day_type). Unserved blocks are emitted
    explicitly rather than omitted -- so a downstream lookup can always tell
    'no audience' apart from 'no data', and so block 1 has a row to carry its
    residual estimate."""
    rbr = route_block_riders(raw)
    rbt = route_block_trips(raw)

    lbp = location_block_people(raw, rbr, rbt)
    vbp = vehicle_block_people(raw, rbr)
    sbp = corridor_street_people(raw, rbt, rbr)

    sc = raw["screens"].copy()
    sc["mechanism"] = [mechanism_of(t, p)
                       for t, p in zip(sc["screen_type"], sc["position"])]
    sc = sc.merge(raw["locations"][["location_id", "zone_id", "city_zone"]],
                  on="location_id", how="left")
    sc = sc.merge(raw["vehicles"][["vehicle_id", "corridor_id", "vehicle_type"]],
                  on="vehicle_id", how="left")

    blocks = sorted(raw["dim_slot"]["time_block_id"].tolist())
    frames = []
    for day_type in ("weekday", "weekend"):
        grid = sc.assign(key=1).merge(
            pd.DataFrame({"time_block_id": blocks, "key": 1}), on="key").drop(columns="key")
        grid["day_type"] = day_type

        g = grid.merge(
            lbp[lbp["day_type"] == day_type].drop(columns="day_type"),
            on=["location_id", "time_block_id"], how="left")
        g = g.merge(
            vbp[vbp["day_type"] == day_type][["corridor_id", "time_block_id",
                                               "people_per_vehicle", "corridor_stops"]],
            on=["corridor_id", "time_block_id"], how="left")
        g = g.merge(
            sbp[sbp["day_type"] == day_type][["corridor_id", "time_block_id",
                                               "street_people"]],
            on=["corridor_id", "time_block_id"], how="left")
        frames.append(g)

    cube = pd.concat(frames, ignore_index=True)
    for c in ("people", "people_per_vehicle", "street_people", "trips", "n_routes",
              "wait_seconds", "corridor_stops"):
        cube[c] = cube[c].fillna(0.0)

    is_loc = cube["mechanism"].isin(["wait", "pass"])
    is_ride = cube["mechanism"] == "ride"
    is_street = cube["mechanism"] == "street"

    cube["raw_people"] = np.select(
        [is_loc, is_ride, is_street],
        [cube["people"], cube["people_per_vehicle"], cube["street_people"]],
        default=0.0)

    # --- provenance, before block 1 is filled in ------------------------- #
    cube["profile_basis"] = np.where(is_street, "street_proxy", "measured_ridership")

    # --- block 1: residual instead of zero ------------------------------- #
    # Anchor each screen's block 1 on its OWN block 6, so a busy terminal and
    # a quiet suburban stop get proportionate residuals instead of a flat
    # network-wide constant.
    b6 = (cube[cube["time_block_id"] == 6]
          .set_index(["screen_id", "day_type"])["raw_people"])
    key = pd.MultiIndex.from_arrays([cube["screen_id"], cube["day_type"]])
    b6_aligned = pd.Series(b6.reindex(key).to_numpy(), index=cube.index).fillna(0.0)

    is_b1 = cube["time_block_id"] == 1
    residual = block1_residual_people(b6_aligned, price_ratios)
    cube.loc[is_b1, "raw_people"] = residual[is_b1]
    cube.loc[is_b1, "profile_basis"] = "residual_no_service"

    ride = (cube["corridor_stops"] * PRIORS["ride_seconds_per_stop_hop"]
            * PRIORS["ride_trip_fraction"])
    cube["dwell_seconds"] = np.select(
        [cube["mechanism"] == "wait", cube["mechanism"] == "pass", is_ride, is_street],
        [cube["wait_seconds"], PRIORS["pass_dwell_seconds"], ride,
         PRIORS["street_dwell_seconds"]],
        default=0.0)
    cube.loc[cube["raw_people"] <= 0, "dwell_seconds"] = 0.0
    # A residual block-1 viewer is not waiting for a train that is coming --
    # they are dwelling. Give them a stated dwell rather than a headway one,
    # which would otherwise be computed from zero trips.
    cube.loc[is_b1 & (cube["raw_people"] > 0), "dwell_seconds"] = \
        PRIORS["block1_dwell_seconds"]

    vis = [VISIBILITY.get((t, p), 0.5) for t, p in
           zip(cube["screen_type"], cube["position"])]
    cube["visibility"] = np.array(vis) * cube["screen_size"].map(SIZE_FACTOR).fillna(1.0)
    cube["visibility"] = cube["visibility"].clip(0.0, 1.0)

    # pool_id: who shares these bodies. wait/pass -> location, ride -> vehicle,
    # street -> corridor. Critical for D4's reach dedup -- summing over
    # screens without deduplicating by pool double/triple-counts bodies.
    cube["pool_id"] = np.where(
        is_loc, "loc:" + cube["location_id"].astype(str),
        np.where(is_ride, "veh:" + cube["vehicle_id"].astype(str),
                 "str:" + cube["corridor_id"].astype(str)))
    cube["pool_id"] = (cube["pool_id"] + ":" + cube["time_block_id"].astype(str)
                        + ":" + cube["day_type"])

    cube["people"] = cube["raw_people"]

    # anchor_id: the thing whose CONTEXT this screen inherits -- its own
    # location for a fixed screen, its corridor for a vehicle-mounted one.
    # Composition is memoized on this, which is why 133,956 cells need only
    # ~12k distinct composition computations.
    cube["anchor_id"] = np.where(is_loc, cube["location_id"].astype(str),
                                  cube["corridor_id"].astype(str))
    cube["anchor_kind"] = np.where(is_loc, "location", "corridor")

    cube["exposure_index_city"] = _normalized_index(
        cube, ["city_id", "day_type", "time_block_id"], "exposure_index_city")
    cube["exposure_index_mode"] = _normalized_index(
        cube, ["screen_type", "day_type", "time_block_id"], "exposure_index_mode")

    keep = ["screen_id", "city_id", "screen_type", "position", "screen_size",
            "mechanism", "location_id", "vehicle_id", "corridor_id", "zone_id",
            "city_zone", "anchor_id", "anchor_kind", "time_block_id", "day_type",
            "people", "dwell_seconds", "visibility", "pool_id", "trips", "n_routes",
            "profile_basis", "exposure_index_city", "exposure_index_mode"]
    return cube[keep].sort_values(["screen_id", "day_type", "time_block_id"]) \
                     .reset_index(drop=True)


# --------------------------------------------------------------------------- #
# POI context -- the proximity-rule / "filter noise" half of D1
# --------------------------------------------------------------------------- #

def poi_mix(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Weighted POI mix per location, after the proximity filter. Weight is
    footfall, discounted for distance and for being across the road."""
    p = raw["poi"].copy()
    near = p[p["distance_to_location_km"] <= POI_MAX_KM].copy()
    decay = 1.0 - (near["distance_to_location_km"] / POI_MAX_KM) * 0.5
    side = np.where(near["side_of_road"] == "far_side", POI_FAR_SIDE_PENALTY, 1.0)
    near["w"] = near["est_daily_footfall"] * decay * side
    out = (near.groupby(["anchor_location_id", "poi_type"], observed=True)
               .w.sum().reset_index()
               .rename(columns={"anchor_location_id": "location_id"}))
    tot = out.groupby("location_id")["w"].transform("sum")
    out["share"] = out["w"] / tot.replace(0, np.nan)
    return out.dropna(subset=["share"])


def dominant_poi(pm: pd.DataFrame) -> pd.Series:
    if pm.empty:
        return pd.Series(dtype=object)
    idx = pm.groupby("location_id")["share"].idxmax()
    return pm.loc[idx].set_index("location_id")["poi_type"]


def corridor_poi_mix(raw: Dict[str, pd.DataFrame], pm: pd.DataFrame) -> pd.DataFrame:
    """POI mix along a whole corridor, for vehicle-mounted screens (a bus has
    no location_id of its own -- it inherits the context of what it passes)."""
    stops = raw["route_stops"][["corridor_id", "location_id"]].drop_duplicates()
    m = stops.merge(pm, on="location_id", how="inner")
    if m.empty:
        return pd.DataFrame(columns=["corridor_id", "poi_type", "w", "share"])
    out = (m.groupby(["corridor_id", "poi_type"], observed=True)
            .w.sum().reset_index())
    tot = out.groupby("corridor_id")["w"].transform("sum")
    out["share"] = out["w"] / tot.replace(0, np.nan)
    return out.dropna(subset=["share"])


def dominant_corridor_poi(cpm: pd.DataFrame) -> pd.Series:
    if cpm.empty:
        return pd.Series(dtype=object)
    idx = cpm.groupby("corridor_id")["share"].idxmax()
    return cpm.loc[idx].set_index("corridor_id")["poi_type"]


# --------------------------------------------------------------------------- #
# corridor zone blend -- what v1 threw away for 23.4% of inventory
# --------------------------------------------------------------------------- #

ZONE_NUMERIC_COLS = ["income_index", "income_index_city",
                     "daytime_population_multiplier",
                     "population_density_per_sqkm", "pct_age_18_34", "median_age"]


def corridor_zone_context(czw: pd.DataFrame, zctx: pd.DataFrame) -> pd.DataFrame:
    """Ridership-weighted zone ATTRIBUTES for each corridor, plus a readable
    blend string.

    A vehicle-mounted screen has no zone_id, so every zone-level number was
    null for all 2,615 of them -- including daytime_population_multiplier,
    which is the most discriminating field in the dataset. Averaging those
    attributes over the zones a corridor actually carries riders through
    gives a vehicle screen the same explanatory surface a fixed screen has.

    The blend string is the point, though. "mixed" tells a client nothing;
    "Riverside Junction 39%, North Terminus 17%, Financial Row 11% (+6 more
    zones)" is a sentence a rep can read out loud, and it is exactly what the
    design doc's worked example asks D1 to be able to say about corridor
    LH-RT-M010.
    """
    if czw.empty:
        return pd.DataFrame(columns=["corridor_id", "zone_blend_str",
                                      "n_zones"] + ZONE_NUMERIC_COLS)
    avg = (czw.groupby(["corridor_id", "zone_id"], observed=True)["weight"]
              .mean().reset_index())
    avg["weight"] = avg["weight"] / avg.groupby("corridor_id")["weight"].transform("sum")

    m = avg.merge(zctx[["zone_id", "zone_name"] + ZONE_NUMERIC_COLS],
                  on="zone_id", how="left")
    for c in ZONE_NUMERIC_COLS:
        m[f"_w_{c}"] = m[c] * m["weight"]
    agg = m.groupby("corridor_id")[[f"_w_{c}" for c in ZONE_NUMERIC_COLS]].sum()
    agg.columns = ZONE_NUMERIC_COLS
    agg = agg.reset_index()

    rows = []
    for cid, g in m.sort_values("weight", ascending=False).groupby(
            "corridor_id", observed=True):
        top = g.head(3)
        parts = [f"{n} {w:.0%}" for n, w in zip(top["zone_name"], top["weight"])]
        extra = len(g) - len(top)
        if extra > 0:
            parts.append(f"(+{extra} more zone{'s' if extra > 1 else ''})")
        rows.append((str(cid), ", ".join(parts), int(len(g))))
    blend = pd.DataFrame(rows, columns=["corridor_id", "zone_blend_str", "n_zones"])
    return agg.merge(blend, on="corridor_id", how="left")


def corridor_zone_weights(raw: Dict[str, pd.DataFrame],
                           lbp: pd.DataFrame) -> pd.DataFrame:
    """Ridership-weighted share of each zone along a corridor, per block and
    day_type.

    v1 read dominant_occupation off zone_id, which is NULL for every
    vehicle-mounted screen, so all 2,615 of them fell through to the "mixed"
    default. But a corridor crosses 1-9 zones (mean 3.3): corridor LH-RT-M010
    spans nine, led by Riverside Junction at 39%, North Terminus 17% and
    Financial Row 11%. "Mixed" and "a weighted blend of nine named
    neighbourhoods" are not the same claim, and only the second one can be
    explained to a client.

    Weights come from where the riders actually are (location_block_people),
    not from a flat count of stops, so a corridor's blend shifts across the
    day exactly as its ridership does.
    """
    stops = raw["route_stops"][["corridor_id", "location_id"]].drop_duplicates()
    loc_zone = raw["locations"][["location_id", "zone_id"]]
    m = (stops.merge(loc_zone, on="location_id", how="left")
              .merge(lbp[["location_id", "time_block_id", "day_type", "people"]],
                     on="location_id", how="inner"))
    g = (m.groupby(["corridor_id", "time_block_id", "day_type", "zone_id"],
                    observed=True)["people"].sum().reset_index())
    tot = g.groupby(["corridor_id", "time_block_id", "day_type"],
                    observed=True)["people"].transform("sum")
    g["weight"] = g["people"] / tot.replace(0, np.nan)
    return g.dropna(subset=["weight"])


# --------------------------------------------------------------------------- #
# composition -- now per (screen, day_type, time_block)
# --------------------------------------------------------------------------- #

def _normalise(d: Dict[str, float]) -> Dict[str, float]:
    tot = sum(d.values())
    if tot <= 0:
        return {}
    return {k: round(v / tot, 4) for k, v in sorted(d.items(), key=lambda kv: -kv[1])
            if v / tot >= 0.005}


def composition_for(base: Dict[str, float], poi_shares: Dict[str, float],
                     block: int, day_type: str) -> Dict[str, float]:
    """Who is in front of this screen: zone/corridor occupation base + POI
    context tilt + time-of-day tilt, combined openly (not model-derived).

    The day_type argument is the whole point of v2. The same base and the
    same POIs produce a genuinely different mix on a Saturday afternoon than
    on a Tuesday morning, because BLOCK_TILT_WEEKEND suppresses the commuter
    segment that BLOCK_TILT_WEEKDAY amplifies -- which is what lets a profile
    say "office professionals at 08:00 midweek, visitors and shoppers at
    14:00 on a weekend" about one screen.
    """
    out = dict(base)
    for poi_type, share in poi_shares.items():
        seg = POI_SEGMENTS.get(poi_type)
        if seg and share >= POI_MIN_WEIGHT:
            out[seg] = out.get(seg, 0.0) + POI_TILT_STRENGTH * share
    tilt = (BLOCK_TILT_WEEKEND if day_type == "weekend"
            else BLOCK_TILT_WEEKDAY).get(block, {})
    for seg, mult in tilt.items():
        if seg in out:
            out[seg] *= mult
    return _normalise(out)


def _base_mix_from_occupation(occupation: str) -> Dict[str, float]:
    return dict(OCCUPATION_SEGMENTS.get(occupation, OCCUPATION_SEGMENTS["mixed"]))


def _blend_occupation_mixes(zone_weights: Dict[str, float],
                             zone_occupation: Dict[str, str]) -> Dict[str, float]:
    """Weighted mixture of per-zone occupation base mixes."""
    out: Dict[str, float] = {}
    for zone_id, w in zone_weights.items():
        occ = zone_occupation.get(zone_id, "mixed")
        for seg, v in OCCUPATION_SEGMENTS.get(occ, OCCUPATION_SEGMENTS["mixed"]).items():
            out[seg] = out.get(seg, 0.0) + w * v
    return out or dict(OCCUPATION_SEGMENTS["mixed"])


def build_composition_index(
    cube: pd.DataFrame,
    pm: pd.DataFrame,
    cpm: pd.DataFrame,
    czw: pd.DataFrame,
    zone_occupation: Dict[str, str],
    location_zone: Dict[str, str],
) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """Compute every DISTINCT composition once, keyed by
    (anchor_id, day_type, time_block_id).

    Composition depends only on the anchor's zone/corridor blend, the
    anchor's POI mix, the block and the day type -- never on the individual
    screen. There are 910 locations + 94 corridors, so 1,004 anchors x 2 day
    types x 6 blocks = at most 12,048 computations covering all 133,956
    cells. v1 instead re-filtered the whole cube once per screen inside a
    Python loop (an 11,163 x 133,956 scan), which is where its runtime went.
    """
    loc_poi: Dict[str, Dict[str, float]] = {
        str(k): dict(zip(g["poi_type"], g["share"]))
        for k, g in pm.groupby("location_id", observed=True)
    }
    cor_poi: Dict[str, Dict[str, float]] = {
        str(k): dict(zip(g["poi_type"], g["share"]))
        for k, g in cpm.groupby("corridor_id", observed=True)
    } if not cpm.empty else {}

    cor_zone_w: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for (cid, blk, dt), g in czw.groupby(
            ["corridor_id", "time_block_id", "day_type"], observed=True):
        cor_zone_w[(str(cid), str(dt), int(blk))] = dict(zip(g["zone_id"], g["weight"]))

    keys = cube[["anchor_id", "anchor_kind", "day_type", "time_block_id"]].drop_duplicates()
    index: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for anchor_id, kind, day_type, block in keys.itertuples(index=False):
        anchor_id, day_type, block = str(anchor_id), str(day_type), int(block)
        if kind == "location":
            base = _base_mix_from_occupation(
                zone_occupation.get(location_zone.get(anchor_id, ""), "mixed"))
            shares = loc_poi.get(anchor_id, {})
        else:
            zw = cor_zone_w.get((anchor_id, day_type, block))
            if zw:
                base = _blend_occupation_mixes(zw, zone_occupation)
            else:
                # Corridor with no rider-weighted zone blend for this cell
                # (block 1, where nothing runs). Fall back to the corridor's
                # own block-6 blend rather than to a flat "mixed" -- the
                # neighbourhoods a line passes through do not change at
                # midnight, only the volume does.
                zw6 = cor_zone_w.get((anchor_id, day_type, 6)) or {}
                base = (_blend_occupation_mixes(zw6, zone_occupation) if zw6
                        else dict(OCCUPATION_SEGMENTS["mixed"]))
            shares = cor_poi.get(anchor_id, {})
        index[(anchor_id, day_type, block)] = composition_for(
            base, shares, block, day_type)
    return index


def build_mix_long(cube: pd.DataFrame,
                    comp_index: Dict[Tuple[str, str, int], Dict[str, float]]
                    ) -> pd.DataFrame:
    """The fct_screen_audience_mix mart: screen x day_type x block x segment.

    A long table rather than wide columns, so the taxonomy can grow (it just
    did -- healthcare_visitors, tourists and travellers_intercity are new in
    v2) without a schema migration on every consumer.
    """
    rows: List[Tuple] = []
    for sid, dt, blk, anchor, ppl in zip(
            cube["screen_id"], cube["day_type"], cube["time_block_id"],
            cube["anchor_id"], cube["people"]):
        comp = comp_index.get((str(anchor), str(dt), int(blk)))
        if not comp:
            continue
        for seg, share in comp.items():
            rows.append((sid, dt, int(blk), seg, share, float(ppl) * share))
    return pd.DataFrame(rows, columns=["screen_id", "day_type", "time_block_id",
                                        "segment", "share", "segment_people"])


def add_affinity_index(mix: pd.DataFrame, cube: pd.DataFrame) -> pd.DataFrame:
    """Affinity = this screen's share of a segment vs its own city's baseline
    share for the same segment/block/day_type, x100.

    A raw share says "22% of this audience are students". An affinity index
    says "2.4x as student-heavy as the city average at this hour", which is
    the claim a media buyer actually acts on -- and it is what makes the
    ranked list in D2 defensible rather than just ordered.
    """
    city = cube[["screen_id", "city_id"]].drop_duplicates()
    m = mix.merge(city, on="screen_id", how="left")
    base = (m.groupby(["city_id", "day_type", "time_block_id", "segment"],
                      observed=True)["share"].transform("mean"))
    m["affinity_index"] = np.where(base > 0, m["share"] / base * 100.0, 100.0)
    return m.drop(columns="city_id")


# --------------------------------------------------------------------------- #
# cold-start corroboration ladder
# --------------------------------------------------------------------------- #

def history_ladder(screens: pd.DataFrame, bookings: pd.DataFrame,
                    locations: pd.DataFrame, vehicles: pd.DataFrame,
                    cities: pd.DataFrame) -> pd.DataFrame:
    """For every screen, the highest rung of HISTORY_LADDER that can supply
    commercial corroboration, plus how many peers it drew on and their median
    price.

    Returns columns: screen_id, history_basis, history_peer_n,
    history_peer_median_price, history_confidence_mult, history_basis_phrase.

    Read the output as provenance, not as a prediction: a screen on
    'zone_type_peer' has no bookings of its own and no booked neighbour at
    its own stop, so anything commercial said about it comes from comparable
    screens in its zone. Storing that per screen is what lets the agent say
    "estimated from 47 comparable bus stops in standard-tier zones" instead
    of quietly presenting a peer average as fact.
    """
    price_col = "contracted_price_per_slot_per_day"
    booked_price = bookings.groupby("screen_id")[price_col].median()
    booked = set(booked_price.index)

    s = screens[["screen_id", "city_id", "screen_type", "position",
                 "location_id", "vehicle_id"]].copy()
    s = s.merge(locations[["location_id", "zone_id"]], on="location_id", how="left")
    s = s.merge(vehicles[["vehicle_id", "corridor_id"]], on="vehicle_id", how="left")
    s = s.merge(cities[["city_id", "market_tier"]], on="city_id", how="left")
    s["has_own"] = s["screen_id"].isin(booked)

    # A vehicle screen has no zone, so its zone-level rung is its corridor --
    # the closest available "same neighbourhood" notion for something that
    # moves. Documented substitution, not a silent one.
    s["geo_key"] = s["zone_id"].fillna(s["corridor_id"]).fillna("UNKNOWN")
    s["anchor_key"] = s["location_id"].fillna(s["vehicle_id"]).fillna("UNKNOWN")

    s["own_price"] = s["screen_id"].map(booked_price)

    def peer_stats(keys: List[str]) -> pd.DataFrame:
        """Booked-peer count and median price per group, excluding the screen
        itself so a screen never corroborates its own number."""
        g = (s.groupby(keys, observed=True)
              .agg(grp_n=("has_own", "sum"), grp_price_sum=("own_price", "sum"),
                   grp_price_n=("own_price", "count")).reset_index())
        return g

    levels = [
        ("same_anchor",             ["anchor_key"]),
        ("zone_type_peer",          ["geo_key", "screen_type"]),
        ("tier_type_position_peer", ["market_tier", "screen_type", "position"]),
        ("mode_peer",               ["screen_type"]),
    ]
    for name, keys in levels:
        g = peer_stats(keys)
        s = s.merge(g.rename(columns={
            "grp_n": f"n_{name}", "grp_price_sum": f"ps_{name}",
            "grp_price_n": f"pn_{name}"}), on=keys, how="left")
        # exclude self
        own = s["own_price"].fillna(0.0)
        s[f"n_{name}"] = (s[f"n_{name}"].fillna(0) - s["has_own"].astype(int)).clip(lower=0)
        s[f"ps_{name}"] = s[f"ps_{name}"].fillna(0.0) - own
        s[f"pn_{name}"] = (s[f"pn_{name}"].fillna(0)
                            - s["own_price"].notna().astype(int)).clip(lower=0)

    global_price = float(booked_price.median())

    basis: List[str] = []
    peer_n: List[int] = []
    peer_price: List[float] = []
    for row in s.itertuples(index=False):
        if row.has_own:
            basis.append("own_history")
            peer_n.append(1)
            peer_price.append(float(row.own_price))
            continue
        chosen = "global_prior"
        n_used, price_used = 0, global_price
        for name, _keys in levels:
            n = int(getattr(row, f"n_{name}"))
            pn = int(getattr(row, f"pn_{name}"))
            if n > 0 and pn > 0:
                chosen = name
                n_used = n
                price_used = float(getattr(row, f"ps_{name}")) / pn
                break
        basis.append(chosen)
        peer_n.append(n_used)
        peer_price.append(price_used)

    out = pd.DataFrame({
        "screen_id": s["screen_id"].to_numpy(),
        "history_basis": basis,
        "history_peer_n": peer_n,
        "history_peer_median_price": np.round(peer_price, 2),
    })
    out["history_confidence_mult"] = out["history_basis"].map(_LADDER_MULT)
    out["history_basis_phrase"] = out["history_basis"].map(_LADDER_PHRASE)
    return out


# --------------------------------------------------------------------------- #
# confidence
# --------------------------------------------------------------------------- #

def confidence_for(mechanism: str, people: float, n_routes: float,
                    profile_basis: str, history_mult: float) -> Tuple[float, str]:
    """How much to trust this exposure number, and why.

    Three cases the caller must be able to tell apart, which v1 collapsed
    into one:

      people == 0                 a real, explained absence. High confidence:
                                  we know exactly why nobody is there.
      profile_basis == residual   a stated prior, not a measurement. LOW
                                  confidence by construction -- v1 gave this
                                  0.99, which is the opposite of honest.
      otherwise                   measured ridership, adjusted for mechanism,
                                  route thinness, and how far down the
                                  cold-start ladder its corroboration came
                                  from.
    """
    if people <= 0:
        return 0.99, "no audience and no scheduled service in this block"

    if profile_basis == "residual_no_service":
        conf = 0.30 * float(history_mult)
        return round(min(conf, 0.5), 3), (
            "residual overnight presence; no scheduled service, so this is a "
            "stated prior anchored on the block-6 audience and the observed "
            "block-1 price ratio -- an estimate, not a measurement")

    conf = MECHANISM_CONFIDENCE.get(mechanism, 0.5)
    parts = [f"{mechanism} exposure from measured ridership"]
    if mechanism == "ride":
        parts.append("ride duration is a stated prior (no trip times in data)")
    if mechanism == "street":
        parts.append("street volume is a proxy; no traffic or pedestrian counts exist")
        conf = min(conf, PRIORS["street_confidence_cap"])

    if mechanism in ("wait", "pass"):
        if n_routes >= 3:
            conf = min(1.0, conf + 0.05)
            parts.append(f"averaged over {int(n_routes)} routes")
        elif n_routes <= 1:
            conf *= 0.9
            parts.append("single route, thin averaging")

    conf *= float(history_mult)
    return round(min(conf, 0.99), 3), "; ".join(parts)


def attach_cell_confidence(cube: pd.DataFrame, ladder: pd.DataFrame) -> pd.DataFrame:
    """Vectorized confidence + basis text for all 133,956 cells."""
    c = cube.merge(ladder[["screen_id", "history_basis", "history_confidence_mult",
                            "history_basis_phrase"]],
                   on="screen_id", how="left")
    c["history_confidence_mult"] = c["history_confidence_mult"].fillna(0.70)

    base = c["mechanism"].map(MECHANISM_CONFIDENCE).fillna(0.5)
    base = np.where(c["mechanism"] == "street",
                    np.minimum(base, PRIORS["street_confidence_cap"]), base)
    adj = np.where(c["n_routes"] >= 3, np.minimum(1.0, base + 0.05),
                   np.where(c["n_routes"] <= 1, base * 0.9, base))
    adj = np.where(c["mechanism"].isin(["wait", "pass"]), adj, base)
    measured = np.minimum(adj * c["history_confidence_mult"], 0.99)

    residual = np.minimum(0.30 * c["history_confidence_mult"], 0.5)
    is_resid = c["profile_basis"] == "residual_no_service"

    c["cell_confidence"] = np.round(
        np.where(c["people"] <= 0, 0.99,
                 np.where(is_resid, residual, measured)), 3)
    return c


# --------------------------------------------------------------------------- #
# archetype labels -- LLM once per archetype, cached, rules-guardrailed
# --------------------------------------------------------------------------- #

def archetype_key(screen_type: str, position: str, occupation: str, dom_poi: str) -> str:
    return f"{screen_type}|{position or 'interior'}|{occupation}|{dom_poi}"


def _template_label(screen_type: str, position: str, mechanism: str,
                     occupation: str, dom_poi: str) -> str:
    """Deterministic fallback -- runs with NO API key and NO network, and is
    also the guardrail's landing spot when a model returns something the
    rules will not vouch for."""
    who = occupation.replace("_", " ")
    where = dom_poi.replace("_", " ")
    return (f"{who} audience at {where} locations, "
            f"{_MECH_PHRASE.get(mechanism, 'nearby')} "
            f"({screen_type.replace('_', ' ')}, {position or 'interior'})")


_LABEL_SYSTEM = (
    "You write one-sentence audience descriptions for digital out-of-home "
    "advertising screens on urban transit. You are given measured facts about "
    "a group of near-identical screens. Describe WHO is in front of these "
    "screens, WHEN, and WHY, in at most 30 words. Use only the segments and "
    "places given to you -- never introduce an audience the facts do not "
    "support, and never invent a number. Return strict JSON: "
    '{"label": "...", "rationale": "..."}'
)


def _archetype_facts(g: pd.DataFrame, top_segments: List[Tuple[str, float]]) -> str:
    first = g.iloc[0]
    segs = ", ".join(f"{s} {share:.0%}" for s, share in top_segments)
    return json.dumps({
        "screen_type": str(first["screen_type"]),
        "mount_position": str(first["position"]) or "interior",
        "exposure_mechanism": str(first["mechanism"]),
        "mechanism_meaning": _MECH_PHRASE.get(str(first["mechanism"]), ""),
        "zone_dominant_occupation": str(first["dominant_occupation"]),
        "dominant_nearby_poi": str(first["dominant_poi"]),
        "peak_time_block": int(first["peak_block"]) if pd.notna(first["peak_block"]) else 0,
        "audience_segments_at_peak": segs,
        "screens_in_this_group": int(len(g)),
    }, indent=None)


def _llm_client() -> Optional[Tuple[object, str]]:
    """Azure OpenAI client, or None if this machine cannot reach one.

    Deliberately silent-and-optional: the whole build must succeed with no
    credentials, because every NUMBER in D1 is deterministic and only the
    prose label is model-written. Credentials turn a template sentence into a
    better sentence; they are never load-bearing.
    """
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    # NANO first. This is the archetype labeller: one short call per
    # archetype (42 on the current cube) writing a prose label over numbers
    # that are already computed. It is the single largest call count in the
    # project and the least demanding job in it, so it must not bill at mini
    # rates -- roughly 3.7x per token -- against a $5 budget shared by four
    # people. _DEPLOYMENT is kept as the fallback so a machine that only sets
    # the generic name still works.
    deployment = (os.environ.get("AZURE_OPENAI_DEPLOYMENT_NANO")
                  or os.environ.get("AZURE_OPENAI_DEPLOYMENT"))
    if not (key and endpoint and deployment):
        return None
    try:
        from openai import AzureOpenAI
    except ImportError:
        print("  [llm] AZURE_OPENAI_* set but the 'openai' package is not "
              "installed -- falling back to templates.")
        return None
    try:
        client = AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        return client, deployment
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [llm] client construction failed ({exc}) -- using templates.")
        return None


def _guardrail(label: str, allowed_segments: List[str],
                dom_poi: str) -> Optional[str]:
    """Reject a label that asserts an audience the signals do not support.

    This is the "rules guardrail the model" half of the design. The model is
    allowed to phrase things; it is not allowed to introduce a segment that
    is not in this archetype's own composition. Returns None if the label is
    acceptable, otherwise the reason it was rejected.
    """
    if not label or len(label) > 400:
        return "empty or over-long"
    low = label.lower()
    allowed_words = set()
    for seg in allowed_segments:
        allowed_words.update(seg.split("_"))
    allowed_words.update(dom_poi.split("_"))
    for seg in SEGMENT_TAXONOMY:
        if seg in allowed_segments:
            continue
        # a segment is "asserted" if every word of its name appears
        words = [w for w in seg.split("_") if len(w) > 4]
        if words and all(w in low for w in words) and not (set(words) & allowed_words):
            return f"asserts unsupported segment '{seg}'"
    return None


def label_archetypes(frame: pd.DataFrame,
                     comp_at_peak: Dict[str, List[Tuple[str, float]]],
                     cache_path: Path,
                     use_llm: bool = True) -> Tuple[Dict[str, str], Dict[str, int]]:
    """One label per ARCHETYPE, never per screen.

    11,163 screens collapse to a few hundred archetypes
    (screen_type | position | occupation | dominant_poi), so this is a few
    hundred cached calls rather than 11k live ones -- which is the only
    reason an LLM can be in this pipeline at all while D1 still answers a
    query in sub-second time. Serving never calls a model; it reads these
    strings.

    Cache key includes PROMPT_VERSION, so changing the prompt or the facts
    fed into it invalidates old labels instead of silently keeping them.
    """
    cache: Dict[str, Dict[str, str]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  [llm] cache at {cache_path.name} unreadable -- rebuilding.")
            cache = {}

    # Gate only: is there any point attempting a call? llm_agent resolves its
    # own deployment and enforces the fence, so this no longer decides HOW to
    # call, only WHETHER. None here means "go straight to templates".
    conn = (_llm_client() if (use_llm and _llm is not None) else None)
    stats = {"archetypes": 0, "from_cache": 0, "from_llm": 0,
             "from_template": 0, "guardrail_rejects": 0, "llm_errors": 0}

    labels: Dict[str, str] = {}
    for key, g in frame.groupby("archetype_id", observed=True):
        key = str(key)
        stats["archetypes"] += 1
        first = g.iloc[0]
        template = _template_label(
            str(first["screen_type"]), str(first["position"]),
            str(first["mechanism"]), str(first["dominant_occupation"]),
            str(first["dominant_poi"]))

        cache_key = f"{PROMPT_VERSION}|{key}"
        if cache_key in cache:
            labels[key] = cache[cache_key]["label"]
            stats["from_cache"] += 1
            continue

        if conn is None:
            labels[key] = template
            stats["from_template"] += 1
            continue

        top = comp_at_peak.get(key, [])
        try:
            # ROUTED THROUGH llm_agent, and that is the whole point of this
            # change. This loop used to hold its own AzureOpenAI client, so:
            #   * none of its calls reached llm_spend.json -- every
            #     "we have spent $0.018" claim silently excluded 224 calls;
            #   * nothing capped it. LLM_MAX_CALLS never applied here, and the
            #     count is DATA-DEPENDENT: the docstring above says 42
            #     archetypes, the current cube has 224. It grew 5x when the
            #     audience mix went time-varying and nobody noticed, because
            #     there was no meter to notice with;
            #   * output length was unbounded -- no max_completion_tokens;
            #   * `temperature=0` is REJECTED by gpt-5.4 with a 400, so every
            #     call failed into the template fallback and the except below
            #     swallowed it. The labels looked fine because templates are
            #     decent. The model was never actually consulted.
            # One call site, fenced, metered, cached, 429-aware.
            raw = _llm.call(_LABEL_SYSTEM, _archetype_facts(g, top),
                            tier="nano", max_tokens=200, json_mode=True,
                            verbose=False)
            if raw is None:
                raise RuntimeError("no completion (fence, credentials or API)")
            payload = json.loads(raw)
            candidate = str(payload.get("label", "")).strip()
            reason = _guardrail(candidate, [s for s, _ in top],
                                str(first["dominant_poi"]))
            if reason:
                stats["guardrail_rejects"] += 1
                labels[key] = template
                stats["from_template"] += 1
            else:
                labels[key] = candidate
                cache[cache_key] = {"label": candidate,
                                     "rationale": str(payload.get("rationale", ""))}
                stats["from_llm"] += 1
        except Exception as exc:                                # noqa: BLE001
            stats["llm_errors"] += 1
            if stats["llm_errors"] <= 3:
                print(f"  [llm] call failed for {key}: {exc}")
            labels[key] = template
            stats["from_template"] += 1

    if stats["from_llm"]:
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return labels, stats


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #

class AudienceEngine:
    """Builds and serves audience profiles for the whole network.

    Everything numeric here is deterministic and reproducible with no network
    access. The ONLY model-written artifact is the per-archetype prose label
    (see label_archetypes), which is cached and guardrailed by rules. That
    split is deliberate: "AI offline, rules online" keeps query-time serving
    to plain lookups, which is what makes Performance & Scalability defensible
    at 11,163 screens.
    """

    def __init__(self, raw: Dict[str, pd.DataFrame],
                 bookings: Optional[pd.DataFrame] = None,
                 use_llm: bool = True):
        self.raw = raw
        self.use_llm = use_llm
        self.bookings = bookings if bookings is not None else pd.DataFrame(
            columns=["screen_id", "contracted_price_per_slot_per_day"])
        self.price_ratios = {}
        self.cube: Optional[pd.DataFrame] = None
        self.mix: Optional[pd.DataFrame] = None
        self.ladder: Optional[pd.DataFrame] = None
        self.label_stats: Dict[str, int] = {}
        self._frame: Optional[pd.DataFrame] = None
        self._archetype_labels: Dict[str, str] = {}
        self._comp_index: Dict[Tuple[str, str, int], Dict[str, float]] = {}
        self._poi_mix: Optional[pd.DataFrame] = None
        self._corridor_poi_mix: Optional[pd.DataFrame] = None
        self._zone_ctx: Optional[pd.DataFrame] = None

    # -- zone context ----------------------------------------------------- #

    def zone_context(self) -> pd.DataFrame:
        """Zone attributes worth carrying onto a profile, including the
        within-city income index the source data does not provide.

        income_index in zone_demographics is built off a single global base
        (city means 106.7 / 107.1 / 110.3) even though real median household
        incomes differ by 1.8x across the three cities. Used naively it makes
        an ACS zone look as affluent as an LH zone. income_index_city
        re-bases it inside each city (city mean = 100) so "affluent for this
        market" and "affluent nationally" stop being the same number.

        daytime_population_multiplier (range 0.58-3.39) is carried through
        untouched because it is the single most discriminating field in the
        dataset: it separates a business district that triples during the
        working day from an affluent suburb that empties, when no residential
        demographic can tell them apart.
        """
        if self._zone_ctx is not None:
            return self._zone_ctx
        z = self.raw["zones"].copy()
        city_mean = z.groupby("city_id")["income_index"].transform("mean")
        z["income_index_city"] = (z["income_index"] / city_mean * 100.0).round(2)
        self._zone_ctx = z[["zone_id", "city_id", "zone_name", "dominant_occupation",
                            "income_index", "income_index_city",
                            "daytime_population_multiplier",
                            "population_density_per_sqkm", "pct_age_18_34",
                            "median_age"]]
        return self._zone_ctx

    # -- build ------------------------------------------------------------ #

    def build(self) -> "AudienceEngine":
        self.price_ratios = observed_block_price_ratios(DATA / "bookings.csv")
        print("  observed block price ratios vs peak: "
              + ", ".join(f"b{b}={self.price_ratios[b]:.3f}"
                          for b in sorted(self.price_ratios)))

        self.cube = build_audience_cube(self.raw, self.price_ratios)

        pm = poi_mix(self.raw)
        self._poi_mix = pm
        self._corridor_poi_mix = corridor_poi_mix(self.raw, pm)
        dom = dominant_poi(pm)
        dom_cor = dominant_corridor_poi(self._corridor_poi_mix)

        # corridor zone blend needs location-level people, recomputed here
        # from the same primitives the cube used
        rbr = route_block_riders(self.raw)
        rbt = route_block_trips(self.raw)
        lbp = location_block_people(self.raw, rbr, rbt)
        czw = corridor_zone_weights(self.raw, lbp)

        zctx = self.zone_context()
        zone_occ = dict(zip(zctx["zone_id"], zctx["dominant_occupation"]))
        loc_zone = dict(zip(self.raw["locations"]["location_id"],
                            self.raw["locations"]["zone_id"]))

        print("  computing time-varying composition "
              f"({self.cube[['anchor_id', 'day_type', 'time_block_id']].drop_duplicates().shape[0]:,} "
              "distinct anchor x day_type x block combinations)...")
        self._comp_index = build_composition_index(
            self.cube, pm, self._corridor_poi_mix, czw, zone_occ, loc_zone)

        print("  running cold-start corroboration ladder...")
        self.ladder = history_ladder(
            self.raw["screens"], self.bookings, self.raw["locations"],
            self.raw["vehicles"], self.raw["cities"])
        self.cube = attach_cell_confidence(self.cube, self.ladder)

        print("  building long audience-mix mart...")
        mix = build_mix_long(self.cube, self._comp_index)
        self.mix = add_affinity_index(mix, self.cube)

        # -- per-screen frame ---------------------------------------------- #
        wd = self.cube[self.cube["day_type"] == "weekday"]
        agg = (wd.groupby("screen_id", observed=True)
                 .agg(daily_people=("people", "sum"),
                      n_routes=("n_routes", "max"),
                      mean_confidence=("cell_confidence", "mean"),
                      exposure_index_city=("exposure_index_city", "mean"),
                      exposure_index_mode=("exposure_index_mode", "mean"))
                 .reset_index())
        # Peak block, measured on served blocks only -- otherwise a screen
        # whose only non-zero block is the block-1 residual would report
        # midnight as its peak.
        served = wd[wd["profile_basis"] != "residual_no_service"]
        peak = (served.loc[served.groupby("screen_id", observed=True)["people"].idxmax(),
                           ["screen_id", "time_block_id"]]
                      .rename(columns={"time_block_id": "peak_block"}))
        we = self.cube[self.cube["day_type"] == "weekend"]
        we_served = we[we["profile_basis"] != "residual_no_service"]
        peak_we = (we_served.loc[we_served.groupby("screen_id", observed=True)["people"].idxmax(),
                                 ["screen_id", "time_block_id"]]
                            .rename(columns={"time_block_id": "peak_block_weekend"}))

        f = (self.cube[["screen_id", "city_id", "screen_type", "position",
                        "mechanism", "location_id", "corridor_id", "zone_id",
                        "anchor_id"]]
             .drop_duplicates("screen_id")
             .merge(agg, on="screen_id", how="left")
             .merge(peak, on="screen_id", how="left")
             .merge(peak_we, on="screen_id", how="left")
             .merge(self.ladder, on="screen_id", how="left"))
        f["peak_block"] = f["peak_block"].fillna(3).astype(int)
        f["peak_block_weekend"] = f["peak_block_weekend"].fillna(4).astype(int)

        f = f.merge(zctx.drop(columns="city_id"), on="zone_id", how="left")

        # Vehicle screens have no zone of their own. v1 left every zone-level
        # field null for all 2,615 of them and fell back to the "mixed"
        # occupation label; both are now filled from the corridor's
        # ridership-weighted blend of the zones it actually serves.
        cor_dom = self._corridor_dominant_occupation(czw, zone_occ)
        f["dominant_occupation"] = np.where(
            f["dominant_occupation"].isna(),
            f["corridor_id"].map(cor_dom),
            f["dominant_occupation"])
        f["dominant_occupation"] = f["dominant_occupation"].fillna("mixed")

        cor_ctx = corridor_zone_context(czw, zctx).set_index("corridor_id")
        for c in ZONE_NUMERIC_COLS:
            if c in f.columns:
                f[c] = f[c].fillna(f["corridor_id"].map(cor_ctx[c]))
        blend = f["corridor_id"].map(cor_ctx["zone_blend_str"])
        n_zones = f["corridor_id"].map(cor_ctx["n_zones"])
        # zone_name stays the single zone for a fixed screen, and becomes an
        # explicit "corridor blend" marker for a moving one -- never a silent
        # blank, and never a single zone name that would imply a precision a
        # vehicle screen does not have.
        f["zone_name"] = f["zone_name"].fillna(
            "corridor blend of " + n_zones.astype("Int64").astype(str) + " zones")
        f["zone_blend_str"] = np.where(
            f["zone_id"].notna(), f["zone_name"].astype(str) + " 100%",
            blend.fillna("unknown"))
        f["n_zones"] = np.where(f["zone_id"].notna(), 1, n_zones.fillna(0)).astype(int)

        f["dominant_poi"] = (f["location_id"].map(dom)
                              .fillna(f["corridor_id"].map(dom_cor))
                              .fillna("none"))
        f["archetype_id"] = [
            archetype_key(t, p, o, dp) for t, p, o, dp in
            zip(f["screen_type"], f["position"], f["dominant_occupation"],
                f["dominant_poi"])]
        self._frame = f

        comp_at_peak = self._composition_at_peak_by_archetype()
        print(f"  labelling {f['archetype_id'].nunique()} archetypes...")
        self._archetype_labels, self.label_stats = label_archetypes(
            f, comp_at_peak, OUT / "d1_archetype_labels.cache.json",
            use_llm=self.use_llm)
        return self

    def _corridor_dominant_occupation(self, czw: pd.DataFrame,
                                       zone_occ: Dict[str, str]) -> Dict[str, str]:
        """The occupation of the highest-weighted zone on each corridor,
        averaged over the day. Used only as a LABEL -- the composition itself
        uses the full blend, not this single winner."""
        if czw.empty:
            return {}
        avg = (czw.groupby(["corridor_id", "zone_id"], observed=True)["weight"]
                  .mean().reset_index())
        idx = avg.groupby("corridor_id")["weight"].idxmax()
        top = avg.loc[idx]
        return {str(c): zone_occ.get(str(z), "mixed")
                for c, z in zip(top["corridor_id"], top["zone_id"])}

    def _composition_at_peak_by_archetype(self) -> Dict[str, List[Tuple[str, float]]]:
        """Top segments at each archetype's weekday peak block -- the facts
        the label prompt is allowed to describe."""
        out: Dict[str, List[Tuple[str, float]]] = {}
        assert self._frame is not None
        for key, g in self._frame.groupby("archetype_id", observed=True):
            first = g.iloc[0]
            comp = self._comp_index.get(
                (str(first["anchor_id"]), "weekday", int(first["peak_block"])), {})
            out[str(key)] = sorted(comp.items(), key=lambda kv: -kv[1])[:4]
        return out

    # -- serving ---------------------------------------------------------- #

    def composition(self, screen_id: str, day_type: str = "weekday",
                     time_block_id: int = 3) -> Dict[str, float]:
        """The agent-facing lookup: who is at this screen, at this hour, on
        this kind of day. A dict lookup -- no model, no scan."""
        assert self._frame is not None
        row = self._frame[self._frame["screen_id"] == screen_id]
        if row.empty:
            return {}
        anchor = str(row.iloc[0]["anchor_id"])
        return self._comp_index.get((anchor, day_type, int(time_block_id)), {})

    def table(self) -> pd.DataFrame:
        """Whole-network view, one row per screen -- D2's scoring input.

        Schema is a superset of v1's: every column D2 already reads
        (composition_str, poi_mix_str, confidence, has_booking_history,
        peak_block, daily_people, label, mechanism, archetype_id) is still
        here with the same meaning. composition_str remains the WEEKDAY PEAK
        mix so existing consumers are unaffected; the weekend peak mix and
        the full time-varying detail are new columns / the long mart.
        """
        if self._frame is None:
            self.build()
        assert self._frame is not None and self.cube is not None
        f = self._frame.copy()

        def comp_str(anchor: str, day_type: str, block: int) -> str:
            d = self._comp_index.get((str(anchor), day_type, int(block)), {})
            return "; ".join(f"{k}:{v:.2f}" for k, v in d.items())

        f["composition_str"] = [comp_str(a, "weekday", b) for a, b in
                                zip(f["anchor_id"], f["peak_block"])]
        f["composition_weekend_str"] = [comp_str(a, "weekend", b) for a, b in
                                        zip(f["anchor_id"], f["peak_block_weekend"])]

        loc_poi = {str(k): _normalise(dict(zip(g["poi_type"], g["share"])))
                   for k, g in self._poi_mix.groupby("location_id", observed=True)}
        cor_poi = ({str(k): _normalise(dict(zip(g["poi_type"], g["share"])))
                    for k, g in self._corridor_poi_mix.groupby("corridor_id", observed=True)}
                   if not self._corridor_poi_mix.empty else {})

        def poi_str(row) -> str:
            d = loc_poi.get(str(row.location_id)) or cor_poi.get(str(row.corridor_id)) or {}
            return "; ".join(f"{k}:{v:.2f}" for k, v in d.items())

        f["poi_mix_str"] = [poi_str(r) for r in f.itertuples(index=False)]
        f["label"] = f["archetype_id"].map(self._archetype_labels).fillna("")
        f["confidence"] = f["mean_confidence"].round(3)
        f["has_booking_history"] = f["history_basis"] == "own_history"

        cols = ["screen_id", "city_id", "zone_id", "zone_name", "zone_blend_str",
                "n_zones", "mechanism",
                "archetype_id", "peak_block", "peak_block_weekend", "daily_people",
                "exposure_index_city", "exposure_index_mode", "confidence",
                "has_booking_history", "history_basis", "history_basis_phrase",
                "history_peer_n", "history_peer_median_price",
                "dominant_occupation", "dominant_poi", "income_index",
                "income_index_city", "daytime_population_multiplier",
                "label", "composition_str", "composition_weekend_str", "poi_mix_str"]
        return f[[c for c in cols if c in f.columns]]

    def profile_cards(self) -> List[dict]:
        """One JSON card per screen -- the denormalized object D5's agent
        reads instead of re-joining marts at query time. Carries the mix for
        every served block so the agent can answer "who is here at 8am" and
        "who is here on Saturday afternoon" from one object, plus the basis
        and confidence for every claim it makes."""
        assert self._frame is not None and self.cube is not None
        cube_by_screen = {
            sid: g for sid, g in self.cube.groupby("screen_id", observed=True)}
        cards: List[dict] = []
        for r in self._frame.itertuples(index=False):
            g = cube_by_screen.get(r.screen_id)
            blocks = []
            if g is not None:
                for c in g.itertuples(index=False):
                    if c.people <= 0:
                        continue
                    comp = self._comp_index.get(
                        (str(r.anchor_id), str(c.day_type), int(c.time_block_id)), {})
                    blocks.append({
                        "day_type": c.day_type,
                        "time_block_id": int(c.time_block_id),
                        "people": round(float(c.people), 1),
                        "dwell_seconds": round(float(c.dwell_seconds), 1),
                        "exposure_index_city": round(float(c.exposure_index_city), 1),
                        "profile_basis": c.profile_basis,
                        "confidence": float(c.cell_confidence),
                        "top_segments": dict(list(comp.items())[:4]),
                    })
            cards.append({
                "screen_id": r.screen_id,
                "city_id": r.city_id,
                "zone_id": r.zone_id,
                "screen_type": r.screen_type,
                "position": r.position or "interior",
                "mechanism": r.mechanism,
                "mechanism_meaning": _MECH_PHRASE.get(str(r.mechanism), ""),
                "archetype_id": r.archetype_id,
                "label": self._archetype_labels.get(str(r.archetype_id), ""),
                "dominant_occupation": r.dominant_occupation,
                "dominant_poi": r.dominant_poi,
                "peak_block_weekday": int(r.peak_block),
                "peak_block_weekend": int(r.peak_block_weekend),
                "confidence": round(float(r.mean_confidence), 3),
                "history_basis": r.history_basis,
                "history_basis_phrase": r.history_basis_phrase,
                "history_peer_n": int(r.history_peer_n),
                "zone_context": {
                    "zone_name": getattr(r, "zone_name", None),
                    "zone_blend": getattr(r, "zone_blend_str", None),
                    "n_zones": int(getattr(r, "n_zones", 0) or 0),
                    "income_index": _opt_float(getattr(r, "income_index", None)),
                    "income_index_city": _opt_float(getattr(r, "income_index_city", None)),
                    "daytime_population_multiplier":
                        _opt_float(getattr(r, "daytime_population_multiplier", None)),
                },
                "blocks": blocks,
            })
        return cards


def _opt_float(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# build artifacts
# --------------------------------------------------------------------------- #

def write_taxonomy(path: Path) -> None:
    """The segment contract, published so D2/D3/D5 have exactly one source of
    truth for the vocabulary rather than each hardcoding its own guess."""
    payload = {
        "version": "d1-taxonomy-v2",
        "note": "Segment keys are the strings that appear in "
                "audience_mix_long.segment and in composition_str. "
                "research_alias records the longer name used in the design "
                "doc. No gender facet exists: the dataset has no gender data.",
        "segments": SEGMENT_TAXONOMY,
        "poi_type_to_segment": POI_SEGMENTS,
        "occupation_base_mix": OCCUPATION_SEGMENTS,
        "block_tilt_weekday": BLOCK_TILT_WEEKDAY,
        "block_tilt_weekend": BLOCK_TILT_WEEKEND,
        "profile_basis_values": {
            "measured_ridership": "exposure derived from actual_ridership joined "
                                   "through route_schedules to stops/vehicles",
            "street_proxy": "exterior bus panel; no traffic or pedestrian counts "
                             "exist in the dataset, so volume is a stated proxy",
            "residual_no_service": "block 1 (00:00-04:00); no scheduled service, "
                                    "so exposure is anchored on block 6 and the "
                                    "observed block-1 price ratio",
        },
        "history_ladder": [
            {"rung": i + 1, "basis": name, "confidence_multiplier": mult,
             "phrase": phrase}
            for i, (name, mult, phrase) in enumerate(HISTORY_LADDER)
        ],
        "priors": PRIORS,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_report(eng: AudienceEngine, raw: Dict[str, pd.DataFrame]) -> dict:
    """Coverage + data-quality summary, emitted as a build artifact so a
    reviewer can see what the run actually produced without rerunning it."""
    cube, mix, ladder = eng.cube, eng.mix, eng.ladder
    assert cube is not None and mix is not None and ladder is not None
    wd = cube[cube["day_type"] == "weekday"]
    n_screens = int(cube["screen_id"].nunique())

    share_sums = mix.groupby(["screen_id", "day_type", "time_block_id"],
                              observed=True)["share"].sum()
    off = share_sums[(share_sums - 1.0).abs() > 0.02]

    return {
        "rows": {
            "audience_cube": int(len(cube)),
            "audience_mix_long": int(len(mix)),
            "screens": n_screens,
            "expected_cube_rows": n_screens * 6 * 2,
        },
        "coverage": {
            "cells_with_audience_pct": round(100.0 * float((cube["people"] > 0).mean()), 2),
            "screens_with_any_audience": int((cube.groupby("screen_id")["people"].sum() > 0).sum()),
            "block1_cells_with_residual": int(
                ((cube["time_block_id"] == 1) & (cube["people"] > 0)).sum()),
            "block1_residual_share_of_weekday_people": round(float(
                wd.loc[wd["time_block_id"] == 1, "people"].sum()
                / max(wd["people"].sum(), 1e-9)), 4),
        },
        "profile_basis_counts": {k: int(v) for k, v in
                                 cube["profile_basis"].value_counts().items()},
        "history_ladder_counts": {k: int(v) for k, v in
                                   ladder["history_basis"].value_counts().items()},
        "confidence": {
            "mean_cell_confidence": round(float(cube["cell_confidence"].mean()), 3),
            "mean_where_measured": round(float(
                cube.loc[cube["profile_basis"] == "measured_ridership",
                         "cell_confidence"].mean()), 3),
            "mean_where_residual": round(float(
                cube.loc[cube["profile_basis"] == "residual_no_service",
                         "cell_confidence"].mean()), 3),
            "mean_where_street_proxy": round(float(
                cube.loc[cube["profile_basis"] == "street_proxy",
                         "cell_confidence"].mean()), 3),
        },
        "quality_checks": {
            "cube_row_count_matches_grain": int(len(cube)) == n_screens * 6 * 2,
            "no_duplicate_cube_keys": bool(
                not cube.duplicated(["screen_id", "day_type", "time_block_id"]).any()),
            "mix_shares_sum_to_one": int(len(off)) == 0,
            "mix_share_violations": int(len(off)),
            "all_screens_present": n_screens == int(raw["screens"]["screen_id"].nunique()),
            "segments_used": sorted(mix["segment"].unique().tolist()),
            "unknown_segments": sorted(
                set(mix["segment"].unique()) - set(SEGMENT_TAXONOMY)),
        },
        "observed_block_price_ratios": {str(k): round(v, 4)
                                        for k, v in sorted(eng.price_ratios.items())},
        "label_stats": eng.label_stats,
        "priors": PRIORS,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build D1 audience profiles.")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the archetype label pass entirely (templates only)")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    print("loading raw tables...")
    raw = load_raw()
    for name, df in raw.items():
        print(f"  loaded {name}: {df.shape}")

    bookings = pd.read_csv(
        DATA / "bookings.csv",
        usecols=["screen_id", "time_block_id", "contracted_price_per_slot_per_day"])
    n_booked = bookings["screen_id"].nunique()
    n_screens = raw["screens"]["screen_id"].nunique()
    print(f"\n{n_booked:,} of {n_screens:,} screens have booking history "
          f"({n_booked / n_screens:.1%}); {n_screens - n_booked:,} are cold-start")

    print("\nbuilding audience cube (screen x block x day_type)...")
    eng = AudienceEngine(raw, bookings=bookings, use_llm=not args.no_llm).build()
    cube, mix, ladder = eng.cube, eng.mix, eng.ladder
    assert cube is not None and mix is not None and ladder is not None

    print(f"\n  cube: {len(cube):,} rows "
          f"({cube['screen_id'].nunique():,} screens x "
          f"{cube['time_block_id'].nunique()} blocks x "
          f"{cube['day_type'].nunique()} day types)")
    served = cube[cube["people"] > 0]
    print(f"  cells with an audience: {len(served):,} "
          f"({100 * len(served) / len(cube):.1f}%)")

    # Totals must be deduplicated by pool -- up to dozens of screens can share
    # one metro station's waiting audience; summing over screens without
    # dedup counts those bodies once per screen, wildly overstating reach.
    wd = cube[cube["day_type"] == "weekday"]

    def dedup(frame: pd.DataFrame) -> float:
        return frame.groupby("pool_id")["people"].first().sum()

    print("\ndistinct people per weekday by block (deduplicated by pool):")
    for b, g in wd.groupby("time_block_id"):
        basis = g["profile_basis"].mode().iat[0]
        print(f"  block {b}: {dedup(g):11,.0f} people   "
              f"{100 * (g['people'] > 0).mean():5.1f}% of screens served   "
              f"[{basis}, mean conf {g['cell_confidence'].mean():.2f}]")

    print("\nby mechanism (weekday, deduplicated):")
    for m, g in wd.groupby("mechanism"):
        dwv = g[g["dwell_seconds"] > 0]["dwell_seconds"]
        print(f"  {m:7s} {g['screen_id'].nunique():6,} screens / "
              f"{g['pool_id'].nunique():5,} pools  {dedup(g):10,.0f} people/day  "
              f"median dwell {dwv.median() if len(dwv) else 0:5.0f}s")

    print("\ncold-start corroboration ladder (rung actually used per screen):")
    for name, mult, phrase in HISTORY_LADDER:
        n = int((ladder["history_basis"] == name).sum())
        if n:
            print(f"  {name:24s} {n:6,} screens  conf x{mult:.2f}  -- {phrase}")

    # --- write artifacts -------------------------------------------------- #
    cube_path = OUT / "audience_cube.csv"
    cube.to_csv(cube_path, index=False)
    print(f"\nsaved: {cube_path.name} -> {cube.shape[0]:,} rows x {cube.shape[1]} cols")

    mix_path = OUT / "audience_mix_long.csv"
    mix.to_csv(mix_path, index=False)
    print(f"saved: {mix_path.name} -> {mix.shape[0]:,} rows x {mix.shape[1]} cols "
          f"({mix['segment'].nunique()} segments)")

    print("\nbuilding per-screen audience profiles (D2's scoring input)...")
    profiles = eng.table()
    prof_path = OUT / "audience_profiles.csv"
    profiles.to_csv(prof_path, index=False)
    print(f"saved: {prof_path.name} -> {profiles.shape[0]:,} rows x "
          f"{profiles.shape[1]} cols")

    # --- the claim v1 could not make ------------------------------------- #
    # Self-selecting demo: the screens whose audience MIX moves most between a
    # weekday commuter peak (block 2) and a weekend afternoon (block 4). v1
    # could not express this at all -- it had one composition per screen, so
    # both columns below would have been identical by construction.
    print("\nsame screen, two different audiences (what time-varying mix buys):")
    shift = (mix[((mix["day_type"] == "weekday") & (mix["time_block_id"] == 2))
                  | ((mix["day_type"] == "weekend") & (mix["time_block_id"] == 4))]
             .pivot_table(index=["screen_id", "segment"], columns="day_type",
                          values="share", fill_value=0.0))
    if {"weekday", "weekend"}.issubset(shift.columns):
        lookup = profiles.set_index("screen_id")
        l1 = ((shift["weekday"] - shift["weekend"]).abs()
              .groupby(level="screen_id").sum().sort_values(ascending=False))
        # One screen per anchor: neighbouring screens at the same stop share
        # an anchor and therefore an identical mix, so an undeduplicated top-N
        # just prints the same profile twice.
        anchor_of = dict(zip(cube["screen_id"], cube["anchor_id"]))
        picks, seen = [], set()
        for sid in l1.index:
            a = anchor_of.get(sid)
            if a in seen:
                continue
            seen.add(a)
            picks.append(sid)
            if len(picks) == 2:
                break
        for sid in picks:
            row = lookup.loc[sid] if sid in lookup.index else None
            if row is not None:
                print(f"  {sid}  ({row['zone_name']}, {row['dominant_occupation']}, "
                      f"daytime mult {row['daytime_population_multiplier']:.2f}, "
                      f"nearest POI {row['dominant_poi']})")
            for dt, blk in (("weekday", 2), ("weekend", 4)):
                comp = eng.composition(sid, dt, blk)
                top = ", ".join(f"{k} {v:.0%}" for k, v in list(comp.items())[:3])
                print(f"      {dt:8s} block {blk}: {top}")

    cards = eng.profile_cards()
    cards_path = OUT / "d1_screen_profile_cards.jsonl"
    with cards_path.open("w", encoding="utf-8") as fh:
        for c in cards:
            fh.write(json.dumps(c) + "\n")
    print(f"saved: {cards_path.name} -> {len(cards):,} cards")

    write_taxonomy(OUT / "d1_audience_taxonomy.json")
    print(f"saved: d1_audience_taxonomy.json -> {len(SEGMENT_TAXONOMY)} segments")

    report = build_report(eng, raw)
    (OUT / "d1_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("saved: d1_build_report.json")

    ls = eng.label_stats
    print(f"\narchetype labels: {ls.get('archetypes', 0)} archetypes -> "
          f"{ls.get('from_llm', 0)} from LLM, {ls.get('from_cache', 0)} cached, "
          f"{ls.get('from_template', 0)} templated"
          + (f", {ls['guardrail_rejects']} guardrail rejects"
             if ls.get("guardrail_rejects") else ""))
    if not ls.get("from_llm") and not ls.get("from_cache"):
        print("  (no AZURE_OPENAI_* credentials found -- every NUMBER above is "
              "unaffected; only the prose label is templated)")

    checks = report["quality_checks"]
    failed = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    if failed:
        print(f"\nFAILED quality checks: {failed}")
        return 1
    print("\nAll quality checks passed. D1 v2 built with no hard LLM/API dependency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

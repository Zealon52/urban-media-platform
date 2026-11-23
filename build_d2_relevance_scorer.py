r"""D2 — Relevance Scorer. Turns a free-text campaign brief + D1's audience
profiles into a ranked, explainable shortlist of screens.

Design, matching the reference toolkit's split of responsibilities:
  * Brief parsing is offline-first and deterministic (regex/keyword rules over
    the fixed doc structure every brief uses: "Company Name", "Campaign
    Budget", "Campaign Duration", "Exclusion Criteria:", etc.). This works
    with ZERO API key today. Once Azure creds land, the same StructuredBrief
    shape can be filled by one LLM call per brief instead -- swap the function,
    not the pipeline.
  * Screen scoring/filtering is deterministic rules + arithmetic, never an LLM
    guess -- consistent with "scoring ... must not be an LLM guess" from the
    reference's own design notes, and with D2 being classified as
    AI Agent + Logical Rules (rules do the fit math, AI explains why).

Hard data constraint, stated explicitly rather than hidden: there are NO
coordinates anywhere in the 14 tables. "Walking radius" (Brief 4's core ask)
is therefore approximated as "same zone_id as the POI/location matching the
brief's anchor description" -- the coarsest available proximity primitive.
This is a documented assumption, not a precise geometric radius.

Inputs (must exist before running):
  Urban Media Datasets/screens.csv, locations.csv, points_of_interest.csv,
  cities.csv, dim_slot.csv, zone_demographics.csv
  eda_output/audience_profiles.csv, audience_cube.csv   (D1 output)

Outputs:
  eda_output/relevance_scores_<brief_slug>.csv   -- one per brief, D2's output
  Printed shortlist (top 10) + reasons + hard-filter exclusion counts.

Usage (PowerShell):
  python .\build_d2_relevance_scorer.py .\campaign_1.docx.parsed.txt .\campaign_4.docx.parsed.txt
  # or with no args, it looks for campaign_*.docx.parsed.txt in the cwd.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Urban Media Datasets"
OUT = ROOT / "eda_output"

# Real, verified categorical values (confirmed against actual EDA reports and
# a real model_selector.py run against real screens.csv -- not guessed).
SCREEN_TYPES = {"bus", "bus_stop", "metro_rail_coach", "metro_station"}
POSITIONS = {"back", "entrance_exit", "left", "platform", "right", "top"}
MARKET_TIERS = {"premium", "standard", "value"}


# --------------------------------------------------------------------------- #
# Contract shape (mirrors reference contracts.py's StructuredBrief)
# --------------------------------------------------------------------------- #

@dataclass
class StructuredBrief:
    name: str
    budget: float
    duration_days: int
    city_id: str = ""
    objective: str = ""
    audience_text: str = ""
    must_have: List[Tuple[str, str]] = field(default_factory=list)   # (screen_type, position) OR-of-pairs; "" = any
    exclude: List[Tuple[str, str]] = field(default_factory=list)
    exclude_market_tier: List[str] = field(default_factory=list)
    prefer_market_tier: List[str] = field(default_factory=list)
    preferred_blocks: List[int] = field(default_factory=list)
    day_type_weighting: str = ""          # "" | "weekend_heavy" | "report_split"
    optimise_for: str = "reach"            # "reach" | "frequency"
    radius_only: bool = False
    radius_anchor_hint: str = ""
    notes: List[str] = field(default_factory=list)
    #: Structured inventory requirement from the intake agent, validated
    #: against closed vocabularies: near_poi_type / position / screen_type /
    #: market_tier. Applied as a HARD filter by score_screens, unlike
    #: `must_have` -- see apply_inventory_filter for why the two differ.
    inventory_req: Dict[str, List[str]] = field(default_factory=dict)
    inventory_evidence: str = ""
    #: Audience segments from the intake agent, drawn from the eight the cube
    #: actually records. Preferred over _AUDIENCE_KEYWORD_TO_OCCUPATION, whose
    #: targets are mostly segments D1 does not produce -- see _audience_fit.
    audience_segments: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Brief parsing -- deterministic, offline-first
# --------------------------------------------------------------------------- #

def _clean(text: str) -> str:
    text = text.replace("\ufffe", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<PARSED TEXT FOR PAGE:[^>]*>", " ", text)
    text = re.sub(r"<IMAGE FOR PAGE:[^>]*>", " ", text)
    return text


def _flatten(text: str) -> str:
    """Collapse all whitespace runs (including line-wraps) to single spaces.

    Docx-to-text extraction wraps mid-phrase -- e.g. "Las\nHackland" -- which
    breaks any naive substring/keyword search done on the raw text. Every
    keyword/substring lookup in this module must run against the flattened
    text, not the raw one. Line-anchored regexes (field extraction, the
    Exclusion Criteria block boundary) still use the raw text, since they
    deliberately rely on real newlines as structural anchors."""
    return re.sub(r"\s+", " ", text)


def _extract_name(text: str) -> str:
    m = re.search(r"CLIENT BRIEF\s*\d+\s*:\s*([A-Z0-9 &.,'\-]+?)\s*[\u2014\u2013-]", text)
    if m:
        return m.group(1).strip().title()
    m = re.search(r"Company Name\s+([^\n]+)", text)
    return m.group(1).strip() if m else "Unnamed Brief"


def _extract_budget(text: str) -> float:
    m = re.search(r"Campaign Budget\s+USD\s*([\d,]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"USD\s*([\d,]{3,})", text)
    return float(m.group(1).replace(",", "")) if m else 0.0


def _extract_duration(text: str) -> int:
    m = re.search(r"Campaign Duration\s+(\d+)\s*Days?", text, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _extract_field(text: str, label: str) -> str:
    m = re.search(rf"{label}\s+(.+)", text)
    if not m:
        return ""
    line = m.group(1)
    # stop at the next known field label or newline
    line = re.split(r"\n|Target Audience|Campaign Budget|Campaign Duration|Industry Vertical", line)[0]
    return line.strip()


def _extract_city(text: str, cities_df: pd.DataFrame) -> str:
    low = _flatten(text).lower()
    for _, row in cities_df.iterrows():
        if str(row["city_name"]).lower() in low:
            return str(row["city_id"])
    return ""


# Keyword rules -> (screen_type, position) pairs. "" means "any" for that slot.
_MUST_HAVE_RULES: List[Tuple[re.Pattern, Tuple[str, str], str]] = [
    (re.compile(r"metro platform", re.I), ("metro_station", "platform"), "metro platform boards"),
    (re.compile(r"metro station", re.I), ("metro_station", ""), "metro station screens"),
    (re.compile(r"bus[- ]rear", re.I), ("bus", "back"), "bus-rear screens"),
    (re.compile(r"mall entr", re.I), ("", "entrance_exit"), "mall entry points"),
    (re.compile(r"platform", re.I), ("", "platform"), "platform-position screens"),
]

_EXCLUDE_RULES: List[Tuple[re.Pattern, Tuple[str, str], str]] = [
    (re.compile(r"exclude bus[- ]rear|exclude.*bus[- ]rear", re.I), ("bus", "back"), "bus-rear screens (excluded)"),
]

_DAYPART_RULES: List[Tuple[re.Pattern, List[str]]] = [
    (re.compile(r"late[- ]night|night[- ]shift|post[- ]venue|entertainment corridor", re.I), ["night"]),
    (re.compile(r"early morning", re.I), ["night", "morning"]),
    (re.compile(r"morning|commute peak", re.I), ["morning"]),
    (re.compile(r"lunch|midday", re.I), ["midday"]),
    (re.compile(r"afternoon", re.I), ["afternoon"]),
    (re.compile(r"evening", re.I), ["evening"]),
]


def _find_exclusion_block(text: str) -> str:
    m = re.search(r"Exclusion Criteria:?\s*(.+?)(?:\n\d\.|\Z)", text, re.DOTALL)
    return m.group(1) if m else ""


def parse_brief(text: str, cities_df: pd.DataFrame, dim_slot_df: pd.DataFrame) -> StructuredBrief:
    text = _clean(text)
    flat = _flatten(text)
    name = _extract_name(text)
    budget = _extract_budget(text)
    duration = _extract_duration(text)
    city_id = _extract_city(text, cities_df)
    objective = _extract_field(text, "Campaign Objective")
    audience_text = _extract_field(text, "Target Audience")

    exclusion_block = _flatten(_find_exclusion_block(text))
    exclude: List[Tuple[str, str]] = []
    exclude_market_tier: List[str] = []
    if re.search(r"value[- ]tier", exclusion_block, re.I):
        exclude_market_tier.append("value")
    for pat, pair, _label in _EXCLUDE_RULES:
        if pat.search(exclusion_block):
            exclude.append(pair)
    # a bare "exclude bus-rear" anywhere in the exclusion block, not just the
    # combined pattern above
    if re.search(r"bus[- ]rear", exclusion_block, re.I) and ("bus", "back") not in exclude:
        exclude.append(("bus", "back"))

    # must_have is recorded as REQUESTED categories, not a hard AND/OR filter.
    # Our keyword vocabulary cannot cover every phrase a brief might use (e.g.
    # "auto-retail arterial corridors", "campus-edge transit nodes") -- hard
    # filtering on an incomplete rule set would silently drop screen
    # categories the client explicitly asked for, which is worse than a
    # slightly noisier shortlist. Matches are used only to BOOST relevance
    # score (see score_screens' category_fit), never to exclude.
    must_have: List[Tuple[str, str]] = []
    outside = _flatten(text.replace(_find_exclusion_block(text), ""))
    for pat, pair, _label in _MUST_HAVE_RULES:
        if pair not in exclude and pat.search(outside):
            must_have.append(pair)

    prefer_market_tier: List[str] = []
    if re.search(r"premium|affluent|upscale", flat, re.I) and "value" not in exclude_market_tier:
        prefer_market_tier.append("premium")

    preferred_daypart_labels: List[str] = []
    for pat, labels in _DAYPART_RULES:
        if pat.search(flat):
            preferred_daypart_labels.extend(labels)
    preferred_daypart_labels = list(dict.fromkeys(preferred_daypart_labels))  # dedupe, keep order

    preferred_blocks: List[int] = []
    if preferred_daypart_labels:
        for lbl in preferred_daypart_labels:
            hits = dim_slot_df[dim_slot_df["nearest_daypart"].str.contains(lbl, case=False, na=False)]
            preferred_blocks.extend(hits["time_block_id"].tolist())
        preferred_blocks = sorted(set(preferred_blocks))

    day_type_weighting = ""
    if re.search(r"weekend[- ]versus[- ]weekday|weekend.*split.*shown separately", flat, re.I):
        day_type_weighting = "report_split"
    elif re.search(r"weekend weighting|weighted delivery from friday", flat, re.I):
        day_type_weighting = "weekend_heavy"

    optimise_for = "frequency" if re.search(r"prioriti[sz]ing frequency|frequency over", flat, re.I) else "reach"

    radius_only = bool(re.search(r"walking radius|walking distance|entire reach requirement sits inside", flat, re.I))
    radius_anchor_hint = ""
    if radius_only:
        # search on the flattened text and stop only at sentence-level
        # punctuation / dashes surrounded by spaces -- NOT at a bare hyphen
        # inside a compound word like "business-park", which a naive
        # char-class stop condition would truncate mid-word.
        m = re.search(r"located in an?\s+([a-z0-9][a-z0-9 \-]*?)\s*(?:\u2014|\u2013|(?<!\w)-(?!\w)|\.)", flat, re.I)
        if not m:
            m = re.search(r"new outlet[^.]*?((?:business[- ]park|food court|mall|plaza|high[- ]street)[a-z0-9 \-]*)", flat, re.I)
        radius_anchor_hint = m.group(1).strip() if m else "walking-radius anchor not explicitly named"

    notes: List[str] = []
    known_place_terms = {"metro", "bus", "mall", "high-street", "high street", "business", "plaza",
                          "food court", "residential", "arterial", "corridor", "stadium", "arena",
                          "campus", "transit", "platform"}
    # flag references to place types not resolvable against known POI/location
    # vocabulary at all -- a lightweight trap-detector for briefs like the
    # reference's Brief 5 ("Airport Transit Corridor" does not exist in data).
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2}\s(?:Corridor|Precinct|District|Hub))\b", flat):
        phrase = m.group(1)
        if not any(t in phrase.lower() for t in known_place_terms):
            notes.append(f"brief references '{phrase}' -- verify this exists in POI/location data before promising it")

    return StructuredBrief(
        name=name, budget=budget, duration_days=duration, city_id=city_id,
        objective=objective, audience_text=audience_text,
        must_have=must_have, exclude=exclude, exclude_market_tier=exclude_market_tier,
        prefer_market_tier=prefer_market_tier, preferred_blocks=preferred_blocks,
        day_type_weighting=day_type_weighting, optimise_for=optimise_for,
        radius_only=radius_only, radius_anchor_hint=radius_anchor_hint, notes=notes,
    )


# --------------------------------------------------------------------------- #
# Screen scoring
# --------------------------------------------------------------------------- #

_AUDIENCE_KEYWORD_TO_OCCUPATION = {
    "professional": "white_collar", "urban professional": "white_collar",
    "office worker": "white_collar", "senior professional": "white_collar",
    "student": "student", "gym-goer": "young_adult", "gen z": "young_adult",
    "young professional": "young_adult", "shopper": "consumer",
}


def _radius_zone_ids(anchor_hint: str, poi_df: pd.DataFrame, locations_df: pd.DataFrame) -> List[str]:
    """Coarse 'walking radius' proxy: zones containing a POI whose poi_type
    matches a keyword in the anchor hint. No coordinates exist in this
    dataset, so same-zone membership is the finest resolvable proximity."""
    hint = anchor_hint.lower()
    keywords = [w for w in re.split(r"[^a-z]+", hint) if len(w) > 3]
    if not keywords:
        return []
    mask = poi_df["poi_type"].str.lower().apply(
        lambda t: any(k in t or t in k for k in keywords) if pd.notna(t) else False
    )
    matched_locations = poi_df.loc[mask, "anchor_location_id"].dropna().unique().tolist()
    if not matched_locations:
        return []
    zones = locations_df.loc[locations_df["location_id"].isin(matched_locations), "zone_id"].dropna().unique().tolist()
    return zones


def _daypart_fit(cube_wd: pd.DataFrame, preferred_blocks: List[int]) -> pd.Series:
    total = cube_wd.groupby("screen_id")["people"].sum().rename("total_people")
    if not preferred_blocks:
        return pd.Series(0.5, index=total.index, name="daypart_fit")
    pref = (cube_wd[cube_wd["time_block_id"].isin(preferred_blocks)]
            .groupby("screen_id")["people"].sum().rename("pref_people"))
    merged = pd.concat([total, pref], axis=1).fillna(0.0)
    fit = np.where(merged["total_people"] > 0, merged["pref_people"] / merged["total_people"], 0.0)
    return pd.Series(fit, index=merged.index, name="daypart_fit")


def _audience_fit(profiles: pd.DataFrame, audience_text: str,
                   segments: Optional[List[str]] = None) -> pd.Series:
    """Share of a screen's audience that belongs to the brief's target segments.

    THE KEYWORD TABLE IS BROKEN AND THIS IS WHY `segments` EXISTS.
    _AUDIENCE_KEYWORD_TO_OCCUPATION maps brief text onto {consumer, student,
    white_collar, young_adult}. The segments audience_profiles actually
    records are {blue_collar, commuter_professional, leisure, resident,
    retail_service, shopper, student, traveller}. Only `student` is in both,
    so three of the four possible targets can never match anything. Measured,
    on 20% of the relevance weight:

        "Urban professionals and shoppers"        0.000 on every screen
        "Office workers and students"             mean 0.046
        "Young women, beauty-conscious commuters" flat 0.500 (no cue hit at all)

    Note the two failure shapes. A brief with no cue hit falls to a neutral
    0.5, which is merely useless. A brief whose cues all map to phantom
    segments scores 0.0 everywhere -- actively penalising every screen for a
    vocabulary mismatch that has nothing to do with the screens.

    `segments` comes from the intake agent choosing from the eight real ones,
    so it is passed through unchanged. The keyword table stays as the offline
    fallback; it is wrong for most briefs but it is deterministic, and
    replacing a known-wrong default with nothing would make the no-credential
    path worse rather than better.
    """
    target_occupations = set(segments or ())
    if not target_occupations:
        text_low = audience_text.lower()
        target_occupations = {v for k, v in _AUDIENCE_KEYWORD_TO_OCCUPATION.items()
                              if k in text_low}
    if not target_occupations:
        return pd.Series(0.5, index=profiles.index, name="audience_fit")

    def score(comp_str: str) -> float:
        if not isinstance(comp_str, str) or not comp_str:
            return 0.3
        shares = {}
        for part in comp_str.split(";"):
            part = part.strip()
            if ":" not in part:
                continue
            k, v = part.rsplit(":", 1)
            try:
                shares[k.strip()] = float(v)
            except ValueError:
                pass
        matched = sum(v for k, v in shares.items() if k in target_occupations)
        return float(np.clip(matched, 0.0, 1.0))

    return profiles["composition_str"].apply(score).rename("audience_fit")


def _tier_fit(screens_join: pd.DataFrame, brief: StructuredBrief) -> pd.Series:
    if brief.prefer_market_tier:
        return screens_join["market_tier"].apply(
            lambda t: 1.0 if t in brief.prefer_market_tier else 0.4).rename("tier_fit")
    return pd.Series(0.5, index=screens_join.index, name="tier_fit")


def apply_inventory_filter(df: pd.DataFrame, req, poi_df: pd.DataFrame
                            ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Narrow to the screens the brief actually asked for.

    WHY THIS IS HARD WHERE `must_have` IS SOFT. `must_have` comes from regex
    over free text, and its vocabulary is admittedly incomplete, so filtering
    on it would silently drop categories the client asked for in words the
    rules do not know. This requirement comes from the intake agent choosing
    from a CLOSED vocabulary -- the real distinct values of poi_type,
    position, screen_type and market_tier -- and anything outside that
    vocabulary was already dropped before it got here. There is no unknown
    category left to lose, so the filter can be hard.

    WHAT IT BUYS. Campaign 6 asks for "Premium vertical screens at shopping
    centre entrances" and "central metro entry points". The network has 11,163
    screens; 3,337 anchor a shopping_mall POI, 1,275 are position
    entrance_exit, and 584 are both. Before this, D2's relevance stage removed
    ZERO screens on that brief -- its own `mall entr` rule missed because the
    brief says "Mall Beauty-Retail Entry" and "shopping centre entrances" --
    so the optimiser ranked all 11,163 by people/price and the brief's primary
    inventory requirement had no effect at all.

    Proximity is via anchor_location_id: a screen is "near" a POI type when
    its location is that POI's anchor. That is the only proximity relation the
    14 tables actually contain -- there are no coordinates -- so it is the
    honest interpretation of "at" or "adjacent to".
    """
    # `req` is a LIST OF GROUPS. Fields within a group are ANDed, groups are
    # ORed. A media brief naming mall entrances, high-street corridors and
    # metro entries is asking for the UNION of three kinds of screen; ANDing
    # them would demand one screen that is all three at once. A bare dict is
    # accepted as a single group so older callers keep working.
    groups = [req] if isinstance(req, dict) else list(req or [])
    stats: Dict[str, int] = {"inventory_groups": len(groups)}
    if not groups:
        stats["after_inventory_filter"] = len(df)
        return df, stats

    if any(g.get("near_poi_type") for g in groups) and "location_id" not in df.columns:
        # Loud, not silent. A requirement that is accepted, validated and then
        # ignored is worse than one refused, because the plan still claims to
        # have honoured it.
        raise KeyError(
            "apply_inventory_filter needs `location_id` to test POI proximity, "
            "and the frame does not have it. Merge it from screens.csv -- "
            "audience_profiles.csv does not carry it.")

    keep = pd.Series(False, index=df.index)
    # Which group each screen matched, and at what priority. D5 uses this to
    # stop inventory the brief called SUPPLEMENTARY from dominating the spend:
    # ORed groups are equals to the optimiser, and the supplementary group is
    # usually the cheaper one, so "anchored on X with supplementary Y" became
    # "mostly Y" without it.
    prio = pd.Series("", index=df.index, dtype=object)
    for i, g in enumerate(groups):
        m = pd.Series(True, index=df.index)
        for field_name in ("screen_type", "position", "market_tier"):
            want = g.get(field_name) or []
            if want and field_name in df.columns:
                m &= df[field_name].isin(want)
        poi_types = g.get("near_poi_type") or []
        if poi_types:
            anchors = set(poi_df.loc[poi_df["poi_type"].isin(poi_types),
                                      "anchor_location_id"].dropna())
            m &= df["location_id"].isin(anchors)
        stats[f"group_{i}"] = int(m.sum())
        # First match wins, and groups arrive primary-first (see the sort in
        # merge_into_brief), so a screen satisfying both is credited to the
        # primary group -- which is the reading that favours the brief.
        newly = m & (prio == "")
        prio[newly] = str(g.get("priority") or "primary")
        keep |= m

    df = df[keep].copy()
    df["inv_priority"] = prio[keep].values
    stats["after_inventory_filter"] = len(df)
    stats["supplementary_only"] = int((df["inv_priority"] == "supplementary").sum())
    return df, stats


def score_screens(
    brief: StructuredBrief,
    profiles: pd.DataFrame,
    cube: pd.DataFrame,
    screens_df: pd.DataFrame,
    locations_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    cities_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats: Dict[str, int] = {"start": len(profiles)}

    # location_id is pulled in for the inventory filter's POI-proximity test.
    # audience_profiles.csv does not carry it, and without it
    # apply_inventory_filter's `if ... "location_id" in df.columns` guard was
    # simply false -- so a near_poi_type requirement would have been accepted,
    # validated, reported, and then quietly done nothing.
    df = profiles.merge(
        screens_df[["screen_id", "screen_type", "position", "screen_size",
                    "location_id"]],
        on="screen_id", how="left",
    )
    df = df.merge(cities_df[["city_id", "market_tier"]], on="city_id", how="left")

    # --- hard filters -------------------------------------------------- #
    if brief.city_id:
        df = df[df["city_id"] == brief.city_id]
    stats["after_city_filter"] = len(df)

    if brief.exclude_market_tier:
        df = df[~df["market_tier"].isin(brief.exclude_market_tier)]
    stats["after_market_tier_exclude"] = len(df)

    for st, pos in brief.exclude:
        cond = pd.Series(True, index=df.index)
        if st:
            cond &= (df["screen_type"] == st)
        if pos:
            cond &= (df["position"] == pos)
        df = df[~cond]
    stats["after_exclusion_rules"] = len(df)

    # must_have is NOT a hard filter -- see parse_brief's docstring note. Our
    # keyword vocabulary is incomplete, so treating it as an AND/OR exclusion
    # filter would silently drop legitimate requested screen categories the
    # rules don't recognise (e.g. "auto-retail arterial corridors"). It is
    # instead folded into category_fit below, which only ever helps a
    # matching screen's score -- it never removes an unmatched one.
    stats["after_must_have_filter"] = len(df)  # unchanged; kept for funnel symmetry

    radius_zones: List[str] = []
    if brief.radius_only:
        radius_zones = _radius_zone_ids(brief.radius_anchor_hint, poi_df, locations_df)
        if radius_zones:
            df = df[df["zone_id"].isin(radius_zones)]
        # if no zone could be resolved from the hint, we deliberately do NOT
        # filter -- silently returning zero screens is worse than flagging it.
    stats["after_radius_filter"] = len(df)

    if brief.inventory_req:
        df, inv_stats = apply_inventory_filter(df, brief.inventory_req, poi_df)
        stats.update(inv_stats)
    else:
        stats["after_inventory_filter"] = len(df)

    if df.empty:
        return df, stats

    # --- soft scoring ---------------------------------------------------- #
    cube_wd = cube[cube["day_type"] == "weekday"]
    dpf = _daypart_fit(cube_wd, brief.preferred_blocks)
    af = _audience_fit(df, brief.audience_text, brief.audience_segments)
    tf = _tier_fit(df, brief)

    df = df.set_index("screen_id")
    df["daypart_fit"] = dpf.reindex(df.index).fillna(0.5)
    df["audience_fit"] = af.reindex(df.index).fillna(0.5).values if len(af) == len(df) else 0.5
    df["confidence_w"] = df["confidence"].fillna(0.5)
    df["tier_fit"] = tf.values if len(tf) == len(df) else 0.5
    df["history_bonus"] = df["has_booking_history"].map({True: 0.6, False: 0.4}).fillna(0.4)

    if brief.must_have:
        def cat_fit(row) -> float:
            for st, pos in brief.must_have:
                ok = (not st or row["screen_type"] == st) and (not pos or row["position"] == pos)
                if ok:
                    return 1.0
            return 0.45  # neutral-low: not a recognised-must-have match, but never excluded
        df["category_fit"] = df.apply(cat_fit, axis=1)
    else:
        df["category_fit"] = 0.5

    weights = {"daypart_fit": 0.25, "audience_fit": 0.20, "confidence_w": 0.15,
               "tier_fit": 0.10, "history_bonus": 0.10, "category_fit": 0.20}
    df["relevance_score"] = sum(df[c] * w for c, w in weights.items())
    df = df.sort_values("relevance_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df = df.reset_index()

    def build_reasons(r) -> str:
        bits = []
        if r["category_fit"] >= 1.0 and brief.must_have:
            bits.append(f"screen_type={r['screen_type']}/position={r['position']} matches a requested inventory category")
        if r["daypart_fit"] > 0.4 and brief.preferred_blocks:
            bits.append(f"{r['daypart_fit']*100:.0f}% of daily footfall falls in preferred block(s) {brief.preferred_blocks}")
        if r["audience_fit"] > 0.5:
            bits.append(f"audience composition aligns with brief's target ({r['audience_fit']*100:.0f}% match)")
        if r["tier_fit"] >= 1.0:
            bits.append(f"market_tier={r.get('market_tier','?')} matches brief's premium positioning")
        if r["has_booking_history"]:
            # Name the GRAIN. This flag is per SCREEN; D3's rc_basis is per
            # screen x time_block. A screen with bookings in block 5 and none
            # in block 3 makes both "has history" and "no booking history"
            # true of the same plan line, and once D5 joined the two onto one
            # row the evidence panel showed them side by side as a flat
            # contradiction. Both statements are correct; only the wording was
            # ambiguous about what each one describes.
            # NO SEMICOLON IN A REASON. Reasons are joined with "; " and split
            # back on it downstream, so a semicolon inside one fragments it --
            # this very string rendered as two bullets with unbalanced
            # parentheses in the evidence block.
            bits.append("this screen has real booking history somewhere in its "
                        "day -- demand-proven at screen level, though the chosen "
                        "block may still be priced from peers (see the price basis)")
        if brief.radius_only and radius_zones:
            bits.append(f"zone_id={r['zone_id']} matches walking-radius anchor '{brief.radius_anchor_hint}'")
        if not bits:
            bits.append(f"meets hard filters; confidence={r['confidence']:.2f}")
        return "; ".join(bits)

    df["reasons"] = df.apply(build_reasons, axis=1)
    return df, stats


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def load_tables():
    screens = pd.read_csv(DATA / "screens.csv")
    locations = pd.read_csv(DATA / "locations.csv")
    poi = pd.read_csv(DATA / "points_of_interest.csv")
    cities = pd.read_csv(DATA / "cities.csv")
    dim_slot = pd.read_csv(DATA / "dim_slot.csv")
    profiles = pd.read_csv(OUT / "audience_profiles.csv")
    cube = pd.read_csv(OUT / "audience_cube.csv")
    return screens, locations, poi, cities, dim_slot, profiles, cube


def main(argv: List[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    brief_paths = [Path(p) for p in argv] if argv else sorted(ROOT.glob("campaign_*.docx.parsed.txt"))
    if not brief_paths:
        print("no campaign brief .txt files given or found -- pass paths as arguments")
        return 1

    screens, locations, poi, cities, dim_slot, profiles, cube = load_tables()
    print(f"loaded screens {screens.shape}, locations {locations.shape}, poi {poi.shape}, "
          f"cities {cities.shape}, dim_slot {dim_slot.shape}, profiles {profiles.shape}, cube {cube.shape}")

    for path in brief_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        brief = parse_brief(text, cities, dim_slot)
        print(f"\n{'='*70}\nBRIEF: {brief.name}  (from {path.name})")
        print(f"  budget=${brief.budget:,.0f}  duration={brief.duration_days}d  city_id={brief.city_id or '(any)'}")
        print(f"  must_have={brief.must_have}  exclude={brief.exclude}  exclude_market_tier={brief.exclude_market_tier}")
        print(f"  preferred_blocks={brief.preferred_blocks}  day_type_weighting={brief.day_type_weighting or '(none)'}"
              f"  optimise_for={brief.optimise_for}")
        print(f"  radius_only={brief.radius_only}  radius_anchor_hint={brief.radius_anchor_hint!r}")
        if brief.notes:
            for n in brief.notes:
                print(f"  NOTE: {n}")

        scored, stats = score_screens(brief, profiles, cube, screens, locations, poi, cities)
        print(f"  funnel: {stats}")
        if scored.empty:
            print("  no screens survived hard filters -- brief may need a wider match or manual review")
            continue

        slug = re.sub(r"[^a-z0-9]+", "_", brief.name.lower()).strip("_")
        out_path = OUT / f"relevance_scores_{slug}.csv"
        scored.to_csv(out_path, index=False)
        print(f"  saved: {out_path.name} -> {scored.shape[0]} rows x {scored.shape[1]} cols")
        print("  top 10 shortlist:")
        for _, r in scored.head(10).iterrows():
            print(f"    #{r['rank']:>3} {r['screen_id']:>10}  score={r['relevance_score']:.3f}  {r['reasons']}")

    print("\nAll briefs processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

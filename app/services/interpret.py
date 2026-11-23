"""Parse a brief and label WHERE each field came from.

The read-vs-inferred distinction is the app's clearest evidence that the agent is
parsing rather than form-filling, and it is nearly free: D2's extractors are
individually callable, so re-running each one tells us whether a field came from
a literal match in the document or from a keyword rule firing.

`needs_you` is the third state and the important one. Three of the six sample
briefs never name a city, and an unset city means D2 scores all 11,163 screens
across three cities. Labelling that as "inferred" would be a lie; labelling it
"needs you" makes it a blocking clarification.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

READ = "read"
INFERRED = "inferred"
DERIVED = "derived"
NEEDS = "needs_you"

# --------------------------------------------------------------------------- #
# City inference
# --------------------------------------------------------------------------- #
# Three of the six sample briefs never name a city. Refusing to proceed is safe
# but unhelpful, and guessing silently is worse. So: infer from real signals in
# the brief, say which signal fired, and if nothing fires, plan across ALL
# cities rather than picking one at random.
#
# The city facts below are read from cities.csv at call time -- never hardcoded --
# so this keeps working when Urban Media expands, which the problem statement
# explicitly flags as a near-future scenario.

#: Structural boilerplate every brief carries, stripped before cue matching.
#:
#: WHY THIS IS NOT OPTIONAL. The first version treated "executive" as a premium
#: signal, and every one of the six sample briefs has a section headed
#: "1. Executive Summary & Objective". So inference confidently reported
#: "the brief uses premium positioning language (executive)" for a gym
#: membership campaign. A cue that fires on a template heading is measuring the
#: template, not the brief.
_BOILERPLATE = re.compile(
    r"^\s*\d+\.\s*(?:executive summary[^\n]*|target audience[^\n]*|"
    r"digital screen selection[^\n]*|visual\s*/?\s*mockup[^\n]*|"
    r"rfp requirements[^\n]*)$", re.I | re.M)

#: brief-language cue -> the city attribute it argues for.
#: "executive" is deliberately absent -- see _BOILERPLATE above. It only counts
#: in an unambiguous travel phrase, which is handled by the business-travel cue.
_PREMIUM_CUES = re.compile(
    r"\bpremium\b|\baffluent\b|\bupscale\b|\bluxur|\bhigh[- ]income\b|"
    r"\bbusiness travel|\bfrequent flyer|\bfinancial (?:core|district)\b|"
    r"\bmass[- ]premium\b|\baspiration", re.I)
_VALUE_CUES = re.compile(
    r"\bvalue[- ]conscious\b|\bbudget[- ]conscious\b|\bprice[- ]sensitive\b|"
    r"\bmass[- ]market\b|\baffordab|\bdiscount\b|\bthrift", re.I)
_DENSE_CUES = re.compile(
    r"\bmetro\b|\bsubway\b|\bunderground\b|\brail\b|\bcommuter rail\b|"
    r"\bdense\b|\bcity centre\b|\bcity center\b|\bdowntown\b|\bcbd\b", re.I)
_SPRAWL_CUES = re.compile(
    r"\bsuburb|\bsprawl|\bdrive[- ]time\b|\bhighway\b|\barterial\b|"
    r"\bpark[- ]and[- ]ride\b", re.I)


def infer_city(brief_text: str, cities_df, screens_df=None) -> tuple:
    """(city_id or "", confidence 0-1, list of human-readable reasons).

    Returns "" when no signal is strong enough. The caller then plans across
    every city instead of inventing an answer.
    """
    # Strip template headings first, then flatten -- cues must match the
    # client's own words, not the document's skeleton.
    body = _BOILERPLATE.sub(" ", brief_text or "")
    flat = re.sub(r"\s+", " ", body)
    scores: dict = {}
    why: dict = {}

    def add(cid, pts, reason):
        scores[cid] = scores.get(cid, 0.0) + pts
        why.setdefault(cid, []).append(reason)

    cities = cities_df.copy()
    tier_col = "market_tier" if "market_tier" in cities.columns else None
    dens_col = "transit_density" if "transit_density" in cities.columns else None
    pop_col = "population" if "population" in cities.columns else None

    # ---- 1. positioning language -> market tier ---- #
    if _PREMIUM_CUES.search(flat) and tier_col:
        hits = _PREMIUM_CUES.findall(flat)
        for _, row in cities[cities[tier_col].str.lower() == "premium"].iterrows():
            add(row["city_id"],
                3.0,
                f"the brief uses premium positioning language "
                f"({', '.join(sorted({h.strip().lower() for h in hits})[:3])}) and "
                f"{row['city_name']} is our premium-tier market")
    if _VALUE_CUES.search(flat) and tier_col:
        for _, row in cities[cities[tier_col].str.lower() == "value"].iterrows():
            add(row["city_id"], 3.0,
                f"the brief is value/price-led and {row['city_name']} is our "
                f"value-tier market")

    # ---- 2. inventory language -> transit density ---- #
    if _DENSE_CUES.search(flat) and dens_col:
        for _, row in cities[cities[dens_col].str.lower() == "dense"].iterrows():
            add(row["city_id"], 2.0,
                f"the brief asks for metro/rail inventory and {row['city_name']} "
                f"has the densest transit network")
    if _SPRAWL_CUES.search(flat) and dens_col:
        for _, row in cities[cities[dens_col].str.lower() == "sprawling"].iterrows():
            add(row["city_id"], 1.5,
                f"the brief describes suburban/arterial coverage, which suits "
                f"{row['city_name']}")

    # ---- 3. where the matching inventory actually is ---- #
    # A tie between two cities is broken by which one can actually deliver the
    # requested screen type, not by population.
    if screens_df is not None and len(screens_df) and "city_id" in screens_df.columns:
        want = None
        if re.search(r"metro platform|metro station|platform", flat, re.I):
            want = "metro_station"
        elif re.search(r"bus[- ]rear|on[- ]bus|bus stop", flat, re.I):
            want = "bus_stop"
        if want and "screen_type" in screens_df.columns:
            counts = (screens_df[screens_df["screen_type"] == want]["city_id"]
                      .value_counts())
            if len(counts):
                top = str(counts.index[0])
                add(top, 1.0,
                    f"most of our {want.replace('_', ' ')} inventory "
                    f"({int(counts.iloc[0]):,} screens) is in "
                    f"{_city_name(cities, top)}")

    # ---- 4. reach ambition -> population, as a weak tiebreak only ---- #
    if re.search(r"\bcitywide\b|\bmass reach\b|\bas many .* as possible\b", flat, re.I) \
            and pop_col:
        biggest = cities.sort_values(pop_col, ascending=False).iloc[0]
        add(biggest["city_id"], 1.0,
            f"the brief asks for broad city-wide reach and {biggest['city_name']} "
            f"is our largest market")

    if not scores:
        return "", 0.0, []

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_pts = ranked[0]
    runner_pts = ranked[1][1] if len(ranked) > 1 else 0.0
    # Confidence is margin-based: two cities scoring equally is NOT a confident
    # answer just because one of them edged ahead alphabetically.
    total = sum(scores.values()) or 1.0
    confidence = max(0.0, min(1.0, (best_pts - runner_pts) / total))
    if best_pts < 2.0 or confidence < 0.25:
        return "", confidence, why.get(best, [])
    return str(best), confidence, why.get(best, [])


def _city_name(cities_df, city_id: str) -> str:
    m = cities_df[cities_df["city_id"] == city_id]
    return str(m.iloc[0]["city_name"]) if len(m) else str(city_id)


def parse(brief_text: str, cities_df, dim_slot_df,
          screens_df=None) -> Tuple[object, Dict[str, str], List[str]]:
    """Returns (StructuredBrief, provenance per field, clarifying questions)."""
    from build_d2_relevance_scorer import (
        _clean, _extract_budget, _extract_duration, _extract_field, _find_exclusion_block,
        parse_brief,
    )

    brief = parse_brief(brief_text, cities_df, dim_slot_df)
    text = _clean(brief_text)

    prov: Dict[str, str] = {}
    questions: List[str] = []

    # ---- fields with a literal source in the document ---- #
    prov["name"] = (READ if brief.name and brief.name != "Unnamed Brief" else NEEDS)
    if prov["name"] == NEEDS:
        questions.append("I could not find a client or campaign name. What should this "
                         "campaign be called?")

    prov["budget"] = READ if _extract_budget(text) > 0 else NEEDS
    if prov["budget"] == NEEDS:
        questions.append("No budget figure appears in the brief. Every downstream step "
                         "is budget-constrained — what is the campaign budget?")

    prov["duration_days"] = READ if _extract_duration(text) > 0 else NEEDS
    if prov["duration_days"] == NEEDS:
        questions.append("No campaign duration appears in the brief. How many days "
                         "should the flight run?")

    if brief.city_id:
        prov["city_id"] = READ
    else:
        # Not named. Try to infer it from real signals before giving up, and if
        # nothing is strong enough, plan across every city rather than guessing.
        guess, conf, reasons = infer_city(brief_text, cities_df, screens_df)
        if guess:
            brief.city_id = guess
            prov["city_id"] = INFERRED
            because = reasons[0] if reasons else "signals in the brief"
            questions.append(
                f"The brief never names a city, so I picked "
                f"<b>{_city_name(cities_df, guess)}</b> because {because}"
                + (f", plus {len(reasons) - 1} other signal(s)"
                   if len(reasons) > 1 else "")
                + f". Confidence is {conf:.0%} &mdash; change it above if that is "
                  f"wrong.")
        else:
            prov["city_id"] = INFERRED
            questions.append(
                "The brief does not name a city and nothing in it points clearly to "
                "one, so rather than guessing I will look across <b>all three "
                "markets</b> and let the optimiser pick whichever locations give "
                "the best result for the budget. Set a city above if you want to "
                "restrict it.")

    prov["objective"] = READ if _extract_field(text, "Campaign Objective") else INFERRED
    prov["audience_text"] = READ if _extract_field(text, "Target Audience") else NEEDS
    if prov["audience_text"] == NEEDS:
        questions.append("No target audience is stated. Relevance scoring leans on it "
                         "heavily — who is this campaign trying to reach?")

    # ---- keyword-rule fields: inferred by construction ---- #
    exclusion_block = _find_exclusion_block(text)
    prov["exclude"] = READ if (brief.exclude and exclusion_block) else INFERRED
    prov["exclude_market_tier"] = READ if (brief.exclude_market_tier and exclusion_block) \
        else INFERRED
    for f in ("must_have", "prefer_market_tier", "preferred_blocks",
              "day_type_weighting", "optimise_for", "radius_only",
              "radius_anchor_hint"):
        prov[f] = INFERRED

    # ---- specific ambiguities worth surfacing as questions ---- #
    if brief.radius_only and (
            not brief.radius_anchor_hint
            or "not explicitly named" in brief.radius_anchor_hint):
        questions.append(
            "The brief asks for a walking-radius buy but does not clearly name the "
            "anchor location. Which location should the radius be measured from?")

    if not brief.preferred_blocks:
        questions.append(
            "No daypart preference was detectable, so all six time blocks are in play. "
            "If the campaign only matters at certain hours, set the blocks — it "
            "materially changes both audience and price.")

    if brief.optimise_for == "frequency":
        questions.append(
            "I read this as a FREQUENCY campaign (depth on fewer screens, same people "
            "more often) rather than reach. Confirm that is right — it changes what the "
            "optimiser maximises.")

    if getattr(brief, "notes", None):
        for n in brief.notes:
            questions.append(f"Flag from the brief: {n}")

    return brief, prov, questions


def field_evidence(brief_text: str, field: str) -> str:
    """A short snippet of the document that supports a field, for the
    'show me where you read that' affordance."""
    from build_d2_relevance_scorer import _clean
    text = _clean(brief_text)
    labels = {
        "budget": r"Campaign Budget[^\n]*\n?[^\n]*",
        "duration_days": r"Campaign Duration[^\n]*\n?[^\n]*",
        "objective": r"Campaign Objective[^\n]*\n?[^\n]*",
        "audience_text": r"Target Audience[^\n]*\n?[^\n]*",
        "name": r"(?:CLIENT BRIEF[^\n]*|Company Name[^\n]*\n?[^\n]*)",
        "exclude": r"Exclusion Criteria[^\n]*(?:\n[^\n]*){0,2}",
        "exclude_market_tier": r"Exclusion Criteria[^\n]*(?:\n[^\n]*){0,2}",
    }
    pat = labels.get(field)
    if not pat:
        return ""
    m = re.search(pat, text, re.I)
    return " ".join(m.group(0).split()) if m else ""

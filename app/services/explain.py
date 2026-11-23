"""Turns what the engines computed into sentences a sales rep can say out loud.

The reasoning already exists -- it is just written for engineers. D2 emits
`reasons` per screen, D3 emits a `components` ledger of signed multipliers with
a `rate_card_basis`, D5 emits a cohort posture with its justification, a
narrowing funnel and an ordered refusal. A rep cannot forward any of that.

So this module translates. Three rules:

  1. NO JARGON. No "D2", "cell", "pool", "daypart", "cold start", "solver".
     `config.GLOSSARY`, `BLOCK_PLAIN`, `SCREEN_TYPE_PLAIN` hold the mapping.
  2. AGGREGATE, DON'T LIST. "11 of 13 screens were chosen because the audience
     matches" beats thirteen near-identical reason strings.
  3. NEVER INVENT A NUMBER. Every figure traces to a computed column. Where the
     data cannot support a claim (conversion rates), say so instead of
     estimating.

Deterministic throughout -- no LLM. D5's `explain()` narrative is layered on top
of this in the UI, never in place of it, so the "why" survives with no
credentials and no budget.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app import config
from app.services.metrics import compact, money, pct


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #

def plain_block(b) -> str:
    try:
        return config.BLOCK_PLAIN.get(int(b), str(b))
    except (TypeError, ValueError):
        return str(b)


def plain_type(t) -> str:
    return config.SCREEN_TYPE_PLAIN.get(str(t), str(t).replace("_", " "))


def plain_position(p) -> str:
    return config.POSITION_PLAIN.get(str(p), str(p).replace("_", " "))


#: Display-time jargon translation, applied to text this app did not author.
#:
#: D5's own narrative -- both the LLM one and `_fallback_narrative()` -- is
#: written for an operator: "11 lines across 11 screens in daypart(s) [3]".
#: D1-D5 are frozen, so the only place to fix that for a sales rep is on the way
#: to the screen. Order matters: the bracketed-list forms must run before the
#: bare-word forms, or "daypart(s) [3]" becomes "time of day (s) [3]".
_HUMANISE: List[tuple] = [
    # "daypart(s) [3, 5]" / "block(s) [3]" -> named windows
    (re.compile(r"\b(?:daypart|time[_ ]?block|block)\(?s?\)?\s*\[([\d,\s]+)\]", re.I),
     lambda m: _name_blocks(m.group(1))),
    (re.compile(r"\bdaypart(s)?\b", re.I), r"time of day"),
    (re.compile(r"\btime[_ ]block(s)?\b", re.I), r"time slot"),
    (re.compile(r"\bdeduped reach\b", re.I), "people reached per day"),
    (re.compile(r"\bcp[_-]?sat\b", re.I), "the optimiser"),
    (re.compile(r"\bgreedy_half_1_minus_1_over_e\b", re.I), "the optimiser"),
    (re.compile(r"\bgreedy\b(?! )", re.I), "the optimiser"),
    (re.compile(r"\bsolver\b", re.I), "optimiser"),
    (re.compile(r"\bcold[- ]start\b", re.I), "not sold before"),
    (re.compile(r"\bpool(s)?\b", re.I), r"shared audience group\1"),
    (re.compile(r"\bscreen_id\b", re.I), "screen"),
    (re.compile(r"\bpool_id\b", re.I), "audience group"),
    (re.compile(r"\bD([1-5])\b"), "the planning engine"),
    (re.compile(r"\blines?\b(?= across)"), "ad slots"),
]


def _name_blocks(raw: str) -> str:
    ids = [int(x) for x in re.findall(r"\d+", raw or "")]
    if not ids:
        return "the requested hours"
    names = [config.BLOCK_PLAIN.get(i, str(i)) for i in ids]
    return _oxford(names, limit=3)


def humanise(text: str) -> str:
    """Strip engine vocabulary out of text written by the engines.

    Display-only: it never touches a number, so a translated sentence still says
    exactly what the original said.
    """
    out = str(text or "")
    for pat, rep in _HUMANISE:
        out = pat.sub(rep, out)
    return out


def _oxford(items: List[str], limit: int = 3) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) > limit:
        head, extra = items[:limit], len(items) - limit
        return ", ".join(head) + f" and {extra} more"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


#: D2's reason fragments -> what a rep would say. Order matters: first match on
#: a reason string wins, so the most specific patterns come first.
_REASON_THEMES: List[tuple] = [
    (re.compile(r"walking-radius|radius anchor", re.I),
     "sits inside the walking catchment the brief asked for"),
    (re.compile(r"matches a requested inventory category", re.I),
     "is exactly the type of screen the brief asked for"),
    (re.compile(r"audience composition aligns", re.I),
     "the people who pass it match the target audience"),
    (re.compile(r"footfall falls in preferred block", re.I),
     "its busiest hours line up with the campaign's target time of day"),
    (re.compile(r"market_tier=premium|premium positioning", re.I),
     "sits in a premium location, matching the brand's positioning"),
    (re.compile(r"real booking history", re.I),
     "has a proven booking record, so its performance is measured rather than modelled"),
    (re.compile(r"meets hard filters", re.I),
     "meets the campaign's location and inventory requirements"),
]


def _theme_reasons(scored: Optional[pd.DataFrame],
                   selected_ids: set) -> List[str]:
    """Group per-screen reasons into a handful of themes with counts."""
    if scored is None or not len(scored) or "reasons" not in scored.columns:
        return []
    s = scored[scored["screen_id"].astype(str).isin(selected_ids)]
    if not len(s):
        return []
    counts: Counter = Counter()
    for raw in s["reasons"].astype(str):
        for pat, phrase in _REASON_THEMES:
            if pat.search(raw):
                counts[phrase] += 1
    n = len(s)
    out = []
    for phrase, c in counts.most_common(5):
        if c == n:
            out.append(f"<b>Every one</b> of the {n} chosen locations {phrase}.")
        elif c >= 2:
            out.append(f"<b>{c} of {n}</b> {phrase}.")
        else:
            out.append(f"<b>One</b> of the {n} {phrase}.")
    return out


# --------------------------------------------------------------------------- #
# the explanation
# --------------------------------------------------------------------------- #

@dataclass
class Explanation:
    headline: str = ""
    why_screens: List[str] = field(default_factory=list)
    why_price: List[str] = field(default_factory=list)
    what_they_get: List[str] = field(default_factory=list)
    watch_outs: List[str] = field(default_factory=list)
    talking_points: List[str] = field(default_factory=list)

    def as_plaintext(self) -> str:
        L: List[str] = []

        def strip(s: str) -> str:
            return re.sub(r"<[^>]+>", "", s)

        if self.headline:
            L += [strip(self.headline), ""]
        for title, items in (("Why these locations", self.why_screens),
                             ("Why this price", self.why_price),
                             ("What the campaign delivers", self.what_they_get),
                             ("Worth flagging", self.watch_outs)):
            if items:
                L.append(title.upper())
                L += [f"  - {strip(i)}" for i in items]
                L.append("")
        return "\n".join(L).strip()


def build(*, result, quoted_totals: Optional[dict] = None,
          adjustments=None) -> Explanation:
    p = result.plan
    b = p.brief
    met = result.metrics
    lines = result.lines
    ex = Explanation()

    if not len(lines):
        ex.headline = ("No package could be built for this brief as it stands. "
                       "The reason is specific and fixable -- see below.")
        for r in p.refusals:
            ex.watch_outs.append(f"<b>{_refusal_plain(r.stage)}</b> {r.reason} "
                                 f"<i>What to do:</i> {r.remedy}")
        return ex

    days = int(b.duration_days or 0)
    quote = (quoted_totals or {}).get("quoted_total", met["spend"])
    n_screens, n_lines = met["n_screens"], met["n_lines"]
    selected = set(lines["screen_id"].astype(str))

    # ---------------- headline ---------------- #
    obj_word = ("as many different people as possible"
                if met["objective"] == "reach" else
                "the same audience repeatedly")
    ex.headline = (
        f"We can reach <b>{compact(met['reach_per_day'])} people a day</b> for "
        f"<b>{money(quote)}</b> across <b>{n_screens} locations</b> over "
        f"{days} days &mdash; {compact(met['impressions_total'])} total views, "
        f"built to reach {obj_word}.")

    # ---------------- why these locations ---------------- #
    types = Counter()
    zones = Counter()
    blocks = Counter()
    if result.scored is not None and len(result.scored):
        j = lines.merge(
            result.scored[[c for c in ["screen_id", "screen_type", "zone_id"]
                           if c in result.scored.columns]].drop_duplicates("screen_id"),
            on="screen_id", how="left")
        for _, r in j.iterrows():
            if pd.notna(r.get("screen_type")):
                types[str(r["screen_type"])] += 1
            if pd.notna(r.get("zone_id")):
                zones[str(r["zone_id"])] += 1
    for bl in lines.get("time_block_id", pd.Series(dtype=int)):
        blocks[int(bl)] += 1

    if types:
        mix = _oxford([f"{c} {plain_type(t)}" for t, c in types.most_common()])
        ex.why_screens.append(f"The plan is <b>{mix}</b>.")
    if blocks:
        when = _oxford([plain_block(k) for k, _ in blocks.most_common(2)], limit=2)
        ex.why_screens.append(
            f"Ads run during the <b>{when}</b>, which is when this audience is "
            f"actually in front of these screens.")
    if len(zones) > 1:
        ex.why_screens.append(
            f"Coverage is spread across <b>{len(zones)} areas</b> rather than "
            f"concentrated in one, so the campaign is not dependent on a single "
            f"location performing.")

    ex.why_screens += _theme_reasons(result.scored, selected)

    if b.radius_only:
        anchor = (b.radius_anchor_hint or "the location named in the brief")
        ex.why_screens.append(
            f"Because this is a local campaign, everything outside walking "
            f"distance of <b>{anchor}</b> was ruled out rather than just ranked "
            f"lower &mdash; reach beyond that radius would be wasted spend.")
    if b.exclude_market_tier:
        ex.why_screens.append(
            f"<b>{_oxford([t + '-tier' for t in b.exclude_market_tier])}</b> "
            f"inventory was excluded, as the brief requested.")
    if met.get("pool_share_of_lines", 0) > 0:
        ex.why_screens.append(
            f"Where two screens share the same audience (same station or same "
            f"route), the plan counts those people <b>once, not twice</b> "
            f"&mdash; so the reach figure above is people, not impressions "
            f"double-counted.")

    ex.why_screens.append(
        f"These {n_screens} were selected from "
        f"<b>{met.get('n_scored_candidates', 0):,} candidate locations</b> that "
        f"were scored against this brief.")

    # ---------------- why this price ---------------- #
    ex.why_price.append(
        f"<b>{money(quote)}</b> is {pct(met['budget_utilisation'])} of the "
        f"{money(b.budget)} budget"
        + (f", leaving {money(met['unspent'])} unspent because no further screen "
           f"added enough reach to justify its cost."
           if met["unspent"] > 1 else ", fully allocated."))

    ex.why_price.append(
        f"The negotiating range for this package is <b>{money(met['package_floor'])} "
        f"to {money(met['package_cap'])}</b>. The floor is what comparable "
        f"bookings actually sold for &mdash; not a discount to give away. Above "
        f"the cap is where comparable deals historically stopped closing.")

    if met.get("weighted_p_win", 0) > 0:
        ex.why_price.append(
            f"At {money(quote)} the modelled chance of the client accepting is "
            f"<b>{pct(met['weighted_p_win'], 0)}</b>, based on real win/loss "
            f"history for similar deals.")

    ex.why_price += _price_drivers_plain(result.quotes)

    ex.why_price.append(
        f"That works out to <b>{money(met['effective_cpm'], 2)} per thousand "
        f"views</b>.")

    if quoted_totals and abs(quoted_totals.get("delta", 0.0)) > 0.5:
        direction = "down" if quoted_totals["delta"] < 0 else "up"
        ex.why_price.append(
            f"{quoted_totals['n_lines_moved']} line(s) were adjusted "
            f"<b>{direction} {pct(abs(quoted_totals['delta_pct']))}</b> based on "
            f"how clients have responded to pricing on similar inventory before. "
            f"Every adjusted price is still inside its own tested range.")

    c = p.cohort
    if getattr(c, "n_clients", 0):
        ex.why_price.append(
            f"This advertiser is new to us, so pricing is benchmarked against "
            f"<b>{c.n_clients} comparable advertisers</b> who typically book "
            f"{money(c.median_budget)} over {c.median_duration_days} days. "
            f"Recommended stance: <b>{_posture_plain(c.posture)}</b>.")

    # ---------------- what they get ---------------- #
    ex.what_they_get = [
        f"<b>{compact(met['reach_per_day'])} different people</b> see the ad on a "
        f"typical day.",
        f"<b>{compact(met['impressions_total'])} total views</b> across the "
        f"{days}-day campaign.",
        f"On average each person who sees it sees it <b>"
        f"{met['avg_frequency_per_day']:.1f} times a day</b> &mdash; enough for "
        f"the message to register, not so much that it wears out.",
        f"<b>{n_lines} ad slots</b> running daily across {n_screens} screens.",
    ]
    if met.get("stretch_worth_offering"):
        ex.what_they_get.append(
            f"If the client can stretch to "
            f"<b>{money(met['spend'] + met['stretch_extra_spend'])}</b> "
            f"({pct(met['stretch_overrun_pct'])} more), that buys "
            f"{compact(met['stretch_extra_reach'])} additional people a day "
            f"&mdash; worth raising if budget is flexible.")

    # ---------------- watch-outs ---------------- #
    ex.watch_outs = _watch_outs(result, met, p)

    # ---------------- talking points ---------------- #
    ex.talking_points = [
        f"Lead with reach: {compact(met['reach_per_day'])} people a day.",
        f"If they push on price, the floor is {money(met['package_floor'])} "
        f"&mdash; do not go below without approval.",
    ]
    if met.get("stretch_worth_offering"):
        ex.talking_points.append(
            f"Upsell ready: +{money(met['stretch_extra_spend'])} buys "
            f"+{compact(met['stretch_extra_reach'])} people a day.")
    if p.blocks.notes:
        ex.talking_points.append(
            "Time-of-day choice is defensible &mdash; see the note in the plan.")
    return ex


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

#: D3 component families -> rep language. The engine speaks in signed
#: multipliers; a rep needs to know which way and why.
_FAMILY_PLAIN = {
    "demand": "how much demand this location is seeing",
    "occupancy": "how full the screen already is",
    "seasonality": "the time of year",
    "event": "events happening nearby during the campaign",
    "relationship": "the client's profile",
    "tier": "the location's market tier",
    "history": "this screen's own booking history",
}


def _price_drivers_plain(quotes: Dict, limit: int = 3) -> List[str]:
    """The two or three factors that actually moved the price, in words."""
    if not quotes:
        return []
    agg: Dict[str, Dict[str, Any]] = {}
    events: List[str] = []
    trend_up = trend_dn = 0
    for q in quotes.values():
        for comp in (q.components or []):
            a = agg.setdefault(comp["name"], {"shift": 0.0, "n": 0,
                                              "family": comp.get("family", "")})
            a["shift"] += float(comp.get("shift", 0.0))
            a["n"] += 1
        if getattr(q, "event_adjustment_pct", 0) and q.event_adjustment_pct > 0:
            if q.event_names:
                events.append(str(q.event_names))
        t = getattr(q, "ridership_trend_pct", 0) or 0
        if t > 0:
            trend_up += 1
        elif t < 0:
            trend_dn += 1

    out: List[str] = []
    ranked = sorted(agg.items(),
                    key=lambda kv: -abs(kv[1]["shift"] / max(kv[1]["n"], 1)))
    for name, a in ranked[:limit]:
        mean = -(a["shift"] / max(a["n"], 1))
        if abs(mean) < 0.005:
            continue
        word = "raises" if mean > 0 else "lowers"
        fam = _FAMILY_PLAIN.get(a["family"], a["family"] or "the location's profile")
        out.append(f"{fam.capitalize()} <b>{word} the price {abs(mean):.0%}</b>.")

    if events:
        uniq = sorted(set(events))
        out.append(
            f"Events during the campaign window push demand up at some locations "
            f"({_oxford(uniq, 2)}), so those screens are priced higher &mdash; "
            f"that is a genuine reason the client can verify.")
    if trend_dn and trend_dn >= trend_up:
        out.append(
            f"Passenger numbers are trending <b>down</b> on {trend_dn} of these "
            f"routes for these specific dates, and the price has been reduced "
            f"accordingly rather than quoting off a stale average.")
    elif trend_up:
        out.append(
            f"Passenger numbers are trending <b>up</b> on {trend_up} of these "
            f"routes for these dates, which supports the price.")
    return out


_POSTURE_PLAIN = {
    "hold_at_target": "hold the quoted price -- comparable clients accept it",
    "hold_at_cap": "there is room above the quoted price if they do not push back",
    "open_below_target": "expect to negotiate down; comparable clients push on price",
}


def _posture_plain(posture: str) -> str:
    return _POSTURE_PLAIN.get(str(posture), str(posture).replace("_", " "))


_REFUSAL_PLAIN = {
    "city": "Wrong city.",
    "daypart": "No inventory at the requested times.",
    "relevance": "Nothing matched the brief's requirements.",
    "capacity": "Everything matching is already sold out for these dates.",
    "budget": "Budget will not cover this.",
    "optimiser": "Internal issue building the package.",
}


def _refusal_plain(stage: str) -> str:
    return _REFUSAL_PLAIN.get(str(stage), f"{stage}:")


def _watch_outs(result, met: Dict[str, Any], p) -> List[str]:
    """Things the rep should know before the client asks."""
    out: List[str] = []

    if met.get("cold_start_spend_share", 0) > 0.2:
        out.append(
            f"<b>{pct(met['cold_start_spend_share'])} of the spend</b> is on "
            f"screens we have not sold before. The audience numbers there are "
            f"modelled from foot traffic and passenger data rather than measured "
            f"from past campaigns &mdash; sound, but say 'estimated' if pressed.")
    if met.get("peer_priced_spend_share", 0) > 0.3:
        out.append(
            f"<b>{met.get('peer_priced_lines', 0)} of {met['n_lines']} prices</b> "
            f"are based on comparable screens rather than that exact screen's own "
            f"history at that time of day. Normal for newer inventory.")
    if p.blocks.unevidenced:
        out.append(
            "Some of the requested hours have no audience measurement at all. "
            "That inventory can be sold, but do <b>not</b> quote view numbers "
            "for it.")
    if met["budget_utilisation"] > 1.0:
        out.append(f"This plan is <b>over</b> the stated budget "
                   f"({pct(met['budget_utilisation'])}). Get it approved before "
                   f"sending.")
    if getattr(p.brief, "notes", None):
        for n in p.brief.notes:
            m = re.search(r"'([^']+)'", str(n))
            place = m.group(1) if m else "a location"
            out.append(
                f"The brief mentions <b>{place}</b>, which does not map cleanly "
                f"to a location in our network. Confirm what the client means "
                f"before promising coverage there.")
    ind = getattr(p, "industry", None)
    if ind is not None and getattr(ind, "confidence", 1.0) <= 0.5:
        out.append(
            "We had limited signal on what industry this client is in, so the "
            "pricing benchmark may not be the right comparison. Worth a sanity "
            "check.")
    for w in p.warnings:
        if "industry" in str(w).lower():
            continue          # already covered above in plain language
        out.append(humanise(str(w)))
    return out

"""Composes the client-facing RFP response.

WHERE THE TEMPLATE COMES FROM. All six sample briefs close with an identically
shaped "5. RFP Requirements (Sales Team Response)" section: a framing sentence
then exactly three numbered asks. Across all six they are always the same three
things --

    1. a scoped/ranked INVENTORY list, with the selection logic named
    2. a PRICING recommendation, reflecting the demand drivers the brief cares about
    3. a REACH / IMPRESSIONS projection for the stated window, with reasoning

-- and then one brief-specific twist on top (weekly impressions and conversion
potential; unique late-night impressions with frequency prioritised; the
weekend-versus-weekday split shown separately; the walking-radius logic stated
explicitly; business-traveller frequency separated from leisure exposure;
reasoning tied to affinity per node type).

So the house DEFAULT template is those three sections plus an executive summary,
an explicit assumptions/disclosures section, and next steps. When a brief DOES
carry its own RFP block, the same three sections are emitted but each one is
labelled with the client's own wording, and a coverage table maps every ask to
the section that answers it -- so a reader can check the response is complete
without reading it twice.

Every number here comes from a run. This module formats; it does not compute.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app import config
from app.services.metrics import compact, money, pct

# --------------------------------------------------------------------------- #
# ask classification -- which section answers which ask
# --------------------------------------------------------------------------- #

_ASK_INVENTORY = re.compile(
    r"inventory|shortlist|curated list|list of screens|screen selection|"
    r"package|nodes?|coverage|anchored on|screens? (within|across)", re.I)
_ASK_PRICING = re.compile(r"pric|rate|cost|quote|cpm|premium.*positioning", re.I)
_ASK_DELIVERY = re.compile(
    r"reach|impression|project|estimate|footfall|frequency|exposure|"
    r"booking potential|conversion", re.I)

SECTION_TITLES = {
    "inventory": "1. Recommended inventory",
    "pricing": "2. Pricing and commercial terms",
    "delivery": "3. Projected delivery",
}


def classify_ask(ask: str) -> List[str]:
    """An ask can legitimately map to more than one section."""
    hits = []
    if _ASK_INVENTORY.search(ask):
        hits.append("inventory")
    if _ASK_PRICING.search(ask):
        hits.append("pricing")
    if _ASK_DELIVERY.search(ask):
        hits.append("delivery")
    return hits or ["delivery"]


# --------------------------------------------------------------------------- #
# brief-specific twists, detected from the brief + its asks
# --------------------------------------------------------------------------- #

@dataclass
class Extras:
    weekly_impressions: bool = False
    split_day_type: bool = False
    radius_logic: bool = False
    frequency_led: bool = False
    per_node_type: bool = False
    segment_split: bool = False
    conversion_oriented: bool = False
    notes: List[str] = field(default_factory=list)


def detect_extras(brief, asks: List[str], plan) -> Extras:
    e = Extras()
    blob = " ".join(asks).lower() + " " + (brief.objective or "").lower()

    if re.search(r"weekly impressions|per week", blob):
        e.weekly_impressions = True
    if (getattr(brief, "day_type_weighting", "") == "report_split"
            or re.search(r"weekend[- ]versus[- ]weekday|split.*separately", blob)):
        e.split_day_type = True
    if getattr(brief, "radius_only", False) or re.search(r"walking (distance|radius)", blob):
        e.radius_logic = True
    if getattr(brief, "optimise_for", "reach") == "frequency" or \
            re.search(r"prioriti[sz]ing frequency|frequency over", blob):
        e.frequency_led = True
    if re.search(r"per (recommended )?node type|each recommended node|node type|"
                 r"affinity at each", blob):
        e.per_node_type = True
    if re.search(r"separate estimates|repeat .*frequency|one[- ]time .*exposure|"
                 r"business .*leisure", blob):
        e.segment_split = True
    if re.search(r"conversion|booking potential|footfall[- ]conversion|test[- ]drive", blob):
        e.conversion_oriented = True
    return e


# --------------------------------------------------------------------------- #
# audience segment decomposition (brief 5's twist)
# --------------------------------------------------------------------------- #

def _parse_composition(s: Any) -> Dict[str, float]:
    """audience_profiles.composition_str looks like
    'commuter_professional:0.56; traveller:0.27; shopper:0.10'."""
    out: Dict[str, float] = {}
    if not isinstance(s, str) or not s.strip():
        return out
    for part in s.split(";"):
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            continue
    return out


def segment_breakdown(lines: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Split the plan's reach across D1's audience segments.

    Reach-weighted average of each selected screen's own composition. This is
    ARITHMETIC OVER EXISTING COLUMNS, not a new model, and the RFP text says so
    -- brief 5 asks for business-traveller versus leisure-traveller estimates,
    and this is the honest way to answer it from what D1 already measured.
    """
    if not len(lines) or profiles is None or not len(profiles):
        return pd.DataFrame()
    comp = dict(zip(profiles["screen_id"].astype(str),
                    profiles.get("composition_str", pd.Series(dtype=object))))
    totals: Dict[str, float] = {}
    grand = 0.0
    for _, r in lines.iterrows():
        sid = str(r.get("screen_id", ""))
        reach = float(r.get("people_reach", 0.0) or 0.0)
        mix = _parse_composition(comp.get(sid))
        if not mix:
            totals["unclassified"] = totals.get("unclassified", 0.0) + reach
            grand += reach
            continue
        s = sum(mix.values()) or 1.0
        for seg, share in mix.items():
            totals[seg] = totals.get(seg, 0.0) + reach * (share / s)
        grand += reach
    if grand <= 0:
        return pd.DataFrame()
    rows = [{"segment": k, "reach_per_day": v, "share": v / grand}
            for k, v in sorted(totals.items(), key=lambda kv: -kv[1])]
    return pd.DataFrame(rows)


def node_type_breakdown(lines: pd.DataFrame, scored: Optional[pd.DataFrame],
                        duration_days: int) -> pd.DataFrame:
    """Per screen_type delivery + the affinity evidence behind it (brief 6)."""
    if not len(lines):
        return pd.DataFrame()
    df = lines.copy()
    if scored is not None and len(scored) and "screen_type" in scored.columns:
        keep = ["screen_id", "screen_type"]
        if "relevance_score" in scored.columns:
            keep.append("relevance_score")
        if "audience_fit" in scored.columns:
            keep.append("audience_fit")
        df = df.merge(scored[keep].drop_duplicates("screen_id"),
                      on="screen_id", how="left")
    if "screen_type" not in df.columns:
        return pd.DataFrame()

    slots_col = ("slots_booked_per_day" if "slots_booked_per_day" in df.columns
                 else "slots")
    if "cost" not in df.columns:
        df["cost"] = df["price"].astype(float) * df[slots_col].astype(float) \
                     * float(duration_days)
    agg = {"screen_id": "nunique", "people_reach": "sum", "cost": "sum"}
    if "relevance_score" in df.columns:
        agg["relevance_score"] = "mean"
    if "audience_fit" in df.columns:
        agg["audience_fit"] = "mean"
    g = df.groupby("screen_type", observed=True).agg(agg).reset_index()
    return g.rename(columns={"screen_id": "screens", "people_reach": "reach_per_day",
                             "cost": "spend"}).sort_values("reach_per_day",
                                                            ascending=False)


# --------------------------------------------------------------------------- #
# the composer
# --------------------------------------------------------------------------- #

def _inventory_table(lines: pd.DataFrame, scored: Optional[pd.DataFrame],
                     quoted: Optional[pd.DataFrame], duration_days: int) -> str:
    if not len(lines):
        return "_No inventory allocated._\n"
    df = lines.copy()
    if scored is not None and len(scored):
        keep = [c for c in ["screen_id", "screen_type", "zone_id", "relevance_score",
                            "reasons"] if c in scored.columns]
        df = df.merge(scored[keep].drop_duplicates("screen_id"),
                      on="screen_id", how="left")
    slots_col = ("slots_booked_per_day" if "slots_booked_per_day" in df.columns
                 else "slots")
    qmap: Dict[Tuple[str, int], float] = {}
    if quoted is not None and len(quoted):
        qmap = {(str(r.screen_id), int(r.time_block_id)): float(r.quoted_price)
                for r in quoted.itertuples()}

    out = ["| # | Screen | Type | Zone | Block | Slots/day | Days | Price/slot/day | Spend | Reach/day | Relevance |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        sid = str(r.get("screen_id", ""))
        blk = int(r.get("time_block_id", 0) or 0)
        slots = float(r.get(slots_col, 0) or 0)
        price = qmap.get((sid, blk), float(r.get("price", 0.0) or 0.0))
        spend = price * slots * duration_days
        rel = r.get("relevance_score")
        out.append(
            f"| {i} | `{sid}` | {r.get('screen_type', '--')} | "
            f"{r.get('zone_id', '--')} | {config.TIME_BLOCKS.get(blk, blk)} | "
            f"{slots:.0f} / 6 | {duration_days} | {money(price, 2)} | "
            f"{money(spend)} | {compact(r.get('people_reach', 0))} | "
            f"{('%.3f' % rel) if pd.notna(rel) else '--'} |")
    return "\n".join(out) + "\n"


def _why_each(lines: pd.DataFrame, scored: Optional[pd.DataFrame],
              limit: int = 8) -> str:
    if scored is None or not len(scored) or not len(lines):
        return ""
    sel = set(lines["screen_id"].astype(str))
    s = scored[scored["screen_id"].astype(str).isin(sel)]
    if not len(s) or "reasons" not in s.columns:
        return ""
    if "relevance_score" in s.columns:
        s = s.sort_values("relevance_score", ascending=False)
    out = []
    for _, r in s.head(limit).iterrows():
        out.append(f"- **`{r['screen_id']}`** — {r['reasons']}")
    if len(s) > limit:
        out.append(f"- _...and {len(s) - limit} further screen(s) on the same criteria._")
    return "\n".join(out) + "\n"


def _price_drivers(quotes: Dict, limit: int = 6) -> str:
    """The multiplier ledger, aggregated across the package."""
    if not quotes:
        return ""
    agg: Dict[str, Dict[str, Any]] = {}
    events: List[str] = []
    trends: List[str] = []
    bases: Dict[str, int] = {}
    for q in quotes.values():
        bases[q.rate_card_basis] = bases.get(q.rate_card_basis, 0) + 1
        for c in (q.components or []):
            a = agg.setdefault(c["name"], {"shift": 0.0, "n": 0,
                                            "family": c.get("family", ""),
                                            "reason": c.get("reason", "")})
            a["shift"] += float(c.get("shift", 0.0))
            a["n"] += 1
        if getattr(q, "event_adjustment_pct", 0) and q.event_adjustment_pct > 0:
            events.append(f"{q.event_names or 'matched event(s)'} "
                          f"(+{q.event_adjustment_pct:.1%})")
        if getattr(q, "ridership_trend_pct", 0):
            trends.append(f"{q.ridership_trend_pct:+.1%}")

    out: List[str] = []
    if bases:
        out.append("**Rate-card basis across the package**\n")
        for b, n in sorted(bases.items(), key=lambda kv: -kv[1]):
            out.append(f"- `{b}` — {n} line(s)")
        out.append("")
    ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]["shift"] / max(kv[1]["n"], 1)))
    if ranked:
        out.append("**Applied adjustments (mean across priced lines)**\n")
        out.append("| Factor | Family | Mean effect | Rationale |")
        out.append("|---|---|---|---|")
        for name, a in ranked[:limit]:
            mean = a["shift"] / max(a["n"], 1)
            out.append(f"| {name} | {a['family']} | {-mean:+.1%} | {a['reason']} |")
        out.append("")
    if events:
        uniq = sorted(set(events))
        out.append(f"**Event overlap in this window** — {len(events)} line(s) carry an "
                   f"upward band shift from events genuinely overlapping the requested "
                   f"dates: {'; '.join(uniq[:4])}"
                   + (" ..." if len(uniq) > 4 else "") + "\n")
    if trends:
        out.append(f"**Ridership trend** — {len(trends)} line(s) carry a band shift from "
                   f"a statistically significant corridor ridership trend for these "
                   f"dates (range {min(trends)} to {max(trends)}).\n")
    return "\n".join(out)


def _assumptions(result, met: Dict[str, Any], extras: Extras,
                 learn_note: str) -> str:
    p = result.plan
    out: List[str] = []

    out.append(f"- **Impressions basis** — {met['impressions_basis']}. Reach and "
               f"impressions are different curves over the same buy: reach saturates "
               f"as `1 - exp(-lambda)` while impressions stay linear in slots, so the "
               f"ratio between them is the average frequency quoted above.")
    out.append(f"- **Reach is per day.** {compact(met['reach_per_day'])} is the distinct "
               f"audience reached on a typical {met['day_type']}, not a cumulative "
               f"unique count over the {met['flight_days']}-day flight.")
    out.append(f"- **Shared audiences are not double counted.** Screens at the same "
               f"location or on the same corridor share an audience pool; the "
               f"optimiser discounts marginal reach within a pool rather than adding "
               f"it twice. {met.get('lines_sharing_a_pool', 0)} of {met['n_lines']} "
               f"line(s) sit in a pool shared with another selected line.")
    out.append(f"- **How the plan was built** — selected by constrained optimisation "
               f"over {met['n_candidate_cells']:,} available screen and time-slot "
               f"combinations, respecting a six-slot-per-time-block ceiling and the "
               f"real unsold capacity for these dates.")

    # Two distinct disclosures. D1's flag is screen-level (ever booked at all);
    # D3's rate_card_basis is cell-level (was THIS screen x block's band
    # credible, or borrowed from peers). Reporting one without the other made
    # the document look like it contradicted itself.
    if met.get("cold_start_spend_share", 0) > 0:
        out.append(
            f"- **Inventory with no booking history at all** — "
            f"{pct(met['cold_start_spend_share'])} of spend "
            f"({met.get('n_cold_start_screens', 0)} screen(s)) sits on screens that "
            f"have never been booked. Their audience is modelled from POI proximity, "
            f"route ridership and zone demographics rather than inferred from "
            f"observed sales.")
    if met.get("peer_priced_spend_share", 0) > 0:
        out.append(
            f"- **Lines priced from comparable inventory** — "
            f"{met.get('peer_priced_lines', 0)} of {met['n_lines']} lines "
            f"({pct(met['peer_priced_spend_share'])} of spend) had too little booking "
            f"history *in that specific time block* for a rate built on their own "
            f"record, so the band is borrowed from peer screens of the same type and "
            f"mount in the same city and blended with the ML price model. This is a "
            f"narrower statement than the line above: a screen can have a strong "
            f"booking record in one daypart and none in another.")
    if extras.radius_logic:
        out.append(
            "- **Walking-radius scoping** — the source tables carry no latitude or "
            "longitude, so 'within walking distance' is resolved as zone membership "
            "around the named anchor rather than a metric distance. Screens outside "
            "those zones were excluded from consideration entirely.")
    if extras.segment_split:
        out.append(
            "- **Segment split method** — the audience split below is a reach-weighted "
            "average of each selected screen's measured composition. It reallocates the "
            "same modelled reach across segments; it is arithmetic over the audience "
            "profiles already measured for each screen, not a separately trained model.")
    if p.blocks.unevidenced:
        out.append(
            f"- **Unmeasured dayparts** — block(s) {p.blocks.unevidenced} have no "
            f"measurable audience in the network's data. Inventory there is quotable "
            f"but no impressions are claimed for it.")
    for w in p.warnings:
        out.append(f"- **Note** — {w}")
    if getattr(p.brief, "notes", None):
        for n in p.brief.notes:
            out.append(f"- **Brief flag** — {n}")
    if learn_note:
        out.append(f"- **Learned adjustments** — {learn_note}")
    return "\n".join(out) + "\n"


def compose(*, result, quoted: Optional[pd.DataFrame] = None,
            quoted_totals: Optional[Dict[str, Any]] = None,
            profiles: Optional[pd.DataFrame] = None,
            asks: Optional[List[str]] = None,
            learn_note: str = "",
            rep_email: str = "") -> Tuple[str, str, List[str]]:
    """Returns (markdown, template_used, asks_used)."""
    p = result.plan
    b = p.brief
    met = result.metrics
    asks = [a for a in (asks or []) if a.strip()]
    template = "brief_rfp" if asks else "default"
    extras = detect_extras(b, asks, p)

    days = int(b.duration_days or 0)
    quoted_total = (quoted_totals or {}).get("quoted_total", met["spend"])

    L: List[str] = []
    A = L.append

    # ---------------- header ---------------- #
    A(f"# Campaign proposal — {b.name or 'Untitled campaign'}")
    A("")
    A(f"**Prepared by** Urban Media commercial intelligence"
      + (f" · {rep_email}" if rep_email else ""))
    A(f"**City** {config.CITY_CHOICES.get(b.city_id, b.city_id or 'unspecified')}  ")
    A(f"**Flight** {days} days, {met['day_type']} delivery  ")
    A(f"**Client budget** {money(b.budget)}  ")
    A(f"**Objective** {b.objective or '--'} (optimised for "
      f"**{met['objective']}**, {met['day_type']} audience)")
    A("")

    if not p.presentable:
        A("> ## This proposal is not ready to send")
        A(">")
        for r in p.refusals:
            A(f"> **{r.stage}** — {r.reason}  ")
            A(f"> _Remedy:_ {r.remedy}")
            A(">")
        A(f"> Funnel: " + " → ".join(f"{n} {v:,}" for n, v in p.funnel.rows()))
        A("")

    # ---------------- executive summary ---------------- #
    A("## Executive summary")
    A("")
    if len(result.lines):
        A(f"We recommend a **{met['n_lines']}-line package across "
          f"{met['n_screens']} screen(s)** over a **{days}-day** flight, delivering an "
          f"estimated **{compact(met['reach_per_day'])} distinct people per day** and "
          f"**{compact(met['impressions_total'])} impressions** across the window at an "
          f"average frequency of **{met['avg_frequency_per_day']:.2f}x per day**.")
        A("")
        A(f"Recommended investment is **{money(quoted_total)}** "
          f"({pct(met['budget_utilisation'])} of the stated {money(b.budget)} budget), "
          f"an effective CPM of **{money(met['effective_cpm'], 2)}**.")
        A("")
        if extras.frequency_led:
            A("This plan is solved for **frequency**, not breadth: it buys depth on "
              "fewer, better-matched screens so the target audience sees the message "
              "repeatedly. The reach figure is therefore what this plan happens to "
              "reach, not the quantity being maximised.")
            A("")
    else:
        A("No allocatable package could be produced for this brief as scoped. "
          "See the refusal above.")
        A("")

    # ---------------- RFP coverage ---------------- #
    if asks:
        A("## Response to your stated RFP requirements")
        A("")
        A("| Your requirement | Answered in |")
        A("|---|---|")
        for ask in asks:
            secs = ", ".join(SECTION_TITLES[s] for s in classify_ask(ask))
            A(f"| {ask} | {secs} |")
        A("")
    else:
        A("_This brief did not include a formal RFP requirements section, so Urban "
          "Media's standard three-part response template is used below: recommended "
          "inventory, pricing, and projected delivery._")
        A("")

    # ---------------- 1. inventory ---------------- #
    A(f"## {SECTION_TITLES['inventory']}")
    A("")
    if extras.radius_logic:
        anchor = getattr(b, "radius_anchor_hint", "") or "the named anchor"
        A(f"**Scoping logic.** This is a hyper-local buy. Every screen below sits "
          f"within the walking catchment of **{anchor}**; everything outside that "
          f"catchment was excluded from consideration rather than ranked lower, "
          f"because reach beyond the radius is wasted spend for a single-location "
          f"campaign.")
        A("")
    A(f"Selected from **{met.get('n_scored_candidates', 0):,} candidate locations** "
      f"scored against this brief in "
      f"{config.CITY_CHOICES.get(b.city_id, 'our network').split(' (')[0]}, "
      f"narrowed to {met['n_candidate_cells']:,} available screen and time-slot "
      f"combinations.")
    A("")
    A(_inventory_table(result.lines, result.scored, quoted, days))
    A(f"**Shortlist sizing.** {p.sizing.reason}")
    A("")
    why = _why_each(result.lines, result.scored)
    if why:
        A("**Why each node was selected**")
        A("")
        A(why)

    if extras.per_node_type:
        nb = node_type_breakdown(result.lines, result.scored, days)
        if len(nb):
            A("**Delivery and affinity by node type**")
            A("")
            A("| Node type | Screens | Reach/day | Spend | Mean relevance | Mean audience fit |")
            A("|---|---|---|---|---|---|")
            for _, r in nb.iterrows():
                A(f"| {r['screen_type']} | {int(r['screens'])} | "
                  f"{compact(r['reach_per_day'])} | {money(r['spend'])} | "
                  f"{r.get('relevance_score', float('nan')):.3f} | "
                  f"{r.get('audience_fit', float('nan')):.3f} |")
            A("")

    # ---------------- 2. pricing ---------------- #
    A(f"## {SECTION_TITLES['pricing']}")
    A("")
    A("| | Package total |")
    A("|---|---|")
    A(f"| Negotiating floor | {money(met['package_floor'])} |")
    A(f"| **Target (recommended)** | **{money(met['package_target'])}** |")
    A(f"| Cap | {money(met['package_cap'])} |")
    A(f"| Quoted in this proposal | {money(quoted_total)} |")
    A("")
    A(f"The floor–cap range is the defensible negotiating window for this package: "
      f"the floor is a real percentile of comparable bookings rather than a discount, "
      f"and quoting above the cap is where comparable deals historically stopped "
      f"closing. At the quoted price the modelled win probability is "
      f"**{pct(met['weighted_p_win'], 0)}** (spend-weighted across lines), for an "
      f"expected value of {money(met['expected_revenue'])}.")
    A("")
    drivers = _price_drivers(result.quotes)
    if drivers:
        A(drivers)
    if quoted_totals and abs(quoted_totals.get("delta", 0.0)) > 0.5:
        A(f"**Learned adjustment.** {quoted_totals['n_lines_moved']} line(s) were "
          f"repriced by {pct(quoted_totals['delta_pct'])} versus the model's own "
          f"recommendation, from accumulated feedback on comparable inventory. Every "
          f"adjusted price remains inside its own modelled floor–cap band.")
        A("")
    if met.get("stretch_worth_offering"):
        A(f"**Optional stretch.** An additional {money(met['stretch_extra_spend'])} "
          f"({pct(met['stretch_overrun_pct'])} above the stated budget, "
          f"{'inside' if met['stretch_inside_tolerance'] else 'outside'} the "
          f"{pct(met['cohort_tolerance_pct'], 0)} tolerance typical of comparable "
          f"advertisers) would add {compact(met['stretch_extra_reach'])} people per "
          f"day — {met['stretch_reach_per_1k']:,.0f} additional people per extra "
          f"$1,000 invested. Presented as an option, not built into the price above.")
        A("")

    # ---------------- 3. delivery ---------------- #
    A(f"## {SECTION_TITLES['delivery']}")
    A("")
    A("| Metric | Value |")
    A("|---|---|")
    A(f"| Distinct people per day | {compact(met['reach_per_day'])} |")
    A(f"| Impressions over the {days}-day flight | {compact(met['impressions_total'])} |")
    if extras.weekly_impressions and days:
        weeks = max(days / 7.0, 1e-9)
        A(f"| **Impressions per week** | **{compact(met['impressions_total'] / weeks)}** |")
    A(f"| Impressions per day | {compact(met['impressions_per_day'])} |")
    A(f"| Average frequency per day | {met['avg_frequency_per_day']:.2f}x |")
    A(f"| Effective CPM | {money(met['effective_cpm'], 2)} |")
    A(f"| Cost per 1,000 daily reach | {money(met['cost_per_1k_daily_reach'], 2)} |")
    A(f"| Slots per day / available | {met.get('slots_bought_per_day', 0):.0f} / "
      f"{met.get('slots_available_per_day', 0):.0f} |")
    A("")

    if extras.split_day_type:
        A("### Weekday and weekend delivery, shown separately")
        A("")
        if result.split:
            A(result.split_note)
            A("")
            A("| Day type | Lines | Spend | Reach/day | Impressions over flight |")
            A("|---|---|---|---|---|")
            for dt, s in result.split.items():
                A(f"| {dt} | {s.n_lines} | {money(s.spend)} | {compact(s.reach)} | "
                  f"{compact(s.impressions)} |")
            A("")
        else:
            A("_The brief asks for this split. It could not be produced for this run "
              "— see the notes below._")
            A("")

    if extras.segment_split:
        sb = segment_breakdown(result.lines, profiles)
        if len(sb):
            A("### Audience split")
            A("")
            A("| Segment | Reach per day | Share |")
            A("|---|---|---|")
            for _, r in sb.iterrows():
                A(f"| {r['segment']} | {compact(r['reach_per_day'])} | "
                  f"{pct(r['share'], 1)} |")
            A("")
            A(f"Repeat-exposure segments (commuters and professionals moving through "
              f"these nodes on a predictable schedule) see the message at the "
              f"{met['avg_frequency_per_day']:.2f}x daily frequency above. "
              f"One-time or infrequent segments (travellers, visitors) are better read "
              f"as single-exposure reach.")
            A("")

    if extras.conversion_oriented:
        A("### On conversion")
        A("")
        A("Urban Media measures delivered audience and impressions. Converting those "
          "into footfall, bookings or trial requires a response rate this network has "
          "no data to supply — it depends on the creative, the offer and the client's "
          "own funnel. We therefore state delivery precisely and deliberately do not "
          "quote a conversion figure we cannot evidence. The daily frequency above is "
          "the most useful input to the client's own conversion model.")
        A("")

    # ---------------- 4. audience rationale ---------------- #
    A("## 4. Why this audience matches the brief")
    A("")
    A(f"**Brief's stated target** — {b.audience_text or 'not stated'}")
    A("")
    narrative = getattr(result, "narrative", None)
    if narrative:
        A(str(narrative))
        A("")
    c = p.cohort
    if getattr(c, "n_clients", 0):
        A(f"**Comparable advertisers.** No account history exists for this advertiser, "
          f"so positioning is benchmarked against {c.n_clients} comparable "
          f"{p.industry.industry or 'network'} advertisers: they book a median "
          f"{money(c.median_budget)} over {c.median_duration_days} days with a "
          f"{pct(c.variance_pct, 0)} budget tolerance. Recommended negotiating posture "
          f"is **{c.posture}** — {c.posture_reason}")
        A("")

    # ---------------- 5. assumptions ---------------- #
    A("## 5. Assumptions and disclosures")
    A("")
    A(_assumptions(result, met, extras, learn_note))

    # ---------------- 6. next steps ---------------- #
    A("## 6. Next steps")
    A("")
    A(f"1. Confirm the flight dates and the {money(quoted_total)} investment level.")
    A("2. Urban Media places a soft hold on the listed screen-slot inventory; "
      "availability reflects current forward occupancy and is not reserved until "
      "confirmed.")
    A("3. Creative specifications and delivery deadlines issued on confirmation.")
    A("")
    A("---")
    A(f"_Prepared by the Urban Media campaign planning platform. Inventory selected "
      f"by constrained optimisation across {met['n_candidate_cells']:,} available "
      f"screen and time-slot combinations; pricing from our rate card blended with a "
      f"win-probability model fitted on historical deals; audience modelled from "
      f"nearby points of interest, route passenger data and area demographics. Every "
      f"figure above is computed from data — none are estimates typed by hand._")

    return "\n".join(L), template, asks

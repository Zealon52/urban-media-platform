"""Builds the client-facing email a rep reads, approves, and sends.

This is the last mile. A rep's job after approving a plan is sending it, and
what they send is an email -- not a markdown file. So the draft has to be
finished prose: no placeholders, no jargon, no square brackets to fill in
except the recipient's name.

Structure follows what the sample briefs actually ask for, in the order a buyer
reads: what we're proposing, what it delivers, what it costs, then each of their
stated RFP requirements answered by name. The reasoning is one short paragraph,
not five -- the detail lives in the attached proposal.

Every number comes from the run. Nothing here computes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app import config
from app.services.explain import plain_block, plain_type
from app.services.metrics import compact, money, pct


def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", str(s))
    s = re.sub(r"&mdash;", "-", s)
    s = re.sub(r"&nbsp;", " ", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Draft:
    subject: str
    body: str
    to_hint: str = ""

    @property
    def as_text(self) -> str:
        return f"Subject: {self.subject}\n\n{self.body}"

    def as_eml(self, sender: str = "") -> bytes:
        """RFC-822 enough for a mail client to open with everything populated."""
        headers = [f"Subject: {self.subject}", "X-Unsent: 1",
                   "Content-Type: text/plain; charset=utf-8"]
        if sender:
            headers.insert(0, f"From: {sender}")
        if self.to_hint:
            headers.insert(0, f"To: {self.to_hint}")
        return ("\r\n".join(headers) + "\r\n\r\n"
                + self.body.replace("\n", "\r\n")).encode("utf-8")


def build(*, result, explanation, asks: Optional[List[str]] = None,
          quoted_totals: Optional[dict] = None, rep_name: str = "",
          rep_email: str = "", contact_name: str = "",
          include_reasoning: bool = True) -> Draft:
    p = result.plan
    b = p.brief
    met = result.metrics
    days = int(b.duration_days or 0)
    quote = (quoted_totals or {}).get("quoted_total", met["spend"])
    client = (b.name or "your campaign").strip()
    asks = [a for a in (asks or []) if a.strip()]

    city = config.CITY_CHOICES.get(b.city_id, b.city_id or "the network")
    city_short = city.split(" (")[0]

    # ---------------- subject ---------------- #
    subject = (f"{client}: {compact(met['reach_per_day'])} people a day across "
               f"{met['n_screens']} {city_short} locations - proposal")

    L: List[str] = []
    A = L.append

    greeting = f"Hi {contact_name}," if contact_name else "Hi,"
    A(greeting)
    A("")

    # ---------------- the offer ---------------- #
    obj = (b.objective or "").strip().rstrip(".")
    A(f"Thanks for the brief{f' on {obj.lower()}' if obj else ''}. Here is what we "
      f"can put together for {client}.")
    A("")
    A(f"We're proposing {met['n_lines']} ad slots across {met['n_screens']} digital "
      f"screens in {city_short}, running for {days} days. That reaches an estimated "
      f"{compact(met['reach_per_day'])} different people every day, and "
      f"{compact(met['impressions_total'])} total views across the campaign.")
    A("")
    A(f"Total investment: {money(quote)}.")
    A("")

    # ---------------- what it delivers ---------------- #
    A("WHAT THE CAMPAIGN DELIVERS")
    A("")
    A(f"  People reached per day     {compact(met['reach_per_day'])}")
    A(f"  Total views                {compact(met['impressions_total'])}")
    A(f"  Average times seen per day {met['avg_frequency_per_day']:.1f}")
    A(f"  Cost per thousand views    {money(met['effective_cpm'], 2)}")
    A(f"  Campaign length            {days} days")
    A(f"  Locations                  {met['n_screens']}")
    A("")

    # ---------------- where ---------------- #
    lines = result.lines
    if len(lines):
        types: Dict[str, int] = {}
        blocks: Dict[int, int] = {}
        if result.scored is not None and len(result.scored) \
                and "screen_type" in result.scored.columns:
            j = lines.merge(
                result.scored[["screen_id", "screen_type"]].drop_duplicates("screen_id"),
                on="screen_id", how="left")
            for t in j["screen_type"].dropna():
                types[str(t)] = types.get(str(t), 0) + 1
        for bl in lines.get("time_block_id", []):
            blocks[int(bl)] = blocks.get(int(bl), 0) + 1

        A("WHERE THE ADS RUN")
        A("")
        if types:
            for t, c in sorted(types.items(), key=lambda kv: -kv[1]):
                A(f"  {c} x {plain_type(t)}")
        if blocks:
            when = ", ".join(plain_block(k) for k, _ in
                            sorted(blocks.items(), key=lambda kv: -kv[1])[:2])
            A(f"  Running during: {when}")
        A("")

    # ---------------- reasoning ---------------- #
    if include_reasoning and explanation is not None:
        A("WHY THIS PLAN")
        A("")
        for item in (explanation.why_screens or [])[:3]:
            A(f"  - {_strip_html(item)}")
        for item in (explanation.why_price or [])[:2]:
            A(f"  - {_strip_html(item)}")
        A("")

    # ---------------- their RFP requirements ---------------- #
    if asks:
        A("YOUR REQUIREMENTS, ADDRESSED")
        A("")
        for i, ask in enumerate(asks, start=1):
            A(f"{i}. {_strip_html(ask)}")
            A(f"   {_answer_ask(ask, result, met, quote, days)}")
            A("")

    # ---------------- split / segment extras ---------------- #
    if result.split:
        A("WEEKDAY AND WEEKEND, SHOWN SEPARATELY")
        A("")
        for dt, s in result.split.items():
            A(f"  {dt.capitalize():<9} {compact(s.reach)} people/day, "
              f"{compact(s.impressions)} views over the campaign")
        A("")
        A("  These are two alternative ways to run the same budget, not additive.")
        A("")

    # ---------------- stretch ---------------- #
    if met.get("stretch_worth_offering"):
        A("OPTIONAL EXTENSION")
        A("")
        A(f"  For an additional {money(met['stretch_extra_spend'])}, we can add "
          f"{compact(met['stretch_extra_reach'])} more people per day. Happy to "
          f"include this if the budget allows.")
        A("")

    # ---------------- caveats worth stating up front ---------------- #
    honest = _client_safe_caveats(met, p)
    if honest:
        A("NOTES")
        A("")
        for h in honest:
            A(f"  - {h}")
        A("")

    # ---------------- close ---------------- #
    A("The full proposal with the location-by-location breakdown is attached.")
    A("")
    A("Availability reflects current bookings and isn't held until confirmed, so "
      "do let me know if you'd like us to place a hold while you review.")
    A("")
    A("Happy to walk through any of it on a call.")
    A("")
    A("Best regards,")
    A(rep_name or "[Your name]")
    if rep_email:
        A(rep_email)
    A("Urban Media")

    return Draft(subject=subject, body="\n".join(L), to_hint="")


# --------------------------------------------------------------------------- #
# answering the brief's own numbered requirements
# --------------------------------------------------------------------------- #

def _answer_ask(ask: str, result, met: Dict[str, Any], quote: float,
                days: int) -> str:
    """One sentence answering this specific requirement, with real figures."""
    a = ask.lower()

    if re.search(r"weekly impressions|per week", a):
        weeks = max(days / 7.0, 1e-9)
        return (f"{compact(met['impressions_total'] / weeks)} views per week. "
                f"On conversion: we measure delivered audience, and the response "
                f"rate depends on your creative and offer, so we've given you the "
                f"exposure figures rather than a conversion estimate we can't "
                f"stand behind.")
    if re.search(r"weekend[- ]versus[- ]weekday|split.*separat", a):
        if result.split:
            parts = ", ".join(
                f"{dt} {compact(s.impressions)} views"
                for dt, s in result.split.items())
            return f"Shown separately above: {parts}."
        return "Weekday and weekend delivery are shown separately above."
    if re.search(r"walking (distance|radius)", a):
        anchor = getattr(result.plan.brief, "radius_anchor_hint", "") or "the outlet"
        return (f"Every location is within walking distance of {anchor}. Anything "
                f"outside that radius was excluded rather than ranked lower, since "
                f"reach beyond a short walk doesn't drive footfall to a single site.")
    if re.search(r"frequency|unique late[- ]night", a):
        return (f"Built for frequency: {met['avg_frequency_per_day']:.1f} views per "
                f"person per day across {met['n_screens']} locations, concentrated "
                f"rather than spread thin.")
    if re.search(r"repeat|one[- ]time|business .*leisure", a):
        return ("Audience split by traveller type is in the attached proposal; "
                "regular commuters see the ad repeatedly at the frequency above, "
                "while occasional travellers are single-exposure.")
    if re.search(r"node type|each recommended node|affinity", a):
        return ("The attached proposal breaks delivery and audience fit down by "
                "location type.")
    if re.search(r"pric|rate|cost|quote", a):
        return (f"{money(quote)} total, {money(met['effective_cpm'], 2)} per thousand "
                f"views. Pricing reflects demand at each specific location and time "
                f"of day, and the per-location rates are in the attached proposal.")
    if re.search(r"inventory|shortlist|screens|list|package|anchored", a):
        return (f"{met['n_screens']} locations, listed individually in the attached "
                f"proposal with the reason each one was selected.")
    if re.search(r"reach|impression|project|estimate|footfall", a):
        return (f"{compact(met['reach_per_day'])} people per day and "
                f"{compact(met['impressions_total'])} total views over the {days}-day "
                f"window.")
    return (f"Covered in the attached proposal "
            f"({compact(met['reach_per_day'])} people/day, {money(quote)}).")


def _client_safe_caveats(met: Dict[str, Any], p) -> List[str]:
    """Only what a client should see. Internal risk notes stay internal.

    A rep forwarding this must not accidentally disclose our own pricing
    confidence or which inventory we have never sold -- but overstating
    precision is worse, so the estimate basis IS stated.
    """
    out: List[str] = []
    out.append("Audience figures are modelled from passenger counts, foot traffic "
               "and location data, and are estimates rather than guarantees.")
    out.append("People reached per day counts each person once per day, even where "
               "two screens are at the same station or on the same route.")
    if p.blocks.unevidenced:
        out.append("A small part of the schedule runs at hours we don't measure "
                   "audience for; no view numbers are claimed for those slots.")
    return out

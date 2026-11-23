"""The feedback router: rep comments -> a concrete, validated change set.

BATCHING IS THE WHOLE DESIGN. Feedback is collected on every screen but applied
NOWHERE until the rep approves a re-run from the recommendation screen. So a
full walkthrough where the rep comments on audience, drops two screens and
objects to the price costs exactly ONE solve and ONE router call -- not five of
each. That is what keeps the app usable against a $5 LLM ceiling.

TRUST BOUNDARY. The model is allowed to propose changes to a strict ALLOWLIST of
scalar brief fields, and every value is coerced and validated before it reaches
`StructuredBrief`. A hallucinated field name, a budget of "lots", or a city that
does not exist is dropped with a recorded reason rather than corrupting the
brief. The structured tuple fields (`must_have`, `exclude`) are deliberately NOT
routable -- they are (screen_type, position) pairs where a wrong guess silently
changes the candidate pool, so they stay rep-only via the UI's own multiselects.

Everything works without an LLM. `_route_with_rules` covers the cases that
actually recur -- price objections, audience and geography corrections, day-type
and objective flips -- using the same `loss_reason` vocabulary the historical
data uses.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app import config
from app.services import llm

# --------------------------------------------------------------------------- #
# the change set
# --------------------------------------------------------------------------- #

#: Scalar brief fields the router may touch, with their coercion type.
ROUTABLE_FIELDS: Dict[str, str] = {
    "budget": "float",
    "duration_days": "int",
    "city_id": "city",
    "objective": "str",
    "audience_text": "str",
    # audience_text alone is NOT enough: D2's _audience_fit only consults it
    # through a keyword table its own docstring documents as broken (three of
    # four targets map to segments that do not exist, scoring 0.0 everywhere).
    # audience_segments is the field that actually moves the score.
    "audience_segments": "segments",
    "preferred_blocks": "blocks",
    "optimise_for": "objective",
    "day_type_weighting": "day_type_weighting",
    "exclude_market_tier": "tiers",
    "prefer_market_tier": "tiers",
    "radius_only": "bool",
    "radius_anchor_hint": "str",
}

PRICE_POSTURES = ("toward_floor", "hold", "toward_cap")


@dataclass
class Actions:
    """A validated, applicable change set."""
    brief_edits: Dict[str, Any] = field(default_factory=dict)
    drop_screen_ids: List[str] = field(default_factory=list)
    day_type: Optional[str] = None
    price_posture: str = "hold"
    want_split: Optional[bool] = None
    loss_reasons: List[str] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    routed_by: str = "rules"
    router_note: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.brief_edits or self.drop_screen_ids or self.day_type
                    or self.price_posture != "hold" or self.want_split is not None)

    def summary_lines(self) -> List[str]:
        out: List[str] = []
        for k, v in self.brief_edits.items():
            out.append(f"set `{k}` to `{v!r}`")
        if self.drop_screen_ids:
            n = len(self.drop_screen_ids)
            shown = ", ".join(self.drop_screen_ids[:4])
            out.append(f"remove {n} screen(s) from the candidate pool: {shown}"
                       + (" ..." if n > 4 else ""))
        if self.day_type:
            out.append(f"solve against the `{self.day_type}` audience")
        if self.price_posture != "hold":
            out.append(f"quote `{self.price_posture.replace('_', ' ')}` within the D3 band")
        if self.want_split:
            out.append("report the weekday/weekend split separately")
        return out


# --------------------------------------------------------------------------- #
# coercion / validation
# --------------------------------------------------------------------------- #

def _coerce(field_name: str, kind: str, value: Any,
            rejected: List[str]) -> Tuple[bool, Any]:
    try:
        if kind == "float":
            v = float(str(value).replace(",", "").replace("$", "").strip())
            if v <= 0:
                rejected.append(f"{field_name}: {value!r} is not a positive amount")
                return False, None
            return True, v
        if kind == "int":
            v = int(float(str(value).replace(",", "").strip()))
            if v <= 0:
                rejected.append(f"{field_name}: {value!r} is not a positive count")
                return False, None
            return True, v
        if kind == "bool":
            if isinstance(value, bool):
                return True, value
            s = str(value).strip().lower()
            if s in {"true", "yes", "1"}:
                return True, True
            if s in {"false", "no", "0"}:
                return True, False
            rejected.append(f"{field_name}: {value!r} is not a boolean")
            return False, None
        if kind == "str":
            s = str(value).strip()
            if not s:
                rejected.append(f"{field_name}: empty")
                return False, None
            return True, s
        if kind == "city":
            s = str(value).strip().upper()
            if s not in config.CITY_CHOICES:
                rejected.append(f"city_id: {value!r} is not one of "
                                f"{list(config.CITY_CHOICES)}")
                return False, None
            return True, s
        if kind == "objective":
            s = str(value).strip().lower()
            if s not in {"reach", "frequency"}:
                rejected.append(f"optimise_for: {value!r} is not reach|frequency")
                return False, None
            return True, s
        if kind == "day_type_weighting":
            s = str(value).strip().lower()
            if s not in {"", "weekend_heavy", "report_split"}:
                rejected.append(f"day_type_weighting: {value!r} not allowed")
                return False, None
            return True, s
        if kind == "blocks":
            raw = value if isinstance(value, (list, tuple)) else [value]
            out = sorted({int(x) for x in raw if str(x).strip().isdigit()
                          and 1 <= int(x) <= 6})
            if not out:
                rejected.append(f"preferred_blocks: {value!r} yielded no block in 1..6")
                return False, None
            return True, out
        if kind == "segments":
            raw = value if isinstance(value, (list, tuple)) else [value]
            out = [str(x).strip().lower() for x in raw
                   if str(x).strip().lower() in config.AUDIENCE_SEGMENTS]
            if not out:
                rejected.append(
                    f"{field_name}: {value!r} names no segment this dataset "
                    f"records -- it has only "
                    f"{', '.join(config.AUDIENCE_SEGMENTS)}")
                return False, None
            return True, sorted(set(out))
        if kind == "tiers":
            raw = value if isinstance(value, (list, tuple)) else [value]
            out = [str(x).strip().lower() for x in raw
                   if str(x).strip().lower() in config.MARKET_TIERS]
            if not out:
                rejected.append(f"{field_name}: {value!r} has no valid market tier")
                return False, None
            return True, sorted(set(out))
    except (TypeError, ValueError):
        rejected.append(f"{field_name}: {value!r} could not be read as {kind}")
        return False, None
    rejected.append(f"{field_name}: unknown field kind {kind}")
    return False, None


def _validate(raw: dict, known_screen_ids: Optional[set]) -> Actions:
    a = Actions()
    edits = raw.get("brief_edits") or {}
    if isinstance(edits, dict):
        for k, v in edits.items():
            kind = ROUTABLE_FIELDS.get(str(k))
            if kind is None:
                a.rejected.append(f"`{k}` is not a routable brief field -- ignored")
                continue
            ok, coerced = _coerce(str(k), kind, v, a.rejected)
            if ok:
                a.brief_edits[str(k)] = coerced

    drops = raw.get("drop_screen_ids") or []
    if isinstance(drops, (list, tuple)):
        for s in drops:
            sid = str(s).strip()
            if not sid:
                continue
            if known_screen_ids is not None and sid not in known_screen_ids:
                a.rejected.append(f"screen `{sid}` is not in the current candidate "
                                  f"set -- ignored")
                continue
            a.drop_screen_ids.append(sid)
        a.drop_screen_ids = sorted(set(a.drop_screen_ids))

    dt = str(raw.get("day_type") or "").strip().lower()
    if dt in {"weekday", "weekend"}:
        a.day_type = dt
    elif dt:
        a.rejected.append(f"day_type `{dt}` is not weekday|weekend -- ignored")

    posture = str(raw.get("price_posture") or "hold").strip().lower()
    a.price_posture = posture if posture in PRICE_POSTURES else "hold"
    if posture not in PRICE_POSTURES and posture:
        a.rejected.append(f"price_posture `{posture}` unknown -- held at target")

    ws = raw.get("want_split")
    if isinstance(ws, bool):
        a.want_split = ws

    reasons = raw.get("loss_reasons") or []
    if isinstance(reasons, (list, tuple)):
        a.loss_reasons = [str(r).strip() for r in reasons
                          if str(r).strip() in config.LOSS_REASONS]

    rat = raw.get("rationale")
    if isinstance(rat, str) and rat.strip():
        a.rationale = [rat.strip()]
    elif isinstance(rat, (list, tuple)):
        a.rationale = [str(x).strip() for x in rat if str(x).strip()]

    return a


# --------------------------------------------------------------------------- #
# deterministic router -- always available, always the fallback
# --------------------------------------------------------------------------- #

_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"too expensive|too high|cheaper|lower the price|price is high|"
                r"discount|over budget|can'?t afford|reduce (the )?price", re.I),
     "price_down"),
    (re.compile(r"charge more|under[- ]?priced|too cheap|premium price|raise the price",
                re.I), "price_up"),
    (re.compile(r"\bstudents?\b|\bpupils?\b|university|college|campus", re.I),
     "aud_student"),
    (re.compile(r"office|professional|white[- ]collar|corporate", re.I),
     "aud_commuter_professional"),
    (re.compile(r"commuter|commute", re.I), "aud_commuter_professional"),
    (re.compile(r"shopper|retail audience|mall goer|\bshopping\b", re.I),
     "aud_shopper"),
    (re.compile(r"tourist|traveller|traveler|visitor", re.I), "aud_traveller"),
    (re.compile(r"\bresidents?\b|\blocals?\b|neighbourhood|neighborhood", re.I),
     "aud_resident"),
    (re.compile(r"blue[- ]collar|manual worker|tradespeople|tradesmen|"
                r"factory worker", re.I), "aud_blue_collar"),
    (re.compile(r"\bleisure\b|nightlife|diners|going out|entertainment", re.I),
     "aud_leisure"),
    (re.compile(r"retail staff|shop staff|shop worker|service staff", re.I),
     "aud_retail_service"),
    (re.compile(r"\bweekend\b", re.I), "weekend"),
    (re.compile(r"\bweekday\b", re.I), "weekday"),
    (re.compile(r"both weekend and weekday|split.*weekend|weekend.*split|"
                r"separate.*weekend", re.I), "split"),
    (re.compile(r"frequency|more slots|same people more often|depth", re.I), "frequency"),
    (re.compile(r"broader reach|more screens|wider coverage|breadth|more people", re.I),
     "reach"),
    (re.compile(r"late[- ]night|after midnight|night shift", re.I), "blocks_night"),
    (re.compile(r"\blunch\b|midday|noon", re.I), "blocks_midday"),
    (re.compile(r"morning|am peak|commute peak", re.I), "blocks_morning"),
    (re.compile(r"evening|pm peak|after work", re.I), "blocks_evening"),
    (re.compile(r"drop value|no value[- ]tier|avoid value|premium only", re.I),
     "tier_no_value"),
]

#: tag -> (real D1 segment, human phrase). The segment is what D2 actually
#: scores on; the phrase is what the rep and the proposal read.
_AUDIENCE_SEGMENT = {
    "aud_student": ("student", "students"),
    "aud_commuter_professional": ("commuter_professional",
                                  "commuting professionals"),
    "aud_shopper": ("shopper", "shoppers"),
    "aud_traveller": ("traveller", "travellers"),
    "aud_resident": ("resident", "residents"),
    "aud_blue_collar": ("blue_collar", "blue-collar workers"),
    "aud_leisure": ("leisure", "leisure visitors"),
    "aud_retail_service": ("retail_service", "retail and service staff"),
}

#: "we want residents, not tourists" must not ADD travellers. Checked against
#: the ~26 characters immediately before a segment cue; a negation there means
#: the rep is ruling that segment out, so the cue is skipped rather than
#: inverted -- the router has no way to express "exclude this segment", and
#: doing the opposite of what was asked is worse than doing nothing.
_NEGATION_BEFORE = re.compile(
    r"\b(?:not|no|non|never|without|exclude|excluding|excludes|drop|dropping|"
    r"avoid|avoiding|remove|removing|minus|except|besides|"
    r"rather than|instead of|less)\b[\s,;-]*$", re.I)


def _is_negated(text: str, at: int) -> bool:
    return bool(_NEGATION_BEFORE.search(text[max(0, at - 26):at]))


#: Audience words a rep will reasonably type that this dataset has no column
#: for. Silently dropping them is what produced a bare "Nothing actionable was
#: found"; they get declined by name instead.
_UNSUPPORTED_AUDIENCE = re.compile(
    r"\bchild(?:ren)?\b|\bkids?\b|\bfamil(?:y|ies)\b|\bteen(?:ager)?s?\b|"
    r"\bparents?\b|\bmums?\b|\bmoms?\b|\bdads?\b|\bseniors?\b|"
    r"\bpensioners?\b|\belderly\b|\bretirees?\b|\bover[- ]?60s?\b|"
    r"\bbabies\b|\btoddlers?\b|\bschoolchildren\b", re.I)

# --------------------------------------------------------------------------- #
# numeric extraction -- budget and duration
# --------------------------------------------------------------------------- #
#
# `budget` and `duration_days` were declared routable from the start but no rule
# ever emitted one, so "make the best in $15K" routed to nothing at all whenever
# the LLM path was unavailable -- which is the default, since the app ships
# without credentials. Two guards keep this honest:
#
#   * an amount only counts as money if it carries a `$`, a k/m suffix, or is
#     >= 1000. That is what stops "in 15 days" and "add 3 screens" from being
#     read as budgets.
#   * a bare price COMPLAINT never moves the budget -- that is a posture change
#     within D3's band. Only an explicit ceiling ("budget", "cap", "no more
#     than", "won't go above") retargets the solve.

_AMOUNT = r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*([kKmM])?"

_BUDGET_CEILING = re.compile(
    r"(?:budget|spend|spending|cap(?:ped)?|max(?:imum)?|"
    r"no more than|not more than|won'?t go (?:above|over|past)|"
    r"can'?t go (?:above|over|past)|keep it (?:to|under|below)|"
    r"stay (?:under|below|within)|under|below|within|up to|"
    r"best (?:in|with|for|of)|work (?:in|with|to)|"
    r"make the best (?:in|of|with))"
    # Chained filler, so "budget is only 8000" and "spend of about 12k" land.
    r"\s*(?:(?:a|an|of|to|at|is|are|was|around|about|only|just|roughly|"
    r"approximately|now|maybe|say)\s+)*" + _AMOUNT, re.I)

_DURATION = re.compile(r"(\d+)\s*(?:-\s*)?(day|days|week|weeks)\b", re.I)


def _as_money(digits: str, suffix: str) -> float | None:
    """A number is only a budget if it looks like one. Returns None if not."""
    try:
        v = float(digits.replace(",", ""))
    except ValueError:
        return None
    if suffix and suffix.lower() == "k":
        v *= 1_000
    elif suffix and suffix.lower() == "m":
        v *= 1_000_000
    elif v < 1_000:
        # No $-scale cue and too small to be a campaign budget.
        return None
    return v if v > 0 else None

_BLOCKS = {
    "blocks_night": [1, 6],
    "blocks_morning": [2],
    "blocks_midday": [3],
    "blocks_evening": [5],
}

_SCREEN_ID = re.compile(r"\b((?:LH|ACS|DAT)-SCR-\d{4,})\b", re.I)


def _route_with_rules(items: Sequence[dict], brief,
                      known_screen_ids: Optional[set]) -> Actions:
    raw: Dict[str, Any] = {"brief_edits": {}, "drop_screen_ids": [],
                           "loss_reasons": [], "rationale": []}
    audience_add: List[str] = []
    segments_add: List[str] = []
    blocks_add: List[int] = []
    notices: List[str] = []

    for it in items:
        mode = it.get("mode")
        payload = json.loads(it["payload_json"]) if it.get("payload_json") else {}
        text = it.get("raw_text") or ""

        # Explicit, structured feedback needs no interpretation.
        if mode == "drop_screen":
            raw["drop_screen_ids"].extend(payload.get("screen_ids", []))
            continue
        if mode == "param":
            for k, v in (payload.get("brief_edits") or {}).items():
                raw["brief_edits"][k] = v
            if payload.get("day_type"):
                raw["day_type"] = payload["day_type"]
            if payload.get("want_split") is not None:
                raw["want_split"] = payload["want_split"]
            continue
        if mode == "price_override":
            raw["price_posture"] = payload.get("posture", "hold")
            continue
        if mode == "loss_reason":
            lr = it.get("loss_reason")
            if lr:
                raw["loss_reasons"].append(lr)
                if lr in config.PRICE_LOSS_REASONS:
                    raw["price_posture"] = "toward_floor"
                if lr == "targeting_mismatch":
                    raw["rationale"].append(
                        "targeting_mismatch recorded -- audience text needs a rep edit "
                        "to be actionable")
                if lr == "inventory_unavailable":
                    raw["rationale"].append(
                        "inventory_unavailable recorded -- drop the affected screens "
                        "to act on it")
            if text:
                mode = "nl"      # fall through and also read the note
            else:
                continue

        if mode != "nl" or not text:
            continue

        for sid in _SCREEN_ID.findall(text):
            raw["drop_screen_ids"].append(sid.upper())

        for pat, tag in _RULES:
            if not pat.search(text):
                continue
            if tag == "price_down":
                raw["price_posture"] = "toward_floor"
                raw["loss_reasons"].append("price_too_high")
            elif tag == "price_up":
                raw["price_posture"] = "toward_cap"
            elif tag in _AUDIENCE_SEGMENT:
                # Only count a segment cue the rep did not negate.
                if all(_is_negated(text, m.start()) for m in pat.finditer(text)):
                    continue
                seg, phrase = _AUDIENCE_SEGMENT[tag]
                segments_add.append(seg)
                audience_add.append(phrase)
            elif tag == "weekend":
                raw["day_type"] = "weekend"
            elif tag == "weekday":
                raw["day_type"] = "weekday"
            elif tag == "split":
                raw["want_split"] = True
                raw["brief_edits"]["day_type_weighting"] = "report_split"
            elif tag == "frequency":
                raw["brief_edits"]["optimise_for"] = "frequency"
            elif tag == "reach":
                raw["brief_edits"]["optimise_for"] = "reach"
            elif tag in _BLOCKS:
                blocks_add.extend(_BLOCKS[tag])
            elif tag == "tier_no_value":
                raw["brief_edits"]["exclude_market_tier"] = ["value"]

        # ---- an explicit budget ceiling retargets the solve --------------- #
        for m in _BUDGET_CEILING.finditer(text):
            amount = _as_money(m.group(1), m.group(2) or "")
            if amount is not None:
                raw["brief_edits"]["budget"] = amount
                break

        # ---- an explicit run length ------------------------------------- #
        for m in _DURATION.finditer(text):
            n = int(m.group(1))
            days = n * 7 if m.group(2).lower().startswith("week") else n
            if 0 < days <= 365:
                raw["brief_edits"]["duration_days"] = days
                break

        # ---- audience words this dataset cannot express ------------------ #
        unsupported = sorted({w.lower() for w in _UNSUPPORTED_AUDIENCE.findall(text)}
                             or {m.group(0).lower()
                                 for m in _UNSUPPORTED_AUDIENCE.finditer(text)})
        if unsupported:
            notices.append(
                "audience_segments: this dataset has no "
                + ", ".join(f"'{w}'" for w in unsupported)
                + " segment. audience_profiles.csv records only "
                + ", ".join(config.AUDIENCE_SEGMENTS)
                + " -- the plan was NOT changed to target it, rather than "
                  "implying a reach figure the data cannot support.")

        raw["rationale"].append(f'rule-matched: "{text[:110]}"')

    if segments_add:
        # This is the edit that actually moves D2's audience_fit. It is merged
        # with whatever the brief already targets rather than replacing it --
        # "add children too" is additive language, and so is "add students".
        merged = sorted(set(list(getattr(brief, "audience_segments", []) or [])
                            + segments_add))
        raw["brief_edits"]["audience_segments"] = merged
    if audience_add:
        existing = (getattr(brief, "audience_text", "") or "").strip()
        additions = [a for a in dict.fromkeys(audience_add)
                     if a.lower() not in existing.lower()]
        if additions:
            joined = ", ".join(additions)
            raw["brief_edits"]["audience_text"] = (
                f"{existing}; also {joined}" if existing else joined)
    if blocks_add:
        merged = sorted(set(list(getattr(brief, "preferred_blocks", []) or [])
                            + blocks_add))
        raw["brief_edits"]["preferred_blocks"] = merged

    a = _validate(raw, known_screen_ids)
    a.routed_by = "rules"
    a.rejected.extend(notices)
    return a


# --------------------------------------------------------------------------- #
# LLM router -- one call for the whole batch
# --------------------------------------------------------------------------- #

_ROUTER_SYSTEM = """You convert a media sales rep's feedback into a strict JSON change set for a campaign planning system. You do not plan campaigns and you never invent numbers.

Return ONE JSON object, no prose, with only these keys:
{
  "brief_edits":      object,   // only these keys are allowed: budget (number), duration_days (integer), city_id ("LH"|"ACS"|"DAT"), objective (string), audience_text (string), audience_segments (array from the ONLY eight segments this dataset records: blue_collar, commuter_professional, leisure, resident, retail_service, shopper, student, traveller), preferred_blocks (array of ints 1-6), optimise_for ("reach"|"frequency"), day_type_weighting (""|"weekend_heavy"|"report_split"), exclude_market_tier (array), prefer_market_tier (array), radius_only (bool), radius_anchor_hint (string)
  "drop_screen_ids":  array of screen ids the rep explicitly rejected,
  "day_type":         "weekday"|"weekend"|null,
  "price_posture":    "toward_floor"|"hold"|"toward_cap",
  "want_split":       true|false|null,
  "loss_reasons":     array from this fixed vocabulary only: price_too_high, budget_mismatch, went_with_competitor, targeting_mismatch, geography_mismatch, inventory_unavailable, timing_conflict, creative_rejected, client_went_quiet, no_decision,
  "rationale":        array of short strings, one per change, saying WHY
}

Rules you must follow:
- Omit any key you have no evidence for. Never guess a budget or a city.
- Audience feedback belongs in audience_segments, not only in audience_text: the scorer consults audience_text through a keyword table that matches almost nothing, so a text-only edit changes no number.
- If the rep names an audience this dataset has no segment for (children, kids, families, teenagers, parents, seniors, over-60s), emit NO audience_segments for it and say so in rationale. Do not substitute a near-miss segment such as student for children.
- A segment the rep rules out ("residents, not tourists") must not be added. There is no way to express an exclusion here, so omit it.
- Time blocks are 4-hour: 1=00-04, 2=04-08, 3=08-12, 4=12-16, 5=16-20, 6=20-24.
- "reach the same people more often" => optimise_for "frequency". "reach more people" => "reach".
- A complaint that the price is too high => price_posture "toward_floor" and loss_reason "price_too_high". Never change the budget because of a price complaint.
- To ADD an audience segment, return the FULL new audience_text including what was already there.
- If the rep only asked a question or gave no actionable instruction, return empty objects/arrays."""


def _route_with_llm(items: Sequence[dict], brief, known_screen_ids: Optional[set],
                    context: str) -> Tuple[Optional[Actions], str]:
    lines = []
    for it in items:
        bits = [f"- [{it.get('screen')}/{it.get('mode')}]"]
        if it.get("raw_text"):
            bits.append(f'note: "{it["raw_text"]}"')
        if it.get("loss_reason"):
            bits.append(f"reason: {it['loss_reason']}")
        if it.get("payload_json"):
            bits.append(f"structured: {it['payload_json']}")
        lines.append(" ".join(bits))

    user = (f"CURRENT BRIEF\n{context}\n\n"
            f"REP FEEDBACK ({len(items)} item(s), collected across the walkthrough)\n"
            + "\n".join(lines)
            + "\n\nReturn the JSON change set.")

    res = llm.chat(purpose="feedback_router", system=_ROUTER_SYSTEM, user=user,
                   max_completion_tokens=config.LLM_ROUTER_MAX_COMPLETION_TOKENS)
    if not res.ok:
        return None, f"{res.source}: {res.reason}"

    obj, err = llm.parse_json_object(res.text)
    if obj is None:
        return None, f"router returned unparseable JSON ({err})"

    a = _validate(obj, known_screen_ids)
    a.routed_by = "llm" if res.source == "llm" else "llm-cache"
    return a, ""


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def brief_context(brief, day_type: str) -> str:
    return json.dumps({
        "name": brief.name, "city_id": brief.city_id, "budget": brief.budget,
        "duration_days": brief.duration_days, "objective": brief.objective,
        "audience_text": brief.audience_text,
        "preferred_blocks": list(brief.preferred_blocks or []),
        "optimise_for": brief.optimise_for,
        "day_type_weighting": brief.day_type_weighting,
        "day_type": day_type,
        "exclude_market_tier": list(brief.exclude_market_tier or []),
        "prefer_market_tier": list(brief.prefer_market_tier or []),
        "radius_only": brief.radius_only,
    }, indent=None)


def route(items: Sequence[dict], *, brief, day_type: str,
          known_screen_ids: Optional[set] = None,
          use_llm: bool = True) -> Actions:
    """Turn a batch of queued feedback into one validated change set.

    Tries the LLM once for the whole batch, then falls back to rules. Rules also
    run when the LLM succeeds, and anything the rules found that the model missed
    is merged in -- an explicit screen id or an unambiguous price objection
    should not be lost to a vague completion.
    """
    if not items:
        return Actions(router_note="no feedback queued")

    rules = _route_with_rules(items, brief, known_screen_ids)
    if not use_llm:
        rules.router_note = "LLM routing disabled -- deterministic rules only"
        return rules

    llm_actions, why = _route_with_llm(items, brief, known_screen_ids,
                                       brief_context(brief, day_type))
    if llm_actions is None:
        rules.router_note = f"fell back to deterministic rules ({why})"
        return rules

    # Merge: the model wins on wording, the rules win on anything unambiguous
    # they positively detected.
    for sid in rules.drop_screen_ids:
        if sid not in llm_actions.drop_screen_ids:
            llm_actions.drop_screen_ids.append(sid)
    llm_actions.drop_screen_ids = sorted(set(llm_actions.drop_screen_ids))
    if rules.price_posture != "hold" and llm_actions.price_posture == "hold":
        llm_actions.price_posture = rules.price_posture
        llm_actions.rationale.append(
            f"price posture {rules.price_posture} added by rule match")
    for lr in rules.loss_reasons:
        if lr not in llm_actions.loss_reasons:
            llm_actions.loss_reasons.append(lr)
    if rules.want_split and llm_actions.want_split is None:
        llm_actions.want_split = True

    llm_actions.router_note = "routed by the model, cross-checked against rules"
    return llm_actions


# --------------------------------------------------------------------------- #
# application
# --------------------------------------------------------------------------- #

def apply_to_brief(brief, actions: Actions):
    """Return (new_brief, changed_fields). The original is never mutated, so a
    revision can always be compared against the one before it."""
    import copy
    new = copy.deepcopy(brief)
    changed: Dict[str, Tuple[Any, Any]] = {}
    for k, v in actions.brief_edits.items():
        if not hasattr(new, k):
            continue
        old = getattr(new, k)
        if old != v:
            setattr(new, k, v)
            changed[k] = (old, v)
    return new, changed


def quote_at_posture(quote, posture: str) -> Tuple[float, str]:
    """Where to quote inside D3's band, given the rep's posture.

    Deliberately conservative: a price objection moves the quote to the MIDPOINT
    of floor..target, not to the floor. Going straight to the floor on one
    comment throws away the whole negotiating range on the first push-back, and
    D3's floor is a real percentile of comparable bookings, not a discount to
    hand out.
    """
    floor, target, cap = float(quote.floor), float(quote.target), float(quote.cap)
    rec = float(quote.recommended)
    if posture == "toward_floor":
        v = (floor + target) / 2.0
        return v, (f"midpoint of floor {floor:,.2f} and target {target:,.2f} -- "
                   f"a price objection was recorded")
    if posture == "toward_cap":
        v = (target + cap) / 2.0
        return v, (f"midpoint of target {target:,.2f} and cap {cap:,.2f} -- "
                   f"headroom was requested")
    return rec, "D3's recommended price, unchanged"

"""Session state, step gating, and the single place a run is executed.

ONE RUN PER REVISION. The rep confirms the brief on Interpretation; D5 runs
once; screens 03-06 are views onto that one result. Feedback queues up as they
review and is applied by exactly one further run, triggered from the
Recommendation screen. That keeps the LLM router to one call per revision and the
solver to one solve per revision, which is what makes the app affordable against
a $5 key.
"""
from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import streamlit as st

from app import config
from app.db import store
from app.services import engines, feedback as fb, learning, pipeline

# Fields the Interpretation screen edits, in render order, with their kind.
BRIEF_FIELDS = [
    ("name", "text", "Campaign / client name"),
    ("city_id", "city", "City"),
    ("budget", "money", "Budget (USD)"),
    ("duration_days", "int", "Duration (days)"),
    ("objective", "text", "Stated objective"),
    ("audience_text", "textarea", "Target audience"),
    ("optimise_for", "objective", "Optimise for"),
    ("preferred_blocks", "blocks", "Preferred time blocks"),
    ("day_type_weighting", "dtw", "Day-type weighting"),
    ("must_have", "pairs", "Requested inventory categories"),
    ("exclude", "pairs", "Excluded inventory categories"),
    ("prefer_market_tier", "tiers", "Preferred market tier"),
    ("exclude_market_tier", "tiers", "Excluded market tier"),
    ("radius_only", "bool", "Restrict to a walking radius"),
    ("radius_anchor_hint", "text", "Radius anchor"),
]

DEFAULTS: Dict[str, Any] = {
    "mode": config.MODE_CLIENT,
    "step": "intake",
    "campaign_id": None,
    "document_id": None,
    "brief_text": "",
    "source_mode": None,
    "source_label": None,
    "brief": None,
    "provenance": {},
    "win_start": None,
    "win_end": None,
    "day_type": "weekday",
    "allow_unevidenced": False,
    "result": None,
    "run_id": None,
    "revision": 0,
    "rfp_md": "",
    "rfp_template": "",
    "rfp_asks": [],
    "rfp_id": None,
    "quoted": None,
    "quoted_totals": None,
    "learning_enabled": config.LEARNING_ENABLED_DEFAULT,
    "llm_enabled": config.LLM_ENABLED_DEFAULT,
    "force_greedy": config.FORCE_GREEDY_DEFAULT,
    "solve_limit": config.SOLVE_TIME_LIMIT_S,
    "dropped_screens": [],
    "run_error": "",
    "last_actions": None,
    "adjustments": None,
    "history": [],
    "rep_name": "",
    "rep_email": "",
    "email_draft": None,
    "email_edited": "",
    "email_subject": "",
}


def init() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.win_start is None:
        st.session_state.win_start = date(2026, 9, 1)
    if st.session_state.win_end is None:
        st.session_state.win_end = date(2026, 9, 30)


def reset_campaign() -> None:
    mode = st.session_state.get("mode", config.MODE_CLIENT)
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.mode = mode           # the mode is a preference, not campaign state
    st.session_state.win_start = date(2026, 9, 1)
    st.session_state.win_end = date(2026, 9, 30)


# --------------------------------------------------------------------------- #
# mode
# --------------------------------------------------------------------------- #

def mode() -> str:
    return st.session_state.get("mode", config.MODE_CLIENT)


def is_dev() -> bool:
    return mode() == config.MODE_DEVELOPER


def visible_steps() -> List[str]:
    return config.step_keys_for(mode())


def set_mode(new_mode: str) -> None:
    """Switching mode must never leave the rep on a screen that no longer
    exists -- client mode has no Audience or Screen-match step."""
    if new_mode == st.session_state.get("mode"):
        return
    st.session_state.mode = new_mode
    if st.session_state.step not in config.step_keys_for(new_mode):
        st.session_state.step = ("package" if st.session_state.result is not None
                                 else "interpret" if st.session_state.brief_text
                                 else "intake")


# --------------------------------------------------------------------------- #
# gating
# --------------------------------------------------------------------------- #

def unlocked_steps() -> List[str]:
    out = ["intake"]
    if st.session_state.brief_text:
        out.append("interpret")
    if st.session_state.result is not None:
        out += ["audience", "relevance", "package", "recommend"]
    return [s for s in out if s in visible_steps()]


def next_step(after: str) -> str:
    """The following step IN THE CURRENT MODE, so 'Continue' from Review goes
    straight to Plan & pricing for a rep and via Audience for a developer."""
    keys = visible_steps()
    if after not in keys:
        return keys[-1]
    i = keys.index(after)
    return keys[min(i + 1, len(keys) - 1)]


def goto(step: str) -> None:
    st.session_state.step = step
    st.rerun()


def all_cities() -> bool:
    """No city set: the plan runs across every market.

    This used to be a hard block. It is not any more -- the brief may genuinely
    not name a city (three of the six samples do not), and the right answer is
    to infer it where possible and otherwise let the optimiser choose the best
    locations across all markets. Slower, but a real answer instead of a refusal.
    """
    b = st.session_state.brief
    return b is not None and not (b.city_id or "").strip()


def blocking_issues() -> List[str]:
    out = []
    b = st.session_state.brief
    if b is None:
        return ["No brief has been interpreted yet."]
    if not b.budget or b.budget <= 0:
        out.append("**Budget is missing or zero.** Every downstream step is budget "
                   "constrained, so the plan would refuse at the affordability stage.")
    if not b.duration_days or b.duration_days <= 0:
        out.append("**Duration is missing or zero.** Line costs are price x slots x "
                   "days, so a zero-day flight prices at zero.")
    if st.session_state.win_start and st.session_state.win_end:
        if st.session_state.win_end < st.session_state.win_start:
            out.append("**Flight window ends before it starts.**")
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def execute_run(*, origin: str, origin_note: str = "",
                actions: Optional[fb.Actions] = None,
                applied_feedback_ids: Optional[List[int]] = None) -> bool:
    """Persist the brief version, run D5, persist the run. Returns success."""
    b = st.session_state.brief
    cid = st.session_state.campaign_id
    st.session_state.run_error = ""

    bv = store.add_brief_version(
        campaign_id=cid, brief_json=dataclasses.asdict(b),
        provenance=st.session_state.provenance,
        campaign_start=str(st.session_state.win_start),
        campaign_end=str(st.session_state.win_end),
        day_type=st.session_state.day_type,
        allow_unevidenced=bool(st.session_state.allow_unevidenced),
        origin=origin, origin_note=origin_note)

    config.SOLVE_TIME_LIMIT_S = float(st.session_state.solve_limit)

    status = st.status("Starting the agent…", expanded=True)
    steps: List[str] = []

    def progress(msg: str) -> None:
        steps.append(msg)
        status.update(label=msg)
        status.write(f"• {msg}")

    try:
        wstate = engines.warmup_state()
        if not wstate.ready:
            progress("waiting for the engines to finish loading "
                     f"({wstate.stage}: {wstate.detail or '...'})")
        orch = engines.get_orchestrator()

        result = pipeline.run(
            orch, brief=b, brief_text=st.session_state.brief_text,
            campaign_start=str(st.session_state.win_start),
            campaign_end=str(st.session_state.win_end),
            day_type=st.session_state.day_type,
            allow_unevidenced=bool(st.session_state.allow_unevidenced),
            force_greedy=bool(st.session_state.force_greedy),
            excluded_screen_ids=st.session_state.dropped_screens,
            want_split=(actions.want_split if actions else None),
            progress=progress)
    except Exception as exc:  # noqa: BLE001
        import traceback
        status.update(label="The run failed", state="error")
        st.session_state.run_error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        store.log(campaign_id=cid, actor="system", event="run_failed",
                  detail=str(exc)[:400])
        return False

    # ---- cross-session learning, applied as a display/quote layer ---- #
    progress("applying learned adjustments from earlier campaigns")
    adj = learning.load(campaign_id=cid, scored=result.scored,
                        enabled=bool(st.session_state.learning_enabled))
    quoted, totals = (None, None)
    if len(result.lines):
        quoted, totals = learning.apply_to_quotes(
            result.lines, result.quotes, adj, int(b.duration_days or 0))
        if actions and actions.price_posture != "hold":
            quoted, totals = _apply_posture(result, quoted, totals,
                                             actions.price_posture,
                                             int(b.duration_days or 0))

    met = dict(result.metrics)
    run = store.add_run(
        campaign_id=cid, brief_version_id=bv["id"],
        plan_json=pipeline.plan_to_dict(result), metrics=met,
        solver_used=met.get("solver_used", "?"), presentable=result.presentable,
        spend=met.get("spend", 0.0), reach=met.get("reach_per_day", 0.0),
        impressions=met.get("impressions_total", 0.0),
        n_lines=met.get("n_lines", 0), n_screens=met.get("n_screens", 0),
        duration_ms=result.duration_ms,
        refusals=[dataclasses.asdict(r) for r in result.plan.refusals],
        warnings=list(result.plan.warnings),
        learning_applied=bool(st.session_state.learning_enabled and adj.n_applied),
        learning_note=adj.summary())

    st.session_state.result = result
    st.session_state.run_id = run["id"]
    st.session_state.revision = run["revision"]
    st.session_state.adjustments = adj
    st.session_state.quoted = quoted
    st.session_state.quoted_totals = totals
    st.session_state.rfp_md = ""
    st.session_state.rfp_id = None

    if applied_feedback_ids:
        store.mark_feedback_applied(
            applied_feedback_ids, run_id=run["id"],
            routed_by=(actions.routed_by if actions else "rules"))

    store.log(campaign_id=cid, run_id=run["id"], actor="agent",
              event=f"run_r{run['revision']}",
              detail=(f"{origin}: {met.get('n_lines')} lines, "
                      f"spend {met.get('spend', 0):,.0f}, "
                      f"reach {met.get('reach_per_day', 0):,.0f}, "
                      f"solver {met.get('solver_used')}, "
                      f"{result.duration_ms}ms. {origin_note}"))

    status.update(label=f"Revision {run['revision']} ready "
                        f"({result.duration_ms / 1000:.1f}s)", state="complete",
                  expanded=False)
    st.session_state.history = store.list_runs(cid)
    return True


def _apply_posture(result, quoted, totals, posture: str, days: int):
    """Move the quoted price within D3's band per the rep's price posture."""
    import pandas as pd
    if quoted is None or not len(quoted):
        return quoted, totals
    rows = []
    for r in quoted.itertuples():
        q = result.quotes.get((str(r.screen_id), int(r.time_block_id)))
        price, why = (float(r.quoted_price), "")
        if q is not None:
            price, why = fb.quote_at_posture(q, posture)
            # Learning still applies on top, then re-clamp into the band.
            price *= float(r.learned_multiplier)
            price = max(float(q.floor), min(float(q.cap), price))
        d = r._asdict()
        d["quoted_price"] = price
        d["quoted_cost"] = price * float(r.units)
        d["reason"] = "; ".join(x for x in [why, str(r.reason)] if x)
        rows.append({k: v for k, v in d.items() if k != "Index"})
    df = pd.DataFrame(rows)
    d3_total = float(df["d3_cost"].sum())
    q_total = float(df["quoted_cost"].sum())
    return df, {"d3_total": d3_total, "quoted_total": q_total,
                "delta": q_total - d3_total,
                "delta_pct": (q_total / d3_total - 1.0) if d3_total > 0 else 0.0,
                "n_lines_moved": int((df["quoted_price"] != df["d3_price"]).sum())}


def apply_feedback_and_rerun() -> bool:
    """The ONE place feedback is acted on: route the whole queue, mutate the
    brief, record what it taught us, then run once."""
    cid = st.session_state.campaign_id
    items = store.queued_feedback(cid)
    if not items:
        return False

    b = st.session_state.brief
    result = st.session_state.result
    known = (set(result.scored["screen_id"].astype(str))
             if result is not None and result.scored is not None
             and len(result.scored) else None)

    with st.spinner("Routing your feedback into a change set…"):
        actions = fb.route(items, brief=b, day_type=st.session_state.day_type,
                           known_screen_ids=known,
                           use_llm=bool(st.session_state.llm_enabled))

    new_brief, changed = fb.apply_to_brief(b, actions)
    st.session_state.brief = new_brief
    for k in changed:
        st.session_state.provenance[k] = "feedback_applied"
    if actions.day_type:
        st.session_state.day_type = actions.day_type
    if actions.drop_screen_ids:
        st.session_state.dropped_screens = sorted(
            set(st.session_state.dropped_screens) | set(actions.drop_screen_ids))
    st.session_state.last_actions = actions

    # Teach the cross-session memory, BEFORE the run so the log order reads true.
    if actions.loss_reasons:
        affected = (actions.drop_screen_ids
                    or (list(result.lines["screen_id"].astype(str))
                        if result is not None and len(result.lines) else []))
        learning.record(
            campaign_id=cid, loss_reasons=actions.loss_reasons,
            affected_screen_ids=affected,
            attrs=learning.screen_attributes(result.scored if result else None),
            source_feedback_id=items[0]["id"],
            note=f"rev {st.session_state.revision} feedback")

    store.log(campaign_id=cid, run_id=st.session_state.run_id, actor="agent",
              event="feedback_routed",
              detail=f"routed_by={actions.routed_by}; "
                     f"{'; '.join(actions.summary_lines()) or 'no actionable change'}; "
                     f"rejected={actions.rejected}")

    ok = execute_run(origin="feedback_rerun",
                     origin_note="; ".join(actions.summary_lines()),
                     actions=actions,
                     applied_feedback_ids=[i["id"] for i in items])
    return ok

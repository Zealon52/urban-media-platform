"""Screen 02 — Interpretation.

The most important screen in the app. The agent shows its reading of the brief
field by field, labels what it read verbatim against what it inferred, asks about
what it genuinely cannot resolve, and lets the rep correct all of it. Every edit
here is free: nothing has been solved yet.

Confirming from this screen is what triggers the single D5 run.
"""
from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import streamlit as st

from app import config
from app.db import store
from app.services import engines, interpret
from app.ui import components as C
from app.ui import state as S

_PAIR_OPTIONS = [(st_, pos) for st_ in ("", *config.SCREEN_TYPES)
                 for pos in ("", *config.POSITIONS)]


def _pair_label(p) -> str:
    st_, pos = p
    if not st_ and not pos:
        return "(any)"
    return f"{st_ or 'any type'} / {pos or 'any position'}"


def _ensure_parsed() -> None:
    if st.session_state.brief is not None:
        return
    lt = engines.light_tables()
    with st.spinner("Reading the brief…"):
        brief, prov, questions = interpret.parse(
            st.session_state.brief_text, lt["cities"], lt["dim_slot"],
            screens_df=lt.get("screens"))
    st.session_state.brief = brief
    st.session_state.provenance = prov
    st.session_state["_questions"] = questions
    if brief.duration_days and brief.duration_days > 0:
        st.session_state.win_end = st.session_state.win_start + timedelta(
            days=int(brief.duration_days) - 1)


def render() -> None:
    if not st.session_state.brief_text:
        st.info("Start from the Intake step.")
        return
    _ensure_parsed()
    b = st.session_state.brief
    prov = st.session_state.provenance
    dev = S.is_dev()

    C.page_header(
        eyebrow="Step 2 of 4" if not dev else "Step 2 of 6 · Parser",
        title="Check what we understood",
        lede="Green was read straight from the brief. Blue was worked out from it. "
             "Red needs you. Fix anything that's wrong — nothing has been priced "
             "yet, so changes here cost nothing.")

    legend = " ".join([C.badge("read"), C.badge("inferred"), C.badge("rep_edited"),
                       C.badge("feedback_applied"), C.badge("needs_you")])
    st.markdown(legend, unsafe_allow_html=True)
    st.markdown("")

    def mark(field: str) -> None:
        if prov.get(field) not in ("rep_edited", "feedback_applied"):
            prov[field] = "rep_edited"

    def lbl(field: str, text: str) -> str:
        return f"{text} &nbsp;{C.badge(prov.get(field, 'derived'))}"

    left, right = st.columns([1, 1], gap="large")

    # ------------------------------------------------------------- left col #
    with left:
        st.markdown(lbl("name", "**Campaign / client name**"), unsafe_allow_html=True)
        v = st.text_input("name", b.name, label_visibility="collapsed")
        if v != b.name:
            b.name = v
            mark("name")

        st.markdown(lbl("city_id", "**City**"), unsafe_allow_html=True)
        opts = [""] + list(config.CITY_CHOICES)
        cur = b.city_id if b.city_id in config.CITY_CHOICES else ""
        v = st.selectbox(
            "city", opts, index=opts.index(cur), label_visibility="collapsed",
            format_func=lambda c: config.CITY_CHOICES.get(c, "— not set —"))
        if v != b.city_id:
            b.city_id = v
            mark("city_id")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(lbl("budget", "**Budget (USD)**"), unsafe_allow_html=True)
            v = st.number_input("budget", min_value=0.0, value=float(b.budget or 0),
                                step=1000.0, format="%.0f",
                                label_visibility="collapsed")
            if v != b.budget:
                b.budget = float(v)
                mark("budget")
        with c2:
            st.markdown(lbl("duration_days", "**Duration (days)**"),
                        unsafe_allow_html=True)
            v = st.number_input("days", min_value=0, value=int(b.duration_days or 0),
                                step=1, label_visibility="collapsed")
            if v != b.duration_days:
                b.duration_days = int(v)
                mark("duration_days")
                st.session_state.win_end = st.session_state.win_start + timedelta(
                    days=max(int(v) - 1, 0))

        st.markdown(lbl("objective", "**Stated objective**"), unsafe_allow_html=True)
        v = st.text_input("obj", b.objective, label_visibility="collapsed")
        if v != b.objective:
            b.objective = v
            mark("objective")

        st.markdown(lbl("audience_text", "**Target audience**"), unsafe_allow_html=True)
        v = st.text_area("aud", b.audience_text, height=80,
                         label_visibility="collapsed")
        if v != b.audience_text:
            b.audience_text = v
            mark("audience_text")

        st.markdown("**Flight window** &nbsp;" + C.badge("derived", "you set"),
                    unsafe_allow_html=True)
        st.caption("Real dates matter: pricing checks which events fall inside this "
                   "exact window and how passenger numbers are trending for it.")
        d1, d2 = st.columns(2)
        s = d1.date_input("start", st.session_state.win_start,
                          label_visibility="collapsed")
        e = d2.date_input("end", st.session_state.win_end,
                          label_visibility="collapsed")
        st.session_state.win_start, st.session_state.win_end = s, e
        span = (e - s).days + 1
        if b.duration_days and span != b.duration_days:
            st.caption(f":orange[Window spans {span} days but duration says "
                       f"{b.duration_days}. Line costs use the duration field "
                       f"({b.duration_days} days); the window drives event and "
                       f"ridership adjustments.]")

    # ------------------------------------------------------------ right col #
    with right:
        st.markdown(lbl("optimise_for", "**Optimise for**"), unsafe_allow_html=True)
        v = st.radio("optf", ["reach", "frequency"],
                     index=0 if b.optimise_for != "frequency" else 1,
                     horizontal=True, label_visibility="collapsed",
                     captions=["distinct people (breadth)",
                               "impressions (depth on fewer screens)"])
        if v != b.optimise_for:
            b.optimise_for = v
            mark("optimise_for")

        st.markdown(lbl("preferred_blocks", "**Preferred time blocks**"),
                    unsafe_allow_html=True)
        v = st.multiselect("blk", list(config.TIME_BLOCKS),
                           default=list(b.preferred_blocks or []),
                           format_func=lambda i: config.TIME_BLOCKS[i],
                           label_visibility="collapsed",
                           help="Empty means all six blocks are candidates.")
        if sorted(v) != sorted(b.preferred_blocks or []):
            b.preferred_blocks = sorted(v)
            mark("preferred_blocks")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Day type**", unsafe_allow_html=True)
            dt = st.radio("dt", ["weekday", "weekend"],
                          index=0 if st.session_state.day_type == "weekday" else 1,
                          horizontal=True, label_visibility="collapsed")
            st.session_state.day_type = dt
        with c2:
            st.markdown(lbl("day_type_weighting", "**Weighting**"),
                        unsafe_allow_html=True)
            dtw_opts = ["", "weekend_heavy", "report_split"]
            v = st.selectbox("dtw", dtw_opts,
                             index=dtw_opts.index(b.day_type_weighting or ""),
                             label_visibility="collapsed",
                             format_func=lambda x: x or "— none —",
                             help="report_split makes the proposal report weekday and "
                                  "weekend delivery separately, which brief 3 asks for "
                                  "explicitly. It costs one extra solve.")
            if v != b.day_type_weighting:
                b.day_type_weighting = v
                mark("day_type_weighting")

        st.markdown(lbl("must_have", "**Requested inventory categories**"),
                    unsafe_allow_html=True)
        st.caption("These improve a location's match score but never rule anything "
                   "out. Our keyword list can't cover every phrase a brief might "
                   "use, so filtering on it would silently drop inventory the "
                   "client actually asked for.")
        v = st.multiselect("mh", _PAIR_OPTIONS,
                           default=[tuple(p) for p in (b.must_have or [])],
                           format_func=_pair_label, label_visibility="collapsed")
        if [tuple(x) for x in v] != [tuple(p) for p in (b.must_have or [])]:
            b.must_have = [tuple(x) for x in v]
            mark("must_have")

        st.markdown(lbl("exclude", "**Excluded inventory categories**"),
                    unsafe_allow_html=True)
        v = st.multiselect("ex", _PAIR_OPTIONS,
                           default=[tuple(p) for p in (b.exclude or [])],
                           format_func=_pair_label, label_visibility="collapsed",
                           help="This IS a hard filter.")
        if [tuple(x) for x in v] != [tuple(p) for p in (b.exclude or [])]:
            b.exclude = [tuple(x) for x in v]
            mark("exclude")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(lbl("prefer_market_tier", "**Prefer tier**"),
                        unsafe_allow_html=True)
            v = st.multiselect("pt", list(config.MARKET_TIERS),
                               default=list(b.prefer_market_tier or []),
                               label_visibility="collapsed")
            if sorted(v) != sorted(b.prefer_market_tier or []):
                b.prefer_market_tier = sorted(v)
                mark("prefer_market_tier")
        with c2:
            st.markdown(lbl("exclude_market_tier", "**Exclude tier**"),
                        unsafe_allow_html=True)
            v = st.multiselect("et", list(config.MARKET_TIERS),
                               default=list(b.exclude_market_tier or []),
                               label_visibility="collapsed")
            if sorted(v) != sorted(b.exclude_market_tier or []):
                b.exclude_market_tier = sorted(v)
                mark("exclude_market_tier")

        st.markdown(lbl("radius_only", "**Walking-radius buy**"),
                    unsafe_allow_html=True)
        v = st.checkbox("Restrict to the anchor's zones only",
                        value=bool(b.radius_only))
        if v != b.radius_only:
            b.radius_only = v
            mark("radius_only")
        if b.radius_only:
            v = st.text_input("anchor", b.radius_anchor_hint,
                              label_visibility="collapsed",
                              placeholder="e.g. business-park food court")
            if v != b.radius_anchor_hint:
                b.radius_anchor_hint = v
                mark("radius_anchor_hint")
            st.caption("No coordinates exist in the 14 source tables, so 'walking "
                       "distance' resolves to zone membership around this anchor. The "
                       "proposal states that limitation rather than implying metres.")

    # -------------------------------------------------- clarifying questions #
    questions = st.session_state.get("_questions") or []
    if questions:
        C.section("Worth a look before you continue")
        for q in questions:
            C.why_box(q, tag="Check this")

    if S.all_cities():
        C.why_box(
            "No city is set, so the plan will look across <b>all three markets</b> "
            "and pick whichever locations deliver the most for the budget. That "
            "takes a little longer than a single-city plan. Set a city above if the "
            "campaign is only for one market.",
            tag="Planning across all markets")

    # ------------------------------------------------------- evidence + raw #
    with st.expander("Show me where you read that"):
        for f in ("name", "budget", "duration_days", "objective", "audience_text",
                  "exclude"):
            ev = interpret.field_evidence(st.session_state.brief_text, f)
            if ev:
                st.markdown(f"**{f}** — `{ev[:260]}`")
    with st.expander("Raw extracted brief text"):
        st.code(st.session_state.brief_text[:8000], language=None)
    with st.expander("Structured brief (what the engines receive)"):
        st.json(dataclasses.asdict(b))

    # ------------------------------------------------------------- run gate #
    st.markdown("---")
    issues = S.blocking_issues()
    if issues:
        st.error("Fix these before running:")
        for i in issues:
            st.markdown(f"- {i}")

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        go = st.button("Build the plan →", type="primary",
                       disabled=bool(issues), use_container_width=True)
    with c2:
        if st.button("Re-read the brief", use_container_width=True,
                     help="Discard your edits and re-parse from the document."):
            st.session_state.brief = None
            st.rerun()
    with c3:
        wm = engines.warmup_state()
        if wm.ready:
            st.caption(f":green[Engines ready] · solver `{wm.solver}` · "
                       f"loaded in {wm.elapsed:.0f}s")
        elif wm.failed:
            st.caption(f":red[Engine load failed] — {wm.error}")
        else:
            st.caption(f"Engines still loading ({wm.stage}) — the run will wait for "
                       f"them. This is why you could edit immediately.")

    if go:
        store.log(campaign_id=st.session_state.campaign_id, actor="rep",
                  event="brief_confirmed",
                  detail=f"city={b.city_id} budget={b.budget:,.0f} "
                         f"days={b.duration_days} blocks={b.preferred_blocks} "
                         f"objective={b.optimise_for}")
        if S.execute_run(origin="rep_edit" if any(
                v in ("rep_edited",) for v in prov.values()) else "parser",
                origin_note="initial run from the confirmed brief"):
            # Sales view has no Audience step -- go to whatever comes next in
            # THIS mode rather than a hardcoded screen that may not exist.
            S.goto(S.next_step("interpret"))
        else:
            st.error("The run failed.")
            with st.expander("Details"):
                st.code(st.session_state.run_error)

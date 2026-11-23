"""Screen 04 — Relevance (D2).

The ranked shortlist with a visible score breakdown. A total score alone invites
"why?", so the six real component weights are shown per row.

Dropping a screen QUEUES a change; it does not re-solve. The re-run happens once,
from the Recommendation step, with every queued item applied together.
"""
from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
import streamlit as st

from app import config
from app.db import store
from app.ui import components as C
from app.ui import state as S

#: D2's real weights (build_d2_relevance_scorer.py, score_screens).
WEIGHTS: Dict[str, float] = {
    "daypart_fit": 0.25,
    "category_fit": 0.20,
    "audience_fit": 0.20,
    "confidence_w": 0.15,
    "tier_fit": 0.10,
    "history_bonus": 0.10,
}

LABELS = {
    "daypart_fit": "Daypart fit",
    "category_fit": "Requested category",
    "audience_fit": "Audience match",
    "confidence_w": "Profile confidence",
    "tier_fit": "Market tier fit",
    "history_bonus": "Booking history",
}


def render() -> None:
    result = st.session_state.result
    if result is None:
        st.info("Run the agent from the Interpretation step first.")
        return

    scored = result.scored
    lines = result.lines
    adj = st.session_state.adjustments
    b = result.plan.brief
    p = result.plan

    st.subheader("Ranked candidates")

    if scored is None or not len(scored):
        st.warning("No locations matched this brief. See the reason below.")
        for r in p.refusals:
            C.refusal_box(r.stage, r.reason, r.remedy)
        return

    selected_ids = set(lines["screen_id"].astype(str)) if len(lines) else set()

    C.tiles([
        {"value": f"{len(scored):,}", "label": "Screens scored against this brief"},
        {"value": f"{p.funnel.after_capacity:,}",
         "label": "With free capacity in the window"},
        {"value": f"{p.funnel.shortlisted:,}", "label": "Shortlisted for the solver",
         "sub": "sized from the budget, not a fixed top-N"},
        {"value": f"{len(selected_ids)}", "label": "Bought by the optimiser",
         "tone": "good"},
    ])

    C.why_box(p.sizing.reason, tag="Shortlist size")

    with st.expander("The narrowing funnel — where candidates went"):
        rows = p.funnel.rows()
        fd = pd.DataFrame(rows, columns=["Stage", "Cells remaining"])
        st.dataframe(fd, hide_index=True, use_container_width=True)
        st.caption("Read outside-in. If a plan comes back empty, the FIRST stage that "
                   "hits zero is the real cause — 'no screens matched' and 'matched "
                   "thousands, none affordable' need opposite fixes.")
        if p.blocks.notes:
            st.markdown("**Daypart resolution**")
            for n in p.blocks.notes:
                st.markdown(f"- {n}")

    # ---------------------------------------------------------------- table #
    st.markdown("##### Shortlist")
    st.caption("Untick a screen to remove it from the candidate pool. That queues a "
               "change — it does not re-solve now. Everything you queue is applied in "
               "one re-run from the Recommendation step.")

    show_only_selected = st.checkbox(
        "Show only the screens the optimiser bought", value=False)

    df = scored.copy()
    if "relevance_score" in df.columns:
        df = df.sort_values("relevance_score", ascending=False)
    df["in_plan"] = df["screen_id"].astype(str).isin(selected_ids)
    if show_only_selected:
        df = df[df["in_plan"]]
    df = df.head(60)

    already_dropped = set(st.session_state.dropped_screens)
    queued_drops: set = set()
    for it in store.queued_feedback(st.session_state.campaign_id):
        if it["mode"] == "drop_screen" and it.get("payload_json"):
            try:
                queued_drops |= set(json.loads(it["payload_json"]).get("screen_ids", []))
            except (TypeError, ValueError):
                pass

    view = pd.DataFrame({
        "keep": [not (str(s) in already_dropped or str(s) in queued_drops)
                 for s in df["screen_id"]],
        "in plan": df["in_plan"].values,
        "screen_id": df["screen_id"].astype(str).values,
        "type": df.get("screen_type", pd.Series(["--"] * len(df))).values,
        "position": df.get("position", pd.Series(["--"] * len(df))).values,
        "zone": df.get("zone_id", pd.Series(["--"] * len(df))).values,
        "score": df["relevance_score"].astype(float).round(3).values
        if "relevance_score" in df.columns else [0.0] * len(df),
    })
    if adj is not None and adj.enabled:
        deltas, whys = [], []
        for s in view["screen_id"]:
            d, w = adj.score_delta(s)
            deltas.append(round(d, 3))
            whys.append("; ".join(w))
        view["learned"] = deltas
        view["learned why"] = whys
    view["leading reason"] = (df.get("reasons",
                                     pd.Series([""] * len(df))).astype(str).values)

    edited = st.data_editor(
        view, hide_index=True, use_container_width=True, height=420,
        disabled=[c for c in view.columns if c != "keep"],
        column_config={
            "keep": st.column_config.CheckboxColumn(
                "keep", help="Untick to queue this screen's removal", width="small"),
            "in plan": st.column_config.CheckboxColumn(
                "bought", help="Did the optimiser buy it", width="small"),
            "score": st.column_config.ProgressColumn(
                "relevance", min_value=0.0, max_value=1.0, format="%.3f"),
            "learned": st.column_config.NumberColumn(
                "learned", help="Adjustment from feedback on earlier campaigns",
                format="%+.3f"),
            "leading reason": st.column_config.TextColumn("why", width="large"),
        }, key="rel_editor")

    newly_unticked = sorted(
        {str(r["screen_id"]) for _, r in edited.iterrows() if not r["keep"]}
        - already_dropped - queued_drops)
    re_ticked = sorted(
        (already_dropped | queued_drops)
        & {str(r["screen_id"]) for _, r in edited.iterrows() if r["keep"]})

    c1, c2, c3 = st.columns([2, 2, 4])
    with c1:
        if st.button(f"Queue removal of {len(newly_unticked)} screen(s)",
                     disabled=not newly_unticked, type="primary"):
            store.add_feedback(
                campaign_id=st.session_state.campaign_id,
                run_id=st.session_state.run_id, screen="relevance",
                mode="drop_screen", payload={"screen_ids": newly_unticked},
                loss_reason="targeting_mismatch")
            store.log(campaign_id=st.session_state.campaign_id, actor="rep",
                      event="queued_drop", detail=", ".join(newly_unticked))
            st.rerun()
    with c2:
        if st.button(f"Restore {len(re_ticked)} screen(s)", disabled=not re_ticked):
            st.session_state.dropped_screens = sorted(
                set(st.session_state.dropped_screens) - set(re_ticked))
            for it in store.queued_feedback(st.session_state.campaign_id):
                if it["mode"] == "drop_screen":
                    store.dismiss_feedback(it["id"])
            st.rerun()
    with c3:
        if already_dropped:
            st.caption(f"Already excluded from this revision: "
                       f"{', '.join(sorted(already_dropped)[:6])}"
                       + (" …" if len(already_dropped) > 6 else ""))

    # -------------------------------------------------------- decomposition #
    st.markdown("---")
    st.markdown("##### Score decomposition")
    pick = st.selectbox("Screen", view["screen_id"].tolist(), key="decomp_pick")
    row = scored[scored["screen_id"].astype(str) == pick]
    if len(row):
        r = row.iloc[0]
        left, right = st.columns([3, 2], gap="large")
        with left:
            total = 0.0
            for comp, w in WEIGHTS.items():
                if comp not in row.columns:
                    continue
                val = float(r[comp])
                contrib = val * w
                total += contrib
                st.markdown(
                    f"{C.bar(val, 150)} &nbsp;**{LABELS[comp]}** &nbsp;"
                    f"<span class='mono muted'>{contrib:.3f} / {w:.2f} "
                    f"(raw {val:.2f})</span>", unsafe_allow_html=True)
            st.markdown(f"**Total relevance — `{total:.3f}`**")
            st.caption("Weighting: time-of-day fit 0.25, requested category 0.20, "
                       "audience match 0.20, profile confidence 0.15, market tier "
                       "0.10, booking history 0.10.")
        with right:
            C.why_box(str(r.get("reasons", "")), tag="Why it scored this way")
            C.stat_row([
                ("Rank", f"{int(r.get('rank', 0))} of {len(scored):,}"),
                ("Zone", f"`{r.get('zone_id', '--')}`"),
                ("Type", f"{r.get('screen_type', '--')} / {r.get('position', '--')}"),
                ("Market tier", str(r.get("market_tier", "--"))),
                ("History", "yes" if r.get("has_booking_history") else "no"),
            ])
            if adj is not None and adj.enabled:
                d, why = adj.score_delta(pick)
                if why:
                    st.markdown(f"**Learned adjustment** `{d:+.3f}`")
                    for w in why:
                        st.caption(f"• {w}")

    st.markdown("---")
    C.feedback_panel(
        screen="relevance", campaign_id=st.session_state.campaign_id,
        run_id=st.session_state.run_id,
        placeholder="e.g. drop the bus-rear inventory, the client thinks it looks cheap")

    c1, c2 = st.columns([1, 5])
    if c1.button("Continue →", type="primary"):
        S.goto(S.next_step("relevance"))

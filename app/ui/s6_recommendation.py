"""Screen 06 — Proposal.

What the rep actually needs from this page, in order: can I send this, why does
it look like this, what do I send, and what do I do if the client pushes back.
So the tabs are: The pitch / Email draft / Numbers / Changes — and in developer
view, two more with the machinery.

The reasoning is composed by `services/explain.py` from what the engines already
computed, translated out of engine language. D5's LLM narrative is layered on
top when credentials exist, never in place of it.

This is also the only place a re-run happens.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import config
from app.db import store
from app.services import docs, email_draft, engines, explain as explain_mod
from app.services import exports, rfp
from app.services.metrics import compact, money, pct
from app.ui import components as C
from app.ui import state as S


# --------------------------------------------------------------------------- #
# builders (cached per run so switching tabs does not recompose)
# --------------------------------------------------------------------------- #

def _explanation():
    if st.session_state.get("_expl_run") != st.session_state.run_id:
        st.session_state._expl = explain_mod.build(
            result=st.session_state.result,
            quoted_totals=st.session_state.quoted_totals,
            adjustments=st.session_state.adjustments)
        st.session_state._expl_run = st.session_state.run_id
    return st.session_state._expl


def _rfp_asks():
    _has, asks = docs.detect_rfp_asks(st.session_state.brief_text or "")
    return asks


def _build_rfp() -> None:
    result = st.session_state.result
    orch = engines.warmup_state().orchestrator
    adj = st.session_state.adjustments
    md, template, asks_used = rfp.compose(
        result=result, quoted=st.session_state.quoted,
        quoted_totals=st.session_state.quoted_totals,
        profiles=(orch.profiles if orch is not None else None),
        asks=_rfp_asks(),
        learn_note=(adj.summary() if adj is not None else ""),
        rep_email=st.session_state.get("rep_email", ""))
    st.session_state.rfp_md = md
    st.session_state.rfp_template = template
    st.session_state.rfp_asks = asks_used
    st.session_state.rfp_id = None
    if st.session_state.run_id is not None:
        st.session_state.rfp_id = store.add_rfp_response(
            run_id=st.session_state.run_id, template=template,
            rfp_asks=asks_used, markdown=md)
        store.log(campaign_id=st.session_state.campaign_id,
                  run_id=st.session_state.run_id, actor="agent",
                  event="rfp_composed",
                  detail=f"template={template}, {len(asks_used)} ask(s)")


def _build_email(force: bool = False) -> None:
    if (not force and st.session_state.email_draft is not None
            and st.session_state.get("_email_run") == st.session_state.run_id):
        return
    d = email_draft.build(
        result=st.session_state.result, explanation=_explanation(),
        asks=_rfp_asks(), quoted_totals=st.session_state.quoted_totals,
        rep_name=st.session_state.get("rep_name", ""),
        rep_email=st.session_state.get("rep_email", ""),
        contact_name=st.session_state.get("_contact_name", ""))
    st.session_state.email_draft = d
    st.session_state.email_subject = d.subject
    st.session_state.email_edited = d.body
    st.session_state._email_run = st.session_state.run_id


# --------------------------------------------------------------------------- #

def render() -> None:
    result = st.session_state.result
    if result is None:
        st.info("Start a campaign from the first step.")
        return

    p = result.plan
    met = result.metrics
    b = p.brief
    cid = st.session_state.campaign_id
    dev = S.is_dev()
    quote = (st.session_state.quoted_totals or {}).get("quoted_total", met["spend"])

    C.page_header(
        eyebrow="Step 4 of 4" if not dev else "Step 6 of 6 · orchestrator",
        title="Proposal",
        lede="Everything the client needs, and the reasoning behind it in language "
             "you can forward.",
        context=[("Investment", money(quote)),
                 ("Reach / day", compact(met["reach_per_day"])),
                 ("Locations", met["n_screens"])])

    if p.presentable:
        st.success(f"**Ready to send.** {money(quote)} is "
                   f"{pct(met['budget_utilisation'])} of the client's stated budget.")
    else:
        st.error("**Not ready to send yet.** See what needs fixing below.")
        for r in p.refusals:
            C.refusal_box(explain_mod._refusal_plain(r.stage), r.reason, r.remedy)

    _agent_rail(p)

    ex = _explanation()

    tabs = ["The pitch", "Email draft", "Numbers", "Changes & feedback", "Downloads"]
    if dev:
        tabs += ["Full proposal", "Engine detail"]
    T = st.tabs(tabs)

    # ==================================================================== #
    # 1. THE PITCH
    # ==================================================================== #
    with T[0]:
        C.why_box(ex.headline, tag="The offer")

        C.section("Why these locations")
        C.reason_list(ex.why_screens)

        C.section("Why this price")
        C.reason_list(ex.why_price)

        C.section("What the client gets")
        C.reason_list(ex.what_they_get)

        if ex.watch_outs:
            C.section("Worth knowing before you send")
            C.reason_list(ex.watch_outs)

        if ex.talking_points:
            C.section("If they push back")
            C.reason_list(ex.talking_points)

        # The LLM narrative is a bonus paragraph, clearly attributed.
        if st.session_state.get("_nar_run") != st.session_state.run_id:
            with st.spinner("Writing a summary…"):
                try:
                    from build_d5_orchestrator import explain as d5_explain
                    st.session_state._nar = d5_explain(
                        p, use_llm=bool(st.session_state.llm_enabled))
                except Exception:  # noqa: BLE001
                    st.session_state._nar = None
            st.session_state._nar_run = st.session_state.run_id
        nar = st.session_state.get("_nar")
        if nar is not None:
            result.narrative = nar
            C.section("One-paragraph summary")
            # D5 writes this for an operator ("11 lines ... in daypart(s) [3]").
            # D1-D5 are frozen, so translate on the way to the screen. Numbers
            # are untouched.
            st.markdown(f"> {explain_mod.humanise(str(nar))}")
            if dev:
                st.caption(f"Provenance: {nar.provenance()}")
                with st.expander("Untranslated original"):
                    st.markdown(f"> {str(nar)}")

    # ==================================================================== #
    # 2. EMAIL DRAFT
    # ==================================================================== #
    with T[1]:
        st.markdown("#### Email to the client")
        st.caption("Read it, edit anything you want, then download or copy. It "
                   "already includes the package, the numbers, and an answer to "
                   "each requirement the brief asked for.")

        c1, c2, c3 = st.columns([2, 2, 1.4])
        with c1:
            contact = st.text_input("Client contact first name (optional)",
                                    key="_contact_name",
                                    placeholder="e.g. Priya")
        with c2:
            st.text_input("Your name", key="rep_name", placeholder="e.g. Rohan Doda")
            st.text_input("Your email", key="rep_email",
                          placeholder="you@urbanmedia.com",
                          label_visibility="collapsed")
        with c3:
            include_why = st.toggle("Include reasoning", value=True,
                                    help="The 'why this plan' bullets.")

        if st.button("Rebuild draft", type="secondary"):
            _build_email(force=True)
            st.rerun()
        _build_email()

        d = st.session_state.email_draft
        subject = st.text_input("Subject", value=st.session_state.email_subject,
                                key="email_subject")

        st.markdown(
            f'<div class="maildraft"><div class="mhead">'
            f'<div class="row"><span>Subject</span><b>{subject}</b></div>'
            f'<div class="row"><span>Attach</span>'
            f'<span>proposal_{(b.name or "campaign")[:24]}.docx</span></div>'
            f'</div></div>', unsafe_allow_html=True)

        body = st.text_area("Body", value=st.session_state.email_edited,
                            key="email_edited", height=520,
                            label_visibility="collapsed")

        stem = "".join(ch for ch in (b.name or "campaign").lower().replace(" ", "_")
                       if ch.isalnum() or ch in "_-")
        rev = st.session_state.revision
        final = email_draft.Draft(subject=subject, body=body)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("Download as .eml (opens in Outlook)",
                               final.as_eml(st.session_state.get("rep_email", "")),
                               file_name=f"{stem}_email_r{rev}.eml",
                               mime="message/rfc822", use_container_width=True,
                               type="primary")
        with d2:
            st.download_button("Download as .txt", final.as_text.encode("utf-8"),
                               file_name=f"{stem}_email_r{rev}.txt",
                               mime="text/plain", use_container_width=True)
        with d3:
            if st.button("Mark approved", use_container_width=True,
                         disabled=not p.presentable):
                if not st.session_state.rfp_md:
                    _build_rfp()
                if st.session_state.rfp_id:
                    store.approve_rfp(st.session_state.rfp_id)
                store.log(campaign_id=cid, run_id=st.session_state.run_id,
                          actor="rep", event="email_approved",
                          detail=f"revision {rev}, {money(quote)}")
                st.success("Approved and logged.")

        st.caption("The .eml opens as a ready-to-send draft in Outlook with the "
                   "subject and body already filled in. Attach the proposal from "
                   "the Downloads tab.")

    # ==================================================================== #
    # 3. NUMBERS
    # ==================================================================== #
    with T[2]:
        st.markdown("#### What the client is buying")
        C.tiles([
            {"value": money(quote), "label": "Total investment", "tone": "",
             "sub": f"{pct(met['budget_utilisation'])} of budget"},
            {"value": compact(met["reach_per_day"]), "label": "People per day"},
            {"value": compact(met["impressions_total"]), "label": "Total views"},
            {"value": f"{met['avg_frequency_per_day']:.1f}x",
             "label": "Times seen per person per day"},
            {"value": money(met["effective_cpm"], 2),
             "label": "Cost per 1,000 views"},
        ])

        st.markdown("#### Your negotiating room")
        if met["package_cap"] > met["package_floor"]:
            C.price_band(met["package_floor"], met["package_target"],
                         met["package_cap"], quote, "quoting")
        n1, n2, n3 = st.columns(3)
        n1.metric("Don't go below", money(met["package_floor"]),
                  help="What comparable bookings actually sold for.")
        n2.metric("Quoting", money(quote))
        n3.metric("Room above", money(met["package_cap"]),
                  help="Above this, comparable deals historically stopped closing.")
        if met.get("weighted_p_win", 0):
            st.caption(f"At this price the modelled chance of acceptance is "
                       f"**{pct(met['weighted_p_win'], 0)}**, from real win/loss "
                       f"history on similar deals.")

        st.markdown("#### The locations")
        st.dataframe(
            _client_table(result, st.session_state.quoted,
                          int(b.duration_days or 0)),
            hide_index=True, use_container_width=True,
            column_config={
                "People / day": st.column_config.NumberColumn(format="%,d"),
                "Investment": st.column_config.NumberColumn(format="$%.0f"),
            })

        if result.split:
            st.markdown("#### Weekday vs weekend")
            st.caption(result.split_note)
            st.dataframe(pd.DataFrame([{
                "Day type": s.day_type.capitalize(), "Locations": s.n_lines,
                "Investment": s.spend, "People / day": s.reach,
                "Total views": s.impressions}
                for s in result.split.values()]),
                hide_index=True, use_container_width=True,
                column_config={
                    "Investment": st.column_config.NumberColumn(format="$%.0f"),
                    "People / day": st.column_config.NumberColumn(format="%.0f"),
                    "Total views": st.column_config.NumberColumn(format="%.0f")})

    # ==================================================================== #
    # 4. CHANGES & FEEDBACK
    # ==================================================================== #
    with T[3]:
        st.markdown("#### Need something changed?")
        st.caption("Tell the agent in plain English what to fix and it will rebuild "
                   "the plan. Anything you flagged on earlier steps is included.")

        items = C.feedback_tray(cid, compact_mode=False, key_prefix="s6")
        C.feedback_panel(
            screen="recommend", campaign_id=cid, run_id=st.session_state.run_id,
            key_prefix="recommend",
            placeholder="e.g. client won't go above $18k and wants students in the "
                        "audience, drop the bus screens")

        c1, c2 = st.columns([2, 3])
        with c1:
            if st.button(f"Rebuild the plan ({len(items)} change(s))",
                         type="primary", disabled=not items,
                         use_container_width=True):
                if S.apply_feedback_and_rerun():
                    st.rerun()
                else:
                    st.error("The rebuild failed.")
                    if dev:
                        st.code(st.session_state.run_error)
        with c2:
            st.caption("One rebuild applies everything at once — quicker and "
                       "cheaper than changing things one at a time.")

        last = st.session_state.last_actions
        if last is not None:
            st.markdown("#### What changed last time")
            lines_ = last.summary_lines()
            if lines_:
                C.reason_list(lines_)
            elif last.rejected:
                # "Nothing actionable" on its own is the least useful thing we
                # can say: the rep cannot tell a parser miss from a request the
                # data genuinely cannot answer. When we know which it was, say
                # it here rather than only behind the developer expander.
                st.warning("Nothing in the plan changed. Here is why:")
                C.reason_list(last.rejected)
            else:
                C.reason_list(["_Nothing actionable was found in that feedback._"])
            if last.rejected and dev and lines_:
                with st.expander(f"Rejected by validation ({len(last.rejected)})"):
                    for r in last.rejected:
                        st.markdown(f"- {r}")

        hist = store.list_runs(cid)
        if len(hist) > 1:
            st.markdown("#### Version history")
            a, bb = hist[-2], hist[-1]
            # Two money() calls in ONE markdown string means two '$', and
            # Streamlit reads the text between them as LaTeX -- which swallowed
            # the figures and printed the bold markers literally. Escape them.
            def _m(v):
                return money(v).replace("$", r"\$")
            st.markdown(
                f"Version {bb['revision']} vs {a['revision']}: investment "
                f"{_m(a['spend'] or 0)} → **{_m(bb['spend'] or 0)}**, reach "
                f"{compact(a['reach'] or 0)} → **{compact(bb['reach'] or 0)}**, "
                f"locations {a['n_screens']} → **{bb['n_screens']}**.")

    # ==================================================================== #
    # 5. DOWNLOADS
    # ==================================================================== #
    with T[4]:
        if not st.session_state.rfp_md:
            _build_rfp()
        rev = st.session_state.revision
        stem = "".join(ch for ch in (b.name or "campaign").lower().replace(" ", "_")
                       if ch.isalnum() or ch in "_-")
        orch = engines.warmup_state().orchestrator

        st.markdown("#### Send to the client")
        c1, c2 = st.columns(2)
        with c1:
            dx = exports.rfp_docx(st.session_state.rfp_md, title=f"{b.name} proposal")
            if dx:
                st.download_button("Proposal (Word)", dx,
                                   file_name=f"{stem}_proposal_r{rev}.docx",
                                   mime="application/vnd.openxmlformats-officedocument."
                                        "wordprocessingml.document",
                                   use_container_width=True, type="primary")
        with c2:
            st.download_button("Media plan (Excel-ready CSV)",
                               exports.media_plan_csv(result, st.session_state.quoted),
                               file_name=f"{stem}_media_plan_r{rev}.csv",
                               mime="text/csv", use_container_width=True)

        st.markdown("#### For your records")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Proposal (Markdown)",
                               exports.markdown_bytes(st.session_state.rfp_md),
                               file_name=f"{stem}_proposal_r{rev}.md",
                               mime="text/markdown", use_container_width=True)
        with c2:
            st.download_button("Pricing detail",
                               exports.rate_card_csv(result),
                               file_name=f"{stem}_pricing_r{rev}.csv",
                               mime="text/csv", use_container_width=True)
        with c3:
            st.download_button("Audience detail",
                               exports.audience_appendix_csv(
                                   result, orch.profiles if orch else None,
                                   orch.cube if orch else None),
                               file_name=f"{stem}_audience_r{rev}.csv",
                               mime="text/csv", use_container_width=True)

        if dev:
            st.markdown("#### Audit")
            c1, c2, c3 = st.columns(3)
            from app.services import pipeline
            with c1:
                st.download_button("Full plan (JSON)",
                                   exports.plan_json_bytes(
                                       pipeline.plan_to_dict(result)),
                                   file_name=f"{stem}_plan_r{rev}.json",
                                   mime="application/json",
                                   use_container_width=True)
            with c2:
                st.download_button("Decision log",
                                   exports.decision_log_md(
                                       store.get_campaign(cid),
                                       store.decision_log(cid),
                                       store.list_runs(cid)),
                                   file_name=f"{stem}_decision_log.md",
                                   mime="text/markdown", use_container_width=True)
            with c3:
                st.download_button("Feedback log",
                                   exports.feedback_csv(store.all_feedback(cid)),
                                   file_name=f"{stem}_feedback.csv",
                                   mime="text/csv", use_container_width=True)

    # ==================================================================== #
    # developer-only tabs
    # ==================================================================== #
    if dev:
        with T[5]:
            if not st.session_state.rfp_md:
                _build_rfp()
            tmpl = st.session_state.rfp_template
            if tmpl == "brief_rfp":
                st.success(f"The brief carried its own RFP section with "
                           f"{len(st.session_state.rfp_asks)} requirement(s), each "
                           f"answered explicitly.")
            else:
                st.info("No RFP section in this brief — the standard three-part "
                        "template was used.")
            if st.button("Rebuild proposal"):
                _build_rfp()
                st.rerun()
            st.markdown("---")
            st.markdown(st.session_state.rfp_md)

        with T[6]:
            st.markdown("##### Plan quality and risk")
            C.tiles([
                {"value": f"{met.get('relevance_mean', 0):.3f}",
                 "label": "Mean match score (bought)",
                 "sub": f"min {met.get('relevance_min', 0):.3f}"},
                {"value": pct(met.get("cold_start_spend_share", 0)),
                 "label": "Spend on never-booked screens",
                 "tone": "warn" if met.get("cold_start_spend_share", 0) > 0.5 else ""},
                {"value": pct(met.get("peer_priced_spend_share", 0)),
                 "label": "Spend priced from peers"},
                {"value": met["solver_used"], "label": "Solver",
                 "sub": f"{met['n_candidate_cells']:,} candidate cells"},
                {"value": f"{met.get('capacity_utilisation', 0) * 100:.0f}%",
                 "label": "Slot capacity used"},
            ])
            c = p.cohort
            if getattr(c, "n_clients", 0):
                st.markdown("##### Cold-prospect benchmark")
                C.stat_row([
                    ("Inferred industry",
                     f"`{p.industry.industry or 'unknown'}` "
                     f"({pct(p.industry.confidence, 0)})"),
                    ("Comparable clients", f"{c.n_clients}"),
                    ("They book", f"{money(c.median_budget)} / "
                                  f"{c.median_duration_days} days"),
                    ("Budget tolerance", pct(c.variance_pct, 0)),
                    ("Posture", f"**{c.posture}** — {c.posture_reason}"),
                ])
            adj = st.session_state.adjustments
            if adj is not None:
                st.markdown("##### Cross-session learning")
                st.caption(adj.summary())
                rows = adj.active_rows()
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 use_container_width=True)
            with st.expander("Full deterministic plan page"):
                st.code(p.render(), language=None)
            with st.expander("Narrative fact sheet"):
                from build_d5_orchestrator import narrative_facts
                st.code(narrative_facts(p), language=None)


def _client_table(result, quoted, days: int) -> pd.DataFrame:
    """The location list in client language — no screen ids, no block numbers."""
    lines = result.lines
    if not len(lines):
        return pd.DataFrame()
    df = lines.copy()
    sc = "slots_booked_per_day" if "slots_booked_per_day" in df.columns else "slots"
    if result.scored is not None and len(result.scored):
        keep = [c for c in ["screen_id", "screen_type", "position", "zone_id"]
                if c in result.scored.columns]
        df = df.merge(result.scored[keep].drop_duplicates("screen_id"),
                      on="screen_id", how="left")
    qmap = {}
    if quoted is not None and len(quoted):
        qmap = {(str(r.screen_id), int(r.time_block_id)): float(r.quoted_price)
                for r in quoted.itertuples()}

    rows = []
    for _, r in df.iterrows():
        sid, blk = str(r["screen_id"]), int(r["time_block_id"])
        price = qmap.get((sid, blk), float(r["price"]))
        slots = float(r[sc])
        rows.append({
            "Location type": explain_mod.plain_type(r.get("screen_type", "")),
            "Position": explain_mod.plain_position(r.get("position", "")),
            "Area": str(r.get("zone_id", "--")),
            "When it runs": explain_mod.plain_block(blk),
            "Slots / day": int(slots),
            "People / day": float(r.get("people_reach", 0) or 0),
            "Investment": price * slots * days,
        })
    return pd.DataFrame(rows).sort_values("People / day", ascending=False)


#: Lever identifier -> what it actually did, in words a rep can repeat to a
#: client. Anything unmapped falls back to the identifier with underscores
#: opened up, so a new lever degrades to readable rather than to a KeyError.
_LEVER_WORDS = {
    "grow_shortlist": "widen the search to more candidate locations",
    "raise_supplementary_share": "allow more of the secondary inventory",
    "drop_staging": "release inventory that was being held back",
    "shrink_shortlist": "tighten the shortlist",
    "accept": "stop and keep the best plan found",
    "baseline": "first attempt",
}


def _lever_words(lever: str) -> str:
    return _LEVER_WORDS.get(lever, str(lever).replace("_", " "))


def _agent_rail(plan) -> None:
    """What the planner tried, what it kept, and what a fixed rule would have
    done instead.

    This is the audit trail, not decoration. Every attempt is either KEPT or
    reverted on a measured comparison, so the rail also shows the honest cases:
    an attempt that made the plan worse and was rolled back, and a plan where
    nothing improved on the first solve. A rail that only ever showed wins
    would be marketing.
    """
    log = list(getattr(plan, "agent_log", None) or [])
    if not log:
        return

    tried = [s for s in log if s.lever not in ("baseline", "accept")]
    kept = [s for s in tried if s.accepted]
    metric = log[0].metric
    first = log[0].score
    best = max((s.score for s in log if s.accepted), default=first)
    lift = (best / first - 1.0) if first else 0.0

    if kept:
        head = (f"improved the plan by {lift:.1%} "
                f"({metric} {first:,.0f} to {best:,.0f})" if metric ==
                "distinct people" else
                f"improved the plan by {lift:.1%} ({metric} "
                f"{first:,.2f} to {best:,.2f})")
    elif tried:
        head = ("tried alternatives and kept the original plan -- none of them "
                "did better")
    else:
        head = "found no alternative worth trying on this brief"

    with st.expander(f"How this plan was reached -- {len(tried)} "
                     f"alternative(s) tried, {head}"):
        st.caption(
            "The planner proposes one change at a time from a fixed menu, "
            "re-solves, and keeps the result only if it beats what it already "
            "had on this brief's own objective -- so a change can never make "
            "the plan worse. 'A fixed rule would have chosen' shows what a "
            "non-learning version would have done, which is how we check the "
            "reasoning is earning its place."
        )
        for s in log:
            if s.lever == "baseline":
                st.markdown(
                    f"**First solve** — {s.lines} location(s), "
                    f"${s.spend:,.0f} committed, {s.metric} "
                    f"**{s.score:,.2f}**" if s.objective == "frequency" else
                    f"**First solve** — {s.lines} location(s), "
                    f"${s.spend:,.0f} committed, {s.metric} "
                    f"**{s.score:,.0f}**")
                continue
            verdict = "**kept**" if s.accepted else "reverted — it did not help"
            delta = (f"{s.delta_score:+,.2f}" if s.objective == "frequency"
                     else f"{s.delta_score:+,.0f}")
            chose = ("proposed by the model" if s.source.startswith("llm")
                     else "proposed by the fallback rule")
            line = (f"**Attempt {s.iteration}** — {_lever_words(s.lever)} "
                    f"({chose}). Reason given: *{s.why}* → {s.lines} "
                    f"location(s), ${s.spend:,.0f}, {s.metric} {delta} → "
                    f"{verdict}")
            if s.counterfactual and s.counterfactual != s.lever:
                line += ("  " + chr(10) + "_A fixed rule would have chosen "
                         + _lever_words(s.counterfactual) + "._")
            st.markdown(line)

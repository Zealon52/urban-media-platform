"""Screen 01 — Intake.

Four input modes converge on ONE normalized brief text, so nothing downstream
branches on how the brief arrived. Uploaded bytes are persisted to SQLite as a
BLOB (deduped on sha256), which is what makes the app runnable from a zip on a
machine that is not ours: one self-contained database, no upload folder to lose
and no relative path to resolve.
"""
from __future__ import annotations

import streamlit as st

from app import config
from app.db import store
from app.services import docs, engines
from app.ui import components as C
from app.ui import state as S


def _start_campaign(*, brief_text: str, source_mode: str, source_label: str,
                    document_id=None, client_name=None) -> None:
    cid = store.create_campaign(
        brief_text=brief_text, source_mode=source_mode, source_label=source_label,
        document_id=document_id, client_name=client_name,
        rep_email=st.session_state.get("rep_email") or None)
    st.session_state.campaign_id = cid
    st.session_state.brief_text = brief_text
    st.session_state.source_mode = source_mode
    st.session_state.source_label = source_label
    st.session_state.document_id = document_id
    st.session_state.brief = None
    st.session_state.result = None
    st.session_state.dropped_screens = []
    store.log(campaign_id=cid, actor="rep", event="intake",
              detail=f"{source_mode}: {source_label} ({len(brief_text):,} chars)")
    S.goto("interpret")


def render() -> None:
    dev = S.is_dev()
    C.page_header(
        eyebrow="Step 1 of 4" if not dev else "Step 1 of 6",
        title="Start a new campaign",
        lede="Give it whatever the client sent you — a brief, a PDF, or just the "
             "email. It reads it and shows you what it understood before doing "
             "anything else.")

    mode = st.radio("Input", ["Upload a document", "Paste an email",
                              "Replay a past lead", "Sample brief"],
                    horizontal=True, label_visibility="collapsed")

    # ---------------------------------------------------------------- upload #
    if mode == "Upload a document":
        up = st.file_uploader("Brief document", type=["docx", "pdf", "txt", "md"],
                              help="DOCX and PDF are parsed including tables — the "
                                   "brief's budget, duration and audience live in a "
                                   "table, not a paragraph.")
        if up is not None:
            data = up.getvalue()
            with st.spinner(f"Extracting {up.name}…"):
                ex = docs.extract(up.name, data)

            c1, c2, c3 = st.columns(3)
            c1.metric("Size", f"{len(data) / 1024:,.0f} KB")
            c2.metric("Extractor", ex.extractor)
            c3.metric("Text", f"{len(ex.text):,} chars"
                              + (f" · {ex.n_pages}p" if ex.n_pages else ""))

            if ex.warning:
                st.warning(ex.warning)
            if ex.looks_empty:
                st.error("No usable text was extracted, so there is nothing to "
                         "interpret. Try the DOCX version, or paste the text.")
            else:
                has_rfp, asks = docs.detect_rfp_asks(ex.text)
                if has_rfp:
                    st.success(f"Found a formal RFP section with {len(asks)} "
                               f"requirement(s) — the proposal will answer each one "
                               f"explicitly.")
                    for a in asks:
                        st.markdown(f"- {a}")
                else:
                    st.info("No formal RFP section found. Urban Media's standard "
                            "three-part response template will be used instead.")

                with st.expander("Extracted text"):
                    st.code(ex.text[:6000] + ("\n…" if len(ex.text) > 6000 else ""),
                            language=None)

                if st.button("Interpret brief →", type="primary"):
                    doc_id = store.upsert_document(
                        filename=up.name, mime=docs.mime_for(up.name), data=data,
                        extracted_text=ex.text, extractor=ex.extractor,
                        n_pages=ex.n_pages, extract_warning=ex.warning or None)
                    _start_campaign(brief_text=ex.text, source_mode="upload",
                                    source_label=up.name, document_id=doc_id)

    # ----------------------------------------------------------------- paste #
    elif mode == "Paste an email":
        txt = st.text_area(
            "Paste the brief or the email body", height=260,
            placeholder="Hi — we're Meridian Capital, looking at the eastern business "
                        "corridor in Las Hackland for a month from Sept 1. Budget "
                        "around $50K. We want to reach as many young professionals as "
                        "possible on their commute…")
        st.caption("A pasted email rarely has a formal RFP section, so the standard "
                   "response template will be used.")
        if st.button("Interpret brief →", type="primary",
                     disabled=len(txt.strip()) < 40):
            _start_campaign(brief_text=txt, source_mode="paste",
                            source_label="pasted text")

    # ------------------------------------------------------------ lead replay #
    elif mode == "Replay a past lead":
        st.caption("Replay a real lost lead through the whole system. The outcome and "
                   "the recorded loss reason are already known, so this doubles as an "
                   "evaluation case rather than only a demo convenience.")
        try:
            leads = engines.lost_leads_sample(300)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read lost_leads.csv: {exc}")
            return

        label_col = next((c for c in ["lead_id", "id"] if c in leads.columns), None)
        if label_col is None:
            st.error("lost_leads.csv has no recognisable id column.")
            return

        def fmt(i: int) -> str:
            r = leads.iloc[i]
            bits = [str(r[label_col])]
            for c in ["industry_vertical", "city_id", "zone_id", "loss_reason"]:
                if c in leads.columns:
                    bits.append(str(r[c]))
            for c in ["requested_budget", "budget"]:
                if c in leads.columns:
                    bits.append(f"${float(r[c] or 0):,.0f}")
                    break
            return " · ".join(bits)

        idx = st.selectbox("Lead", range(len(leads)), format_func=fmt)
        row = leads.iloc[int(idx)].to_dict()
        cities = engines.light_tables()["cities"]
        city_name = ""
        if row.get("city_id") is not None:
            m = cities[cities.city_id == row.get("city_id")]
            city_name = str(m.iloc[0]["city_name"]) if len(m) else ""

        text = docs.lead_to_brief_text(row, city_name)
        with st.expander("Synthesised brief"):
            st.code(text, language=None)
        if st.button("Interpret brief →", type="primary"):
            _start_campaign(brief_text=text, source_mode="lead",
                            source_label=str(row[label_col]),
                            client_name=str(row.get("company_name") or ""))

    # ---------------------------------------------------------------- sample #
    else:
        samples = docs.list_sample_briefs(config.CAMPAIGNS_DIR)
        if not samples:
            st.warning(f"No campaign_*.docx found in {config.CAMPAIGNS_DIR}.")
            return
        st.caption("The six briefs shipped with the project. Each closes with its own "
                   "formal RFP requirements section, which the proposal answers "
                   "point by point.")
        pick = st.selectbox("Brief", samples, format_func=lambda p: p.name)
        data = pick.read_bytes()
        with st.spinner("Extracting…"):
            ex = docs.extract(pick.name, data)
        has_rfp, asks = docs.detect_rfp_asks(ex.text)
        if has_rfp:
            st.markdown("**Its RFP requirements**")
            for a in asks:
                st.markdown(f"- {a}")
        with st.expander("Extracted text"):
            st.code(ex.text[:6000], language=None)
        if st.button("Interpret brief →", type="primary"):
            doc_id = store.upsert_document(
                filename=pick.name, mime=docs.mime_for(pick.name), data=data,
                extracted_text=ex.text, extractor=ex.extractor,
                n_pages=ex.n_pages, extract_warning=ex.warning or None)
            _start_campaign(brief_text=ex.text, source_mode="sample",
                            source_label=pick.name, document_id=doc_id)

    # ------------------------------------------------------------- past work #
    prior = store.list_campaigns(12)
    if prior:
        with st.expander(f"Recent campaigns in the database ({len(prior)})"):
            for c in prior:
                cols = st.columns([3, 2, 1, 1])
                cols[0].markdown(f"**{c['source_label'] or '--'}**  \n"
                                 f"<span class='muted'>{c['created_at']}</span>",
                                 unsafe_allow_html=True)
                cols[1].markdown(f"<span class='muted'>{c['source_mode']} · "
                                 f"{c['n_runs']} run(s)</span>",
                                 unsafe_allow_html=True)
                if cols[3].button("Reopen", key=f"reopen_{c['id']}"):
                    st.session_state.campaign_id = c["id"]
                    st.session_state.brief_text = c["brief_text"]
                    st.session_state.source_mode = c["source_mode"]
                    st.session_state.source_label = c["source_label"]
                    st.session_state.document_id = c["document_id"]
                    st.session_state.brief = None
                    st.session_state.result = None
                    S.goto("interpret")

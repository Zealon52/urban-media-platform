"""D6 — Unified Platform. Streamlit entry point.

Run:
    .venv\\Scripts\\python.exe -m streamlit run app/main.py

Flow: Intake -> Interpretation -> [D5 runs once] -> Audience (D1) ->
Relevance (D2) -> Package & Price (D3+D4) -> Recommendation & RFP (D5).

Feedback is collected on every screen and applied by exactly one re-run,
triggered from the Recommendation step.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Make `import app.*` and `import build_d5_orchestrator` work regardless of how
# Streamlit was invoked or what the CWD is.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from app.db import store  # noqa: E402
from app.services import engines, llm  # noqa: E402
from app.ui import components as C  # noqa: E402
from app.ui import state as S  # noqa: E402
from app.ui import (s1_intake, s2_interpret, s3_audience, s4_relevance,  # noqa: E402
                    s5_package, s6_recommendation)

# D1-D5 emit DtypeWarnings from their own read_csv calls (audience_cube columns
# 7-8 have mixed types). Those files are frozen, so silence the noise here
# rather than letting it clutter the app surface.
warnings.filterwarnings("ignore", category=Warning, module="build_.*")
warnings.filterwarnings("ignore", message=".*mixed types.*")

st.set_page_config(page_title="Urban Media · Commercial Intelligence",
                   page_icon="◧", layout="wide",
                   initial_sidebar_state="expanded")


def boot() -> None:
    config.load_dotenv_if_present()
    config.ensure_app_data()
    if "db_ready" not in st.session_state:
        store.init_db()
        st.session_state.db_ready = True
    S.init()
    C.inject_css()
    # Kick the heavy load off immediately; screens 01-02 do not wait on it.
    engines.start_warmup()


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### Urban Media")
        st.caption("Campaign planning for the sales desk")

        mode = S.mode()
        dev = S.is_dev()

        C.step_rail(st.session_state.step, S.unlocked_steps(), mode)

        unlocked = S.unlocked_steps()
        labels = dict((s[0], s[1]) for s in config.steps_for(mode))
        if unlocked:
            nav = st.selectbox(
                "Jump to", unlocked,
                index=unlocked.index(st.session_state.step)
                if st.session_state.step in unlocked else 0,
                format_func=lambda k: labels.get(k, k),
                label_visibility="collapsed")
            if nav != st.session_state.step:
                S.goto(nav)

        if st.session_state.campaign_id:
            st.markdown("---")
            st.markdown("**Pending feedback**")
            C.feedback_tray(st.session_state.campaign_id, compact_mode=False,
                            key_prefix="rail")

        st.markdown("---")

        # ---- readiness. Plain language for a rep, detail for a developer ---- #
        wm = engines.warmup_state()
        if wm.ready:
            if dev:
                st.markdown(f":green[● Engines ready] `{wm.solver}`")
                st.caption(f"loaded in {wm.elapsed:.0f}s · "
                           + ", ".join(f"{k} {v:.1f}s"
                                       for k, v in wm.stage_timings.items()))
            else:
                st.markdown(":green[● Ready]")
        elif wm.failed:
            st.markdown(":red[● Something went wrong loading the data]")
            if dev:
                with st.expander("Error"):
                    st.code(wm.error)
            else:
                st.caption("Contact support — the campaign data could not be loaded.")
        else:
            st.markdown(f":orange[● Getting ready… {wm.elapsed:.0f}s]")
            st.caption("You can start entering the brief now."
                       if not dev else (wm.detail or wm.stage))

        # ---- developer-only diagnostics ---- #
        if dev:
            bs = llm.budget_status()
            st.markdown("---")
            if bs["configured"]:
                st.markdown("**Model budget**")
                st.progress(min(bs["spend_pct"], 1.0),
                            text=f"${bs['spend_usd']:.4f} of "
                                 f"${bs['max_spend_usd']:.2f} guard")
                # The key is shared across the whole team, so this app's own
                # meter is only half the picture -- the engines bill the same
                # key through their own fence.
                st.caption(
                    f"app ${bs['spend_usd']:.4f} + engines "
                    f"${bs['engine_spend_usd']:.4f} = "
                    f"**${bs['total_spend_usd']:.4f}** estimated against the "
                    f"shared key. Pessimistic rate -- a safety estimate, not "
                    f"an invoice.")
                st.caption(f"{bs['billable_calls']} of {bs['max_calls']} billable, "
                           f"{bs['cached_hits']} cached · "
                           f"{bs['prompt_tokens']:,}+{bs['completion_tokens']:,} tokens")
                if bs["distinct_tiers"]:
                    st.caption("Feedback routing on **nano**; the one narrative per "
                               "plan on **mini**. New purposes default to nano.")
                else:
                    st.caption(":orange[One deployment only — everything bills at that "
                               "tier. Set AZURE_OPENAI_DEPLOYMENT_NANO.]")
                if bs["exhausted"]:
                    st.warning("Budget guard reached — deterministic paths in use.")
            else:
                st.markdown("**Model budget** — no credentials")
                st.caption("Routing uses deterministic rules; narrative uses D5's "
                           "template. Everything still works.")

            with st.expander("Engine settings"):
                st.session_state.llm_enabled = st.toggle(
                    "Use the model where available",
                    value=st.session_state.llm_enabled)
                st.session_state.learning_enabled = st.toggle(
                    "Apply cross-session learning",
                    value=st.session_state.learning_enabled)
                preset = st.selectbox(
                    "Solver effort", list(config.SOLVE_PRESETS),
                    index=list(config.SOLVE_PRESETS.values()).index(
                        st.session_state.solve_limit)
                    if st.session_state.solve_limit in config.SOLVE_PRESETS.values()
                    else 1,
                    help="D5 runs two solves per plan, so wall clock is roughly "
                         "twice this.")
                st.session_state.solve_limit = config.SOLVE_PRESETS[preset]
                st.session_state.force_greedy = st.toggle(
                    "Force the greedy solver",
                    value=st.session_state.force_greedy,
                    help="Exercises the no-ortools fallback deliberately.")
                st.session_state.allow_unevidenced = st.toggle(
                    "Allow unmeasurable time slots",
                    value=bool(st.session_state.allow_unevidenced))

            with st.expander("Database"):
                s = store.db_stats()
                st.caption(f"`{config.DB_PATH.name}` · {s['db_bytes'] / 1024:,.0f} KB")
                st.dataframe(
                    [{"table": k, "rows": v} for k, v in s.items()
                     if k != "db_bytes"],
                    hide_index=True, use_container_width=True)
        # Rep name/email are owned by the Proposal screen's email tab, where they
        # are actually used -- two widgets bound to one session_state key is a
        # duplicate-key crash, and the email is the right place to set a
        # signature anyway.

        # ---- mode ---- #
        st.markdown("---")
        chosen = st.radio(
            "View", [config.MODE_CLIENT, config.MODE_DEVELOPER],
            index=0 if mode == config.MODE_CLIENT else 1,
            format_func=lambda m: ("Sales view" if m == config.MODE_CLIENT
                                   else "Developer view"),
            horizontal=True, label_visibility="collapsed",
            help="Sales view goes straight from the brief to the plan and the "
                 "proposal. Developer view adds the audience and screen-match "
                 "steps, engine labels and diagnostics.")
        if chosen != mode:
            S.set_mode(chosen)
            st.rerun()

        if st.button("Start a new campaign", use_container_width=True):
            S.reset_campaign()
            st.rerun()


def header() -> None:
    """Only the working-campaign context line. Each screen renders its own
    title via C.page_header, so the two do not compete."""
    if not st.session_state.campaign_id:
        return
    b = st.session_state.brief
    bits = []
    if b is not None and b.name and b.name != "Unnamed Brief":
        bits.append(f"<b>{b.name}</b>")
    if b is not None:
        if b.city_id:
            bits.append(config.CITY_CHOICES.get(b.city_id, b.city_id).split(" (")[0])
        if b.budget:
            bits.append(f"${b.budget:,.0f}")
        if b.duration_days:
            bits.append(f"{b.duration_days} days")
    if st.session_state.revision:
        bits.append(f"version {st.session_state.revision}")
    if bits:
        st.markdown(
            f"<div class='muted' style='margin:-.5rem 0 .9rem'>"
            f"{' &nbsp;·&nbsp; '.join(bits)}</div>", unsafe_allow_html=True)


PAGES = {
    "intake": s1_intake.render,
    "interpret": s2_interpret.render,
    "audience": s3_audience.render,
    "relevance": s4_relevance.render,
    "package": s5_package.render,
    "recommend": s6_recommendation.render,
}


def main() -> None:
    boot()
    sidebar()
    # Client mode has no Audience or Screen-match screen. Landing on one (from a
    # stale session or a mode switch) must fall back rather than render nothing.
    if st.session_state.step not in S.visible_steps():
        st.session_state.step = (
            "package" if st.session_state.result is not None
            else "interpret" if st.session_state.brief_text else "intake")
    header()
    page = PAGES.get(st.session_state.step, s1_intake.render)
    page()


main()

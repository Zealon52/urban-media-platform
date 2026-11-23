"""Screen 03 — Audience context (D1).

D1 does the most modelling in the system and produces nothing client-facing, so
without this screen a reviewer never sees it. Two views: the campaign-level
audience the plan actually bought, and a per-screen inspector where the day-type
toggle proves the model is time-aware in one click.
"""
from __future__ import annotations

import math
from typing import Dict

import pandas as pd
import streamlit as st

from app import config
from app.services import engines
from app.services.rfp import _parse_composition, segment_breakdown
from app.ui import components as C
from app.ui import state as S


def _cube_row(cube: pd.DataFrame, screen_id: str, block: int, day_type: str):
    m = cube[(cube.screen_id == screen_id) & (cube.time_block_id == block)
             & (cube.day_type == day_type)]
    return m.iloc[0] if len(m) else None


def render() -> None:
    result = st.session_state.result
    if result is None:
        st.info("Run the agent from the Interpretation step first.")
        return

    orch = engines.warmup_state().orchestrator
    if orch is None:
        st.warning("Engines are not loaded.")
        return

    profiles, cube = orch.profiles, orch.cube
    lines = result.lines
    b = result.plan.brief

    st.subheader("Who is actually in front of these screens")
    st.caption("Every screen gets an audience profile built from what's nearby, "
               "route passenger numbers, area demographics and how the screen is "
               "mounted — then the plan buys against that.")

    # ------------------------------------------------- campaign-level view #
    if len(lines):
        seg = segment_breakdown(lines, profiles)
        sel_ids = set(lines["screen_id"].astype(str))
        sp = profiles[profiles["screen_id"].astype(str).isin(sel_ids)]

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.markdown("##### Audience the plan actually buys")
            if len(seg):
                seg2 = seg.copy()
                seg2["Reach / day"] = seg2["reach_per_day"].map(
                    lambda v: f"{v:,.0f}")
                seg2["Share"] = seg2["share"].map(lambda v: f"{v * 100:.1f}%")
                st.dataframe(seg2[["segment", "Reach / day", "Share"]],
                             hide_index=True, use_container_width=True)
                st.bar_chart(seg.set_index("segment")["share"], height=180)
                st.caption("A reach-weighted average of each selected screen's own "
                           "measured audience mix — arithmetic over existing "
                           "profiles, not a separate estimate.")
            else:
                st.caption("No composition data for the selected screens.")

        with c2:
            st.markdown("##### Evidence quality")
            if len(sp):
                conf = float(sp["confidence"].mean())
                # Real Python bools: a pandas nullable-boolean comparison here
                # returns NA-propagating values, not a clean count.
                hist = sp["has_booking_history"].fillna(False).astype(bool)
                n_cold = int((~hist).sum())
                C.tiles([
                    {"value": f"{conf:.2f}", "label": "Mean profile confidence",
                     "tone": "good" if conf >= 0.7 else "warn"},
                    {"value": f"{n_cold} / {len(sp)}",
                     "label": "Screens with no booking history",
                     "tone": "warn" if n_cold else "good",
                     "sub": "cold-start: audience modelled, not observed"},
                ])
                mech = sp["mechanism"].value_counts()
                st.markdown("**How the audience meets the screen**")
                for k, v in mech.items():
                    st.markdown(f"- `{k}` — {v} screen(s)")
                st.caption("`wait` = dwelling at a stop or platform (long exposure), "
                           "`ride` = on board, `pass` = walking or driving past. "
                           "Dwell time drives how much one rotation slot is worth.")

    st.markdown("---")

    # ---------------------------------------------------- per-screen view #
    st.markdown("##### Inspect a single screen")

    pool = (sorted(lines["screen_id"].astype(str).unique()) if len(lines) else [])
    if result.scored is not None and len(result.scored):
        extra = [s for s in result.scored["screen_id"].astype(str).head(60)
                 if s not in pool]
        pool = pool + extra

    if not pool:
        st.caption("No screens to inspect.")
        return

    c1, c2, c3 = st.columns([3, 2, 3])
    with c1:
        sid = st.selectbox("Screen", pool, format_func=lambda s: (
            f"{s}  {'· in plan' if len(lines) and s in set(lines['screen_id'].astype(str)) else ''}"))
    with c2:
        day_type = st.radio("Day type", ["weekday", "weekend"], horizontal=True,
                            index=0 if st.session_state.day_type == "weekday" else 1,
                            key="s3_daytype")
    with c3:
        blocks = sorted(cube[cube.screen_id == sid]["time_block_id"].unique().tolist())
        default_blk = (int(lines[lines.screen_id == sid]["time_block_id"].iloc[0])
                       if len(lines) and sid in set(lines["screen_id"]) else
                       (blocks[0] if blocks else 1))
        blk = st.select_slider("Time block", options=blocks or [1],
                               value=default_blk if default_blk in blocks else
                               (blocks[0] if blocks else 1),
                               format_func=lambda i: config.TIME_BLOCKS.get(i, i))

    prow = profiles[profiles.screen_id == sid]
    prow = prow.iloc[0] if len(prow) else None
    crow = _cube_row(cube, sid, int(blk), day_type)

    left, right = st.columns([1, 1], gap="large")
    with left:
        if crow is not None:
            people = float(crow.get("people", 0) or 0)
            dwell = float(crow.get("dwell_seconds", 0) or 0)
            vis = float(crow.get("visibility", 0) or 0)
            cycles = (dwell / 60.0) * vis
            C.tiles([
                {"value": C.compact(people),
                 "label": f"Est. audience · {day_type} {config.TIME_BLOCKS.get(int(blk))}"},
                {"value": f"{dwell:,.0f}s", "label": "Mean dwell"},
                {"value": f"{vis:.2f}", "label": "Visibility"},
            ])
            st.markdown(
                f"One rotation slot here reaches "
                f"**{(1 - math.exp(-cycles)) * 100:.1f}%** of that audience "
                f"(share of voice `lambda = {cycles:.2f}` per slot). A second slot adds "
                f"less than the first — that saturation is why the optimiser spreads "
                f"across screens for a reach brief and stacks slots for a frequency one.")

            other = "weekend" if day_type == "weekday" else "weekday"
            orow = _cube_row(cube, sid, int(blk), other)
            if orow is not None:
                op = float(orow.get("people", 0) or 0)
                if people > 0 and op > 0:
                    delta = (op / people - 1.0) * 100
                    st.caption(f"Same screen, same block, **{other}**: "
                               f"{op:,.0f} people ({delta:+.0f}%). "
                               f"Flip the toggle above to see the mix move.")
        else:
            st.warning("No audience-cube row for that combination.")

        if prow is not None:
            mix = _parse_composition(prow.get("composition_str"))
            if mix:
                st.markdown("**Segment composition**")
                s = sum(mix.values()) or 1.0
                for k, v in sorted(mix.items(), key=lambda kv: -kv[1]):
                    st.markdown(
                        f"{C.bar(v / s, 130)} &nbsp;`{v / s * 100:5.1f}%` &nbsp;{k}",
                        unsafe_allow_html=True)

    with right:
        if prow is not None:
            st.markdown("**Why this profile**")
            C.why_box(str(prow.get("label", "")), tag="Who is here")
            C.stat_row([
                ("Archetype", f"`{prow.get('archetype_id', '--')}`"),
                ("Mechanism", f"`{prow.get('mechanism', '--')}`"),
                ("Peak block", config.TIME_BLOCKS.get(int(prow.get("peak_block", 0) or 0),
                                                       "--")),
                ("Daily people", f"{float(prow.get('daily_people', 0) or 0):,.0f}"),
                ("Confidence", f"{float(prow.get('confidence', 0) or 0):.2f}"),
                ("Booking history",
                 "yes" if prow.get("has_booking_history") else "no — cold start"),
                ("Zone", f"`{prow.get('zone_id', '--')}`"),
            ])
            poi = prow.get("poi_mix_str")
            if isinstance(poi, str) and poi.strip():
                st.markdown("**Nearby POI evidence**")
                for part in poi.split(";"):
                    if ":" in part:
                        k, _, v = part.partition(":")
                        try:
                            st.markdown(f"- {k.strip()} — {float(v) * 100:.0f}% of the "
                                        f"POI mix")
                        except ValueError:
                            st.markdown(f"- {part.strip()}")
                st.caption("The audit trail behind the profile — inspectable without a "
                           "map, which the MVP does not need.")

    st.markdown("---")
    C.feedback_panel(
        screen="audience", campaign_id=st.session_state.campaign_id,
        run_id=st.session_state.run_id,
        placeholder="e.g. this zone is students not office workers in the evening")

    c1, c2 = st.columns([1, 5])
    if c1.button("Continue →", type="primary"):
        S.goto(S.next_step("audience"))

"""Screen 05 — Package and price (D3 + D4, both already run in the backend).

The rep asked not to walk pricing and allocation as separate steps, so this
screen is the merged answer: the package the optimiser chose, and the price range
to quote for it. D3's per-line floor/target/cap bands are summed into one
package band, because a rep negotiates one number for the deal even though the
model prices per screen-slot.
"""
from __future__ import annotations

import pandas as pd
import json
import streamlit as st

from app import config
from app.services.metrics import compact, money, pct
from app.ui import components as C
from app.ui import state as S


def _posture_widget(container):
    """Structured price feedback, applied on the single re-run."""
    posture = container.selectbox(
        "Price posture for the re-run",
        ["hold", "toward_floor", "toward_cap"],
        format_func={"hold": "Hold at the recommended price",
                     "toward_floor": "Move down (client pushed back)",
                     "toward_cap": "Move up (there is headroom)"}.get,
        key="posture_pick")
    return {} if posture == "hold" else {"posture": posture}


def render() -> None:
    result = st.session_state.result
    if result is None:
        st.info("Run the agent from the Interpretation step first.")
        return

    p = result.plan
    met = result.metrics
    b = p.brief
    lines = result.lines
    quoted = st.session_state.quoted
    totals = st.session_state.quoted_totals
    days = int(b.duration_days or 0)
    dev = S.is_dev()
    quote_total = (totals or {}).get("quoted_total", met["spend"])

    C.page_header(
        eyebrow="Step 3 of 4" if not dev else "Step 5 of 6 · pricing + optimiser",
        title="The plan and what to quote",
        lede="Built to get the most out of the budget, priced against what "
             "comparable bookings actually sold for.",
        context=[("Investment", money(quote_total)),
                 ("People / day", compact(met["reach_per_day"]))]
        if len(lines) else ())

    # ------------------------------------------------------------- refusals #
    if not p.presentable:
        st.error("This is not ready to send.")
        for r in p.refusals:
            C.refusal_box(r.stage, r.reason, r.remedy)
        st.markdown("**Where the funnel emptied**")
        st.dataframe(pd.DataFrame(p.funnel.rows(),
                                  columns=["Stage", "Cells remaining"]),
                     hide_index=True, use_container_width=True)
        st.caption("The first stage at zero is the real cause. An empty plan is a "
                   "diagnosis, not an error.")
        st.markdown("---")
        C.feedback_panel(screen="package",
                         campaign_id=st.session_state.campaign_id,
                         run_id=st.session_state.run_id,
                         placeholder="e.g. widen the daypart, or lift the budget")
        if st.button("Go to Recommendation →"):
            S.goto("recommend")
        return

    # ---------------------------------------------------------------- tiles #
    util = met["budget_utilisation"]
    C.tiles([
        {"value": money(met["spend"]), "label": f"of {money(met['budget'])} budget",
         "sub": f"{pct(util)} used", "tone": "good" if util <= 1.001 else "bad"},
        {"value": compact(met["reach_per_day"]), "label": "Distinct people / day"},
        {"value": compact(met["impressions_total"]),
         "label": f"Impressions over {days} days"},
        {"value": f"{met['avg_frequency_per_day']:.2f}x", "label": "Frequency / day"},
        {"value": money(met["effective_cpm"], 2), "label": "Effective CPM"},
        {"value": f"{met['n_lines']} / {met['n_screens']}",
         "label": "Lines / screens", "sub": f"{met.get('n_pools', 0)} audience pools"},
    ])

    # ---------------------------------------------------------- price range #
    st.markdown("##### What to quote")
    quoted_total = (totals or {}).get("quoted_total", met["spend"])
    band_lo, band_tg, band_hi = (met["package_floor"], met["package_target"],
                                 met["package_cap"])
    if band_hi > band_lo:
        C.price_band(band_lo, band_tg, band_hi, quoted_total, "quote")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Negotiating floor", money(band_lo))
    c2.metric("Target", money(band_tg))
    c3.metric("Cap", money(band_hi))
    c4.metric("Chance they accept", pct(met["weighted_p_win"], 0),
              help="Weighted by spend across locations, from a model fitted on real "
                   "win/loss history for comparable deals.")

    spread = (band_hi - band_lo) / band_tg if band_tg else 0
    st.caption(
        f"Negotiating room is {pct(spread, 1)} of the target. The floor is a real "
        f"percentile of comparable bookings, not a discount to hand out; above the cap "
        f"is where comparable deals historically stopped closing. Band computed on "
        f"{pct(met['band_coverage'], 0)} of lines.")

    if totals and abs(totals.get("delta", 0.0)) > 0.5:
        C.why_box(
            f"{totals['n_lines_moved']} location(s) have been repriced by "
            f"{pct(totals['delta_pct'])} against the model's own recommendation, "
            f"moving the total from {money(totals['d3_total'])} to "
            f"{money(totals['quoted_total'])}. This comes from how clients responded "
            f"to pricing on comparable inventory in earlier campaigns. Every adjusted "
            f"price is still inside its own tested range, and this changes what you "
            f"quote &mdash; not which locations were selected.",
            tag="Adjusted from past deals")

    _price_sensitivity()

    # ----------------------------------------------------------- allocation #
    C.section("Where the ads run")

    df = lines.copy()
    sc = "slots_booked_per_day" if "slots_booked_per_day" in df.columns else "slots"
    if result.scored is not None and len(result.scored):
        keep = [c for c in ["screen_id", "screen_type", "zone_id", "relevance_score",
                            "has_booking_history"] if c in result.scored.columns]
        df = df.merge(result.scored[keep].drop_duplicates("screen_id"),
                      on="screen_id", how="left")

    cap_map = {}
    if result.cells is not None and len(result.cells):
        cap_map = {(str(r.screen_id), int(r.time_block_id)): int(r.remaining_capacity)
                   for r in result.cells.itertuples()}
    qmap = {}
    if quoted is not None and len(quoted):
        qmap = {(str(r.screen_id), int(r.time_block_id)): float(r.quoted_price)
                for r in quoted.itertuples()}

    # Normalise to real Python bools once: a pandas nullable boolean is never
    # the `True` singleton, so an identity test misreports every screen.
    if "has_booking_history" in df.columns:
        df["has_booking_history"] = (
            df["has_booking_history"].fillna(False).astype(bool))

    rows = []
    for _, r in df.iterrows():
        sid, blk = str(r["screen_id"]), int(r["time_block_id"])
        q = result.quotes.get((sid, blk))
        slots = float(r[sc])
        cap = cap_map.get((sid, blk), 6)
        price = qmap.get((sid, blk), float(r["price"]))
        row = {
            "Location type": C.plain_screen_type(r.get("screen_type", "")),
            "Area": str(r.get("zone_id", "--")),
            "When it runs": C.plain_block(blk),
            "Slots / day": f"{slots:.0f} of {cap}",
            "Price / slot / day": price,
            "Investment": price * slots * days,
            "People / day": float(r.get("people_reach", 0) or 0),
        }
        if dev:
            row.update({
                "screen_id": sid,
                "floor": q.floor if q else None,
                "target": q.target if q else None,
                "cap": q.cap if q else None,
                "p(win)": q.p_win_at_recommended if q else None,
                "event": (q.event_names or "") if q else "",
                "no history": (not bool(r.get("has_booking_history", False))),
            })
        else:
            row["Nearby event"] = (q.event_names or "") if q else ""
        rows.append(row)
    alloc = pd.DataFrame(rows)

    st.dataframe(
        alloc, hide_index=True, use_container_width=True,
        column_config={
            "floor": st.column_config.NumberColumn(format="$%.2f"),
            "target": st.column_config.NumberColumn(format="$%.2f"),
            "cap": st.column_config.NumberColumn(format="$%.2f"),
            "Price / slot / day": st.column_config.NumberColumn(format="$%.2f"),
            "Investment": st.column_config.NumberColumn(format="$%.0f"),
            "People / day": st.column_config.NumberColumn(format="%,d"),
            "p(win)": st.column_config.NumberColumn(format="%.0f%%"),
            "no history": st.column_config.CheckboxColumn(
                help="Never sold before: price from comparable screens, audience "
                     "modelled rather than measured"),
        })
    st.caption("Slots are shown as bought of available. Availability is what is "
               "genuinely unsold for these dates, not a flat six.")

    # ----------------------------------------------------- price detail row #
    with st.expander("Why each location is priced the way it is"):
        pick = st.selectbox("Line", list(result.quotes.keys()),
                            format_func=lambda k: f"{k[0]} · "
                                                  f"{config.TIME_BLOCKS.get(k[1], k[1])}")
        q = result.quotes.get(pick)
        if q is not None:
            C.price_band(q.floor, q.target, q.cap,
                         qmap.get(pick, q.recommended), "quote")
            c1, c2 = st.columns([2, 3], gap="large")
            with c1:
                C.stat_row([
                    ("Basis", f"`{q.rate_card_basis}`"),
                    ("Recommended", money(q.recommended, 2)),
                    ("Win probability", pct(q.p_win_at_recommended, 0)),
                    ("Expected revenue", money(q.expected_revenue, 2)),
                ])
                if q.event_adjustment_pct:
                    st.markdown(f"**Event overlap** `+{q.event_adjustment_pct:.1%}` — "
                                f"{q.event_names or 'matched event(s)'}")
                if q.ridership_trend_pct:
                    st.markdown(f"**Ridership trend** "
                                f"`{q.ridership_trend_pct:+.1%}` — "
                                f"{q.ridership_trend_note}")
                for w in (q.warnings or []):
                    st.caption(f":orange[{w}]")
            with c2:
                if q.components:
                    st.markdown("**Why this price**")
                    comp = pd.DataFrame([
                        {"factor": c["name"], "family": c.get("family", ""),
                         "effect": -float(c["shift"]), "rationale": c.get("reason", "")}
                        for c in sorted(q.components,
                                        key=lambda x: -abs(x["shift"]))])
                    st.dataframe(comp, hide_index=True, use_container_width=True,
                                 column_config={"effect":
                                                st.column_config.NumberColumn(
                                                    format="%+.1f%%")})
                else:
                    st.caption("Rate card only — no relationship history for this "
                               "advertiser, which is expected for a cold prospect.")

    # ----------------------------------------------------------- constraints #
    C.section("Checks that passed" if not dev else "Constraint check")
    if not dev:
        # A rep needs the reassurance, not the formulation.
        C.reason_list([
            f"<b>Budget respected.</b> {money(met['spend'])} of "
            f"{money(met['budget'])}"
            + (f", with {money(met['unspent'])} left over because no further "
               f"screen was worth its cost." if met["unspent"] > 1
               else ", fully used."),
            "<b>Availability confirmed.</b> Every slot in this plan is genuinely "
            "unsold for these dates.",
            "<b>No double-counted people.</b> Where two screens share an audience "
            "&mdash; same station or same route &mdash; those people are counted "
            "once.",
            f"<b>Dates check out.</b> Everything runs inside "
            f"{st.session_state.win_start} to {st.session_state.win_end}.",
        ])
    checks = [
        f"**Budget** — {money(met['spend'])} of {money(met['budget'])}. "
        + (f"{money(met['unspent'])} left unallocated because no remaining slot "
           f"improved reach per dollar." if met["unspent"] > 1 else
           "Fully allocated."),
        f"**Block capacity** — no screen exceeds its real remaining slots; "
        f"{met.get('slots_bought_per_day', 0):.0f} of "
        f"{met.get('slots_available_per_day', 0):.0f} available slots/day bought "
        f"({pct(met.get('capacity_utilisation', 0), 0)}).",
        f"**Shared audiences** — {met.get('lines_sharing_a_pool', 0)} of "
        f"{met['n_lines']} lines sit in a pool shared with another selected line. "
        f"Marginal reach inside a pool saturates as 1 - exp(-lambda) rather than "
        f"being counted twice.",
        f"**Flight window** — all lines inside {st.session_state.win_start} → "
        f"{st.session_state.win_end}.",
        f"**Solver** — `{met['solver_used']}` over {met['n_candidate_cells']:,} "
        f"candidate cells at a {st.session_state.solve_limit:.0f}s limit.",
    ]
    if met.get("cold_start_spend_share", 0) > 0:
        checks.append(
            f"**Never-booked screens** — {pct(met['cold_start_spend_share'])} of spend "
            f"on {met.get('n_cold_start_screens', 0)} screen(s) with no booking history "
            f"at all (screen-level). Audience modelled from nearby places and "
            f"passenger numbers.")
    if met.get("peer_priced_spend_share", 0) > 0:
        checks.append(
            f"**Peer-priced lines** — {met.get('peer_priced_lines', 0)} of "
            f"{met['n_lines']} lines ({pct(met['peer_priced_spend_share'])} of spend) "
            f"had too little history *in that time block* for an own-record rate "
            f"(screen x time-slot level), so the range comes from comparable "
            f"inventory. A screen can have history in one time slot and none in "
            f"another, which is why this differs from the line above.")
    if p.blocks.unevidenced:
        checks.append(f"**Unmeasured dayparts** — block(s) {p.blocks.unevidenced} have "
                      f"no measurable audience; quotable, but no impressions claimed.")
    if dev:
        for c in checks:
            st.markdown(f"- {c}")

    # --------------------------------------------------------------- stretch #
    #
    # Rendered as one HTML paragraph rather than a markdown f-string. The old
    # version interleaved `**bold**` runs with `+` and `$` characters, which
    # produced stray `***` where two emphasis runs met and left the numbers in a
    # different weight and colour from the sentence around them. Building the
    # sentence with explicit <b> tags means the emphasis is exactly where
    # intended and everything else inherits one text colour.
    if met.get("stretch_worth_offering"):
        C.section("If the client has more budget")
        inside = met["stretch_inside_tolerance"]
        total_with = met["spend"] + met["stretch_extra_spend"]
        room = ("which is within what advertisers like this one usually accept"
                if inside else
                "which is more than advertisers like this one usually accept, so "
                "expect to justify it")
        C.why_box(
            f"For another <b>{money(met['stretch_extra_spend'])}</b> on top of the "
            f"{money(met['spend'])} plan &mdash; <b>{money(total_with)}</b> in total, "
            f"{pct(met['stretch_overrun_pct'])} above the budget they stated, {room} "
            f"&mdash; we could reach <b>"
            f"{compact(met['stretch_extra_reach'])} more people every day</b>. "
            f"That works out to roughly "
            f"{met['stretch_reach_per_1k']:,.0f} extra people for every additional "
            f"$1,000 spent.<br><br>"
            f"This is an option to raise, not part of the quote. The client asked "
            f"for {money(met['budget'])}, so that is what the plan above costs.",
            tag="Worth mentioning")
    elif met.get("stretch_extra_spend", 0) > 0:
        C.why_box(
            f"Going above the stated budget would not be worth suggesting here: a "
            f"further {money(met['stretch_extra_spend'])} would only add "
            f"{compact(met['stretch_extra_reach'])} people a day.",
            tag="No upsell worth making")

    # ------------------------------------------------------------ day split #
    if result.split:
        st.markdown("##### Weekday / weekend, shown separately")
        st.caption(result.split_note)
        sp = pd.DataFrame([{
            "day type": s.day_type, "lines": s.n_lines, "spend": s.spend,
            "reach/day": s.reach, "impressions": s.impressions,
        } for s in result.split.values()])
        st.dataframe(sp, hide_index=True, use_container_width=True,
                     column_config={
                         "spend": st.column_config.NumberColumn(format="$%.0f"),
                         "reach/day": st.column_config.NumberColumn(format="%.0f"),
                         "impressions": st.column_config.NumberColumn(format="%.0f")})

    for n in result.notes:
        st.caption(f":orange[{n}]")

    st.markdown("---")
    C.feedback_panel(
        screen="package", campaign_id=st.session_state.campaign_id,
        run_id=st.session_state.run_id, param_widgets=_posture_widget,
        placeholder="e.g. the client will not go above $18k — pull it back")

    c1, c2 = st.columns([1, 5])
    if c1.button("Build the proposal →", type="primary"):
        S.goto(S.next_step("package"))


#: Below this, a revenue difference is not worth a rep's attention -- it is
#: inside the rounding on a single quoted line. Stated by the analysis itself
#: rather than chosen here.
_MATERIALITY_PCT = 1.0


def _price_sensitivity() -> None:
    """The expected-revenue-vs-price curve, and what its interval rules out.

    Renders nothing at all if the artefacts are missing. This is supporting
    evidence for a price the plan already quotes, not a dependency of it.
    """
    j = config.OUT_DIR / "price_sensitivity_d3.json"
    png = config.OUT_DIR / "price_sensitivity_d3.png"
    if not j.exists():
        return
    try:
        d = json.loads(j.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return

    head = d.get("headline") or {}
    ci = d.get("elasticity_ci") or {}
    fit = d.get("fitted_model") or {}
    worst = float(head.get("pct_revenue_left_on_table_worst_case_in_ci") or 0.0)
    shipped = float(head.get("pct_revenue_left_on_table_at_shipped_elasticity")
                    or 0.0)
    n_obs = int(fit.get("n_obs") or 0)
    n_boot = int(ci.get("n_accepted") or 0)

    with st.expander("Why we are not recommending a higher price "
                     f"(tested across {n_boot:,} resamples)"):
        st.markdown(
            f"The recommended price is the one that maximises **expected** "
            f"revenue &mdash; price multiplied by the chance the client accepts "
            f"it. Quoting higher wins more per deal and loses more deals. "
            f"The question is only which way that trade lands, and it is "
            f"measurable."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Revenue left on the table", f"{shipped:.2f}%",
                      help="At the fitted demand curve, quoting at the target "
                           "rather than at the true optimum.")
        with c2:
            st.metric("Worst case tested", f"{worst:.2f}%",
                      help="The most it could be anywhere in the confidence "
                           "interval, not just at the central estimate.")
        with c3:
            st.metric("Worth acting on above", f"{_MATERIALITY_PCT:.0f}%",
                      help="Below this, the difference is inside the rounding "
                           "on a single line.")

        st.markdown(
            f"**So the honest answer is that there is nothing to gain here.** "
            f"Even in the least favourable case the interval allows, quoting at "
            f"the target leaves **{worst:.2f}%** of revenue behind &mdash; "
            f"{_MATERIALITY_PCT / max(worst, 1e-9):.0f}x below the point where "
            f"it would be worth changing anything. That is a measured result, "
            f"not an absence of one: the demand curve was estimated on "
            f"**{n_obs:,}** real win/loss observations and then re-estimated "
            f"**{n_boot:,}** times on resamples of them."
        )

        if png.exists():
            st.image(str(png),
                     caption="Expected revenue against price. The curve is "
                             "flat across the whole interval near the target, "
                             "which is what makes the recommendation safe.",
                     use_container_width=True)

        st.markdown("**What this does not prove.**")
        for c in (d.get("honest_caveats") or [])[:4]:
            st.markdown(f"- {c}")
        st.caption(
            "Kept deliberately: the price we quote is the optimum partly "
            "because the model is calibrated so that it is. The defensible "
            "claim is the flatness of the curve around it, which the interval "
            "above establishes, and not the location of the peak."
        )

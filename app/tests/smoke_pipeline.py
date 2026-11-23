"""End-to-end smoke test: DOCX -> parse -> warmup -> D5 plan -> metrics.

Also the timing harness. The numbers this prints are what the UI is designed
around -- warmup cost, plan cost, and whether the background-warmup strategy
actually removes the stall from the first run.

Run:
    .venv\\Scripts\\python.exe -m app.tests.smoke_pipeline           # brief 4
    .venv\\Scripts\\python.exe -m app.tests.smoke_pipeline 3         # brief 3 (split)
    .venv\\Scripts\\python.exe -m app.tests.smoke_pipeline all
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import config  # noqa: E402
from app.services import docs, engines, pipeline  # noqa: E402
from app.services.metrics import compact, money, pct  # noqa: E402


def run_one(n: int) -> bool:
    path = config.CAMPAIGNS_DIR / f"campaign_{n}.docx"
    print("=" * 78)
    print(f"BRIEF {n}: {path.name}")
    print("=" * 78)

    ex = docs.extract_docx(path.read_bytes())
    lt = engines.light_tables()
    from build_d2_relevance_scorer import parse_brief
    brief = parse_brief(ex.text, lt["cities"], lt["dim_slot"])

    # Briefs 2, 3 and 6 never name a city. Unset, D2 would score all 11,163
    # screens across all three cities -- which is both wrong and slow. The app
    # makes this a blocking clarification on the Interpretation screen; the
    # smoke test defaults it so the run is comparable across briefs.
    if not brief.city_id:
        brief.city_id = "LH"
        print("  [note] no city in brief -> defaulted to LH (UI asks the rep)")

    print(f"  {brief.name!r} city={brief.city_id} budget={brief.budget:,.0f} "
          f"days={brief.duration_days} blocks={brief.preferred_blocks} "
          f"objective={brief.optimise_for} dtw={brief.day_type_weighting!r}")

    import pandas as pd
    start = pd.Timestamp("2026-09-01")
    end = start + pd.Timedelta(days=int(brief.duration_days) - 1)

    orch = engines.get_orchestrator()

    t = time.monotonic()
    res = pipeline.run(
        orch, brief=brief, brief_text=ex.text,
        campaign_start=start.strftime("%Y-%m-%d"),
        campaign_end=end.strftime("%Y-%m-%d"),
        day_type="weekday",
        progress=lambda m: print(f"    .. {m}"),
    )
    took = time.monotonic() - t

    m = res.metrics
    print(f"\n  plan built in {took:.1f}s   (breakdown: "
          + ", ".join(f"{k} {v:.1f}s" for k, v in res.timings.items()) + ")")
    print(f"  presentable   : {res.presentable}   solver={m['solver_used']}")
    print(f"  lines/screens : {m['n_lines']} lines on {m['n_screens']} screens "
          f"({m.get('n_pools', 0)} pools), from {m['n_candidate_cells']:,} candidate cells")
    print(f"  spend         : {money(m['spend'])} of {money(m['budget'])} "
          f"({pct(m['budget_utilisation'])})  -- {m['verdict_state']}")
    print(f"  reach         : {compact(m['reach_per_day'])} distinct people/day")
    print(f"  impressions   : {compact(m['impressions_total'])} over the flight")
    print(f"                  basis: {m['impressions_basis']}")
    print(f"  avg frequency : {m['avg_frequency_per_day']:.2f}x per day")
    print(f"  effective CPM : {money(m['effective_cpm'], 2)}")
    print(f"  QUOTE RANGE   : floor {money(m['package_floor'])} / "
          f"target {money(m['package_target'])} / cap {money(m['package_cap'])}")
    print(f"                  recommended {money(m['package_recommended'])} "
          f"(bands on {pct(m['band_coverage'], 0)} of lines)")
    print(f"  weighted pwin : {pct(m['weighted_p_win'])}   "
          f"expected revenue {money(m['expected_revenue'])}")
    if m.get("relevance_mean") is not None:
        print(f"  relevance     : mean {m.get('relevance_mean', 0):.3f} "
              f"min {m.get('relevance_min', 0):.3f}  of "
              f"{m.get('n_scored_candidates', 0):,} scored candidates")
    if m.get("cold_start_spend_share") is not None:
        print(f"  cold start    : {pct(m['cold_start_spend_share'])} of spend on "
              f"{m.get('n_cold_start_screens', 0)} never-booked screen(s)")
    if m.get("stretch_worth_offering"):
        print(f"  STRETCH       : +{money(m['stretch_extra_spend'])} for "
              f"+{compact(m['stretch_extra_reach'])} people "
              f"({pct(m['stretch_overrun_pct'])} over budget, tolerance "
              f"{pct(m['cohort_tolerance_pct'])}, "
              f"{'inside' if m['stretch_inside_tolerance'] else 'OUTSIDE'})")

    if res.split:
        print(f"\n  DAY-TYPE SPLIT (D2 asked for it, D5 does not do it):")
        for dt, s in res.split.items():
            print(f"    {dt:<8} {s.n_lines:>3} lines  spend {money(s.spend):>10}  "
                  f"reach {compact(s.reach):>8}  impressions {compact(s.impressions):>8}")

    for note in res.notes:
        print(f"  [note] {note}")
    for w in res.plan.warnings:
        print(f"  [warn] {w}")
    for r in res.plan.refusals:
        print(f"  [REFUSAL] {r}")

    # Serialisation must not raise -- it is what gets written to SQLite.
    import json
    blob = json.dumps(pipeline.plan_to_dict(res), default=str)
    print(f"\n  plan_json serialises: {len(blob):,} bytes")
    print()
    return res.presentable


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "4"

    t0 = time.monotonic()
    engines.start_warmup()
    last = ""
    while True:
        st = engines.warmup_state()
        if st.detail != last and st.detail:
            print(f"  [warmup:{st.stage}] {st.detail}")
            last = st.detail
        if st.ready or st.failed:
            break
        time.sleep(0.4)
    st = engines.warmup_state()
    if st.failed:
        print(f"WARMUP FAILED: {st.error}\n{st.traceback}")
        return 1
    print(f"  [warmup] ready in {time.monotonic() - t0:.1f}s  "
          + ", ".join(f"{k} {v:.1f}s" for k, v in st.stage_timings.items())
          + f"   solver={st.solver}\n")

    targets = range(1, 7) if arg == "all" else [int(arg)]
    ok = True
    for n in targets:
        try:
            ok = run_one(n) and ok
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"BRIEF {n} RAISED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

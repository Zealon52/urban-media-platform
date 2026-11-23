"""End-to-end: brief -> plan -> RFP response, for the briefs with the awkward
requirements.

Brief 3 asks for the weekday/weekend split shown separately (D2 detects it, D5
does not act on it -- the app closes that gap). Brief 2 asks for frequency over
coverage. Brief 5 asks for business-traveller frequency separated from
leisure-traveller exposure. Those are the three that a generic template quietly
fails, so they are the ones worth checking.

Run:
    .venv\\Scripts\\python.exe -m app.tests.smoke_rfp 3
    .venv\\Scripts\\python.exe -m app.tests.smoke_rfp all
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import config  # noqa: E402
from app.services import docs, engines, exports, learning, pipeline, rfp  # noqa: E402
from app.services.metrics import compact, money, pct  # noqa: E402

OUT = Path(config.ROOT) / "app_data" / "sample_proposals"


def run_one(n: int) -> bool:
    path = config.CAMPAIGNS_DIR / f"campaign_{n}.docx"
    ex = docs.extract_docx(path.read_bytes())
    lt = engines.light_tables()
    from app.services.interpret import parse
    brief, prov, questions = parse(ex.text, lt["cities"], lt["dim_slot"])
    if not brief.city_id:
        brief.city_id = "LH"

    print("=" * 78)
    print(f"BRIEF {n}: {brief.name}  ({brief.optimise_for}, "
          f"dtw={brief.day_type_weighting!r})")
    print("=" * 78)
    for q in questions[:3]:
        print(f"  [needs you] {q[:120]}")

    import pandas as pd
    start = pd.Timestamp("2026-09-01")
    end = start + pd.Timedelta(days=int(brief.duration_days) - 1)

    orch = engines.get_orchestrator()
    res = pipeline.run(orch, brief=brief, brief_text=ex.text,
                       campaign_start=start.strftime("%Y-%m-%d"),
                       campaign_end=end.strftime("%Y-%m-%d"),
                       day_type="weekday")

    met = res.metrics
    print(f"  plan: {met['n_lines']} lines / {met['n_screens']} screens, "
          f"{money(met['spend'])} of {money(met['budget'])}, "
          f"reach {compact(met['reach_per_day'])}/day, "
          f"impressions {compact(met['impressions_total'])}, "
          f"freq {met['avg_frequency_per_day']:.2f}x, "
          f"solver {met['solver_used']}")
    print(f"  quote band: {money(met['package_floor'])} / "
          f"{money(met['package_target'])} / {money(met['package_cap'])}")

    adj = learning.load(campaign_id=None, scored=res.scored, enabled=True)
    quoted, totals = (None, None)
    if len(res.lines):
        quoted, totals = learning.apply_to_quotes(
            res.lines, res.quotes, adj, int(brief.duration_days))

    has, asks = docs.detect_rfp_asks(ex.text)
    md, template, asks_used = rfp.compose(
        result=res, quoted=quoted, quoted_totals=totals,
        profiles=orch.profiles, asks=asks, learn_note=adj.summary(),
        rep_email="rep@urbanmedia.example")

    extras = rfp.detect_extras(brief, asks_used, res.plan)
    print(f"  RFP template={template}, asks={len(asks_used)}")
    on = [k for k, v in vars(extras).items() if v is True]
    print(f"  extras detected: {', '.join(on) if on else 'none'}")

    ok = True
    # The three canonical sections must always be present.
    for needed in ("1. Recommended inventory", "2. Pricing and commercial terms",
                   "3. Projected delivery", "5. Assumptions and disclosures"):
        if needed not in md:
            print(f"  MISSING SECTION: {needed}")
            ok = False

    # Brief-specific requirements.
    if extras.split_day_type:
        if "Weekday and weekend delivery, shown separately" not in md:
            print("  MISSING: the weekday/weekend split section brief 3 asks for")
            ok = False
        elif res.split:
            print("  split present:", {k: f"{v.n_lines}L "
                                          f"{money(v.spend)} "
                                          f"{compact(v.impressions)}imp"
                                       for k, v in res.split.items()})
        else:
            print("  WARNING: split section rendered but no split solve landed")
    if extras.segment_split and "Audience split" not in md:
        print("  MISSING: the audience segment split brief 5 asks for")
        ok = False
    if extras.weekly_impressions and "Impressions per week" not in md:
        print("  MISSING: weekly impressions brief 1 asks for")
        ok = False
    if extras.per_node_type and "Delivery and affinity by node type" not in md:
        print("  MISSING: per-node-type reasoning brief 6 asks for")
        ok = False
    if extras.radius_logic and "Scoping logic" not in md:
        print("  MISSING: the explicit walking-radius statement brief 4 asks for")
        ok = False

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"campaign_{n}_proposal.md").write_text(md, encoding="utf-8")
    dx = exports.rfp_docx(md, title=brief.name)
    if dx:
        (OUT / f"campaign_{n}_proposal.docx").write_bytes(dx)
    csv = exports.media_plan_csv(res, quoted)
    (OUT / f"campaign_{n}_media_plan.csv").write_bytes(csv)

    print(f"  wrote {len(md):,} chars md"
          + (f", {len(dx):,} bytes docx" if dx else " (docx unavailable)")
          + f", {len(csv):,} bytes csv  -> app_data/sample_proposals/")
    print(f"  {'OK' if ok else 'PROBLEMS'}\n")
    return ok


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "3"
    from app.db import store
    store.init_db()          # learning.load() reads learned_signals
    engines.get_orchestrator()
    wm = engines.warmup_state()
    print(f"warmup: {wm.stage} in {wm.elapsed:.1f}s, solver={wm.solver}\n")

    targets = range(1, 7) if arg == "all" else [int(x) for x in arg.split(",")]
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

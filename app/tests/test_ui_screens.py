"""Renders every screen through Streamlit's own AppTest harness.

An HTTP GET against a running Streamlit server only returns the JS shell -- the
actual script executes over a websocket, so "the port answers 200" proves
nothing about whether a screen renders. AppTest runs the real script in-process
and surfaces any exception, which is the only cheap way to prove all six screens
work without a browser.

Screens 03-06 need a real run, so this executes ONE genuine D5 plan and injects
it into session state, then renders each screen against it. That is deliberately
the same object the app itself would hold.

Run:
    .venv\\Scripts\\python.exe -m app.tests.test_ui_screens
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streamlit.testing.v1 import AppTest  # noqa: E402

from app import config  # noqa: E402

APP = str(config.ROOT / "app" / "main.py")
FAILS: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  [ok  ] {label}")
    else:
        print(f"  [FAIL] {label}" + (f"  -- {detail}" if detail else ""))
        FAILS.append(label)


def _exc(at) -> str:
    if not at.exception:
        return ""
    out = []
    for e in at.exception:
        out.append(f"{getattr(e, 'value', '')}\n{getattr(e, 'stack_trace', '')}")
    return "\n".join(out)


def test_boot() -> None:
    print("\nboot + intake")
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    check(not at.exception, "app boots with no exception", _exc(at)[:1500])
    # The radio's own label is collapsed, so assert on OPTIONS -- and only on
    # the intake radio, not the sidebar's mode switch.
    intake_radios = [r for r in at.radio
                     if any("Upload" in str(o) for o in (r.options or []))]
    check(bool(intake_radios), "intake input-mode radio present")
    if intake_radios:
        opts = [str(o) for o in intake_radios[0].options]
        check(len(opts) == 4 and any("past lead" in o for o in opts),
              "intake offers all four input modes", str(opts))
    check(len(at.sidebar.selectbox) >= 1, "sidebar renders the step nav")

    # Switch to the sample-brief mode and confirm the six briefs are offered.
    at.radio[0].set_value("Sample brief").run()
    check(not at.exception, "switching to sample-brief mode renders",
          _exc(at)[:1500])
    boxes = [sb for sb in at.selectbox if "Brief" in str(sb.label)]
    check(bool(boxes), "sample brief selector present")
    if boxes:
        check(len(boxes[0].options) == 6, "all six sample briefs offered",
              str(len(boxes[0].options)))


def test_interpretation() -> None:
    print("\ninterpretation")
    from app.services import docs
    ex = docs.extract_docx((config.CAMPAIGNS_DIR / "campaign_3.docx").read_bytes())

    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state.step = "interpret"
    at.session_state.brief_text = ex.text
    at.session_state.campaign_id = None
    at.run()
    check(not at.exception, "interpretation renders from a real brief",
          _exc(at)[:2000])

    b = at.session_state.brief
    check(b is not None and b.name.startswith("Loom"), "brief parsed into state",
          str(getattr(b, "name", None)))
    check(b.day_type_weighting == "report_split",
          "brief 3's split requirement carried into the UI state")

    # Brief 3 names no city. That is no longer a blocker: it is inferred where
    # possible, and otherwise the plan spans all markets. Either way the rep can
    # proceed, and is told what happened.
    run_btns = [btn for btn in at.button if "Build the plan" in str(btn.label)]
    check(bool(run_btns) and not run_btns[0].disabled,
          "a brief with no city does NOT block the run any more")
    txt = " ".join(str(m.value) for m in at.markdown)
    check("Check this" in txt or "all three" in txt or "Planning across" in txt,
          "and the rep is told how the city was resolved")
    check(at.session_state.provenance.get("city_id") == "inferred",
          "city is badged as inferred",
          str(at.session_state.provenance.get("city_id")))

    # A rep override must still take precedence and be badged as theirs.
    city = [s for s in at.selectbox if s.label == "city"]
    check(bool(city), "city selector present")
    if city:
        current = at.session_state.brief.city_id
        override = "ACS" if current != "ACS" else "DAT"
        city[0].set_value(override).run()
        check(not at.exception, "changing the city re-renders", _exc(at)[:1500])
        check(at.session_state.brief.city_id == override,
              "the rep's choice overrides the inference")
        check(at.session_state.provenance.get("city_id") == "rep_edited",
              "and the field is re-badged as rep-edited",
              str(at.session_state.provenance.get("city_id")))


def _real_run():
    """One genuine plan, built outside Streamlit and injected."""
    from app.db import store
    from app.services import docs, engines, learning, pipeline
    from app.services.interpret import parse
    store.init_db()
    ex = docs.extract_docx((config.CAMPAIGNS_DIR / "campaign_4.docx").read_bytes())
    lt = engines.light_tables()
    brief, prov, _q = parse(ex.text, lt["cities"], lt["dim_slot"])
    brief.city_id = brief.city_id or "LH"
    orch = engines.get_orchestrator()
    res = pipeline.run(orch, brief=brief, brief_text=ex.text,
                       campaign_start="2026-09-01", campaign_end="2026-09-15",
                       day_type="weekday")
    adj = learning.load(campaign_id=None, scored=res.scored, enabled=True)
    quoted, totals = learning.apply_to_quotes(res.lines, res.quotes, adj,
                                              int(brief.duration_days))
    cid = store.create_campaign(brief_text=ex.text, source_mode="sample",
                                source_label="campaign_4.docx")
    # Persist a real brief version + run so screen 06 exercises the genuine
    # persistence path (rfp_responses.run_id is NOT NULL) rather than a
    # half-populated state the app itself would never produce.
    import dataclasses
    bv = store.add_brief_version(
        campaign_id=cid, brief_json=dataclasses.asdict(brief), provenance=prov,
        campaign_start="2026-09-01", campaign_end="2026-09-15",
        day_type="weekday", allow_unevidenced=False, origin="parser")
    run = store.add_run(
        campaign_id=cid, brief_version_id=bv["id"],
        plan_json=pipeline.plan_to_dict(res), metrics=dict(res.metrics),
        solver_used=res.metrics.get("solver_used", "?"),
        presentable=res.presentable, spend=res.metrics.get("spend", 0.0),
        reach=res.metrics.get("reach_per_day", 0.0),
        impressions=res.metrics.get("impressions_total", 0.0),
        n_lines=res.metrics.get("n_lines", 0),
        n_screens=res.metrics.get("n_screens", 0),
        duration_ms=res.duration_ms, refusals=[], warnings=[],
        learning_applied=False)
    return ex, brief, prov, res, adj, quoted, totals, cid, run["id"]


def test_result_screens() -> None:
    print("\nresult screens (one real D5 run, injected)")
    ex, brief, prov, res, adj, quoted, totals, cid, run_id = _real_run()
    print(f"  run: {res.metrics['n_lines']} lines, "
          f"{res.metrics['n_screens']} screens, presentable={res.presentable}")

    from app import config as _cfg
    cases = [(_cfg.MODE_CLIENT, "package"), (_cfg.MODE_CLIENT, "recommend"),
             (_cfg.MODE_DEVELOPER, "audience"), (_cfg.MODE_DEVELOPER, "relevance"),
             (_cfg.MODE_DEVELOPER, "package"), (_cfg.MODE_DEVELOPER, "recommend")]
    for mode, step in cases:
        at = AppTest.from_file(APP, default_timeout=300)
        at.session_state.mode = mode
        at.session_state.step = step
        at.session_state.brief_text = ex.text
        at.session_state.brief = brief
        at.session_state.provenance = prov
        at.session_state.result = res
        at.session_state.adjustments = adj
        at.session_state.quoted = quoted
        at.session_state.quoted_totals = totals
        at.session_state.campaign_id = cid
        at.session_state.run_id = run_id
        at.session_state.revision = 1
        at.session_state.source_label = "campaign_4.docx"
        at.session_state.llm_enabled = False     # keep the test offline and free
        at.run()
        check(not at.exception, f"[{mode}] screen '{step}' renders",
              _exc(at)[:2500])
        import re as _re
        raw = " ".join(str(m.value) for m in at.markdown)
        # Strip the injected stylesheet before scanning for jargon: CSS hex
        # colours like #B9C3D2 contain "D2" and produced a false positive.
        txt = _re.sub(r"<style>.*?</style>", " ", raw, flags=_re.S)

        # Deliverable codes must appear on NO screen, in EITHER mode -- they mean
        # nothing to a rep and they leak into screenshots and demos. The wider
        # jargon list is only enforced in the sales view, where a developer
        # legitimately needs to see e.g. capacity language.
        hard = [r"\bD[1-6]\b"]
        soft = [r"\bcp_sat\b", r"\bCP-SAT\b", r"\bdaypart", r"\bcold[- ]start",
                r"\bsolver\b", r"\bpool_id\b", r"\bscreen_id\b",
                r"\brelevance score\b"]
        patterns = hard + (soft if mode == _cfg.MODE_CLIENT else [])
        leaked = []
        for term in patterns:
            for m in _re.finditer(term, txt):
                s, e = max(0, m.start() - 55), min(len(txt), m.end() + 55)
                leaked.append(f"{term!r} in ...{txt[s:e]}...")
                break
        check(not leaked, f"[{mode}] '{step}' leaks no engine jargon",
              " || ".join(leaked[:4]))

        if step == "package":
            check("negotiating room" in txt.lower() or "What to quote" in txt
                  or any("Don" in str(m.label) for m in at.metric)
                  or any("Negotiating" in str(m.label) for m in at.metric),
                  f"[{mode}] package screen shows the quote range")
        if step == "recommend":
            expected = 7 if mode == _cfg.MODE_DEVELOPER else 5
            check(len(at.tabs) == expected,
                  f"[{mode}] proposal renders {expected} tabs", str(len(at.tabs)))
            check("Why these locations" in txt,
                  f"[{mode}] the plain-language reasoning is present")
            check(any("Subject" in str(t.label) for t in at.text_input),
                  f"[{mode}] the email draft is built")


def main() -> int:
    print("=" * 74)
    print("D6 UI screen tests (Streamlit AppTest)")
    print("=" * 74)
    for t in (test_boot, test_interpretation, test_result_screens):
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  [RAISED] {t.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            FAILS.append(t.__name__)
    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

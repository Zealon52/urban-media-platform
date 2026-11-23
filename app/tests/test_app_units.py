"""Unit tests for the app's non-UI logic: imports, DB round-trips, the feedback
router's validation boundary, learning decay, and RFP detection.

These are the parts a Streamlit smoke run would only exercise by accident.

Run:
    .venv\\Scripts\\python.exe -m app.tests.test_app_units
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import config  # noqa: E402

# Point the DB at a scratch file BEFORE store is imported anywhere meaningful.
_TMP = Path(tempfile.mkdtemp(prefix="d6test_"))
config.APP_DATA_DIR = _TMP
config.DB_PATH = _TMP / "test.db"

from app.db import store  # noqa: E402
from app.services import docs, feedback as fb, interpret, learning  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  [ok  ] {label}")
    else:
        print(f"  [FAIL] {label}" + (f"  -- {detail}" if detail else ""))
        FAILS.append(label)


# --------------------------------------------------------------------------- #

def test_imports() -> None:
    print("\nimports")
    import app.main  # noqa: F401  -- would raise on any syntax/name error
    print("  [ok  ] app.main and every ui/service module import")


def test_db() -> None:
    print("\ndatabase")
    store.init_db()
    store.init_db()   # idempotent
    check(True, "schema creates and is re-runnable")

    data = b"PK\x03\x04 fake docx bytes"
    d1 = store.upsert_document(filename="a.docx", mime="x", data=data,
                               extracted_text="hello", extractor="python-docx")
    d2 = store.upsert_document(filename="renamed.docx", mime="x", data=data,
                               extracted_text="hello", extractor="python-docx")
    check(d1 == d2, "documents dedupe on content sha256, not filename",
          f"{d1} vs {d2}")
    check(store.document_bytes(d1) == data, "BLOB round-trips byte-exact")

    cid = store.create_campaign(brief_text="t", source_mode="upload",
                                source_label="a.docx", document_id=d1)
    bv = store.add_brief_version(campaign_id=cid, brief_json={"budget": 1},
                                 provenance={"budget": "read"},
                                 campaign_start="2026-09-01",
                                 campaign_end="2026-09-30", day_type="weekday",
                                 allow_unevidenced=False, origin="parser")
    check(bv["version"] == 1, "brief versions start at 1")
    bv2 = store.add_brief_version(campaign_id=cid, brief_json={"budget": 2},
                                  provenance={}, campaign_start=None,
                                  campaign_end=None, day_type="weekday",
                                  allow_unevidenced=False, origin="rep_edit")
    check(bv2["version"] == 2, "brief versions increment per campaign")

    r1 = store.add_run(campaign_id=cid, brief_version_id=bv["id"],
                       plan_json={"a": 1}, metrics={"spend": 10.0},
                       solver_used="cp_sat", presentable=True, spend=10.0,
                       reach=5.0, impressions=50.0, n_lines=2, n_screens=2,
                       duration_ms=1234, refusals=[], warnings=[],
                       learning_applied=False)
    r2 = store.add_run(campaign_id=cid, brief_version_id=bv2["id"],
                       plan_json={"a": 2}, metrics={"spend": 20.0},
                       solver_used="cp_sat", presentable=True, spend=20.0,
                       reach=6.0, impressions=60.0, n_lines=3, n_screens=3,
                       duration_ms=99, refusals=[], warnings=[],
                       learning_applied=True)
    check((r1["revision"], r2["revision"]) == (1, 2), "runs revision per campaign")

    f1 = store.add_feedback(campaign_id=cid, run_id=r1["id"], screen="package",
                            mode="nl", raw_text="too expensive")
    store.add_feedback(campaign_id=cid, run_id=r1["id"], screen="relevance",
                       mode="drop_screen", payload={"screen_ids": ["LH-SCR-0001"]})
    check(len(store.queued_feedback(cid)) == 2, "feedback queues")
    store.mark_feedback_applied([f1], run_id=r2["id"], routed_by="rules")
    check(len(store.queued_feedback(cid)) == 1, "applying removes from the queue")

    store.log_llm_call(purpose="feedback_router", cache_key="k", cached=False,
                       deployment="d", prompt_tokens=1000, completion_tokens=200,
                       est_cost_usd=0.00027, ok=True)
    store.llm_cache_put("k", "feedback_router", "cached text")
    check(store.llm_cache_get("k") == "cached text", "llm cache round-trips")
    summ = store.llm_spend_summary()
    check(summ["n_billable"] == 1 and summ["spend_usd"] > 0,
          "llm ledger totals", str(summ))

    store.log(campaign_id=cid, actor="rep", event="test", detail="d")
    check(len(store.decision_log(cid)) == 1, "decision log appends")


def test_router_validation() -> None:
    print("\nfeedback router -- validation boundary")

    class B:
        name = "T"; city_id = "LH"; budget = 10000.0; duration_days = 20
        objective = "x"; audience_text = "office professionals"
        preferred_blocks = [3]; optimise_for = "reach"; day_type_weighting = ""
        exclude_market_tier: list = []; prefer_market_tier: list = []
        radius_only = False; radius_anchor_hint = ""
        must_have: list = []; exclude: list = []; notes: list = []

    b = B()
    known = {"LH-SCR-0001", "LH-SCR-0002"}

    # A hostile / hallucinated payload must be rejected field by field.
    bad = {
        "brief_edits": {
            "budget": "lots",                  # unparseable
            "duration_days": -5,               # non-positive
            "city_id": "PARIS",                # not a real city
            "optimise_for": "vibes",           # not in enum
            "preferred_blocks": [0, 9, "x"],   # all out of range
            "exclude_market_tier": ["gold"],   # not a real tier
            "must_have": [["bus", "back"]],    # NOT routable by design
            "secret_field": 1,                 # unknown
        },
        "drop_screen_ids": ["LH-SCR-9999"],    # not in the candidate set
        "day_type": "tuesday",
        "price_posture": "nuke_it",
        "loss_reasons": ["made_up_reason", "price_too_high"],
    }
    a = fb._validate(bad, known)
    check(a.brief_edits == {}, "every invalid brief edit is dropped",
          str(a.brief_edits))
    check("must_have" in " ".join(a.rejected),
          "must_have is refused as non-routable")
    check(a.drop_screen_ids == [], "unknown screen ids are refused")
    check(a.day_type is None, "invalid day_type refused")
    check(a.price_posture == "hold", "invalid posture falls back to hold")
    check(a.loss_reasons == ["price_too_high"],
          "loss reasons filtered to the real vocabulary", str(a.loss_reasons))
    check(len(a.rejected) >= 7, f"rejections are recorded ({len(a.rejected)})")

    good = {"brief_edits": {"budget": "12,500", "city_id": "acs",
                            "preferred_blocks": [3, "5"], "optimise_for": "FREQUENCY",
                            "exclude_market_tier": "value"},
            "drop_screen_ids": ["LH-SCR-0002"], "day_type": "weekend",
            "price_posture": "toward_floor", "want_split": True}
    a = fb._validate(good, known)
    check(a.brief_edits["budget"] == 12500.0, "budget coerces from '12,500'")
    check(a.brief_edits["city_id"] == "ACS", "city upper-cases")
    check(a.brief_edits["preferred_blocks"] == [3, 5], "blocks coerce and sort")
    check(a.brief_edits["optimise_for"] == "frequency", "enum lower-cases")
    check(a.brief_edits["exclude_market_tier"] == ["value"], "scalar -> list")
    check(a.drop_screen_ids == ["LH-SCR-0002"], "known screen id accepted")
    check(a.want_split is True and a.day_type == "weekend", "flags pass through")

    # Deterministic rules, no LLM.
    items = [
        {"screen": "package", "mode": "nl",
         "raw_text": "way too expensive, and drop LH-SCR-0001",
         "payload_json": None, "loss_reason": None},
        {"screen": "audience", "mode": "nl",
         "raw_text": "client wants students in the mix too",
         "payload_json": None, "loss_reason": None},
        {"screen": "package", "mode": "nl",
         "raw_text": "prioritise frequency over broad coverage",
         "payload_json": None, "loss_reason": None},
    ]
    a = fb.route(items, brief=b, day_type="weekday", known_screen_ids=known,
                 use_llm=False)
    check(a.price_posture == "toward_floor", "rules catch a price objection")
    check("price_too_high" in a.loss_reasons, "price objection -> loss reason")
    check("LH-SCR-0001" in a.drop_screen_ids, "rules extract a screen id")
    check("students" in a.brief_edits.get("audience_text", "").lower(),
          "audience addition preserves the original text",
          a.brief_edits.get("audience_text", ""))
    check("office professionals" in a.brief_edits.get("audience_text", "").lower(),
          "original audience is NOT lost when adding")
    check(a.brief_edits.get("optimise_for") == "frequency",
          "frequency intent detected")
    check(a.routed_by == "rules", "routed_by reports the deterministic path")

    new_b, changed = fb.apply_to_brief(b, a)
    check(b.optimise_for == "reach", "the original brief is never mutated")
    check(new_b.optimise_for == "frequency", "the copy carries the change")
    check("optimise_for" in changed, "changed fields are reported")


def test_posture() -> None:
    print("\nprice posture")

    class Q:
        floor, target, cap, recommended = 40.0, 50.0, 70.0, 50.0

    v, why = fb.quote_at_posture(Q(), "toward_floor")
    check(v == 45.0, "a price objection goes to the floor/target MIDPOINT, "
                     "not straight to the floor", str(v))
    v, _ = fb.quote_at_posture(Q(), "toward_cap")
    check(v == 60.0, "headroom goes to the target/cap midpoint", str(v))
    v, _ = fb.quote_at_posture(Q(), "hold")
    check(v == 50.0, "hold returns D3's recommendation unchanged")


def test_learning() -> None:
    print("\nlearning -- decay, threshold, caps")
    check(abs(learning.recency_weight(0) - 1.0) < 1e-9, "fresh signal weighs 1.0")
    check(abs(learning.recency_weight(config.LEARNING_HALFLIFE_DAYS) - 0.5) < 1e-9,
          "one halflife weighs 0.5")
    check(learning.recency_weight(config.LEARNING_HALFLIFE_DAYS * 4) < 0.07,
          "old signals decay away")

    import pandas as pd
    scored = pd.DataFrame([
        {"screen_id": "S1", "zone_id": "Z1", "screen_type": "bus_stop"},
        {"screen_id": "S2", "zone_id": "Z2", "screen_type": "metro_station"},
    ])
    attrs = learning.screen_attributes(scored)
    check(attrs["S1"]["zone"] == "Z1", "screen attributes resolve")

    cid = store.create_campaign(brief_text="t", source_mode="paste")
    other = store.create_campaign(brief_text="t2", source_mode="paste")

    # Two events: below the 3-event threshold, so nothing should apply.
    for _ in range(2):
        learning.record(campaign_id=other, loss_reasons=["price_too_high"],
                        affected_screen_ids=["S1"], attrs=attrs)
    adj = learning.load(campaign_id=cid, scored=scored, enabled=True)
    mult, why = adj.price_multiplier("S1")
    check(mult == 1.0, "below the evidence threshold nothing is applied", str(mult))
    check(adj.n_applied == 0, "and nothing reports as applied")

    # A third pushes it over.
    learning.record(campaign_id=other, loss_reasons=["price_too_high"],
                    affected_screen_ids=["S1"], attrs=attrs)
    adj = learning.load(campaign_id=cid, scored=scored, enabled=True)
    mult, why = adj.price_multiplier("S1")
    check(mult < 1.0, "corroborated price feedback lowers the quote", str(mult))
    check(bool(why), "and carries its explanation")

    # Own-campaign feedback must not feed back to itself.
    for _ in range(6):
        learning.record(campaign_id=cid, loss_reasons=["price_too_high"],
                        affected_screen_ids=["S2"], attrs=attrs)
    adj_self = learning.load(campaign_id=cid, scored=scored, enabled=True)
    m2, _ = adj_self.price_multiplier("S2")
    check(m2 == 1.0, "a campaign's own feedback is excluded from its own run",
          str(m2))

    # Cap holds under a pile of evidence.
    for _ in range(80):
        learning.record(campaign_id=other, loss_reasons=["price_too_high"],
                        affected_screen_ids=["S1"], attrs=attrs)
    adj = learning.load(campaign_id=cid, scored=scored, enabled=True)
    mult, _ = adj.price_multiplier("S1")
    check(mult >= 1.0 - config.LEARNING_MAX_PRICE_SHIFT - 1e-9,
          f"price shift is capped at {config.LEARNING_MAX_PRICE_SHIFT:.0%}", str(mult))

    adj_off = learning.load(campaign_id=cid, scored=scored, enabled=False)
    check(adj_off.price_multiplier("S1")[0] == 1.0, "disabling learning is absolute")

    # A reason with no teaching rule must teach nothing.
    ids = learning.record(campaign_id=other, loss_reasons=["client_went_quiet"],
                          affected_screen_ids=["S1"], attrs=attrs)
    check(ids == [], "a non-teaching loss reason records no signal")

    # REGRESSION: an AGED set of exactly MIN_EVIDENCE events must still clear the
    # gate, with a smaller magnitude. Gating on the decayed weight instead of the
    # event count made this unreachable -- 3 events one day old weigh 2.954,
    # which never clears a threshold of 3 again.
    aged = store.create_campaign(brief_text="aged", source_mode="paste")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(
        timespec="seconds")
    with store.connect() as conn:
        for _ in range(config.LEARNING_MIN_EVIDENCE):
            conn.execute(
                """INSERT INTO learned_signals (scope, scope_key, signal, direction,
                          source_feedback_id, campaign_id, note, created_at)
                   VALUES ('zone','Z9','price_too_high',-1.0,NULL,?,'aged',?)""",
                (aged, old_ts))
    scored9 = pd.DataFrame([{"screen_id": "S9", "zone_id": "Z9",
                             "screen_type": "bus"}])
    adj_aged = learning.load(campaign_id=cid, scored=scored9, enabled=True)
    m_aged, why_aged = adj_aged.price_multiplier("S9")
    check(m_aged < 1.0, "3 events aged 90 days STILL clear the corroboration gate",
          str(m_aged))
    # 90 days is exactly two halflives, so the shift must be ~1/4 of a fresh
    # 3-event shift: 3 * 0.5**2 * 2.5% = 1.875% against 7.5%.
    aged_shift, fresh_shift = 1.0 - m_aged, config.LEARNING_MIN_EVIDENCE * 0.025
    check(abs(aged_shift - fresh_shift / 4.0) < 1e-4,
          "and its magnitude has decayed by exactly two halflives",
          f"shift {aged_shift:.5f}, expected {fresh_shift / 4.0:.5f}")
    check(bool(why_aged), "and they still explain themselves")


def test_metrics_cold_start() -> None:
    """REGRESSION: has_booking_history is a pandas nullable boolean.

    `pandas_bool is not True` is ALWAYS true, because a pandas scalar is never
    the `True` singleton. That reported 100% of spend as cold-start on every
    plan, contradicting D2's own per-screen reason text and D3's rate_card_basis
    inside the same generated proposal.
    """
    print("\nmetrics -- cold-start accounting")
    import pandas as pd
    from app.services import metrics

    lines = pd.DataFrame([
        {"screen_id": "A", "time_block_id": 3, "slots_booked_per_day": 1,
         "price": 100.0, "people_reach": 1000.0, "pool_id": "p1"},
        {"screen_id": "B", "time_block_id": 3, "slots_booked_per_day": 1,
         "price": 100.0, "people_reach": 1000.0, "pool_id": "p2"},
        {"screen_id": "C", "time_block_id": 3, "slots_booked_per_day": 1,
         "price": 100.0, "people_reach": 1000.0, "pool_id": "p3"},
        {"screen_id": "D", "time_block_id": 3, "slots_booked_per_day": 1,
         "price": 100.0, "people_reach": 1000.0, "pool_id": "p4"},
    ])
    scored = pd.DataFrame([
        {"screen_id": "A", "relevance_score": 0.6, "confidence": 0.9,
         "has_booking_history": True},
        {"screen_id": "B", "relevance_score": 0.6, "confidence": 0.9,
         "has_booking_history": True},
        {"screen_id": "C", "relevance_score": 0.6, "confidence": 0.9,
         "has_booking_history": False},
        {"screen_id": "D", "relevance_score": 0.6, "confidence": 0.9,
         "has_booking_history": None},          # unknown -> treat as cold
    ]).astype({"has_booking_history": "boolean"})

    class Verdict:
        state, reason = "within_budget", "ok"

    class Cohort:
        variance_pct = 0.2

    class Blocks:
        unevidenced: list = []

    class Brief:
        budget, duration_days, optimise_for = 4000.0, 10, "reach"

    class Plan:
        day_type = "weekday"
        at_budget = {"solver_used": "cp_sat", "n_candidate_cells": 10}
        verdict, cohort, blocks, refusals, warnings = Verdict(), Cohort(), Blocks(), [], []
        brief = Brief()
        presentable = True
        stretch = None
        stretch_gain = (0.0, 0.0, 0.0)
        stretch_overrun_pct = 0.0
        stretch_worth_offering = False
        spend, reach = 4000.0, 4000.0

        @property
        def lines(self):
            return lines

    met = metrics.compute(plan=Plan(), cells=None, quotes={}, scored=scored,
                          brief=Brief())
    check(met["n_cold_start_screens"] == 2,
          "exactly the 2 no-history screens count as cold start (not all 4)",
          str(met["n_cold_start_screens"]))
    check(abs(met["cold_start_spend_share"] - 0.5) < 1e-9,
          "cold-start spend share is 50%, not 100%",
          str(met["cold_start_spend_share"]))


def test_impressions_basis() -> None:
    """REGRESSION: the saturating and linear formulas must not be mixed.

    reach = people * (1 - exp(-lambda)) saturates; impressions = people * lambda
    is linear in slots. When D4's cells frame is absent the fallback is a stated
    LOWER BOUND, and the basis string must say so -- the weekday/weekend split
    reported 3.72M against the headline 8.14M for the identical plan because it
    quietly used the fallback.
    """
    print("\nmetrics -- impressions basis")
    import pandas as pd
    from app.services import metrics

    # Numbers chosen so the two paths DISAGREE -- otherwise the test would pass
    # even if the implementation swapped them.
    lines = pd.DataFrame([{"screen_id": "A", "time_block_id": 3,
                           "slots_booked_per_day": 2, "people_reach": 500.0}])
    cells = pd.DataFrame([{"screen_id": "A", "time_block_id": 3, "people": 2000.0,
                           "cycles_per_slot": 0.5}])

    with_cells = metrics.impressions_from_lines(lines, cells, duration_days=10)
    # lambda = 2 * 0.5 = 1.0 -> 2000 * 1.0 = 2000/day -> 20,000 over 10 days
    check(abs(with_cells - 20_000.0) < 1e-6,
          "with cells: people x slots x cycles_per_slot x days", str(with_cells))
    check("share-of-voice" in metrics.impressions_basis(cells),
          "and the basis names the D4 model")

    without = metrics.impressions_from_lines(lines, None, duration_days=10)
    # 500 * 2 * 10 = 10,000 -- a genuine lower bound, half the modelled figure
    check(abs(without - 10_000.0) < 1e-6,
          "fallback: reach x slots x days", str(without))
    check(without < with_cells,
          "the fallback is a LOWER bound, not an alternative estimate",
          f"{without} vs {with_cells}")
    check("lower bound" in metrics.impressions_basis(None),
          "and the basis explicitly says lower bound")
    check(metrics.impressions_basis(cells) != metrics.impressions_basis(None),
          "the two bases are distinguishable, so a mismatch is visible")


def test_city_inference() -> None:
    """Three of six sample briefs never name a city. Infer it from real signals,
    and when nothing points anywhere, plan across all markets rather than
    guessing or refusing."""
    print("\ncity inference")
    from app.services.engines import light_tables
    lt = light_tables()
    cities, screens = lt["cities"], lt.get("screens")

    # Premium positioning -> the premium-tier market (LH).
    cid, conf, why = interpret.infer_city(
        "A mass-premium electric sedan for affluent urban professionals. Metro "
        "platform boards in the primary financial district.", cities, screens)
    check(cid == "LH", "premium + metro language infers the premium market",
          f"{cid} @ {conf:.0%}")
    check(bool(why), "and states which signal fired")
    if why:
        print(f"         -> {why[0][:110]}")

    # Value/price-led -> the value-tier market (ACS).
    cid, conf, why = interpret.infer_city(
        "Budget-conscious, price-sensitive mass-market shoppers looking for "
        "affordable everyday value.", cities, screens)
    check(cid == "ACS", "value-led language infers the value market",
          f"{cid} @ {conf:.0%}")

    # Nothing to go on -> no guess.
    cid, conf, why = interpret.infer_city(
        "We would like to advertise our new product next month. Budget is "
        "flexible. Please send options.", cities, screens)
    check(cid == "", "a signal-free brief yields NO city rather than a guess",
          f"got {cid!r}")

    # REGRESSION: template boilerplate must not be read as a client signal.
    # Every sample brief is headed "1. Executive Summary & Objective", and
    # treating "executive" as premium language inferred a premium market for a
    # gym-membership campaign.
    cid, conf, why = interpret.infer_city(
        "Company Name\tNorthwind Fitness\n"
        "Target Audience\tYoung professionals aged 25-40\n"
        "1. Executive Summary & Objective\n"
        "We are opening new gyms and want to drive trial.",
        cities, screens)
    check(cid == "", "a boilerplate 'Executive Summary' heading infers nothing",
          f"got {cid!r} because {why[:1]}")

    # And the parser turns that into an all-markets plan, not a block.
    b, prov, qs = interpret.parse(
        "Company Name\tAcme\nCampaign Budget\tUSD 20,000\n"
        "Campaign Duration\t10 Days\nWe want to advertise our product.",
        cities, lt["dim_slot"], screens_df=screens)
    check(b.city_id == "", "brief with no signal leaves city unset")
    check(prov["city_id"] == "inferred",
          "and it is NOT flagged as blocking", prov.get("city_id"))
    check(any("all three" in q for q in qs),
          "and the rep is told the plan will span all markets")


def test_rfp_detection() -> None:
    print("\nRFP detection")
    for n in range(1, 7):
        p = config.CAMPAIGNS_DIR / f"campaign_{n}.docx"
        if not p.exists():
            continue
        ex = docs.extract_docx(p.read_bytes())
        has, asks = docs.detect_rfp_asks(ex.text)
        check(has and len(asks) == 3, f"campaign_{n}: 3 RFP asks found",
              f"got {len(asks)}")

    has, asks = docs.detect_rfp_asks(
        "Hi, we want to advertise in Las Hackland next month, budget $20k. Thanks!")
    check(not has and not asks, "a plain email has no RFP block -> default template")

    from app.services.rfp import classify_ask
    covered = set()
    for a in ["A curated shortlist of screens ranked by affluent-commuter affinity.",
              "An optimal price recommendation reflecting premium-node demand.",
              "Projected weekly impressions and test-drive-booking potential."]:
        covered |= set(classify_ask(a))
    check(covered == {"inventory", "pricing", "delivery"},
          "the three canonical asks map onto all three sections", str(covered))


def test_interpret_provenance() -> None:
    print("\ninterpretation provenance")
    from app.services.engines import light_tables
    lt = light_tables()

    p4 = config.CAMPAIGNS_DIR / "campaign_4.docx"
    ex = docs.extract_docx(p4.read_bytes())
    brief, prov, questions = interpret.parse(ex.text, lt["cities"], lt["dim_slot"])
    check(prov["budget"] == "read", "budget is labelled read from the brief")
    check(prov["city_id"] == "read", "city is labelled read when present")
    check(prov["must_have"] == "inferred", "keyword fields are labelled inferred")

    p3 = config.CAMPAIGNS_DIR / "campaign_3.docx"
    ex3 = docs.extract_docx(p3.read_bytes())
    b3, prov3, q3 = interpret.parse(ex3.text, lt["cities"], lt["dim_slot"],
                                    screens_df=lt.get("screens"))
    check(prov3["city_id"] == "inferred",
          "a brief with no city is inferred, never silently left blank",
          prov3.get("city_id"))
    check(any("city" in q.lower() or "market" in q.lower() for q in q3),
          "and the rep is told what happened and why")
    check(b3.day_type_weighting == "report_split",
          "brief 3's weekday/weekend split requirement is detected")


def main() -> int:
    print("=" * 74)
    print("D6 app unit tests")
    print(f"scratch db: {config.DB_PATH}")
    print("=" * 74)
    for t in (test_imports, test_db, test_router_validation, test_posture,
              test_learning, test_metrics_cold_start, test_impressions_basis,
              test_city_inference, test_rfp_detection, test_interpret_provenance):
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

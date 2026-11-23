"""Parity check: does app/services/docs.py produce text that D2's parser reads
the SAME WAY as the committed, known-good campaign_4.docx.parsed.txt?

This is the highest-value test in the app. If DOCX extraction drops the header
table, `budget` and `duration_days` come back 0 and every plan refuses at the
budget stage -- with no visible cause, because the refusal is honest and the
brief "looks" parsed. Comparing field-by-field against a reference file that is
already known to parse correctly catches that immediately.

Run:
    .venv\\Scripts\\python.exe -m app.tests.test_intake_parity
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import config  # noqa: E402
from app.services import docs  # noqa: E402
from app.services.engines import light_tables  # noqa: E402

from build_d2_relevance_scorer import parse_brief  # noqa: E402

REFERENCE_TXT = config.ROOT / "campaign_4.docx.parsed.txt"
REFERENCE_DOCX = config.CAMPAIGNS_DIR / "campaign_4.docx"

# Fields that legitimately may differ between the reference text dump and our
# extraction, because they depend on incidental line-wrapping rather than on
# any value we care about.
TOLERATED = {"notes"}


def _brief_from_text(text: str):
    lt = light_tables()
    return parse_brief(text, lt["cities"], lt["dim_slot"])


def main() -> int:
    failures: list[str] = []

    if not REFERENCE_TXT.exists():
        print(f"FAIL reference missing: {REFERENCE_TXT}")
        return 1
    if not REFERENCE_DOCX.exists():
        print(f"FAIL docx missing: {REFERENCE_DOCX}")
        return 1

    ref_brief = _brief_from_text(REFERENCE_TXT.read_text(encoding="utf-8-sig"))

    extraction = docs.extract_docx(REFERENCE_DOCX.read_bytes())
    if extraction.warning:
        print(f"  extractor warning: {extraction.warning}")
    if extraction.looks_empty:
        print("FAIL extraction produced (essentially) no text")
        return 1
    got_brief = _brief_from_text(extraction.text)

    print(f"reference text : {len(REFERENCE_TXT.read_text(encoding='utf-8-sig')):,} chars")
    print(f"our extraction : {len(extraction.text):,} chars")
    print()

    ref_d = dataclasses.asdict(ref_brief)
    got_d = dataclasses.asdict(got_brief)

    width = max(len(k) for k in ref_d)
    for key in ref_d:
        r, g = ref_d[key], got_d.get(key)
        same = r == g
        flag = "ok  " if same else ("tol " if key in TOLERATED else "DIFF")
        print(f"  [{flag}] {key:<{width}}  ref={r!r}")
        if not same:
            print(f"         {'':<{width}}  got={g!r}")
            if key not in TOLERATED:
                failures.append(key)

    print()
    # Independently assert the two fields whose silent failure is most damaging.
    if got_brief.budget != 9000.0:
        failures.append(f"budget expected 9000.0, got {got_brief.budget}")
    if got_brief.duration_days != 15:
        failures.append(f"duration_days expected 15, got {got_brief.duration_days}")
    if got_brief.city_id != "LH":
        failures.append(f"city_id expected 'LH', got {got_brief.city_id!r}")

    # The RFP block must be detectable, since screen 06 branches on it.
    has_rfp, asks = docs.detect_rfp_asks(extraction.text)
    print(f"RFP detected: {has_rfp}, {len(asks)} asks")
    for a in asks:
        print(f"   - {a}")
    if not has_rfp or len(asks) != 3:
        failures.append(f"expected 3 RFP asks, got {len(asks)}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        return 1
    print("PASS -- extraction is parser-equivalent to the reference dump")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

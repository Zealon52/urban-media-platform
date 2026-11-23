"""
extract_briefs.py -- turn the six campaign_N.docx client briefs into the
tab-delimited .txt form D2's parser reads.

WHY THIS EXISTS. Only campaign_4 had ever been extracted, so D5's
no-argument mode -- which globs campaign_*.docx.parsed.txt -- had only ever
seen one of the six briefs, and the other five were being approximated in
tests by overriding fields on campaign 4's parsed brief. That is not the same
thing: campaign 2 is the frequency brief and campaign 6 is the one with a
primary/supplementary inventory split and the high-street ask that has no POI
type. Those are the two paths most worth exercising, and neither was being
exercised on its real text.

FORMAT, matched to the existing campaign_4.docx.parsed.txt exactly rather
than invented, because D2's `parse_brief` is regex over this structure:

  * header fields are `Label<TAB>Value` on one line, from the docx's table
  * body paragraphs are plain lines
  * bullets keep their leading marker and a tab, since `_find_exclusion_block`
    keys off "Exclusion Criteria:" appearing in a bullet
  * CRLF line endings, as in the existing file

Run:  python extract_briefs.py            (writes any that are missing)
      python extract_briefs.py --force    (rewrite all six)
      python extract_briefs.py --check    (report, write nothing)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
#: The .docx originals live with the raw datasets, outside the repo, because
#: they are supplied material rather than something we produce.
SOURCES = [ROOT.parent / "DataSets_UrbanMedia_Raw" / "Campaigns",
           ROOT / "Campaigns", ROOT]


def find_docx(n: int) -> Path | None:
    for base in SOURCES:
        p = base / f"campaign_{n}.docx"
        if p.exists():
            return p
    return None


def docx_to_lines(path: Path) -> List[str]:
    """Paragraphs and table rows, in document order.

    python-docx exposes paragraphs and tables as separate collections, so
    iterating body XML children is the only way to keep a table's header
    fields ahead of the prose that follows them -- and the header fields are
    where the budget, duration and city actually live.
    """
    from docx import Document                       # local: optional dependency
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    out: List[str] = []
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = Paragraph(child, doc).text.rstrip()
            out.append(text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells = [c.text.strip() for c in row.cells]
                # Collapse the repeated-cell artefact of merged columns.
                dedup: List[str] = []
                for c in cells:
                    if not dedup or dedup[-1] != c:
                        dedup.append(c)
                if any(dedup):
                    out.append("\t".join(dedup))
    return out


def write_parsed(n: int, force: bool, check: bool) -> str:
    dest = ROOT / f"campaign_{n}.docx.parsed.txt"
    if dest.exists() and not force:
        return f"campaign_{n}: exists, left alone ({dest.stat().st_size:,} bytes)"
    src = find_docx(n)
    if src is None:
        return f"campaign_{n}: NO .docx FOUND in any of {[str(s) for s in SOURCES]}"
    try:
        lines = docx_to_lines(src)
    except ImportError:
        return (f"campaign_{n}: python-docx is not installed in this "
                f"interpreter -- `pip install python-docx`")
    except Exception as exc:                                   # noqa: BLE001
        return f"campaign_{n}: could not read {src.name} ({exc})"
    if check:
        return (f"campaign_{n}: would write {len(lines)} line(s) from "
                f"{src.name}")
    # CRLF and a trailing newline, matching the existing campaign_4 file so
    # the parser sees one consistent shape across all six.
    dest.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return (f"campaign_{n}: wrote {len(lines)} line(s), "
            f"{dest.stat().st_size:,} bytes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rewrite briefs that already have a .txt")
    ap.add_argument("--check", action="store_true",
                    help="report what would happen, write nothing")
    a = ap.parse_args()

    print("extracting client briefs -> campaign_N.docx.parsed.txt")
    bad = 0
    for n in range(1, 7):
        msg = write_parsed(n, a.force, a.check)
        print("  " + msg)
        if "NO .docx" in msg or "could not read" in msg or "not installed" in msg:
            bad += 1
    if bad:
        print(f"\n{bad} brief(s) unavailable. D5's no-argument mode globs "
              f"campaign_*.docx.parsed.txt, so it will only see the ones that "
              f"exist.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

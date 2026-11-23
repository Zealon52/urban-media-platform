"""Brief intake: turn an uploaded file into the exact text shape D2's parser
was built and tested against.

WHY THIS FILE IS FUSSIER THAN IT LOOKS. D2's `parse_brief` is deterministic and
line-anchored:

    _extract_name    r"Company Name\\s+([^\\n]+)"
    _extract_budget  r"Campaign Budget\\s+USD\\s*([\\d,]+)"
    _extract_field   rf"{label}\\s+(.+)"   then split on "\\n"

Every one of those header fields -- company, industry, objective, audience,
budget, duration -- lives inside a DOCX **table**, not a paragraph. So:

  1. `document.paragraphs` alone silently drops all of them. Budget and
     duration would come back 0 and the whole plan would refuse at the budget
     stage for no visible reason.
  2. `document.paragraphs` and `document.tables` are separate lists that lose
     interleaving, so the body has to be walked in real document order.
  3. Table rows must be emitted as `label<TAB>value`, matching the committed
     `campaign_4.docx.parsed.txt` this repo already parses correctly. `\\s+`
     matches a tab, so that shape satisfies every extractor above.

`_clean()` in D2 already strips the `<PARSED TEXT FOR PAGE:...>` and
`<IMAGE FOR PAGE:...>` markers, so PDF extraction is free to leave them in.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Extraction:
    text: str
    extractor: str
    n_pages: Optional[int] = None
    warning: str = ""

    @property
    def looks_empty(self) -> bool:
        return len(self.text.strip()) < 40


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #

def _docx_block_items(document):
    """Yield ("p", Paragraph) / ("t", Table) in true document order.

    python-docx exposes .paragraphs and .tables as separate flat lists, which
    loses the interleaving. Walking the body's XML children is the documented
    way to preserve it, and preserving it matters here: the parser's
    `_extract_field` splits on the first newline after a label, so a table
    landing in the wrong place can attach the wrong value to a field.
    """
    from docx.document import Document as _Doc
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph

    if isinstance(document, _Doc):
        parent = document.element.body
    elif isinstance(document, _Cell):
        parent = document._tc
    else:
        parent = document.element.body

    for child in parent.iterchildren():
        if isinstance(child, CT_P):
            yield "p", Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield "t", Table(child, document)


def extract_docx(data: bytes) -> Extraction:
    try:
        import docx  # python-docx
    except ImportError:
        return Extraction("", "python-docx", warning=
                          "python-docx is not installed -- run "
                          "pip install -r requirements.txt")

    document = docx.Document(io.BytesIO(data))
    lines: List[str] = []

    for kind, block in _docx_block_items(document):
        if kind == "p":
            txt = (block.text or "").strip()
            if txt:
                lines.append(txt)
            continue

        # A table. Emit one line per row, cells joined by TAB -- the shape the
        # committed campaign_4.docx.parsed.txt uses and D2 already parses.
        for row in block.rows:
            cells = []
            seen = set()
            for cell in row.cells:
                # Merged cells repeat the same _tc across row.cells; dedupe on
                # identity so a merged header does not emit "Budget Budget".
                if id(cell._tc) in seen:
                    continue
                seen.add(id(cell._tc))
                cells.append(" ".join((cell.text or "").split()))
            while cells and not cells[-1]:
                cells.pop()
            if any(cells):
                lines.append("\t".join(cells))

    return Extraction("\n".join(lines), "python-docx")


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

def extract_pdf(data: bytes) -> Extraction:
    try:
        import pdfplumber
    except ImportError:
        return Extraction("", "pdfplumber", warning=
                          "pdfplumber is not installed -- run "
                          "pip install -r requirements.txt")

    chunks: List[str] = []
    n_pages = 0
    warning = ""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            # Tables first: a brief's header block is usually a real table, and
            # extract_text() flattens it into an unpredictable word order.
            try:
                for table in page.extract_tables() or []:
                    for row in table:
                        cells = [" ".join((c or "").split()) for c in row]
                        while cells and not cells[-1]:
                            cells.pop()
                        if any(cells):
                            chunks.append("\t".join(cells))
            except Exception:  # noqa: BLE001 -- table detection is best-effort
                pass
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)

    text = "\n".join(chunks)
    if not text.strip():
        warning = ("no extractable text -- this looks like a scanned/image PDF. "
                   "Paste the brief text instead, or upload the DOCX.")
    return Extraction(text, "pdfplumber", n_pages=n_pages, warning=warning)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

def extract(filename: str, data: bytes) -> Extraction:
    suffix = Path(filename).suffix.lower()
    if suffix == ".docx":
        return extract_docx(data)
    if suffix == ".pdf":
        return extract_pdf(data)
    if suffix in {".txt", ".md"}:
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return Extraction(data.decode(enc), "plain")
            except UnicodeDecodeError:
                continue
        return Extraction(data.decode("utf-8", errors="replace"), "plain",
                          warning="file had undecodable bytes; replaced them")
    if suffix == ".doc":
        return Extraction("", "unsupported", warning=
                          "legacy .doc is not supported -- save as .docx or paste the text")
    return Extraction("", "unsupported",
                      warning=f"unsupported file type {suffix or '(none)'}")


def mime_for(filename: str) -> str:
    return {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }.get(Path(filename).suffix.lower(), "application/octet-stream")


# --------------------------------------------------------------------------- #
# Sample briefs shipped in the repo
# --------------------------------------------------------------------------- #

def list_sample_briefs(campaigns_dir: Path) -> List[Path]:
    if not campaigns_dir.exists():
        return []
    return sorted(campaigns_dir.glob("campaign_*.docx"))


# --------------------------------------------------------------------------- #
# Historical lead replay
# --------------------------------------------------------------------------- #

_LEAD_OBJECTIVE = {
    "auto": "Brand Awareness & Test-Drive Bookings",
    "cpg": "Trial & Impulse Purchase",
    "retail": "Seasonal Footfall & Sale Awareness",
    "hospitality": "Footfall & Local Recall",
    "telecom": "Subscriber Acquisition",
    "finance": "Product Awareness & Acquisition",
}


def lead_to_brief_text(lead: dict, city_name: str = "") -> str:
    """Render a real `lost_leads` row into the same header-block shape a client
    brief uses, so a lost lead can be replayed through the identical parser.

    This is the cheapest high-value input in the app: the outcome and the
    `loss_reason` are already known, so a replay doubles as an evaluation case
    rather than only a demo convenience.
    """
    industry = str(lead.get("industry_vertical") or "").strip()
    budget = lead.get("requested_budget") or lead.get("budget") or 0
    duration = lead.get("requested_duration_days") or lead.get("duration_days") or 0
    company = str(lead.get("company_name") or lead.get("lead_id") or "Unknown Lead")
    zone = str(lead.get("zone_id") or "").strip()
    loss = str(lead.get("loss_reason") or "").strip()
    objective = _LEAD_OBJECTIVE.get(industry, "Awareness")

    lines = [
        f"CLIENT BRIEF (REPLAY): {company.upper()} — HISTORICAL LEAD",
        f"Company Name\t{company}",
        f"Industry Vertical\t{industry.upper()}",
        f"Campaign Objective\t{objective}",
        f"Target Audience\tAs originally scoped for this lead",
        f"Campaign Budget\tUSD {int(float(budget or 0)):,}",
        f"Campaign Duration\t{int(float(duration or 0))} Days",
        "",
        "1. Executive Summary & Objective",
        f"Replay of historical lead {lead.get('lead_id', '?')}"
        + (f" in {city_name}" if city_name else "")
        + (f", originally scoped around zone {zone}" if zone else "")
        + ". This lead did not convert"
        + (f"; the recorded loss reason was '{loss}'." if loss else "."),
        "",
        "3. Digital Screen Selection & Location Requirements",
        "Inventory as originally requested for this lead's zone and industry.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# RFP detection -- does the brief carry its own section 5?
# --------------------------------------------------------------------------- #

_RFP_HEADING = re.compile(
    r"RFP\s+Requirements|formalized response containing|"
    r"Sales\s+Team\s+Response", re.I)

_ASK_LINE = re.compile(r"^\s*(\d)[.)]\s+(.{15,})$", re.M)


def detect_rfp_asks(text: str) -> Tuple[bool, List[str]]:
    """Find the brief's own RFP requirement list, if it has one.

    All six sample briefs close with an identically-shaped section 5 -- a
    heading, one framing sentence, then a numbered list of three asks. When
    that block is absent (a pasted email, a replayed lead), the caller falls
    back to the house default template instead.
    """
    m = _RFP_HEADING.search(text)
    if not m:
        return False, []

    tail = text[m.end():]
    # Stop at the next top-level numbered section heading ("6. ...") so a
    # following section's own numbered list cannot leak in as an ask.
    stop = re.search(r"\n\s*[6-9]\.\s+[A-Z]", tail)
    if stop:
        tail = tail[:stop.start()]

    asks: List[str] = []
    for _num, body in _ASK_LINE.findall(tail):
        ask = " ".join(body.split()).rstrip()
        # Guard against the framing sentence being captured as an ask.
        if ask and not _RFP_HEADING.search(ask):
            asks.append(ask)

    return (len(asks) > 0), asks

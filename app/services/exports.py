"""Exports. A rep's job after approving a plan is sending it, so this is MVP
surface rather than polish.

Everything returns bytes, ready to hand to `st.download_button`. Nothing here
recomputes: every export reads the run that is already on screen, so a exported
file can never disagree with what the rep approved.
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from app import config
from app.services.metrics import _slots_col, line_costs


def _csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")   # BOM so Excel opens it correctly


# --------------------------------------------------------------------------- #
# media plan
# --------------------------------------------------------------------------- #

def media_plan_csv(result, quoted: Optional[pd.DataFrame] = None) -> bytes:
    lines = result.lines
    if not len(lines):
        return _csv(pd.DataFrame([{"note": "no lines allocated"}]))

    b = result.plan.brief
    days = int(b.duration_days or 0)
    df = lines.copy()
    sc = _slots_col(df)

    if result.scored is not None and len(result.scored):
        keep = [c for c in ["screen_id", "screen_type", "position", "zone_id",
                            "city_id", "relevance_score", "confidence",
                            "has_booking_history", "reasons"]
                if c in result.scored.columns]
        df = df.merge(result.scored[keep].drop_duplicates("screen_id"),
                      on="screen_id", how="left")

    df["time_block"] = df["time_block_id"].map(config.TIME_BLOCKS)
    df["day_type"] = result.plan.day_type
    df["flight_days"] = days
    df["campaign"] = b.name
    df["d3_price_per_slot_day"] = df["price"]
    df["spend"] = line_costs(lines, days).values

    for key, attr in [("d3_floor", "floor"), ("d3_target", "target"),
                      ("d3_cap", "cap"), ("d3_recommended", "recommended"),
                      ("p_win", "p_win_at_recommended"),
                      ("rate_card_basis", "rate_card_basis"),
                      ("event_adjustment_pct", "event_adjustment_pct"),
                      ("event_names", "event_names"),
                      ("ridership_trend_pct", "ridership_trend_pct")]:
        df[key] = [getattr(result.quotes.get((str(s), int(t))), attr, None)
                   for s, t in zip(df["screen_id"], df["time_block_id"])]

    if quoted is not None and len(quoted):
        q = quoted[["screen_id", "time_block_id", "learned_multiplier",
                    "quoted_price", "quoted_cost", "reason"]]
        df = df.merge(q, on=["screen_id", "time_block_id"], how="left")
        df = df.rename(columns={"reason": "learned_adjustment_reason"})

    order = [c for c in [
        "campaign", "screen_id", "screen_type", "position", "zone_id", "city_id",
        "time_block_id", "time_block", "day_type", sc, "flight_days",
        "d3_floor", "d3_target", "d3_cap", "d3_recommended",
        "d3_price_per_slot_day", "learned_multiplier", "quoted_price",
        "spend", "quoted_cost", "people_reach", "p_win", "rate_card_basis",
        "event_adjustment_pct", "event_names", "ridership_trend_pct",
        "relevance_score", "confidence", "has_booking_history", "overlap_tier",
        "pool_id", "reasons", "learned_adjustment_reason",
    ] if c in df.columns]
    return _csv(df[order])


# --------------------------------------------------------------------------- #
# rate card / quote detail
# --------------------------------------------------------------------------- #

def rate_card_csv(result) -> bytes:
    rows: List[Dict[str, Any]] = []
    for (sid, blk), q in result.quotes.items():
        base = {
            "screen_id": sid, "time_block_id": blk,
            "time_block": config.TIME_BLOCKS.get(blk, blk),
            "floor": q.floor, "target": q.target, "cap": q.cap,
            "recommended": q.recommended, "basis": q.rate_card_basis,
            "p_win_at_recommended": q.p_win_at_recommended,
            "expected_revenue": q.expected_revenue,
            "event_adjustment_pct": q.event_adjustment_pct,
            "event_names": q.event_names,
            "ridership_trend_pct": q.ridership_trend_pct,
            "ridership_trend_note": q.ridership_trend_note,
            "warnings": " | ".join(q.warnings or []),
        }
        if q.components:
            for c in q.components:
                r = dict(base)
                r.update({"component": c.get("name"), "family": c.get("family"),
                          "shift": c.get("shift"), "component_reason": c.get("reason")})
                rows.append(r)
        else:
            rows.append(base)
    if not rows:
        return _csv(pd.DataFrame([{"note": "no quotes"}]))
    return _csv(pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# audience appendix
# --------------------------------------------------------------------------- #

def audience_appendix_csv(result, profiles: pd.DataFrame,
                          cube: Optional[pd.DataFrame] = None) -> bytes:
    if not len(result.lines) or profiles is None or not len(profiles):
        return _csv(pd.DataFrame([{"note": "no audience data"}]))
    sel = set(result.lines["screen_id"].astype(str))
    p = profiles[profiles["screen_id"].astype(str).isin(sel)].copy()
    keep = [c for c in ["screen_id", "city_id", "zone_id", "mechanism",
                        "archetype_id", "peak_block", "daily_people", "confidence",
                        "has_booking_history", "label", "composition_str",
                        "poi_mix_str"] if c in p.columns]
    p = p[keep]
    if cube is not None and len(cube):
        c = cube[(cube["screen_id"].astype(str).isin(sel))
                 & (cube["day_type"] == result.plan.day_type)]
        agg = c.groupby("screen_id", observed=True).agg(
            people_total=("people", "sum"),
            dwell_seconds=("dwell_seconds", "mean"),
            visibility=("visibility", "mean")).reset_index()
        p = p.merge(agg, on="screen_id", how="left")
    return _csv(p)


# --------------------------------------------------------------------------- #
# raw artifacts
# --------------------------------------------------------------------------- #

def plan_json_bytes(plan_dict: dict) -> bytes:
    return json.dumps(plan_dict, indent=2, default=str).encode("utf-8")


def markdown_bytes(md: str) -> bytes:
    return md.encode("utf-8")


def decision_log_md(campaign: dict, rows: List[dict], runs: List[dict]) -> bytes:
    L = [f"# Decision log — {campaign.get('client_name') or campaign.get('source_label') or 'campaign'}",
         "",
         f"Source: `{campaign.get('source_mode')}` · {campaign.get('source_label') or '--'}",
         f"Created: {campaign.get('created_at')}",
         ""]
    if runs:
        L += ["## Revisions", "",
              "| Rev | Solver | Presentable | Spend | Reach/day | Lines | Screens | Took | Learning | At |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            L.append(f"| {r['revision']} | {r['solver_used']} | "
                     f"{'yes' if r['presentable'] else 'no'} | "
                     f"${(r['spend'] or 0):,.0f} | {(r['reach'] or 0):,.0f} | "
                     f"{r['n_lines']} | {r['n_screens']} | "
                     f"{(r['duration_ms'] or 0) / 1000:.1f}s | "
                     f"{'on' if r['learning_applied'] else 'off'} | {r['created_at']} |")
        L.append("")
    L += ["## Event trail", "", "| When | Actor | Event | Detail |", "|---|---|---|---|"]
    for r in rows:
        detail = (r.get("detail") or "").replace("|", "\\|").replace("\n", " ")
        L.append(f"| {r['created_at']} | {r['actor']} | {r['event']} | {detail} |")
    return "\n".join(L).encode("utf-8")


def feedback_csv(rows: List[dict]) -> bytes:
    if not rows:
        return _csv(pd.DataFrame([{"note": "no feedback recorded"}]))
    df = pd.DataFrame(rows)
    keep = [c for c in ["id", "screen", "mode", "raw_text", "loss_reason", "status",
                        "routed_by", "routed_json", "payload_json",
                        "applied_in_run_id", "created_at"] if c in df.columns]
    return _csv(df[keep])


# --------------------------------------------------------------------------- #
# DOCX -- the client-facing artifact
# --------------------------------------------------------------------------- #

def rfp_docx(md: str, title: str = "Campaign proposal") -> Optional[bytes]:
    """Render the RFP markdown to a .docx a rep can actually email.

    A deliberately small markdown subset -- headings, tables, bullets, numbered
    lists, bold spans, blockquotes and rules -- because that is all `rfp.compose`
    emits. Returns None if python-docx is unavailable so the caller can offer
    the .md instead of failing.
    """
    try:
        import docx
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt
    except ImportError:
        return None

    doc = docx.Document()
    for style_name, size in (("Normal", 10.5),):
        try:
            doc.styles[style_name].font.size = Pt(size)
        except KeyError:
            pass

    def add_rich(par, text: str) -> None:
        """Bold spans only -- **like this**."""
        for i, chunk in enumerate(text.split("**")):
            if not chunk:
                continue
            run = par.add_run(chunk.replace("`", ""))
            run.bold = (i % 2 == 1)

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("---") and set(stripped) <= {"-"}:
            doc.add_paragraph()
            i += 1
            continue

        # table: a header row followed by a |---| separator
        if stripped.startswith("|") and i + 1 < len(lines) \
                and set(lines[i + 1].strip().replace("|", "").replace(":", "")
                        .replace(" ", "")) <= {"-"} \
                and lines[i + 1].strip().startswith("|"):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            body: List[List[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            table = doc.add_table(rows=1, cols=len(header))
            try:
                table.style = "Light Grid Accent 1"
            except KeyError:
                table.style = "Table Grid"
            for j, h in enumerate(header):
                cell = table.rows[0].cells[j]
                cell.text = ""
                add_rich(cell.paragraphs[0], h)
                for r in cell.paragraphs[0].runs:
                    r.bold = True
            for row in body:
                cells = table.add_row().cells
                for j, v in enumerate(row[:len(header)]):
                    cells[j].text = ""
                    add_rich(cells[j].paragraphs[0], v)
            doc.add_paragraph()
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            doc.add_heading(stripped.lstrip("#").strip().replace("`", ""),
                            level=min(level, 4))
            i += 1
            continue

        if stripped.startswith(">"):
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Pt(18)
            add_rich(par, stripped.lstrip("> ").lstrip("#").strip())
            for r in par.runs:
                r.italic = True
            i += 1
            continue

        if stripped.startswith(("- ", "* ")):
            par = doc.add_paragraph(style="List Bullet")
            add_rich(par, stripped[2:])
            i += 1
            continue

        num = stripped.split(".", 1)
        if len(num) == 2 and num[0].isdigit() and stripped[len(num[0]) + 1:].startswith(" "):
            par = doc.add_paragraph(style="List Number")
            add_rich(par, num[1].strip())
            i += 1
            continue

        par = doc.add_paragraph()
        add_rich(par, stripped.strip("_"))
        if stripped.startswith("_") and stripped.endswith("_"):
            for r in par.runs:
                r.italic = True
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT
        i += 1

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

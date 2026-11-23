"""SQLite persistence for the D6 app.

A connection is opened per operation rather than cached on the module. That is
deliberate: Streamlit reruns scripts on every interaction and warms the engine
on a background thread, so a single shared connection would need
`check_same_thread=False` plus a lock to be safe. WAL mode makes per-call
connections cheap enough that the simpler, obviously-correct version wins.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from app import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextmanager
def connect():
    config.ensure_app_data()
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Idempotent. Safe to call on every app boot."""
    sql = config.SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(sql)


def _rows(cur) -> List[Dict[str, Any]]:
    return [dict(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #

def upsert_document(*, filename: str, mime: Optional[str], data: bytes,
                    extracted_text: str, extractor: str,
                    n_pages: Optional[int] = None,
                    extract_warning: Optional[str] = None) -> int:
    """Insert, or return the existing id if these exact bytes were already
    uploaded. Dedupe is on content, not filename, so the same brief renamed
    twice does not extract twice."""
    digest = sha256_bytes(data)
    with connect() as conn:
        cur = conn.execute("SELECT id FROM documents WHERE sha256 = ?", (digest,))
        row = cur.fetchone()
        if row:
            return int(row["id"])
        cur = conn.execute(
            """INSERT INTO documents (sha256, filename, mime, n_bytes, file_bytes,
                                      extracted_text, extractor, n_pages,
                                      extract_warning, uploaded_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (digest, filename, mime, len(data), data, extracted_text, extractor,
             n_pages, extract_warning, _now()),
        )
        return int(cur.lastrowid)


def get_document(document_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def document_bytes(document_id: int) -> Optional[bytes]:
    with connect() as conn:
        cur = conn.execute("SELECT file_bytes FROM documents WHERE id = ?", (document_id,))
        row = cur.fetchone()
        return bytes(row["file_bytes"]) if row else None


def list_documents(limit: int = 50) -> List[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute(
            """SELECT id, sha256, filename, mime, n_bytes, extractor, n_pages,
                      uploaded_at, length(extracted_text) AS text_len
               FROM documents ORDER BY id DESC LIMIT ?""", (limit,))
        return _rows(cur)


# --------------------------------------------------------------------------- #
# campaigns
# --------------------------------------------------------------------------- #

def create_campaign(*, brief_text: str, source_mode: str,
                    source_label: Optional[str] = None,
                    document_id: Optional[int] = None,
                    client_name: Optional[str] = None,
                    rep_email: Optional[str] = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO campaigns (document_id, source_mode, source_label,
                                      brief_text, client_name, rep_email, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (document_id, source_mode, source_label, brief_text, client_name,
             rep_email, _now()),
        )
        return int(cur.lastrowid)


def get_campaign(campaign_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_campaigns(limit: int = 25) -> List[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute(
            """SELECT c.*, (SELECT COUNT(*) FROM runs r WHERE r.campaign_id = c.id) AS n_runs
               FROM campaigns c ORDER BY c.id DESC LIMIT ?""", (limit,))
        return _rows(cur)


# --------------------------------------------------------------------------- #
# brief versions
# --------------------------------------------------------------------------- #

def add_brief_version(*, campaign_id: int, brief_json: dict, provenance: dict,
                      campaign_start: Optional[str], campaign_end: Optional[str],
                      day_type: str, allow_unevidenced: bool, origin: str,
                      origin_note: str = "") -> Dict[str, Any]:
    with connect() as conn:
        cur = conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM brief_versions WHERE campaign_id = ?",
            (campaign_id,))
        version = int(cur.fetchone()["v"]) + 1
        cur = conn.execute(
            """INSERT INTO brief_versions (campaign_id, version, brief_json,
                     provenance_json, campaign_start, campaign_end, day_type,
                     allow_unevidenced, origin, origin_note, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (campaign_id, version, json.dumps(brief_json), json.dumps(provenance),
             campaign_start, campaign_end, day_type, int(allow_unevidenced),
             origin, origin_note, _now()),
        )
        return {"id": int(cur.lastrowid), "version": version}


def latest_brief_version(campaign_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute(
            """SELECT * FROM brief_versions WHERE campaign_id = ?
               ORDER BY version DESC LIMIT 1""", (campaign_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #

def add_run(*, campaign_id: int, brief_version_id: int, plan_json: dict,
            metrics: dict, solver_used: str, presentable: bool,
            spend: float, reach: float, impressions: float, n_lines: int,
            n_screens: int, duration_ms: int, refusals: list, warnings: list,
            learning_applied: bool, learning_note: str = "") -> Dict[str, Any]:
    with connect() as conn:
        cur = conn.execute(
            "SELECT COALESCE(MAX(revision), 0) AS r FROM runs WHERE campaign_id = ?",
            (campaign_id,))
        revision = int(cur.fetchone()["r"]) + 1
        cur = conn.execute(
            """INSERT INTO runs (campaign_id, brief_version_id, revision, plan_json,
                     metrics_json, solver_used, presentable, spend, reach, impressions,
                     n_lines, n_screens, duration_ms, refusals_json, warnings_json,
                     learning_applied, learning_note, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (campaign_id, brief_version_id, revision, json.dumps(plan_json, default=str),
             json.dumps(metrics, default=str), solver_used, int(presentable), spend,
             reach, impressions, n_lines, n_screens, duration_ms,
             json.dumps(refusals, default=str), json.dumps(warnings, default=str),
             int(learning_applied), learning_note, _now()),
        )
        return {"id": int(cur.lastrowid), "revision": revision}


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_runs(campaign_id: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute(
            """SELECT id, revision, solver_used, presentable, spend, reach,
                      impressions, n_lines, n_screens, duration_ms, learning_applied,
                      created_at
               FROM runs WHERE campaign_id = ? ORDER BY revision""", (campaign_id,))
        return _rows(cur)


# --------------------------------------------------------------------------- #
# feedback
# --------------------------------------------------------------------------- #

def add_feedback(*, campaign_id: int, run_id: Optional[int], screen: str,
                 mode: str, raw_text: Optional[str] = None,
                 payload: Optional[dict] = None,
                 loss_reason: Optional[str] = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO feedback (campaign_id, run_id, screen, mode, raw_text,
                     payload_json, loss_reason, status, created_at)
               VALUES (?,?,?,?,?,?,?,'queued',?)""",
            (campaign_id, run_id, screen, mode, raw_text,
             json.dumps(payload) if payload is not None else None,
             loss_reason, _now()),
        )
        return int(cur.lastrowid)


def queued_feedback(campaign_id: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute(
            """SELECT * FROM feedback WHERE campaign_id = ? AND status = 'queued'
               ORDER BY id""", (campaign_id,))
        return _rows(cur)


def all_feedback(campaign_id: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute(
            "SELECT * FROM feedback WHERE campaign_id = ? ORDER BY id", (campaign_id,))
        return _rows(cur)


def dismiss_feedback(feedback_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE feedback SET status='dismissed' WHERE id = ?", (feedback_id,))


def mark_feedback_applied(ids: Iterable[int], *, run_id: Optional[int],
                          routed: Optional[Dict[int, dict]] = None,
                          routed_by: str = "rules") -> None:
    routed = routed or {}
    with connect() as conn:
        for fid in ids:
            conn.execute(
                """UPDATE feedback SET status='applied', applied_in_run_id=?,
                          routed_json=?, routed_by=? WHERE id = ?""",
                (run_id, json.dumps(routed.get(fid)) if routed.get(fid) else None,
                 routed_by, fid),
            )


# --------------------------------------------------------------------------- #
# rfp responses
# --------------------------------------------------------------------------- #

def add_rfp_response(*, run_id: int, template: str, rfp_asks: list,
                     markdown: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO rfp_responses (run_id, template, rfp_asks_json, markdown,
                     created_at) VALUES (?,?,?,?,?)""",
            (run_id, template, json.dumps(rfp_asks), markdown, _now()),
        )
        return int(cur.lastrowid)


def approve_rfp(rfp_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE rfp_responses SET approved=1, approved_at=? WHERE id=?",
                     (_now(), rfp_id))


def latest_rfp_for_run(run_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute(
            "SELECT * FROM rfp_responses WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# --------------------------------------------------------------------------- #
# learned signals
# --------------------------------------------------------------------------- #

def add_learned_signal(*, scope: str, scope_key: str, signal: str,
                       direction: float, source_feedback_id: Optional[int],
                       campaign_id: Optional[int], note: str = "") -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO learned_signals (scope, scope_key, signal, direction,
                     source_feedback_id, campaign_id, note, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (scope, scope_key, signal, direction, source_feedback_id, campaign_id,
             note, _now()),
        )
        return int(cur.lastrowid)


def learned_signals(*, exclude_campaign_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """All recorded signals. `exclude_campaign_id` leaves out the campaign
    currently being worked, so a rep never sees this session's own feedback
    fed back to it as if it were independent historical evidence."""
    sql = "SELECT * FROM learned_signals"
    params: list = []
    if exclude_campaign_id is not None:
        sql += " WHERE campaign_id IS NULL OR campaign_id != ?"
        params.append(exclude_campaign_id)
    sql += " ORDER BY id"
    with connect() as conn:
        return _rows(conn.execute(sql, params))


# --------------------------------------------------------------------------- #
# llm ledger + cache
# --------------------------------------------------------------------------- #

def llm_cache_get(cache_key: str) -> Optional[str]:
    with connect() as conn:
        cur = conn.execute("SELECT response_text FROM llm_cache WHERE cache_key = ?",
                           (cache_key,))
        row = cur.fetchone()
        return row["response_text"] if row else None


def llm_cache_put(cache_key: str, purpose: str, response_text: str) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO llm_cache (cache_key, purpose, response_text,
                     created_at) VALUES (?,?,?,?)""",
            (cache_key, purpose, response_text, _now()),
        )


def log_llm_call(*, purpose: str, cache_key: Optional[str], cached: bool,
                 deployment: Optional[str], prompt_tokens: Optional[int],
                 completion_tokens: Optional[int], est_cost_usd: float,
                 ok: bool, error: Optional[str] = None) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO llm_calls (purpose, cache_key, cached, deployment,
                     prompt_tokens, completion_tokens, est_cost_usd, ok, error,
                     created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (purpose, cache_key, int(cached), deployment, prompt_tokens,
             completion_tokens, est_cost_usd, int(ok), error, _now()),
        )


def llm_spend_summary() -> Dict[str, Any]:
    with connect() as conn:
        cur = conn.execute(
            """SELECT COUNT(*) AS n_calls,
                      SUM(CASE WHEN cached=1 THEN 1 ELSE 0 END) AS n_cached,
                      COALESCE(SUM(est_cost_usd), 0.0) AS spend_usd,
                      COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                      COALESCE(SUM(completion_tokens), 0) AS completion_tokens
               FROM llm_calls WHERE ok = 1""")
        row = dict(cur.fetchone())
        # Cached hits cost nothing but are worth showing as avoided spend.
        cur = conn.execute(
            "SELECT COUNT(*) AS n FROM llm_calls WHERE ok=1 AND cached=0")
        row["n_billable"] = int(cur.fetchone()["n"])
        return row


# --------------------------------------------------------------------------- #
# decision log
# --------------------------------------------------------------------------- #

def log(*, campaign_id: Optional[int], run_id: Optional[int] = None,
        actor: str, event: str, detail: str = "") -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO decision_log (campaign_id, run_id, actor, event, detail,
                     created_at) VALUES (?,?,?,?,?,?)""",
            (campaign_id, run_id, actor, event, detail, _now()),
        )


def decision_log(campaign_id: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.execute(
            "SELECT * FROM decision_log WHERE campaign_id = ? ORDER BY id", (campaign_id,))
        return _rows(cur)


def db_stats() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with connect() as conn:
        for t in ("documents", "campaigns", "brief_versions", "runs", "feedback",
                  "rfp_responses", "learned_signals", "llm_calls"):
            cur = conn.execute(f"SELECT COUNT(*) AS n FROM {t}")
            out[t] = int(cur.fetchone()["n"])
    out["db_bytes"] = config.DB_PATH.stat().st_size if config.DB_PATH.exists() else 0
    return out

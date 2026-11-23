-- D6 Unified Platform -- SQLite schema.
--
-- Everything the app persists lives in ONE file (app_data/urban_media.db),
-- including the raw uploaded PDF/DOCX bytes as BLOBs. That is a deliberate
-- choice: the submission is a zip that has to run on a machine that is not
-- ours, and a single self-contained database has no relative-path resolution
-- to get wrong and no uploads/ folder that a zip tool can silently drop.
--
-- The DB ships EMPTY. Schema is created on first run. The six sample briefs
-- are read from Campaigns/ on demand rather than duplicated in here.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- --------------------------------------------------------------------------
-- Uploaded source documents. Deduped on sha256 so re-uploading the same
-- brief during a demo does not create a second row or re-run extraction.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY,
    sha256          TEXT    NOT NULL UNIQUE,
    filename        TEXT    NOT NULL,
    mime            TEXT,
    n_bytes         INTEGER NOT NULL,
    file_bytes      BLOB    NOT NULL,
    extracted_text  TEXT    NOT NULL,
    extractor       TEXT    NOT NULL,   -- python-docx | pdfplumber | plain
    n_pages         INTEGER,
    extract_warning TEXT,
    uploaded_at     TEXT    NOT NULL
);

-- --------------------------------------------------------------------------
-- One campaign per brief the rep works. Holds the normalized brief TEXT --
-- every intake mode (upload / paste / lead replay / sample) converges here,
-- so nothing downstream branches on how the brief arrived.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS campaigns (
    id           INTEGER PRIMARY KEY,
    document_id  INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    source_mode  TEXT    NOT NULL,     -- upload | paste | lead | sample
    source_label TEXT,                 -- filename, LEAD-000018, campaign_3.docx
    brief_text   TEXT    NOT NULL,
    client_name  TEXT,
    rep_email    TEXT,
    created_at   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_campaigns_created ON campaigns(created_at DESC);

-- --------------------------------------------------------------------------
-- The StructuredBrief, versioned. v1 is the parser's output; later versions
-- are rep edits on the Interpretation screen or a feedback-driven re-run.
-- provenance_json records read | inferred | rep_edited | feedback_applied per
-- field, which is what the badges on screen 02 render.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS brief_versions (
    id              INTEGER PRIMARY KEY,
    campaign_id     INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    version         INTEGER NOT NULL,
    brief_json      TEXT    NOT NULL,
    provenance_json TEXT    NOT NULL,
    campaign_start  TEXT,
    campaign_end    TEXT,
    day_type        TEXT,
    allow_unevidenced INTEGER NOT NULL DEFAULT 0,
    origin          TEXT    NOT NULL,  -- parser | rep_edit | feedback_rerun
    origin_note     TEXT,
    created_at      TEXT    NOT NULL,
    UNIQUE(campaign_id, version)
);

-- --------------------------------------------------------------------------
-- One row per orchestrator execution. `revision` counts re-runs within a
-- campaign so screen 06 can diff revision N against N-1.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY,
    campaign_id      INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    brief_version_id INTEGER NOT NULL REFERENCES brief_versions(id) ON DELETE CASCADE,
    revision         INTEGER NOT NULL,
    plan_json        TEXT    NOT NULL,
    metrics_json     TEXT    NOT NULL,
    solver_used      TEXT,
    presentable      INTEGER NOT NULL DEFAULT 0,
    spend            REAL,
    reach            REAL,
    impressions      REAL,
    n_lines          INTEGER,
    n_screens        INTEGER,
    duration_ms      INTEGER,
    refusals_json    TEXT,
    warnings_json    TEXT,
    learning_applied INTEGER NOT NULL DEFAULT 0,
    learning_note    TEXT,
    created_at       TEXT    NOT NULL,
    UNIQUE(campaign_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_runs_campaign ON runs(campaign_id, revision DESC);

-- --------------------------------------------------------------------------
-- Rep feedback. Queued while the rep walks screens 03-06 and applied in ONE
-- batch when they trigger a re-run from the recommendation screen -- so a
-- walkthrough costs at most one extra solve and one router LLM call, not one
-- per comment.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback (
    id                INTEGER PRIMARY KEY,
    campaign_id       INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    run_id            INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    screen            TEXT    NOT NULL,  -- audience|relevance|package|recommend
    mode              TEXT    NOT NULL,  -- nl|param|drop_screen|price_override|loss_reason
    raw_text          TEXT,
    payload_json      TEXT,
    loss_reason       TEXT,
    status            TEXT    NOT NULL,  -- queued | applied | dismissed
    routed_json       TEXT,              -- the router's decision for this item
    routed_by         TEXT,              -- llm | rules
    applied_in_run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    created_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_campaign ON feedback(campaign_id, status);

-- --------------------------------------------------------------------------
-- The client-facing RFP response artifact, versioned per run.
-- template = brief_rfp when the uploaded brief carried its own section 5,
-- default when it did not and the house template was used instead.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rfp_responses (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    template      TEXT    NOT NULL,     -- brief_rfp | default
    rfp_asks_json TEXT,
    markdown      TEXT    NOT NULL,
    approved      INTEGER NOT NULL DEFAULT 0,
    approved_at   TEXT,
    created_at    TEXT    NOT NULL
);

-- --------------------------------------------------------------------------
-- CROSS-SESSION LEARNING.
--
-- One row per corroborating feedback event. Nothing here is a coefficient --
-- weights are decayed and aggregated at READ time (see services/learning.py),
-- so the raw evidence is always inspectable and a bad rule can be changed
-- without rewriting history.
--
-- Applied in the app layer only. D1-D5 never read this table.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS learned_signals (
    id                  INTEGER PRIMARY KEY,
    scope               TEXT    NOT NULL,  -- zone|screen_type|time_block|industry|screen
    scope_key           TEXT    NOT NULL,
    signal              TEXT    NOT NULL,  -- a loss_reason value
    direction           REAL    NOT NULL,  -- -1 penalise, +1 favour
    source_feedback_id  INTEGER REFERENCES feedback(id) ON DELETE SET NULL,
    campaign_id         INTEGER REFERENCES campaigns(id) ON DELETE SET NULL,
    note                TEXT,
    created_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learned_scope ON learned_signals(scope, scope_key, signal);

-- --------------------------------------------------------------------------
-- LLM ledger. The supplied key has a hard ~$5 ceiling, so every call is
-- metered and every response cached. The sidebar reads this table to show a
-- running estimate, and services/llm.py refuses to call once the configured
-- guard is hit.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY,
    purpose           TEXT    NOT NULL,   -- narrative | feedback_router
    cache_key         TEXT,
    cached            INTEGER NOT NULL DEFAULT 0,
    deployment        TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    est_cost_usd      REAL,
    ok                INTEGER NOT NULL,
    error             TEXT,
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key     TEXT PRIMARY KEY,
    purpose       TEXT NOT NULL,
    response_text TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

-- --------------------------------------------------------------------------
-- Append-only decision log -- every rep action, for the exportable audit
-- trail and the "explainability & trust" story.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decision_log (
    id          INTEGER PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id) ON DELETE CASCADE,
    run_id      INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    actor       TEXT NOT NULL,          -- rep | system | agent
    event       TEXT NOT NULL,
    detail      TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_log_campaign ON decision_log(campaign_id, id);

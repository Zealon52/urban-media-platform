# CLAUDE.md — AI coding context

Urban Media commercial intelligence platform. Agentic AI Hackathon 2026, team-091.

## The hard rule

**`app/` must not modify D1–D5 to get an app feature.** `build_d1_audience_profiles.py`,
`build_d2_relevance_scorer.py`, `pricing_engine_d3.py`, `build_d4_allocator.py`,
`build_d5_orchestrator.py`, `llm_agent.py`, `event_overlap.py`,
`ridership_trend.py`, `encode_pricing_grain.py` and `ds-toolkit-package/` are the
engines; everything in `app/` imports and calls them.

Read that precisely. It is a *layering* rule, not a freeze: the engines are
still actively fixed, and several of the numbers in this repo moved this week
because bugs were found in them. What it forbids is reaching into an engine to
make a screen work — that is how two versions of the same logic start drifting
apart. An actual engine bug gets fixed in the engine, and `python
test_pipeline.py` (69 checks, no network, no spend) is the gate it has to pass.

**Never copy engine files in from a zip or another clone without diffing first.**
The app handover shipped a folder labelled `engines-newer-than-git/` that was in
fact 566 lines *behind* `build_d5_orchestrator.py` and 629 behind
`llm_agent.py` — measured against a stale clone. Copying it over the repo would
have silently removed the entire LLM cost fence and the agentic loop.

If a capability seems to need a change in D1–D5, it almost certainly does not.
Two worked examples of getting what you need without editing anything:

- **Restricting the candidate pool.** `Orchestrator.plan()` scores relevance via
  `score_screens(brief, self.profiles, ...)`, reading its own instance attribute.
  So narrowing `orch.profiles` before the call and restoring it after restricts
  the shortlist through D5's own logic. See `_NarrowedProfiles` in
  `app/services/pipeline.py`. `StructuredBrief.exclude` cannot do this — it holds
  (screen_type, position) *category* pairs, so it can drop "all bus-rear screens"
  but never "this one screen", which is what a rep unticking a row means.
- **The weekday/weekend split.** D2 sets
  `day_type_weighting == "report_split"` (brief 3), but D5's `plan()` only
  branches on `"weekend_heavy"`, so the requirement is parsed and then dropped.
  The app closes it by calling `plan()` a second time at the other day type.
  See `_day_type_split` in `app/services/pipeline.py`.

## Layout

```
app/
  main.py                 Streamlit entry + sidebar + router
  config.py               paths, vocabularies, budget guards. Puts repo root on sys.path
  db/schema.sql           SQLite schema (documents as BLOBs)
  db/store.py             all persistence; one connection per call, WAL
  services/
    docs.py               DOCX/PDF extraction, RFP-block detection, lead replay
    interpret.py          parse + per-field provenance (read/inferred/needs_you)
    engines.py            heavy singletons + BACKGROUND WARMUP
    pipeline.py           the only place engines are invoked
    metrics.py            rep-facing metrics; impressions math
    feedback.py           the router: rep comments -> validated change set
    learning.py           cross-session learning (app layer only)
    rfp.py                the client-facing proposal composer
    exports.py            csv / md / docx / json
    llm.py                metered, cached LLM access
  ui/
    state.py              session state, gating, execute_run()
    components.py         rail, tiles, badges, price band, feedback panel
    s1..s6_*.py           the six screens
  tests/                  runnable without pytest: `python -m app.tests.<name>`
```

## Flow (deliberate, not the wireframe's)

`Intake → Interpretation → [ONE D5 run] → Audience (D1) → Relevance (D2) →
Package & Price (D3+D4) → Recommendation & RFP (D5)`

Screens 03–06 are **views onto a single run**. Pricing and allocation are not
separate steps — they run in the backend and surface together on screen 05 as
the package plus the price range to quote.

**Feedback is queued on every screen and applied by exactly one re-run**,
triggered from screen 06. This is a cost constraint, not a style choice: the
supplied Azure key has a ~$5 ceiling, so a walkthrough must cost one batched
router call and one solve, not one of each per comment. Do not "helpfully" make
feedback apply immediately.

## Things that will bite you

- **DOCX header fields live in tables.** Budget, duration, objective and audience
  are all in a DOCX table, not a paragraph. `document.paragraphs` silently drops
  them and every plan then refuses at the budget stage with no visible cause.
  `docs.py` walks the body in document order and emits table rows as
  `label<TAB>value`, matching the committed `campaign_4.docx.parsed.txt` that
  D2 already parses. `app/tests/test_intake_parity.py` asserts field-by-field
  equivalence against that reference — run it after touching extraction.
- **Three of six sample briefs never name a city.** Briefs 2, 3 and 6 return
  `city_id=''`. Unset, D2 scores all 11,163 screens across three cities. This is
  a *blocking* clarification on screen 02, not a soft badge.
- **`ridership_actuals.csv` is 129 MB** and is parsed lazily by the first
  date-aware quote, followed by one OLS fit per corridor × time block. That is
  why `engines.py` warms in a background thread: the expensive state is
  window-independent and cached on the engine instance, so a throwaway probe
  quote pays for it while the rep is still reading the brief. Screens 01–02 need
  only `cities.csv` + `dim_slot.csv`, so they stay instant.
- **Pins matter.** `pandas==3.0.5` exactly — not `<3`. D1–D5 were *written*
  against the 2.x API, which is where the `<3` guess came from, but 3.0.5 is
  what is installed, what all six briefs were solved on, and what every number
  in `eda_output/` and the C4 document was produced against. Downgrading moves
  the submission off its own reproducibility claim for no gain.
  `scikit-learn==1.9.0` exactly, because `model_d3_price.joblib` carries
  `_sklearn_version` 1.9.0 and a mismatch warns or fails on unpickle. LightGBM is
  *not* needed — the saved model is an sklearn `GradientBoostingRegressor`.
- **D5 runs two solves per plan** (at budget, then the cohort-tolerance stretch),
  so wall clock is ~2× `SOLVE_TIME_LIMIT_S`. At 20s the at-budget solve returns
  OPTIMAL; at 12s it returns FEASIBLE with the identical allocation.
- **Reach and impressions are different curves.** λ = slots × (dwell/60) ×
  visibility; reach = people × (1 − e^−λ) (saturating), impressions = people × λ
  (linear in slots). Never invent a second impressions formula — `metrics.py`
  reads `people` and `cycles_per_slot` back out of D4's own cells frame so the
  figure agrees with what the optimiser maximised.
- **Learning gates on event COUNT, not decayed weight.** Decay only reduces
  weight below the count, so `weight >= MIN_EVIDENCE` was satisfiable only at
  zero age — three events one day old weigh 2.954 and would never re-qualify.
  Corroboration doesn't decay; magnitude does. See `Adjustments._shift`.
- **The feedback router has a hard allowlist.** It may only touch scalar brief
  fields in `ROUTABLE_FIELDS`, every value is coerced and validated, and
  `must_have`/`exclude` are deliberately non-routable. A hallucinated field is
  recorded in `Actions.rejected` and dropped. Keep it that way.

## Cost: two guards, and why they are separate

The Azure key is **shared across all four teammates** and the budget is ~$5 of
real dollars. Two independent guards exist and both must stay:

- **`llm_agent.py`** fences every engine-side call on four axes — dollars,
  tokens, prompt size and call count — checked *before* dispatch, so a refused
  call spends nothing. It is the only module in the repo that holds an Azure
  client, which is what makes the determinism boundary structural rather than a
  convention. `python llm_agent.py --journal` shows the ledger;
  `--check-network` distinguishes a 403 (off the company VPN) from a bad key.
- **`app/services/llm.py`** meters the app's own two purposes into SQLite, with
  its ceilings in `app/config.py`.

Two things about that pair are deliberate and easy to break:

1. The app reads **`D6_LLM_MAX_CALLS`**, not `LLM_MAX_CALLS`. They were the same
   variable, which meant exporting `LLM_MAX_CALLS=500` to configure the *engine*
   fence silently widened the *app's* per-session cap from 40 to 500.
2. `LLM_RATES` in `app/config.py` is **pessimistic on purpose** — ~17× the
   published list price, matching what `llm_agent.py` fences on. The access
   email contradicts itself on mini's multiple, so nobody knows the true rate. A
   guard priced off the optimistic reading is not a guard. Over-estimating ends
   a demo early; under-estimating spends a teammate's budget.

`usage()["usd"]` is a *protection* figure at the pessimistic rate, not an
invoice. Do not quote it to a mentor as spend.

## Conventions

- Every LLM path must have a deterministic fallback that is always built and
  always correct. The model improves wording and intent parsing; it is never
  load-bearing for a number.
- Explain in the UI where a number came from. Provenance badges, D2's own
  `reasons` column, D3's `components` ledger, D5's `refusals` and `funnel`.
- Never claim a figure the data cannot support. The proposal explicitly declines
  to quote a conversion rate.
- Tests are plain modules, no pytest: `python -m app.tests.test_app_units`.

## Tests

```powershell
.venv\Scripts\python.exe -m app.tests.test_intake_parity   # extraction == reference
.venv\Scripts\python.exe -m app.tests.test_app_units       # db, router, learning, RFP
.venv\Scripts\python.exe -m app.tests.smoke_pipeline 4     # one brief end to end
.venv\Scripts\python.exe -m app.tests.smoke_rfp all        # all six -> proposals
```

Engine side, and the gate for any D1–D5 change:

```powershell
python test_pipeline.py            # 69 checks, no network, no spend
python test_pipeline.py --fast     # skips the plan-building checks
```

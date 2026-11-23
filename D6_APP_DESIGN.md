# D6 — Unified Platform: screen-by-screen design

What was built, and why. Grounded in the problem statement, the six briefs in
`Campaigns/`, and the real signatures in D1–D5.

**Constraint: D1–D5 are frozen.** Everything in `app/` imports and calls them.
See [CLAUDE.md](CLAUDE.md) for the two techniques used to extend behaviour
without editing an engine.

Network: 11,163 screens across 3 cities (LH Las Hackland / premium, ACS
Accordionshire / value, DAT DA Town / standard).

---

## The flow

```
Intake → Interpretation → [ONE D5 run] → Audience (D1) → Relevance (D2)
       → Package & Price (D3+D4) → Recommendation & RFP (D5)
```

Two decisions shape this, both departures from the original wireframe:

**Pricing and allocation are not separate steps.** They run in the backend and
surface together on one screen as the package plus the price range to quote. A
rep negotiates one number for a deal; per-screen bands are the model's internal
representation, not the thing being sold.

**Feedback is queued on every screen and applied by exactly one re-run**,
triggered from the Recommendation screen. This is a cost constraint made into a
design: the supplied Azure key has a ~$5 ceiling, so a walkthrough must cost one
batched router call and one solve, not one of each per comment. It also happens
to be better UX — the rep reviews the whole plan before committing to a change,
instead of watching it re-solve under them.

### How one run serves four screens

`Orchestrator.plan()` is called once. It returns a `Plan` but not the two things
the UI needs, so `services/pipeline.py` gathers them alongside:

| Screen needs | Source |
|---|---|
| D1 profiles, audience cube | `orch.profiles`, `orch.cube` — already loaded |
| D2 ranked shortlist | `score_screens()` re-run — deterministic, pure pandas, so identical to what the plan used |
| D3 per-line bands | `orch.engine.quote()` per allocated line (5–30 calls, cheap) |
| D4 cells for impressions | `build_candidate_cells()` over just the selected screens |

---

## Screen 00 — Shell

Step rail with `done / now / locked` gating, a pending-feedback tray, live engine
status, the LLM budget meter, and settings (solver effort, learning on/off,
force-greedy, unmeasured dayparts).

**The warmup, which is the main performance decision.** `Orchestrator.__init__`
reads nine CSVs (audience_cube 21 MB, pricing_grain 24 MB, encoded grain 26 MB,
win_loss 12 MB) and builds D3's rate card, win model and relationship table.
Worse, the *first date-aware quote* lazily parses `ridership_actuals.csv` —
129 MB — then fits one OLS per corridor × time block. Left alone that lands
inside the first plan run.

Two facts remove it: screens 01–02 need only `cities.csv` + `dim_slot.csv`
(`parse_brief` takes nothing else), and the expensive lazy state is
*window-independent* and cached on the engine instance. So a background thread
builds the orchestrator and fires one throwaway date-aware probe quote while the
rep reads and corrects the brief.

**Measured:** warmup 9–12s (import 0.1s, orchestrator init 4.1s, lazy prime
4.7s), fully overlapped with human input time. Plan run ~41s at the default 20s
solver limit.

---

## Screen 01 — Intake

Four modes → one normalized brief text: upload (DOCX/PDF/TXT), paste, replay a
real `lost_leads` row, or load one of the six shipped briefs. Detects whether the
document carries its own RFP section and says which template will be used.

**The bit that had to be right.** The brief's budget, duration, objective and
audience all live in a DOCX **table**, not a paragraph. `document.paragraphs`
drops them silently and every plan then refuses at the budget stage with no
visible cause. `services/docs.py` walks the body in document order and emits
table rows as `label<TAB>value`, matching the committed
`campaign_4.docx.parsed.txt` that D2 already parses correctly.
`app/tests/test_intake_parity.py` asserts field-by-field equivalence against that
reference — all 16 fields match, including `radius_anchor_hint` and
`preferred_blocks`.

---

## Screen 02 — Interpretation

Every `StructuredBrief` field editable, each badged **read** / **inferred** /
**you edited** / **from feedback** / **needs you**. Provenance is computed by
re-running D2's individual extractors and asking whether a field came from a
literal match or a keyword rule.

Plus the flight window (D3 needs real dates for event overlap and the ridership
trend), and the **cold-prospect cohort card** — none of the six briefs names a
company in `client_facts.csv`, so personalisation is analogue industry matching,
not history lookup.

**`needs you` is the important state.** Briefs 2, 3 and 6 never name a city.
Unset, D2 scores all 11,163 screens across three cities — slow and wrong. So the
run is *blocked*, not soft-flagged. Other clarifications raised here: an
unresolvable radius anchor, no detectable daypart, a frequency-vs-reach reading
to confirm, and D2's own trap-detector flags (brief 5 trips
`"Financial District" — verify this exists in POI data before promising it`).

---

## Screen 03 — Audience (D1)

Campaign-level: the segment mix the plan actually buys (reach-weighted over each
screen's `composition_str`), mean profile confidence, cold-start count, and the
`wait`/`ride`/`pass` mechanism split.

Per-screen inspector with day-type and time-block toggles, showing estimated
audience, dwell, visibility, the share-of-voice a single slot buys
(`1 − e^(−λ)`), the D1 rationale, and nearby POI evidence. The day-type toggle
prints the delta against the other day type, which is the one-click proof the
model is time-aware.

---

## Screen 04 — Relevance (D2)

Ranked shortlist with the **real** six weights from `score_screens`: daypart 0.25,
requested category 0.20, audience 0.20, confidence 0.15, tier 0.10, history 0.10.
Per-row decomposition showing value × weight → contribution.

The narrowing funnel is shown as a table, because the *first* stage that hits
zero is the real cause of an empty plan. `shortlist_size`'s own arithmetic is
rendered verbatim, which answers "why these 12 and not the top 50" before it is
asked.

Unticking a screen **queues** a removal. It reaches the optimiser on the next run
by narrowing `orch.profiles` — `StructuredBrief.exclude` cannot express it, since
it holds category pairs, not screen ids.

---

## Screen 05 — Package & Price (D3 + D4)

Tiles: spend/budget, reach/day, impressions, frequency, effective CPM, lines and
pools.

**The quote range**, which is the headline the rep needs: per-line floor/target/
cap summed into a package band, with the modelled win probability and the
negotiating room as a percentage of target.

Allocation table with slots as `bought / available` (real forward occupancy, not
a flat 6), per-line bands, p(win), event names, and a cold-start flag. A per-line
expander shows D3's multiplier ledger plus the event-overlap and ridership-trend
band shifts for the specific window.

Constraint checklist: budget with the reason for any remainder, block capacity,
pool saturation for shared audiences, flight window, solver and status, and the
two distinct history disclosures (see below). Then the stretch option at the
cohort's *real* `budget_variance_pct`, offered only when it buys ≥5% more reach.

If the plan refused, this screen renders the funnel and D5's ordered refusals
instead — an empty plan is a diagnosis, not an error page.

---

## Screen 06 — Recommendation, RFP, exports

Five tabs.

**Recommendation** — D5's `explain()` narrative beside the fact sheet it was
written from (proving it invented nothing), the cohort/posture card, warnings,
and D5's full deterministic plan page.

**RFP response** — see below.

**Metrics** — delivery, commercial, quality/risk, and the revision history with a
diff against the previous revision.

**Feedback & re-run** — the queue, the single re-run button, what the last re-run
actually changed, what the router *rejected* and why, and the cross-session
learning table.

**Exports** — proposal (.md/.docx), media plan, rate card detail, audience
appendix, full plan JSON, decision log, feedback CSV.

---

## The RFP response

All six briefs close with an identically-shaped section 5 asking for exactly
three things: a scoped **inventory** list with the selection logic named, a
**pricing** recommendation reflecting the drivers that brief cares about, and a
**reach/impressions projection** for the window with reasoning. When a brief
carries that section, each ask is mapped to the section answering it in a
coverage table. When it does not (a pasted email, a replayed lead), the same
three sections are emitted as the house default template plus executive summary,
audience rationale, assumptions, and next steps.

Brief-specific twists are detected and served individually. Verified on all six:

| Brief | Twist | Detected as | Result |
|---|---|---|---|
| 1 Zephyr EV | weekly impressions + conversion potential | `weekly_impressions`, `conversion_oriented` | 16 lines, $39,995, 13.19M impressions |
| 2 Ember Energy | frequency over coverage | `frequency_led` | **4 screens at 10.68× frequency** |
| 3 Loom & Thread | weekday/weekend split shown separately | `split_day_type` | weekday 8.14M vs weekend 1.87M |
| 4 Basil & Bloom | walking-radius logic stated explicitly | `radius_logic` | 13 lines, $9,000, radius stated |
| 5 SkyNimbus | business vs leisure separated | `segment_split` | segment table from `composition_str` |
| 6 Lumière | affinity per node type | `per_node_type` | delivery grouped by `screen_type` |

Brief 2 against brief 3 is the clean demonstration that the objective switch is
real: frequency buys **4 screens at 10.68×**, reach buys **21 screens at 2.19×**.

**What the proposal declines to say.** It states delivery precisely and refuses
to quote a conversion rate, because the network has no data to support one. That
is a deliberate credibility choice.

---

## Persistence

One SQLite file, `app_data/urban_media.db`, created on first run and shipping
empty. Uploaded documents are stored as **BLOBs** (deduped on sha256), which is
the deliberate answer to "must run from a zip on a machine that is not mine":
one self-contained artifact, no uploads folder a zip tool can drop, no relative
path to resolve.

Tables: `documents`, `campaigns`, `brief_versions` (versioned, with provenance),
`runs` (per revision, full plan JSON), `feedback` (queued → applied, with the
router's decision), `rfp_responses`, `learned_signals`, `llm_calls` + `llm_cache`,
and an append-only `decision_log`.

---

## The agentic feedback loop

Feedback items carry a mode: `nl`, `param`, `drop_screen`, `price_override`,
`loss_reason`. On re-run, the whole batch goes to the router in **one** LLM call,
which returns a JSON change set. Deterministic keyword rules always run too, and
anything unambiguous they found that the model missed is merged in.

**The trust boundary is the point.** The router may only touch an allowlist of
scalar brief fields; every value is coerced and validated; `must_have` and
`exclude` are deliberately non-routable because they are (screen_type, position)
pairs where a wrong guess silently changes the candidate pool. A hallucinated
field, a budget of "lots", or a nonexistent city is dropped into
`Actions.rejected` and shown in the UI. Tested against a hostile payload: 11
rejections, zero corruption.

A price objection moves the quote to the **midpoint of floor..target**, not to
the floor — surrendering the whole negotiating range on one push-back would be
worse than useless, and D3's floor is a real percentile of comparable bookings.

### Cross-session learning

`learned_signals` stores one row per feedback event. Weights decay on a 45-day
halflife (matching D3's own `recency_weight`), a scope needs 3 corroborating
events before it applies, price moves are capped at ±10% and relevance at ±0.08,
a campaign's own feedback is excluded from its own run, and every applied
adjustment carries its explanation. Applied in the app layer only: it moves the
price the rep is advised to **quote**, never the optimiser's cost basis — the
solver always allocates against D3's own recommendation. Learning a price down
*and* spending it as capacity would double-count one signal.

---

## Bugs the tests caught

Worth recording, since each was a real defect in this app (not in D1–D5):

1. **Learning could never activate.** The evidence gate compared `MIN_EVIDENCE`
   against the *decayed* weight, but decay only pushes weight below the event
   count — so `weight >= 3` was satisfiable only at literally zero age. Three
   events one day old weigh 2.954 and would never re-qualify. Corroboration
   doesn't decay; magnitude does.
2. **Every plan reported 100% cold-start spend.** `has_booking_history` is a
   pandas nullable boolean, and `pandas_bool is not True` is *always* true, so
   every screen counted as never-booked — contradicting D2's own reason text and
   D3's `rate_card_basis` in the same generated document.
3. **Impressions disagreed with themselves.** The weekday/weekend split passed
   `cells=None`, silently falling back to the lower-bound formula: 3.72M in the
   split table against 8.14M in the headline for the *identical* plan.
4. **Two history disclosures read as a contradiction.** D1's
   `has_booking_history` is screen-level; D3's `rate_card_basis` is cell-level. A
   screen can have a strong record in one daypart and none in another, so 0%
   cold-start beside 14 peer-priced lines is correct — but only if the document
   says which is which. Now reported as two separately-labelled figures.
5. **A missing `run_id` took down the whole Recommendation screen** on a NOT NULL
   constraint. Composing and persisting are now separate concerns.

---

## Test suite

```powershell
.venv\Scripts\python.exe -m app.tests.test_intake_parity   # extraction == committed reference
.venv\Scripts\python.exe -m app.tests.test_app_units       # db, router, learning, metrics, RFP
.venv\Scripts\python.exe -m app.tests.test_ui_screens      # all 6 screens via Streamlit AppTest
.venv\Scripts\python.exe -m app.tests.smoke_pipeline 4     # one brief end to end, with timings
.venv\Scripts\python.exe -m app.tests.smoke_rfp all        # all six -> app_data/sample_proposals/
```

---

## Known limits, stated rather than discovered later

- **Radius is a zone proxy.** No coordinates exist in the 14 source tables, so
  "within walking distance" resolves to zone membership around the anchor. D2
  owns this limitation; the app inherits it and the proposal says so.
- **Reach is per day**, not a cumulative unique count over the flight. Labelled
  everywhere it appears.
- **D3's rate card averages weekday and weekend economics** (its own logs say so).
  The app's day-type split re-solves the *allocation* per day type, but the
  underlying per-cell rate card is not day-type keyed.
- **CP-SAT may return FEASIBLE rather than OPTIMAL** on the stretch solve at the
  default time limit. Which one it got is displayed, never hidden.
- **The LLM is never load-bearing.** Every model path has a deterministic
  fallback that is always built and always correct.

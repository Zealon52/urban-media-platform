"""Paths, constants and feature flags for the D6 Streamlit app.

Importing this module is what makes `import build_d5_orchestrator` work from
inside `app/`: the repo root goes on sys.path here, once, so no other app
module needs to know about it.

NOTHING in app/ may modify D1-D5. Every engine call is import-and-call.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# app/config.py -> app/ -> repo root
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# D1-D5 live at the repo root and resolve their own data paths relative to
# their own __file__, so they work regardless of the process CWD -- but they
# have to be importable first.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "Urban Media Datasets"
OUT_DIR = ROOT / "eda_output"
def _resolve_campaigns_dir():
    """Find the six client .docx briefs, wherever this clone keeps them.

    A single hardcoded ROOT/"Campaigns" was silently wrong on any layout but
    the one it was written on: the .docx originals are supplied event material
    that lives OUTSIDE the repo here, and they are 25 MB of client documents
    that may or may not ever be committed. Four of the app's own tests and the
    sample-brief selector all failed with FileNotFoundError, which reads like a
    broken app rather than a missing input.

    First candidate that actually contains a campaign_*.docx wins. If none do,
    fall back to the in-repo path so the error message still names the place a
    reader would expect to look.
    """
    override = os.environ.get("D6_CAMPAIGNS_DIR")
    candidates = ([Path(override)] if override else []) + [
        ROOT / "Campaigns",                                    # in-repo
        ROOT.parent / "DataSets_UrbanMedia_Raw" / "Campaigns",  # supplied datasets
        ROOT.parent / "Campaigns",
    ]
    for c in candidates:
        try:
            if c.is_dir() and any(c.glob("campaign_*.docx")):
                return c
        except OSError:
            continue
    return ROOT / "Campaigns"


CAMPAIGNS_DIR = _resolve_campaigns_dir()

APP_DATA_DIR = ROOT / "app_data"
DB_PATH = APP_DATA_DIR / "urban_media.db"
SCHEMA_PATH = APP_DIR / "db" / "schema.sql"

# --------------------------------------------------------------------------- #
# The six real cities / vocabularies. Mirrored from the source tables so the UI
# can render selectors before the heavy engine has finished loading.
# --------------------------------------------------------------------------- #

CITY_CHOICES = {
    "LH": "Las Hackland (premium, dense, 3.2M)",
    "ACS": "Accordionshire (value, sprawling, 850k)",
    "DAT": "DA Town (standard, mixed, 1.45M)",
}

SCREEN_TYPES = ("bus", "bus_stop", "metro_rail_coach", "metro_station")
POSITIONS = ("back", "entrance_exit", "left", "platform", "right", "top")
MARKET_TIERS = ("premium", "standard", "value")

#: The ONLY audience segments audience_profiles.csv actually records, counted
#: straight off `composition_str` (student 2,417 screens ... shopper 11,163).
#: D2's `_audience_fit` scores against exactly these when `audience_segments` is
#: set, and falls back to a keyword table its own docstring calls broken when it
#: is not -- so routing a rep's audience feedback onto this vocabulary is what
#: makes the feedback actually move the score. Anything outside it (children,
#: families, over-60s) has no column in this dataset and must be declined out
#: loud rather than silently dropped.
AUDIENCE_SEGMENTS = ("blue_collar", "commuter_professional", "leisure",
                     "resident", "retail_service", "shopper", "student",
                     "traveller")

#: dim_slot.csv's six 4-hour blocks, with the labels used across the UI.
TIME_BLOCKS = {
    1: "00-04 Night",
    2: "04-08 Morning",
    3: "08-12 Midday",
    4: "12-16 Afternoon",
    5: "16-20 Evening",
    6: "20-24 Night",
}

#: The 10 real `lost_leads.loss_reason` values. Rep feedback is recorded
#: against this exact vocabulary so it lands in the same shape the historical
#: data already uses -- trainable rather than free text nobody parses.
LOSS_REASONS = (
    "price_too_high",
    "budget_mismatch",
    "went_with_competitor",
    "targeting_mismatch",
    "geography_mismatch",
    "inventory_unavailable",
    "timing_conflict",
    "creative_rejected",
    "client_went_quiet",
    "no_decision",
)

#: Subset of the above that means "the price was the problem" -- same set D5
#: uses (PRICE_LOSS_REASONS) so the app's routing agrees with D5's elasticity
#: read instead of quietly using a different definition.
PRICE_LOSS_REASONS = frozenset({"price_too_high", "budget_mismatch",
                                "went_with_competitor"})

# --------------------------------------------------------------------------- #
# LLM budget guard
# --------------------------------------------------------------------------- #
# The supplied Azure key carries a hard ~$5 ceiling, so the app treats LLM
# calls as a metered resource: two purposes only, both cached, both with a
# deterministic fallback that is always built and always correct.

LLM_ENABLED_DEFAULT = True
#: Refuse further calls once this many have been made. A revision needs ~2 (one
#: batched router call + one narrative), so 40 is ~20 revisions.
# Its OWN variable, not the engines' `LLM_MAX_CALLS`. Sharing that name was a
# live cost-runaway path: llm_agent.py's cap is a runaway backstop set against
# its dollar fence and defaults to 500, so a teammate exporting LLM_MAX_CALLS
# =500 to configure the ENGINE fence would silently widen the APP's per-session
# cap from 40 to 500 as a side effect -- and the app's call cap is the guard
# actually doing the work here (see the rate note below).
LLM_MAX_CALLS_PER_SESSION = int(os.environ.get("D6_LLM_MAX_CALLS", "40"))
#: Refuse once estimated spend passes this. Deliberately below the real $5 so
#: the app stops itself before the provider does -- and well below, because the
#: $5 is shared across the team and overspending starves a teammate.
LLM_MAX_SPEND_USD = float(os.environ.get("LLM_MAX_SPEND_USD", "1.25"))

# --------------------------------------------------------------------------- #
# Deployment tiers -- the cost decision
# --------------------------------------------------------------------------- #
# An AzureOpenAI client is NOT bound to a deployment: the deployment name is the
# per-call `model=` argument. So the app reuses D5's own `_llm_client()` (which
# already encodes three fixes established against these exact deployments --
# `model=` must be the DEPLOYMENT name, `max_completion_tokens` not
# `max_tokens`, and the GPT-5 family rejects a non-default `temperature`) while
# still choosing a CHEAPER deployment per purpose. Zero duplication, per-task
# cost control.
#
#   feedback_router -> NANO. It emits a small, strictly-validated JSON change
#                      set. That is extraction, not prose, and every field is
#                      coerced and range-checked afterwards, so a weaker model
#                      cannot corrupt the brief -- it can only fail validation
#                      and fall through to the deterministic rules.
#   narrative       -> MINI, via D5's explain(). Wording quality is visible to a
#                      buyer here. This is the only mini-billed call, once per
#                      plan, capped at 300 completion tokens.

TIER_NANO = "nano"
TIER_MINI = "mini"

#: purpose -> tier. Anything unlisted defaults to nano, so a new feature is
#: cheap by default and has to opt IN to the expensive tier.
LLM_PURPOSE_TIER = {
    "feedback_router": TIER_NANO,
    "narrative": TIER_MINI,
}


def deployment_for(tier: str) -> Optional[str]:
    """Resolve a tier to a deployment name, falling back down the chain rather
    than failing: nano -> mini -> generic."""
    generic = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    mini = os.environ.get("AZURE_OPENAI_DEPLOYMENT_MINI") or generic
    nano = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NANO") or mini
    return nano if tier == TIER_NANO else mini


#: Per-1M-token rates for the meter and the spend guard. Estimates, not
#: billing. The nano:mini ratio follows the team's note that nano is ~3.7x
#: cheaper per token.
LLM_RATES = {
    # PESSIMISTIC ON PURPOSE, and they must stay that way. These previously
    # carried the published list rates (nano $0.041/$0.162, mini $0.15/$0.60),
    # roughly 17x cheaper than the figure llm_agent.py fences on -- which is
    # the shared $5 divided by the 7M tokens the access email quotes. The email
    # is internally inconsistent about mini's multiple (3.7x in one place,
    # "around 6x" in another), so nobody on the team knows the true rate.
    #
    # A guard priced off the optimistic reading is not a guard: it would have
    # let the app bill ~$21 of fence-rate spend before its own $1.25 ceiling
    # tripped. Over-estimating stops the demo early and costs nothing;
    # under-estimating overruns a budget shared with three teammates. Set these
    # back to list prices only against an actual invoice.
    TIER_NANO: (float(os.environ.get("LLM_USD_PER_1M_PROMPT_NANO", "0.714")),
                float(os.environ.get("LLM_USD_PER_1M_COMPLETION_NANO", "0.714"))),
    TIER_MINI: (float(os.environ.get("LLM_USD_PER_1M_PROMPT_MINI", "4.286")),
                float(os.environ.get("LLM_USD_PER_1M_COMPLETION_MINI", "4.286"))),
}

#: Keep the router's output tight -- the change set is a handful of keys, and an
#: unbounded completion is the easiest way to waste a shared budget.
LLM_ROUTER_MAX_COMPLETION_TOKENS = 500

# --------------------------------------------------------------------------- #
# Solver / run behaviour
# --------------------------------------------------------------------------- #

#: Passed through to build_d4_allocator.CPSAT_TIME_LIMIT_S by D5's plan().
#
#: Measured on brief 4: D5 runs TWO solves per plan (at-budget, then the
#: cohort-tolerance stretch), so wall-clock is roughly 2x this plus window prep.
#: At 20s the plan took 41s and the at-budget solve returned OPTIMAL; at 12s it
#: took ~30s and returned FEASIBLE with the IDENTICAL objective value
#: (98,414,686) -- same allocation, just no proof of optimality. 20s is the
#: default because "provably optimal" is worth 8 seconds; the sidebar exposes
#: the trade-off rather than hiding it.
SOLVE_TIME_LIMIT_S = float(os.environ.get("D6_SOLVE_TIME_LIMIT_S", "20"))

#: Sidebar presets for the above.
SOLVE_PRESETS = {"Fast (8s)": 8.0, "Balanced (20s)": 20.0, "Exact (45s)": 45.0}

#: When True the app asks D5 to skip CP-SAT and use the greedy fallback. Useful
#: to demo the no-ortools path deliberately rather than hoping it works.
FORCE_GREEDY_DEFAULT = False

# --------------------------------------------------------------------------- #
# Cross-session learning
# --------------------------------------------------------------------------- #
# Learned signals are applied in the APP LAYER only, on top of what D2/D3
# returned. D1-D5 are never modified and never see these values.

LEARNING_ENABLED_DEFAULT = True
#: Below this many corroborating feedback events, a signal is recorded but not
#: applied. Same reasoning as D5's MIN_COHORT: a 2-event "pattern" is noise.
LEARNING_MIN_EVIDENCE = 3
#: Feedback ages. Halflife in days, mirroring pricing_engine_d3's own
#: recency_weight so the app's decay agrees with D3's.
LEARNING_HALFLIFE_DAYS = 45.0
#: Hard caps on how far learning may move anything, so an accumulated pile of
#: feedback can never move a price more than a rep would accept unexamined.
LEARNING_MAX_PRICE_SHIFT = 0.10      # +/- 10% on the D3 target
LEARNING_MAX_SCORE_SHIFT = 0.08      # +/- 0.08 on a D2 relevance score

# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #

#: Two audiences, two flows.
#:
#: CLIENT mode is what a sales rep uses. It never says "D1" or "relevance
#: scorer", and it skips the two diagnostic screens entirely -- brief in,
#: proposal out, four steps. The engines still all run; the rep just is not
#: made to walk through their internals to get a plan.
#:
#: DEVELOPER mode is the full six-screen build, with engine labels, funnels,
#: score decompositions and solver status. It exists to inspect and demo the
#: machinery.
MODE_CLIENT = "client"
MODE_DEVELOPER = "developer"

#: (key, client label, developer label, stage tag, in_client).
#:
#: `stage` is rendered only in developer mode, and it names the STAGE in plain
#: terms rather than the module ("audience model", not "D1"). Deliverable codes
#: are an internal build detail: they mean nothing to a sales rep and they leak
#: into screenshots and demos, so they do not appear anywhere in the UI.
_STEP_DEFS = [
    ("intake",    "New campaign",   "Intake",          None,                 True),
    ("interpret", "Review brief",   "Interpretation",  "brief parser",       True),
    ("audience",  "Audience",       "Audience",        "audience model",     False),
    ("relevance", "Screen match",   "Relevance",       "match scoring",      False),
    ("package",   "Plan & pricing", "Package & Price", "pricing + optimiser", True),
    ("recommend", "Proposal",       "Recommendation",  "orchestrator",       True),
]


def steps_for(mode: str):
    """[(key, label, engine_or_None)] for the given mode."""
    dev = (mode == MODE_DEVELOPER)
    out = []
    for key, client_label, dev_label, engine, in_client in _STEP_DEFS:
        if not dev and not in_client:
            continue
        out.append((key, dev_label if dev else client_label, engine if dev else None))
    return out


def step_keys_for(mode: str):
    return [s[0] for s in steps_for(mode)]


def step_index(key: str, mode: str = MODE_DEVELOPER) -> int:
    keys = step_keys_for(mode)
    return keys.index(key) if key in keys else 0


#: Business vocabulary. The rep-facing UI uses the right-hand column
#: everywhere; the left-hand terms appear only in developer mode.
GLOSSARY = {
    "relevance score": "match score",
    "deduped reach":   "people reached per day",
    "pool":            "shared audience group",
    "cell":            "screen / time-slot",
    "daypart":         "time of day",
    "funnel":          "shortlisting steps",
    "cold start":      "no booking history yet",
    "solver":          "optimiser",
    "cohort":          "comparable advertisers",
}

#: Six 4-hour blocks, in words a rep would say out loud.
BLOCK_PLAIN = {
    1: "overnight (12am-4am)",
    2: "early morning commute (4am-8am)",
    3: "late morning / lunch (8am-12pm)",
    4: "afternoon (12pm-4pm)",
    5: "evening commute (4pm-8pm)",
    6: "late evening (8pm-12am)",
}

#: Screen types in client language.
SCREEN_TYPE_PLAIN = {
    "bus":              "on-bus screens",
    "bus_stop":         "bus-stop screens",
    "metro_rail_coach": "in-train screens",
    "metro_station":    "metro station screens",
}

POSITION_PLAIN = {
    "platform":      "platform-facing",
    "entrance_exit": "at the entrance",
    "back":          "rear-facing",
    "top":           "overhead",
    "left":          "left-side",
    "right":         "right-side",
}


def azure_configured() -> bool:
    """Same three env vars D5's own _llm_client() requires."""
    return bool(
        os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_ENDPOINT")
        and (os.environ.get("AZURE_OPENAI_DEPLOYMENT_MINI")
             or os.environ.get("AZURE_OPENAI_DEPLOYMENT"))
    )


def load_dotenv_if_present() -> bool:
    """Minimal .env reader -- avoids a python-dotenv dependency for six lines.

    Does not overwrite anything already exported in the real environment.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return False
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
    return True


def ensure_app_data() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

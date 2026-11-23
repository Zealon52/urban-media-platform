"""ADAPT capture layer -- record rep feedback, and show what it WOULD change.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
The problem statement's ADAPT pillar asks for "Act, Learn & Adjust". Today rep
edits change the current solve and nothing else: they never reach D2's scoring
weights or D3's cohort priors. This module closes the *capture* half of that gap
and demonstrates the *update* half as a proposal.

    THE LOOP IS NOT CLOSED. Nothing in this file writes to a model, a weight, a
    prior, or any engine file. `proposed_d2_weight_update()` and
    `proposed_d3_cohort_update()` RETURN dictionaries. Applying them is a human
    decision that has not been made, on a rule that has not been validated.

That is a deliberate scope choice, not an unfinished one: with ~40 rows of
feedback, an automatic update would be fitting noise, and a closed loop nobody
validated is worse than an open one everybody can see. Say this plainly in the
demo.

WRITES ONLY: eda_output/rep_feedback.jsonl (append-only, never truncated). That
path is gitignored -- it is per-clone local state, the same shape as the LLM
ledger and run journal. The committed reference copy is
eda_output/rep_feedback.sample.jsonl (40 synthetic rows); `--seed` regenerates
an equivalent file in a fresh clone.

Makes no API/LLM call. Does not import or modify build_d5_orchestrator.py.

THE FOUR PIECES
---------------
  plan_hash(plan_lines)          stable short id for the plan a rep was shown
  record(...)                    append one feedback row (accept / edit / reject)
  summarise()                    accept rate + mean edit direction, by segment
  proposed_d2_weight_update()    deltas D2's weights WOULD take, arithmetic shown
  proposed_d3_cohort_update()    deltas D3's cohort prior WOULD take, ditto

Usage:
    python feedback_capture.py --seed        # write ~40 synthetic rows, then report
    python feedback_capture.py               # report on whatever has been recorded
    python feedback_capture.py --seed -n 60  # more synthetic rows
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Urban Media Datasets"
OUT_DIR = BASE_DIR / "eda_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_PATH = OUT_DIR / "rep_feedback.jsonl"
QUOTES_PATH = OUT_DIR / "pricing_quotes_d3.csv"
SCREENS_PATH = DATA_DIR / "screens.csv"
CITIES_PATH = DATA_DIR / "cities.csv"
SLOTS_PATH = DATA_DIR / "dim_slot.csv"
LOST_LEADS_PATH = DATA_DIR / "lost_leads.csv"

ACTIONS = ("accept", "edit", "reject")

#: D2's live scoring weights, from build_d2_relevance_scorer.py:549-550. Copied
#: rather than imported: importing that module to read six floats would couple
#: this tool to its import-time behaviour for no benefit. If D2's weights change,
#: this constant has to be re-synced -- `verify_baselines()` checks the sum, which
#: catches a partial edit but not a re-weighting, so the pointer above is the
#: real safeguard.
D2_BASELINE_WEIGHTS: Dict[str, float] = {
    "daypart_fit": 0.25,
    "audience_fit": 0.20,
    "confidence_w": 0.15,
    "tier_fit": 0.10,
    "history_bonus": 0.10,
    "category_fit": 0.20,
}

#: Which feedback segment is evidence about which D2 weight. This mapping is the
#: load-bearing assumption of proposed_d2_weight_update() and is stated here so a
#: reviewer can disagree with it in one place rather than reading the arithmetic.
D2_DIMENSION_TO_WEIGHT: Dict[str, str] = {
    "market_tier": "tier_fit",       # tier_fit is the only weight keyed to tier
    "time_block": "daypart_fit",     # daypart_fit is the only weight keyed to time
    "screen_type": "category_fit",   # category_fit scores screen_type/position
}

#: Weights with no segment in the capture schema. They absorb the redistribution
#: so the vector keeps summing to 1.0; none of them is *evidenced* by feedback.
D2_UNEVIDENCED_WEIGHTS = ("audience_fit", "confidence_w", "history_bonus")

#: Learning rate and clip for the D2 proposal. Deliberately small: this is a
#: demonstration of the mechanism on ~40 rows, not a tuned optimiser.
D2_LEARNING_RATE = 0.10
D2_MAX_ABS_DELTA = 0.05

#: Minimum rows before a segment is allowed to drive a proposal at all.
MIN_SEGMENT_N = 5

#: Shrinkage strength for the D3 cohort prior, in "equivalent observations". The
#: prior is built off 1,450 lost leads; 12 rep edits should barely move it, so
#: the feedback gets weight n/(n + 50).
D3_PRIOR_STRENGTH = 50.0

#: D5's own posture thresholds, from build_d5_orchestrator.py:330-345. Reproduced
#: so the proposal can report whether an update would FLIP the posture.
D3_GAP_THRESHOLD = 0.02


# --------------------------------------------------------------------------- #
# plan identity
# --------------------------------------------------------------------------- #

def plan_hash(plan_lines: Iterable[Any]) -> str:
    """A stable short hash of the selected lines.

    Accepts dicts with screen_id / time_block_id, or (screen_id, time_block_id)
    pairs. Identity is the SET of inventory lines: the list is sorted before
    hashing, so the same plan emitted in a different row order hashes the same.

    Price is deliberately NOT hashed. A rep who edits a price is giving feedback
    ON a plan, not creating a new one; hashing price would give every edit its
    own plan id and make the feedback un-aggregatable -- which is the whole point
    of having a plan id.
    """
    keys: List[str] = []
    for line in plan_lines:
        if isinstance(line, dict):
            sid, tb = line.get("screen_id"), line.get("time_block_id")
        else:
            sid, tb = line[0], line[1]
        if sid is None or tb is None:
            raise ValueError(f"plan line is missing screen_id/time_block_id: {line!r}")
        keys.append(f"{sid}|{int(tb)}")
    if not keys:
        raise ValueError("plan_hash() needs at least one plan line")
    canonical = json.dumps(sorted(keys), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# reference data
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _quote_lookup() -> Dict[tuple, float]:
    """(screen_id, time_block_id) -> the price D3 recommended."""
    if not QUOTES_PATH.exists():
        print(f"[feedback] WARNING: {QUOTES_PATH.name} not found -- quoted_price will "
              f"be null unless passed explicitly.")
        return {}
    q = pd.read_csv(QUOTES_PATH,
                    usecols=["screen_id", "time_block_id", "recommended"])
    return {(r.screen_id, int(r.time_block_id)): float(r.recommended)
            for r in q.itertuples(index=False)}


@lru_cache(maxsize=1)
def _segment_lookup() -> pd.DataFrame:
    """screen_id -> screen_type, city_id, market_tier."""
    s = pd.read_csv(SCREENS_PATH, usecols=["screen_id", "city_id", "screen_type"])
    c = pd.read_csv(CITIES_PATH, usecols=["city_id", "market_tier"])
    return s.merge(c, on="city_id", how="left")


@lru_cache(maxsize=1)
def _slot_labels() -> Dict[int, str]:
    d = pd.read_csv(SLOTS_PATH)
    return {int(r.time_block_id): str(r.time_block_label)
            for r in d.itertuples(index=False)}


@lru_cache(maxsize=1)
def _network_price_baseline() -> Dict[str, float]:
    """The network-wide cohort prior D5 compares against.

    Recomputed here from lost_leads.csv exactly as build_d5_orchestrator.
    build_cohort() does (lines 298-300), so the proposal is stated against the
    real prior rather than a hardcoded guess.
    """
    price_reasons = ("price_too_high", "budget_mismatch", "went_with_competitor")
    ll = pd.read_csv(LOST_LEADS_PATH, usecols=["loss_reason", "price_gap_pct"])
    return {
        "net_price_loss_share": float(ll.loss_reason.isin(price_reasons).mean()),
        "net_price_gap": float(ll.price_gap_pct.median()),
        "n_lost_leads": int(len(ll)),
    }


def verify_baselines() -> None:
    total = sum(D2_BASELINE_WEIGHTS.values())
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(
            f"FATAL: D2_BASELINE_WEIGHTS sums to {total:.6f}, not 1.0 -- re-sync it "
            f"with build_d2_relevance_scorer.py before trusting any proposal.")


# --------------------------------------------------------------------------- #
# 1. record
# --------------------------------------------------------------------------- #

def record(plan_hash: str, screen_id: str, time_block_id: int, action: str,
           note: str = "", *, edited_price: Optional[float] = None,
           quoted_price: Optional[float] = None,
           synthetic: bool = False) -> Dict[str, Any]:
    """Append one row of rep feedback. Never overwrites.

    `plan_hash` here is the STRING id (from plan_hash(), the function of the same
    name -- the brief fixes both names, so inside this function the parameter
    shadows the function; record() does not need to call it).

    `edited_price` is required to make sense of action="edit": without it there is
    no edit DIRECTION, and summarise() cannot say whether reps push price up or
    down. It is keyword-only so the positional signature the brief specifies is
    unchanged.
    """
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
    tb = int(time_block_id)
    if quoted_price is None:
        quoted_price = _quote_lookup().get((screen_id, tb))
    if action == "edit" and edited_price is None:
        raise ValueError(
            "action='edit' needs edited_price -- otherwise the row records that "
            "something changed but not which way, which is the one thing "
            "summarise() needs.")

    edit_pct = None
    if edited_price is not None and quoted_price:
        # Signed: positive = the rep pushed the price UP from what D3 quoted.
        edit_pct = float(edited_price) / float(quoted_price) - 1.0

    row: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan_hash": plan_hash,
        "screen_id": screen_id,
        "time_block_id": tb,
        "action": action,
        "quoted_price": quoted_price,
        "edited_price": float(edited_price) if edited_price is not None else None,
        "edit_pct": edit_pct,
        "note": note,
        "synthetic": bool(synthetic),
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def load_feedback() -> pd.DataFrame:
    """Read the JSONL back, enriched with segment columns."""
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame()
    rows = [json.loads(ln) for ln in FEEDBACK_PATH.read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.merge(_segment_lookup(), on="screen_id", how="left")
    df["time_block"] = df["time_block_id"].map(_slot_labels())
    return df


# --------------------------------------------------------------------------- #
# 2. summarise
# --------------------------------------------------------------------------- #

def _segment_stats(df: pd.DataFrame, dim: str) -> List[Dict[str, Any]]:
    out = []
    for value, g in df.groupby(dim, dropna=False):
        edits = g[g.action == "edit"]
        mean_edit = float(edits.edit_pct.mean()) if len(edits) else None
        out.append({
            "segment": str(value),
            "n": int(len(g)),
            "accept_rate": float((g.action == "accept").mean()),
            "edit_rate": float((g.action == "edit").mean()),
            "reject_rate": float((g.action == "reject").mean()),
            "n_edits": int(len(edits)),
            "mean_edit_pct": mean_edit,
            "edit_direction": (None if mean_edit is None
                               else "up" if mean_edit > 0 else
                               "down" if mean_edit < 0 else "flat"),
        })
    return sorted(out, key=lambda r: r["segment"])


def summarise() -> Dict[str, Any]:
    """Accept rate and mean edit direction, per screen_type / market_tier / time_block."""
    df = load_feedback()
    if df.empty:
        return {"n_rows": 0, "note": "no feedback recorded yet -- run with --seed"}

    edits = df[df.action == "edit"]
    return {
        "n_rows": int(len(df)),
        "n_synthetic": int(df.synthetic.sum()),
        "n_real": int((~df.synthetic.astype(bool)).sum()),
        "n_plans": int(df.plan_hash.nunique()),
        "overall": {
            "accept_rate": float((df.action == "accept").mean()),
            "edit_rate": float((df.action == "edit").mean()),
            "reject_rate": float((df.action == "reject").mean()),
            "n_edits": int(len(edits)),
            "mean_edit_pct": float(edits.edit_pct.mean()) if len(edits) else None,
        },
        "by_screen_type": _segment_stats(df, "screen_type"),
        "by_market_tier": _segment_stats(df, "market_tier"),
        "by_time_block": _segment_stats(df, "time_block"),
    }


# --------------------------------------------------------------------------- #
# 3. proposed D2 weight update
# --------------------------------------------------------------------------- #

def proposed_d2_weight_update(summary: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Deltas D2's scoring weights WOULD take. Returns a proposal; applies nothing.

    The rule, stated once so it can be argued with:

        D2 decides WHICH lines get shortlisted. If one segment of a dimension is
        accepted much less than the book overall, the weight keyed to that
        dimension is admitting lines reps do not want, so it is too HIGH.

        shortfall = overall_accept_rate - worst_segment_accept_rate     (>= 0)
        raw_delta = -learning_rate * shortfall
        delta     = clip(raw_delta, -max_abs_delta, 0)

    Mass removed from the evidenced weights is redistributed equally across the
    weights no segment speaks to, so the vector still sums to 1.0. Every step is
    returned in `arithmetic`.
    """
    verify_baselines()
    summary = summary or summarise()
    if not summary.get("n_rows"):
        return {"status": "no_feedback", "deltas": {},
                "note": "nothing recorded yet -- no proposal can be made"}

    overall = summary["overall"]["accept_rate"]
    steps: List[str] = [
        f"overall accept rate = {overall:.4f} over {summary['n_rows']} rows"]
    deltas: Dict[str, float] = {}
    skipped: List[str] = []

    for dim, weight in D2_DIMENSION_TO_WEIGHT.items():
        segs = [s for s in summary[f"by_{dim}"] if s["n"] >= MIN_SEGMENT_N]
        if not segs:
            skipped.append(f"{dim} -> {weight}: no segment reached n>={MIN_SEGMENT_N}")
            continue
        worst = min(segs, key=lambda s: s["accept_rate"])
        shortfall = overall - worst["accept_rate"]
        if shortfall <= 0:
            skipped.append(
                f"{dim} -> {weight}: worst segment '{worst['segment']}' accepts at "
                f"{worst['accept_rate']:.4f} >= overall {overall:.4f}, no shortfall")
            continue
        raw = -D2_LEARNING_RATE * shortfall
        clipped = max(raw, -D2_MAX_ABS_DELTA)
        deltas[weight] = clipped
        steps.append(
            f"{dim} -> {weight}: worst segment '{worst['segment']}' (n={worst['n']}) "
            f"accepts {worst['accept_rate']:.4f}; shortfall = {overall:.4f} - "
            f"{worst['accept_rate']:.4f} = {shortfall:.4f}; raw delta = "
            f"-{D2_LEARNING_RATE} x {shortfall:.4f} = {raw:+.5f}"
            + (f"; clipped to {clipped:+.5f}" if clipped != raw else ""))

    removed = -sum(deltas.values())
    if removed > 0:
        per = removed / len(D2_UNEVIDENCED_WEIGHTS)
        steps.append(
            f"redistribute {removed:.5f} equally across "
            f"{', '.join(D2_UNEVIDENCED_WEIGHTS)} = {per:+.5f} each "
            f"(these have no segment in the capture schema, so they are not "
            f"evidenced -- they only keep the vector summing to 1.0)")
        for w in D2_UNEVIDENCED_WEIGHTS:
            deltas[w] = deltas.get(w, 0.0) + per

    # Exact arithmetic for the invariant, rounding only for display. Rounding
    # six weights independently at 1e-6 can move their sum by up to 3e-6, which
    # made this check fail on data where the redistribution was correct -- it
    # passed on 40 feedback rows and broke on 50. The real invariant is that the
    # deltas sum to zero, so that is what is asserted.
    exact = {k: D2_BASELINE_WEIGHTS[k] + deltas.get(k, 0.0)
             for k in D2_BASELINE_WEIGHTS}
    total = sum(exact.values())
    delta_sum = sum(deltas.values())
    steps.append(f"deltas sum to {delta_sum:+.2e} (the redistribution conserves "
                 f"mass exactly) so the proposed weights sum to {total:.12f}")

    return {
        "status": "proposal_only_not_applied",
        "baseline_weights": dict(D2_BASELINE_WEIGHTS),
        "deltas": {k: round(v, 6) for k, v in deltas.items()},
        "proposed_weights": {k: round(v, 6) for k, v in exact.items()},
        "proposed_weights_sum_exact": total,
        "deltas_sum_to_zero": abs(delta_sum) < 1e-9,
        "sums_to_one": abs(total - 1.0) < 1e-9,
        "rule": ("delta = clip(-learning_rate * (overall_accept - "
                 "worst_segment_accept), -max_abs_delta, 0), then redistribute"),
        "learning_rate": D2_LEARNING_RATE,
        "max_abs_delta": D2_MAX_ABS_DELTA,
        "min_segment_n": MIN_SEGMENT_N,
        "arithmetic": steps,
        "skipped": skipped,
        "caveat": ("Unvalidated rule on a small sample. The dimension->weight map "
                   "is an assumption (see D2_DIMENSION_TO_WEIGHT), and accept rate "
                   "is a proxy for weight error, not a measurement of it. NOT "
                   "APPLIED anywhere."),
    }


# --------------------------------------------------------------------------- #
# 4. proposed D3 cohort update
# --------------------------------------------------------------------------- #

def proposed_d3_cohort_update(summary: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Deltas D3's cohort priors WOULD take. Returns a proposal; applies nothing.

    The rule:

        A rep's edited price is their estimate of what will actually win. So an
        edit reveals the gap between what we quoted and what the client will pay:

            implied_price_gap = quoted / edited - 1 = 1/(1 + mean_edit_pct) - 1

        That is the same quantity as Cohort.median_price_gap (quoted vs the
        client's own target), so it can be blended with the prior directly.
        Shrunk by evidence weight so a handful of edits cannot swing a prior
        built on 1,450 lost leads:

            w         = n_edits / (n_edits + prior_strength)
            new_gap   = prior_gap * (1 - w) + implied_gap * w

    The proposal also reports whether the update would FLIP D5's posture, using
    D5's own +/-0.02 threshold.
    """
    summary = summary or summarise()
    if not summary.get("n_rows"):
        return {"status": "no_feedback", "deltas": {},
                "note": "nothing recorded yet -- no proposal can be made"}

    base = _network_price_baseline()
    prior_gap = base["net_price_gap"]
    n_edits = summary["overall"]["n_edits"]
    mean_edit = summary["overall"]["mean_edit_pct"]

    steps = [f"prior median_price_gap = {prior_gap:+.5f} (median price_gap_pct over "
             f"{base['n_lost_leads']:,} lost leads)",
             f"prior net_price_loss_share = {base['net_price_loss_share']:.5f}"]

    if not n_edits or mean_edit is None:
        return {"status": "proposal_only_not_applied", "deltas": {},
                "arithmetic": steps + ["no edit rows -- no price evidence to blend"],
                "note": "accepts and rejects alone do not identify a price gap"}

    implied_gap = 1.0 / (1.0 + mean_edit) - 1.0
    w = n_edits / (n_edits + D3_PRIOR_STRENGTH)
    new_gap = prior_gap * (1.0 - w) + implied_gap * w
    delta_gap = new_gap - prior_gap

    steps += [
        f"mean_edit_pct = {mean_edit:+.5f} over {n_edits} edit rows "
        f"(negative = reps cut the price)",
        f"implied_price_gap = 1/(1 + {mean_edit:+.5f}) - 1 = {implied_gap:+.5f}",
        f"evidence weight w = {n_edits}/({n_edits} + {D3_PRIOR_STRENGTH:g}) = {w:.5f}",
        f"new_gap = {prior_gap:+.5f} x (1 - {w:.5f}) + {implied_gap:+.5f} x {w:.5f} "
        f"= {new_gap:+.5f}",
        f"delta median_price_gap = {new_gap:+.5f} - {prior_gap:+.5f} = {delta_gap:+.5f}",
    ]

    # Would this flip the posture? build_cohort() (build_d5_orchestrator.py:330-345)
    # requires BOTH a gap move and a price-loss-share move in the same direction:
    #
    #   if   d_gap > +0.02 and d_loss > 0 -> open_below_target
    #   elif d_gap < -0.02 and d_loss < 0 -> hold_at_cap
    #   else                              -> hold_at_target
    #
    # Reproducing only the d_gap half would over-claim a flip, so both are
    # evaluated. d_loss is 0.0 because this schema cannot evidence a loss-share
    # move (see `not_proposed`), which is precisely why the posture holds.
    d_gap = new_gap - prior_gap
    d_loss = 0.0
    if d_gap > D3_GAP_THRESHOLD and d_loss > 0.0:
        posture = "open_below_target"
    elif d_gap < -D3_GAP_THRESHOLD and d_loss < 0.0:
        posture = "hold_at_cap"
    else:
        posture = "hold_at_target"

    gap_cond = ("met" if abs(d_gap) > D3_GAP_THRESHOLD else "not met")
    steps.append(
        f"posture check needs BOTH conditions (D5 build_cohort): "
        f"gap condition |d_gap {d_gap:+.5f}| > {D3_GAP_THRESHOLD} is {gap_cond}; "
        f"loss condition d_loss {d_loss:+.5f} != 0 is not met "
        f"(no loss-share delta is proposed) -> '{posture}'"
        + (" (UNCHANGED)" if posture == "hold_at_target"
           else " (WOULD FLIP from 'hold_at_target')"))
    if abs(d_gap) > D3_GAP_THRESHOLD and posture == "hold_at_target":
        steps.append(
            "  NOTE: the price-gap evidence alone WOULD clear D5's threshold. The "
            "posture holds only because rejects are not reason-coded, so d_loss "
            "stays 0. That is the concrete cost of the schema gap below.")

    return {
        "status": "proposal_only_not_applied",
        "baseline_priors": base,
        "deltas": {"median_price_gap": round(delta_gap, 6)},
        "proposed_priors": {"median_price_gap": round(new_gap, 6)},
        "implied_posture": posture,
        "posture_would_flip": posture != "hold_at_target",
        "rule": ("implied_gap = 1/(1+mean_edit_pct) - 1; blended with the prior at "
                 "weight n/(n + prior_strength)"),
        "prior_strength": D3_PRIOR_STRENGTH,
        "arithmetic": steps,
        "not_proposed": [
            "net_price_loss_share: rejects are not reason-coded in this schema, so a "
            "reject cannot be attributed to price. Adding a closed reason vocabulary "
            "to record() is the next schema change, and until then no loss-share "
            "delta is proposed rather than one inferred from free-text notes.",
        ],
        "caveat": ("Unvalidated rule on a small sample. Treats the rep's edited "
                   "price as a correct estimate of the winning price, which is "
                   "exactly the assumption a closed loop would need to earn. NOT "
                   "APPLIED anywhere."),
    }


# --------------------------------------------------------------------------- #
# 5. synthetic seed
# --------------------------------------------------------------------------- #

def seed(n: int = 40, rng_seed: int = 20260823, force: bool = False) -> int:
    """Write ~n plausible SYNTHETIC feedback rows, each flagged synthetic=true.

    Exists so summarise() and both proposal functions can be demonstrated before
    any real rep has touched the tool. Appends like everything else.
    """
    existing = load_feedback()
    if len(existing) and not force:
        n_syn = int(existing.synthetic.sum()) if "synthetic" in existing else 0
        print(f"[feedback] {FEEDBACK_PATH.name} already has {len(existing)} rows "
              f"({n_syn} synthetic). Not seeding again -- that would double-count. "
              f"Pass --force to append anyway.")
        return 0

    if not QUOTES_PATH.exists():
        raise SystemExit(f"FATAL: need {QUOTES_PATH} to seed against real screens.")

    q = pd.read_csv(QUOTES_PATH, usecols=["screen_id", "time_block_id", "recommended"])
    seg = _segment_lookup()
    q = q.merge(seg, on="screen_id", how="left")

    rng = random.Random(rng_seed)
    # Spread the sample across screen_type x market_tier x time_block so every
    # segment the summary reports on actually has rows in it.
    picks = []
    for (_st, _mt), g in q.groupby(["screen_type", "market_tier"], dropna=True):
        take = max(1, n // max(1, q.groupby(["screen_type", "market_tier"]).ngroups))
        idx = rng.sample(range(len(g)), min(take + 2, len(g)))
        picks.extend(g.iloc[idx].to_dict("records"))
    rng.shuffle(picks)
    picks = picks[:n]

    # Two plans, so plan_hash() is exercised and feedback is attributable.
    half = len(picks) // 2
    plans = [plan_hash([(r["screen_id"], r["time_block_id"]) for r in picks[:half]]),
             plan_hash([(r["screen_id"], r["time_block_id"]) for r in picks[half:]])]

    # Plausible, not uniform: reps mostly accept, edit a fair bit, reject rarely,
    # and their edits lean DOWN (the behaviour the pricing gap predicts).
    written = 0
    for i, r in enumerate(picks):
        ph = plans[0] if i < half else plans[1]
        roll = rng.random()
        tier = r.get("market_tier")
        # premium inventory gets pushed back on more often -- gives the D2
        # proposal a real tier signal to find.
        reject_p = 0.20 if tier == "premium" else 0.08
        if roll < reject_p:
            action, edited, note = "reject", None, rng.choice(
                ["client wants a different corridor", "off-brief location",
                 "client pushed back on this board"])
        elif roll < reject_p + 0.30:
            action = "edit"
            pct = rng.gauss(-0.035, 0.030)  # centred below quote
            edited = round(float(r["recommended"]) * (1.0 + pct), 2)
            note = rng.choice(["trimmed to match the client's target",
                               "matched a competing quote",
                               "held margin, small trim to close"])
        else:
            action, edited, note = "accept", None, rng.choice(
                ["quoted as recommended", "client accepted on the call", ""])
        record(ph, r["screen_id"], int(r["time_block_id"]), action, note,
               edited_price=edited, quoted_price=float(r["recommended"]),
               synthetic=True)
        written += 1

    print(f"[feedback] wrote {written} synthetic rows to {FEEDBACK_PATH.name} "
          f"across {len(plans)} plans (all flagged \"synthetic\": true)")
    return written


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def _print_segment_table(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    print(f"\n  {title}")
    print(f"    {'segment':<16}{'n':>5}{'accept':>9}{'edit':>8}{'reject':>8}"
          f"{'mean edit':>11}{'dir':>6}")
    for r in rows:
        me = "-" if r["mean_edit_pct"] is None else f"{r['mean_edit_pct']:+.2%}"
        print(f"    {r['segment']:<16}{r['n']:>5}{r['accept_rate']:>9.1%}"
              f"{r['edit_rate']:>8.1%}{r['reject_rate']:>8.1%}{me:>11}"
              f"{r['edit_direction'] or '-':>6}")


def report() -> None:
    s = summarise()
    print("\n" + "=" * 78)
    print("ADAPT FEEDBACK REPORT")
    print("=" * 78)
    if not s.get("n_rows"):
        print("\n  No feedback recorded yet. Run:  python feedback_capture.py --seed\n")
        return

    o = s["overall"]
    print(f"\n  {s['n_rows']} rows over {s['n_plans']} plans "
          f"({s['n_synthetic']} synthetic, {s['n_real']} real)")
    print(f"  overall: accept {o['accept_rate']:.1%} | edit {o['edit_rate']:.1%} | "
          f"reject {o['reject_rate']:.1%}")
    if o["mean_edit_pct"] is not None:
        print(f"  mean edit direction across {o['n_edits']} edits: "
              f"{o['mean_edit_pct']:+.2%} "
              f"({'reps cut price' if o['mean_edit_pct'] < 0 else 'reps raised price'})")
    if s["n_real"] == 0:
        print("\n  NOTE: every row is SYNTHETIC. These numbers demonstrate the "
              "mechanism;\n        they are not evidence about real rep behaviour.")

    _print_segment_table("by screen_type", s["by_screen_type"])
    _print_segment_table("by market_tier", s["by_market_tier"])
    _print_segment_table("by time_block", s["by_time_block"])

    for title, prop in (("PROPOSED D2 WEIGHT UPDATE", proposed_d2_weight_update(s)),
                        ("PROPOSED D3 COHORT UPDATE", proposed_d3_cohort_update(s))):
        print("\n" + "-" * 78)
        print(f"{title}   [{prop['status']}]")
        print("-" * 78)
        for line in prop.get("arithmetic", []):
            print(f"  {line}")
        if prop.get("skipped"):
            print("\n  skipped:")
            for line in prop["skipped"]:
                print(f"    - {line}")
        if prop.get("deltas"):
            print("\n  deltas that WOULD be applied:")
            for k, v in sorted(prop["deltas"].items()):
                base = prop.get("baseline_weights", {}).get(k)
                if base is None:
                    base = prop.get("baseline_priors", {}).get(k)
                tail = ""
                if isinstance(base, (int, float)):
                    tail = f"   ({base:+.5f} -> {base + v:+.5f})"
                print(f"    {k:<22}{v:+.5f}{tail}")
        if prop.get("proposed_weights"):
            print(f"\n  proposed weight vector (sums to "
                  f"{prop['proposed_weights_sum_exact']:.12f}):")
            for k, v in prop["proposed_weights"].items():
                print(f"    {k:<22}{D2_BASELINE_WEIGHTS[k]:.4f} -> {v:.4f}")
        if prop.get("posture_would_flip") is not None:
            print(f"\n  implied posture: {prop['implied_posture']}"
                  f"{'  (WOULD FLIP)' if prop['posture_would_flip'] else '  (unchanged)'}")
        for line in prop.get("not_proposed", []):
            print(f"\n  not proposed: {line}")
        print(f"\n  caveat: {prop['caveat']}")

    print("\n" + "=" * 78)
    print("THE LOOP IS NOT CLOSED. Nothing above was applied to D2, D3, or any")
    print("model file. These are proposals for a human to accept or reject.")
    print("=" * 78 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", action="store_true",
                    help="append ~40 synthetic feedback rows before reporting")
    ap.add_argument("-n", "--rows", type=int, default=40,
                    help="how many synthetic rows to write with --seed")
    ap.add_argument("--force", action="store_true",
                    help="seed even if the file already has rows")
    args = ap.parse_args()

    verify_baselines()
    if args.seed:
        seed(args.rows, force=args.force)
    report()


if __name__ == "__main__":
    main()

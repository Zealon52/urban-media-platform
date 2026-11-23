"""
D3 enhancement -- win/loss expected-revenue pricing + relationship layer,
with the GradientBoostingRegressor (model_d3_price.joblib) now ACTUALLY
wired into the serving path, not just described in a docstring.

--------------------------------------------------------------------------
FIX (P0, from teammate review): the GBM was orphaned, and would have
crashed if called
--------------------------------------------------------------------------
Previously this module's docstring said it "upgrades" the GBM point
estimate, but the serving code never loaded or called model_d3_price.joblib
at all -- it only ever used the empirical rate-card percentiles
(price_p10/p50/p90 from pricing_grain_d3.csv). Separately, the saved GBM
itself could not have been called safely even if it had been wired in:
model_selector.py's training script does `X.fillna(X.median(...))` right
before fitting, but never PERSISTS those median values anywhere the saved
.joblib can reuse at serving time -- so a fresh call to model.predict() on
pricing_grain_d3_encoded.csv would crash on NaN (~70% missing in
location_footfall_log, ~81% in corridor_ridership_log, structural gaps from
the mount_type branch; ~37% in avg_lead_time_days/bundle_share from
cold-start screens; ~23% in income_index from a zone-join gap).

Both are fixed now:
  1. encode_pricing_grain.py (v2) does principled, PERSISTED imputation
     (0 + has_footfall/has_ridership flags for the structural gaps; stored
     training-median for the cold-start gaps) and asserts zero remaining
     NaN in every feature column before saving -- see that file's docstring
     for the full reasoning. Retrain model_d3_price.joblib against THAT
     fixed encoded CSV before using this module.
  2. This module now actually loads model_d3_price.joblib (see
     load_gbm_price_predictions() below) and BLENDS its prediction into the
     rate card's target (price_p50) by booking-history credibility: a
     screen/block with abundant real bookings trusts its own empirical
     history; a screen/block with thin or zero history (the ~1,224
     never-booked screens this whole exercise cares about) leans on the
     ML model instead of a coarse 3-way peer-group average, since the GBM
     saw far more granular features (POI footfall, demand_intensity,
     market_tier, etc.) than a groupby(city, mount_type, screen_type).mean()
     ever could. price_p10/p90 (the floor/cap band) are left as the
     existing empirical/peer-fallback logic -- the GBM was only ever
     trained as a point estimate (price_p50), not a quantile regressor, so
     it has nothing principled to say about the WIDTH of the band, only
     its center.

If model_d3_price.joblib or its .model_card.json is missing (e.g. you
haven't retrained yet), this module still runs correctly -- it prints a
clear warning and falls back to the pre-fix behavior (pure empirical rate
card), rather than crashing. Nothing here is a hard dependency.

--------------------------------------------------------------------------
Everything below this point is UNCHANGED from before this fix:
--------------------------------------------------------------------------
Also gives a defensible floor/target/cap band with a client-specific
recommended price, a win probability, and named discount/premium
components -- ported from the urban_media_D1 reference's src/pricing.py,
re-fit on OUR verified real data instead of that reference's placeholder
logistic prior.

    expected_revenue(p) = p * P(win | p, client)
    recommended          = argmax expected_revenue(p), inside [floor, cap]

Data sources (all verified against real eda_engine.py JSON reports this
session -- do not re-guess column names or category values):

  eda_output/win_loss_table_d3.csv   -- bookings (won=True) + price-driven
                                         lost_leads (won=False), from
                                         build_training_table.py
  eda_output/pricing_grain_d3.csv    -- screen_id x time_block_id rate card
                                         (price_p10/p50/p90, has_history)
  Urban Media Datasets/client_facts.csv, bookings.csv, lost_leads.csv
                                      -- raw, for dates + relationship fields

IMPORTANT -- what is real vs. heuristic in the relationship layer:
  client_facts.csv has NO group_id / parent-company / is_agency column (we
  checked the real schema; only company_name, industry, client_tier,
  bundle_affinity, negotiation_leverage, account_status, etc. exist). So:
    - group_key   is DERIVED from company_name (strip legal suffixes, take
                   the name root). This is a heuristic proxy for a holding
                   group / agency relationship, not verified ground truth --
                   exactly the fallback the reference implementation itself
                   documents ("until [a parent column] lands, derive by
                   shared name-root").
    - category effects (client_tier, negotiation_leverage, bundle_affinity,
                   account_status) are NOT given assumed label semantics
                   (e.g. we do NOT assume "platinum" > "gold" or guess which
                   string means what -- three separate guesses on other
                   columns were wrong this session: market_type/market_tier,
                   booking_status values, loss_reason values). Instead each
                   category's real effect is measured empirically: the mean
                   price_ratio (offered_price / rate_card) among WON deals
                   for that category, relative to the global mean. A rep can
                   say "clients with this attribute value have historically
                   transacted at N% of rate card" -- a claim backed by the
                   data actually in front of us, not an assumed label meaning.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import event_overlap as _evo  # sibling module -- date-aware event overlap
                                # (see event_overlap.py docstring for why this
                                # is separate from the static event_demand_boost_log
                                # already baked into pricing_grain_d3.csv)
import ridership_trend as _rtr  # sibling module -- date-aware ridership TREND +
                                # weekly seasonality, applied as a SECOND,
                                # independent price-band shift on top of the event
                                # overlap shift above. Unlike the event adjustment,
                                # this is bidirectional (a real declining trend can
                                # lower price, not just a rising one raise it) and
                                # only ever fires for vehicle-mounted screens that
                                # resolve to a real corridor_id -- location-mounted
                                # screens correctly fall back to no adjustment (see
                                # ridership_trend.py's forecast_multiplier_for_corridor_block
                                # docstring for the verified fallback behavior).

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Urban Media Datasets"
OUT_DIR = BASE_DIR / "eda_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECENCY_HALFLIFE_DAYS = 90.0
RELATIONSHIP_CATEGORY_COLS = ["client_tier", "negotiation_leverage", "bundle_affinity", "account_status"]

# Common legal/agency suffixes to strip when deriving a group_key from
# company_name. This is a heuristic (see module docstring) -- verify the
# printed sample of derived groups before trusting it for a real quote.
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|co|company|group|holdings?|"
    r"media|communications|partners|agency|worldwide|global|international)\b\.?",
    flags=re.IGNORECASE,
)


def recency_weight(age_days: float, halflife_days: float = RECENCY_HALFLIFE_DAYS) -> float:
    """Weight for a historical win/loss observation -- older leads count less.

    Directly implements the briefing's note that "the lead expiry date is in
    the data -- consider how you weight it."
    """
    if age_days is None or (isinstance(age_days, float) and math.isnan(age_days)):
        return 0.5  # unknown age -> half weight rather than full or zero
    return 0.5 ** (max(0.0, age_days) / max(1.0, halflife_days))


def derive_group_key(company_name: str) -> str:
    """Heuristic parent-group proxy from company_name. See module docstring."""
    if not isinstance(company_name, str) or not company_name.strip():
        return "unknown"
    cleaned = _LEGAL_SUFFIX_RE.sub("", company_name)
    cleaned = re.sub(r"[^a-z0-9 ]", "", cleaned.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    tokens = cleaned.split(" ")
    root = " ".join(tokens[:2]) if tokens and tokens[0] else company_name.lower().strip()
    return root or "unknown"


# --------------------------------------------------------------------------- #
# Win-probability model -- fit on OUR real bookings + lost_leads win/loss data
# --------------------------------------------------------------------------- #

class WinModel:
    """P(win | price_ratio). price_ratio = offered_price / rate_card_price.

    Fit by damped Newton / IRLS on real (price_ratio, won, weight) triples --
    ported from the urban_media_D1 reference's src/pricing.py WinModel, which
    that build could only seed with a placeholder prior because it hadn't
    joined bookings + lost_leads yet. We have both, verified, so this fits on
    real censored win/loss data rather than a guess.
    """

    #: A fit is only accepted if `base_ratio` -- the price ratio at which we win
    #: half the time -- lands somewhere a real market could put it. Outside this
    #: band the curve is not a demand curve, whatever its slope says.
    #:
    #: WHY A SLOPE CHECK IS NOT ENOUGH. `b1 < -1e-3` only asks "does win
    #: probability fall as price rises". It says nothing about WHERE the curve
    #: sits. The committed run passed the slope check and produced
    #: base_ratio = 12.75, i.e. a claim that we win half our deals at 1275% of
    #: reference. At that location p_win is 0.998 flat across the whole
    #: floor-target-cap band -- it varies by 0.00026 -- so expected revenue
    #: rises monotonically with price and argmax pins to the cap on 100.0% of
    #: 133,956 quotes. A slope-only guard cannot see that.
    BASE_RATIO_BOUNDS = (0.80, 1.40)

    #: And a bound on STEEPNESS. base_ratio says where the curve sits;
    #: elasticity says how fast it falls. Both can be wrong independently, and
    #: a location check alone lets a separating hyperplane through.
    #:
    #: HOW THIS WAS FOUND. Restricting the price-loss definition to
    #: {price_too_high, budget_mismatch} makes the two groups almost perfectly
    #: separable by price_gap_pct -- price losses cluster at 0.30-0.33, the rest
    #: at 0.02-0.04 -- so IRLS drives the slope toward infinity. It returned
    #: elasticity 573.22 with base_ratio 1.0111, which is INSIDE the bounds
    #: above and was therefore accepted. At 573 a rep sees p_win fall from 0.99
    #: to 0.01 across a 1.6% price move: a 1% rise loses every deal. No market
    #: does that; it is the classic logistic perfect-separation artefact.
    #:
    #: 40 is generous. The identified fit lands at 14.2 on the full price-loss
    #: definition and 13.1 on price_too_high alone, so anything past 40 is the
    #: solver reporting separation rather than measuring demand.
    ELASTICITY_BOUNDS = (0.5, 40.0)

    def __init__(self, base_ratio: float = 1.0, elasticity: float = 6.0):
        self.base_ratio = base_ratio
        self.elasticity = elasticity
        self.fitted = False
        self.n_obs = 0
        #: Why a fit was refused, or "" if it was accepted. Written into
        #: win_model_fit_report.json so a refusal is visible in the artefact
        #: rather than only in stdout.
        self.reject_reason = ""

    @staticmethod
    def _loglik(obs, b0: float, b1: float) -> float:
        ll = 0.0
        for r, y, wt in obs:
            z = b0 + b1 * r
            softplus = z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))
            ll += wt * (y * z - softplus)
        return ll

    def fit(self, observations: Sequence[Tuple[float, bool, float]]) -> "WinModel":
        obs = [(r, 1.0 if w else 0.0, max(0.0, wt))
               for r, w, wt in observations if r is not None and r > 0 and wt is not None]
        obs = [o for o in obs if o[2] > 0]
        if len(obs) < 20:
            print(f"[win model] only {len(obs)} usable observations -- keeping the prior "
                  f"(base_ratio={self.base_ratio}, elasticity={self.elasticity})")
            return self
        wins = sum(1 for _r, y, _w in obs if y > 0.5)
        if wins == 0 or wins == len(obs):
            print(f"[win model] {wins}/{len(obs)} observations are wins -- no contrast, "
                  f"keeping the prior")
            return self

        b0, b1 = 0.0, 0.0
        ll = self._loglik(obs, b0, b1)
        ridge = 1e-6

        for _ in range(100):
            g0 = g1 = h00 = h01 = h11 = 0.0
            for r, y, wt in obs:
                z = b0 + b1 * r
                pz = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                g0 += wt * (y - pz)
                g1 += wt * (y - pz) * r
                w = wt * pz * (1.0 - pz)
                h00 += w
                h01 += w * r
                h11 += w * r * r
            h00 += ridge
            h11 += ridge
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-12:
                break
            d0 = (h11 * g0 - h01 * g1) / det
            d1 = (h00 * g1 - h01 * g0) / det

            step, improved = 1.0, False
            for _ in range(30):
                cand0, cand1 = b0 + step * d0, b1 + step * d1
                cand_ll = self._loglik(obs, cand0, cand1)
                if cand_ll > ll + 1e-12:
                    b0, b1, ll, improved = cand0, cand1, cand_ll, True
                    break
                step *= 0.5
            if not improved or (abs(step * d0) < 1e-9 and abs(step * d1) < 1e-9):
                break

        # Two checks, not one: the slope must fall with price AND the curve must
        # sit somewhere plausible. See BASE_RATIO_BOUNDS for why the second is
        # load-bearing.
        if b1 >= -1e-3:
            self.reject_reason = (f"slope did not fall with price (b1={b1:.4f})")
            print(f"[win model] REFUSED the fit: {self.reject_reason} -- keeping the "
                  f"prior (base_ratio={self.base_ratio}, elasticity={self.elasticity}) "
                  f"rather than trusting a confounded fit")
            return self

        cand_elasticity = -b1
        cand_base_ratio = -b0 / b1

        # Steepness first: a separating fit can land at a perfectly plausible
        # LOCATION, so checking base_ratio alone would accept it.
        e_lo, e_hi = self.ELASTICITY_BOUNDS
        if not (e_lo <= cand_elasticity <= e_hi):
            self.reject_reason = (
                f"elasticity {cand_elasticity:.3f} outside plausible "
                f"[{e_lo}, {e_hi}] (base_ratio {cand_base_ratio:.3f} was fine)")
            print(f"[win model] REFUSED the fit: {self.reject_reason}")
            if cand_elasticity > e_hi:
                print(f"[win model]   that steep means p_win falls from ~1 to ~0 across "
                      f"a ~{100 * 9.2 / max(cand_elasticity, 1e-9):.2f}% price move -- the "
                      f"signature of perfect separation in the logistic fit, not a "
                      f"demand curve. Check whether the two classes are separable "
                      f"on the feature before trusting any slope.")
            print(f"[win model]   keeping the prior (base_ratio={self.base_ratio}, "
                  f"elasticity={self.elasticity}).")
            return self

        lo, hi = self.BASE_RATIO_BOUNDS
        if not (lo <= cand_base_ratio <= hi):
            wins = sum(1 for _r, y, _w in obs if y > 0.5)
            losses = len(obs) - wins
            self.reject_reason = (
                f"base_ratio {cand_base_ratio:.3f} outside plausible [{lo}, {hi}] "
                f"(elasticity {cand_elasticity:.3f} was fine)")
            print(f"[win model] REFUSED the fit: {self.reject_reason}")
            print(f"[win model]   sample was {wins:,} wins vs {losses:,} losses "
                  f"({100*wins/max(len(obs),1):.2f}% wins). Bookings are TRANSACTED "
                  f"prices, so they are accepted by construction -- that comparison "
                  f"cannot identify a demand curve at any sample size.")
            print(f"[win model]   keeping the prior (base_ratio={self.base_ratio}, "
                  f"elasticity={self.elasticity}). Use fit_from_price_gap() on the "
                  f"identified sample instead.")
            return self

        self.elasticity = cand_elasticity
        self.base_ratio = cand_base_ratio
        self.fitted = True
        self.n_obs = len(obs)
        self.reject_reason = ""
        print(f"[win model] fitted on {self.n_obs} real observations -> "
              f"base_ratio={self.base_ratio:.4f}, elasticity={self.elasticity:.4f}")
        return self

    # ----------------------------------------------------------------------- #
    # The identified fit
    # ----------------------------------------------------------------------- #

    def fit_from_price_gap(self, price_gap_pct, price_driven,
                           weights=None) -> "WinModel":
        """Fit on the comparison that IS identified.

        Bookings-vs-leads is 191,109 wins to 546 losses -- 99.7% wins -- and
        bookings are transacted prices, so they are wins by construction. No
        amount of that data identifies a demand curve; it identifies the
        selection rule that produced the sample.

        The identified comparison lives INSIDE lost_leads: every row is a deal
        we did not win, and `loss_reason` says whether PRICE was why. Same
        population on both sides, so selection cancels.

        `price_gap_pct` = quoted / client_target - 1, i.e. how far above the
        client's own stated target we quoted. A "win" here means the loss was
        NOT price-driven, so the curve reads as p(price was acceptable).
        """
        obs = []
        n = len(price_gap_pct)
        for i in range(n):
            gap = price_gap_pct.iloc[i] if hasattr(price_gap_pct, "iloc") else price_gap_pct[i]
            pd_flag = price_driven.iloc[i] if hasattr(price_driven, "iloc") else price_driven[i]
            if gap is None or (isinstance(gap, float) and math.isnan(gap)):
                continue
            wt = 1.0
            if weights is not None:
                wt = weights.iloc[i] if hasattr(weights, "iloc") else weights[i]
            # ratio relative to the client's own target price
            obs.append((1.0 + float(gap), not bool(pd_flag), float(wt)))
        if not obs:
            self.reject_reason = "no usable price_gap_pct rows"
            print(f"[win model] {self.reject_reason} -- keeping the prior")
            return self
        return self.fit(obs)

    def calibrate_to_reference(self) -> "WinModel":
        """Shift the curve so the band target is the revenue-optimal price.

        Without this the fitted location is relative to the CLIENT's target,
        not to our rate card, and argmax collapses to whichever endpoint
        happens to sit on the steep side. Revenue p * p_win(p) is maximised
        where p_win = 1 - 1/elasticity, so we solve base_ratio for that point.
        """
        e = self.elasticity
        if e <= 1.0:
            self.reject_reason = f"elasticity {e:.3f} too flat to calibrate"
            print(f"[win model] {self.reject_reason} -- leaving the curve as fitted")
            return self
        p_star = min(max(1.0 - 1.0 / e, 1e-6), 1 - 1e-6)
        self.base_ratio = 1.0 + math.log(p_star / (1.0 - p_star)) / e
        print(f"[win model] calibrated to reference -> base_ratio={self.base_ratio:.4f} "
              f"(the optimum now sits at the band target)")
        return self

    def p_win(self, price_ratio: float, relationship_shift: float = 0.0) -> float:
        z = -self.elasticity * (price_ratio - (self.base_ratio + relationship_shift))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def p_win_vec(self, price_ratio: np.ndarray, relationship_shift: np.ndarray) -> np.ndarray:
        z = -self.elasticity * (price_ratio - (self.base_ratio + relationship_shift))
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def to_report(self) -> dict:
        return {"fitted": self.fitted, "n_obs": self.n_obs,
                "base_ratio": self.base_ratio, "elasticity": self.elasticity,
                # A refusal must be visible in the artefact, not only in stdout.
                # The committed report said "fitted": true at base_ratio 12.75,
                # which is why nobody noticed.
                "reject_reason": self.reject_reason,
                "base_ratio_bounds": list(self.BASE_RATIO_BOUNDS),
                "elasticity_bounds": list(self.ELASTICITY_BOUNDS)}


# --------------------------------------------------------------------------- #
# Loading + assembling the real win/loss observations
# --------------------------------------------------------------------------- #

def load_win_loss_and_grain() -> Tuple[pd.DataFrame, pd.DataFrame]:
    wl_path = OUT_DIR / "win_loss_table_d3.csv"
    grain_path = OUT_DIR / "pricing_grain_d3.csv"
    if not wl_path.exists() or not grain_path.exists():
        raise FileNotFoundError(
            f"Missing {wl_path.name} or {grain_path.name} in {OUT_DIR}. "
            f"Run build_training_table.py first -- this module builds ON TOP of "
            f"that output, it does not rebuild the joins from scratch."
        )
    win_loss = pd.read_csv(wl_path)
    grain = pd.read_csv(grain_path)
    print(f"loaded win_loss_table_d3: {win_loss.shape}, pricing_grain_d3: {grain.shape}")
    return win_loss, grain


def load_gbm_price_predictions(
    model_path: Optional[Path] = None,
    model_card_path: Optional[Path] = None,
    encoded_grain_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Loads model_d3_price.joblib + its .model_card.json (the feature-column
    list persisted by model_selector.py), aligns pricing_grain_d3_encoded.csv
    to that EXACT column list, and returns a per-cell GBM price_p50
    prediction keyed by screen_id + time_block_id.

    Returns None (with a printed warning) if the model or its card is
    missing -- callers must treat None as "no ML signal available this run,
    fall back to pure empirical/peer-fallback pricing", never as an error.
    Only RAISES if the files exist but are structurally incompatible (stale
    model vs. a since-changed encoded file, or NaNs slipping through) --
    those are real bugs worth stopping on, not silently working around.
    """
    model_path = model_path or (BASE_DIR / "model_d3_price.joblib")
    model_card_path = model_card_path or Path(str(model_path) + ".model_card.json")
    encoded_grain_path = encoded_grain_path or (OUT_DIR / "pricing_grain_d3_encoded.csv")

    if not model_path.exists() or not model_card_path.exists():
        print(f"[gbm] {model_path.name} or {model_card_path.name} not found -- skipping "
              f"ML blend, using pure empirical/peer-fallback rate card only. (Run "
              f"encode_pricing_grain.py then model_selector.py to produce these and "
              f"enable the blend.)")
        return None
    if not encoded_grain_path.exists():
        print(f"[gbm] {encoded_grain_path.name} not found -- skipping ML blend.")
        return None

    # Imported lazily because this module must still run with zero ML
    # artifacts present. The ImportError guard is NOT theoretical: joblib is
    # absent from both interpreters on the primary dev machine right now, and
    # without this the bare import propagated out of D5's constructor and
    # crashed the entire pipeline -- on a code path whose own docstring
    # promises it degrades to the empirical rate card. A missing FILE was
    # handled; a missing PACKAGE took the demo down. Same failure, one layer up.
    try:
        import joblib
    except ImportError:
        print("[gbm] the 'joblib' package is not installed in this "
              "interpreter -- skipping the ML blend and using the pure "
              "empirical/peer-fallback rate card. Prices remain valid and "
              "fully explainable; they just lose the model's cold-start "
              "contribution. `pip install joblib scikit-learn` to enable it.")
        return None
    model = joblib.load(model_path)
    with open(model_card_path) as f:
        card = json.load(f)
    # model_selector.py writes the key as `feature_columns_in_order`; this
    # loader looked only for `feature_columns`, so even with the .joblib in
    # place the blend silently declined to run and printed "has no
    # 'feature_columns' entry". The committed
    # model_d3_price.joblib.model_card.json proves it: its keys are
    # ['target', 'task_type', 'feature_columns_in_order', 'exclude_columns',
    #  'feature_engineering', 'diagnostics_gate'] -- 41 features, under the
    # other name. Accept both rather than renaming one and breaking the other.
    feature_columns = (card.get("feature_columns")
                       or card.get("feature_columns_in_order"))
    if not feature_columns:
        print(f"[gbm] {model_card_path.name} has neither 'feature_columns' nor "
              f"'feature_columns_in_order' -- "
              f"skipping ML blend.")
        return None

    encoded = pd.read_csv(encoded_grain_path)
    missing_cols = [c for c in feature_columns if c not in encoded.columns]
    if missing_cols:
        raise RuntimeError(
            f"model_card.json expects features {missing_cols} that are not present in "
            f"{encoded_grain_path.name}. The encoded grain and the saved model are out of "
            f"sync -- retrain against the CURRENT encoded file before using this module."
        )

    X = encoded[feature_columns].copy()
    remaining_na = X.isna().sum()
    remaining_na = remaining_na[remaining_na > 0]
    if len(remaining_na) > 0:
        raise RuntimeError(
            f"NaNs found in the GBM feature matrix at serving time -- this is exactly the "
            f"P0 bug this fix addresses. Re-run encode_pricing_grain.py (v2) to regenerate "
            f"{encoded_grain_path.name} with the persisted imputation applied before "
            f"retraining. Columns still missing values: {remaining_na.to_dict()}"
        )

    preds = model.predict(X)
    out = encoded[["screen_id", "time_block_id"]].copy()
    out["gbm_price_p50"] = preds

    # COLLAPSE TO ONE PREDICTION PER CELL, or the merge in build_rate_card()
    # doubles the frame it is merging into.
    #
    # The encoded grain is keyed on day_type, so it holds 133,956 rows over
    # 66,978 cells -- two per cell -- and 9 of the model's 41 features differ
    # between them (booking_count, demand_intensity, exposure_log,
    # corridor_ridership_log, ..., and `day_type_weekday` itself), so the two
    # predictions genuinely differ. Returning both gives
    #     own.merge(gbm_predictions, on=[screen_id, time_block_id], how="left")
    # duplicate keys on BOTH sides -- `own` is still 133,956 rows at that
    # point -- so the merge is 2x2 and produces 267,912 rows, four per cell.
    #
    # Being precise about the damage, because it is not a wrong price: the
    # later collapse averages those four back down, and the arithmetic happens
    # to survive (averaging over both grain rows x both prediction rows equals
    # averaging over the grain rows against the mean prediction). What does NOT
    # survive is everything around it -- the "[rate card] ML-blended ... for
    # N/M cells" diagnostic doubles and reports a denominator that is not a
    # cell count, the non-numeric collapse takes `.first()` of four candidate
    # rows instead of two, and peak memory and merge time double on the widest
    # frame in the pipeline. A silently doubled intermediate is worth removing
    # even when today's output survives it.
    #
    # Averaging is the same choice the grain collapse makes, for the same
    # reason and with the same caveat: day_type is a real dimension being
    # averaged away, and the right long-term fix is to carry it through the
    # quote key instead. Stated here so it is a known approximation rather
    # than an invisible one.
    n_rows = len(out)
    n_cells = out.groupby(["screen_id", "time_block_id"], observed=True).ngroups
    if n_cells != n_rows:
        out = (out.groupby(["screen_id", "time_block_id"], as_index=False,
                            observed=True)["gbm_price_p50"].mean())
        print(f"[gbm] collapsed {n_rows:,} predictions -> {len(out):,} cells "
              f"(the encoded grain is keyed on day_type; averaging weekday and "
              f"weekend, same approximation as the rate card).")

    print(f"[gbm] loaded {model_path.name}, predicted price_p50 for all {len(out):,} cells "
          f"using {len(feature_columns)} features from the persisted model card -- "
          f"zero NaN, zero crash.")
    return out


#: Minimum weight the EMPIRICAL price keeps when blending in the GBM. See the
#: long note at the blend itself: 0.50 means a cell with no booking history
#: weights the peer median and the model equally, because neither is entitled
#: to decide alone. Raising this trusts our own history more; lowering it
#: trusts the model more on cells it has no label for.
ML_MIN_EMPIRICAL_WEIGHT = 0.50

#: Never quote a floor below the cheapest price COMPARABLE inventory has
#: actually transacted at, where comparable is the same peer group
#: (city_id x mount_type x screen_type -- 12 groups, observed minimums 9.89
#: to 32.31). This is a commercial bound, not a model output: a rep can
#: negotiate to the floor, and a floor beneath anything the business has ever
#: accepted is a hole rather than a discount. Set False to disable and see the
#: unclamped model prices.
CLAMP_FLOOR_TO_PEER_MINIMUM = True


def build_rate_card(
    grain: pd.DataFrame,
    gbm_predictions: Optional[pd.DataFrame] = None,
    ml_full_trust_bookings: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Per-screen-block rate card, plus a peer-group fallback for cold-start
    cells (has_history=False -> price_p10/50/90 are NaN in pricing_grain_d3).

    If gbm_predictions is given (see load_gbm_price_predictions), the TARGET
    (rc_p50) is BLENDED with the GBM's prediction by booking-history
    credibility:
        credibility = clip(booking_count / ml_full_trust_bookings, 0, 1)
        rc_p50_blended = credibility * rc_p50_empirical + (1 - credibility) * gbm_price_p50
    A screen/block with >= ml_full_trust_bookings real bookings trusts its
    own empirical history fully (credibility=1, no ML influence); a
    screen/block with 0 bookings (cold-start) leans ENTIRELY on the ML
    model instead of the coarser 3-way peer-group average, since the GBM
    saw far more granular per-screen features (POI footfall, demand
    intensity, market tier, etc.) than a groupby(city, mount_type,
    screen_type).mean() ever could.

    price_p10 / price_p90 (the floor/cap band) are deliberately NOT
    blended -- the GBM was trained as a point estimate only (price_p50),
    not a quantile regressor, so it has nothing principled to say about
    band WIDTH, only the center.
    """
    base_cols = ["screen_id", "time_block_id", "city_id", "mount_type", "screen_type",
                 "price_p10", "price_p50", "price_p90", "has_history"]
    if "booking_count" in grain.columns:
        base_cols.append("booking_count")
    if "event_demand_boost_log" in grain.columns:
        # carried through as the STATIC baseline -- event_price_adjustment()
        # only applies a shift for the INCREMENTAL signal above this, never
        # double-counting what the static grid average already reflects.
        base_cols.append("event_demand_boost_log")
    own = grain[base_cols].copy()
    if "event_demand_boost_log" not in own.columns:
        own["event_demand_boost_log"] = 0.0
    if "booking_count" not in own.columns:
        # defensive fallback if the column is ever absent -- treat any
        # has_history=True cell as "at least 1 booking" rather than crash
        own["booking_count"] = np.where(own["has_history"], 1, 0)

    peer_cols = ["city_id", "mount_type", "screen_type"]
    peer = (
        grain[grain["has_history"]]
        .groupby(peer_cols)[["price_p10", "price_p50", "price_p90"]]
        .mean()
        .reset_index()
        .rename(columns={c: f"peer_{c}" for c in ["price_p10", "price_p50", "price_p90"]})
    )
    global_p50 = grain.loc[grain["has_history"], "price_p50"].median()

    own = own.merge(peer, on=peer_cols, how="left")
    for pct in ["p10", "p50", "p90"]:
        own[f"rc_{pct}"] = own[f"price_{pct}"].where(own["has_history"], own[f"peer_price_{pct}"])
        own[f"rc_{pct}"] = own[f"rc_{pct}"].fillna(global_p50)
    own["rc_basis"] = np.where(own["has_history"], "own_history", "cold_start_peer_fallback")

    own["ml_blend_credibility"] = np.nan
    own["gbm_price_p50"] = np.nan
    own["rc_p50_pre_blend"] = own["rc_p50"]

    if gbm_predictions is not None:
        own = own.merge(gbm_predictions, on=["screen_id", "time_block_id"], how="left",
                         suffixes=("", "_new"))
        if "gbm_price_p50_new" in own.columns:
            own["gbm_price_p50"] = own["gbm_price_p50_new"]
            own = own.drop(columns=["gbm_price_p50_new"])

        has_gbm = own["gbm_price_p50"].notna()
        # CREDIBILITY HAS A FLOOR. Without one, a cell with zero bookings gets
        # credibility 0 and the GBM sets its price outright -- and the target
        # is null for every no-history row, so the model trained on 65,441
        # rows that are 100% has_history=True and then priced 29,894 cells it
        # has no label for. Measured, that unilateral replacement marked
        # cold-start inventory down a mean 33.2% (median 35.5%) and the whole
        # book 16.4%, on exactly the cells where the model is least supported.
        #
        # The floor is 0.50 because that is the only non-arbitrary value here:
        # at 0.50 a no-history cell weights the peer median and the model
        # EQUALLY, which is the honest statement of the situation. The peer
        # median is real evidence about the wrong cell -- the mean of
        # comparable screens that did transact. The GBM is a 41-feature fit
        # about the right cell with no label. Neither dominates, so neither
        # gets to.
        #
        # Measured effect of the floor on the cold-start markdown:
        #     floor 0.00  -33.2% mean   (model weight 96%)
        #     floor 0.30  -23.2%        (67%)
        #     floor 0.50  -16.6%        (48%)   <- chosen
        #     floor 0.60  -13.3%        (38%)
        #
        # Cells WITH history are unaffected in the limit: credibility still
        # reaches 1.0 at ml_full_trust_bookings, so a well-evidenced cell
        # still trusts itself completely.
        credibility = np.clip(
            ML_MIN_EMPIRICAL_WEIGHT + (1.0 - ML_MIN_EMPIRICAL_WEIGHT) * np.clip(
                own["booking_count"].fillna(0) / max(1, ml_full_trust_bookings), 0.0, 1.0),
            0.0, 1.0)
        blended = np.where(
            has_gbm,
            credibility * own["rc_p50_pre_blend"] + (1.0 - credibility) * own["gbm_price_p50"],
            own["rc_p50_pre_blend"],
        )
        own["ml_blend_credibility"] = np.where(has_gbm, credibility, np.nan)
        own["rc_p50"] = blended

        # MOVE THE WHOLE BAND, NOT JUST THE TARGET.
        #
        # This blended rc_p50 but left rc_p10 and rc_p90 at their empirical
        # values, and the target moves by a mean |21.8%|. The result is a band
        # that no longer contains its own target. Measured on the full grid
        # with the GBM enabled:
        #
        #     target < floor   54.23%   (36,323 cells)
        #     target > cap     15.29%   (10,243 cells)
        #     inside           30.48%
        #
        # and the median target sat at -0.58 of the way from floor to cap --
        # i.e. below the floor by more than half a band width.
        #
        # quote() then searches 60 steps between floor and cap for the revenue
        # optimum and clamps the answer into that band, so with an incoherent
        # band the answer is an EDGE: 49.8% of quotes came back at the floor
        # and 20.0% at the cap, against 0.02%/0.02% with the blend off. Turning
        # the ML model on made the pricing visibly worse, which is why "just
        # train and save the artefact" was not enough.
        #
        # Rescaling by the same factor the target moved preserves both the
        # ordering (floor < target < cap) and the band's relative width, which
        # is what the win model's price_ratio is defined against. It does
        # assume the GBM shifts the level rather than the spread -- fair, since
        # the model was trained on price_p50 alone and has nothing to say about
        # dispersion.
        scale = np.where(
            has_gbm & (own["rc_p50_pre_blend"] > 0),
            own["rc_p50"] / own["rc_p50_pre_blend"].replace(0, np.nan),
            1.0,
        )
        scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
        own["rc_p10"] = own["rc_p10"] * scale
        own["rc_p90"] = own["rc_p90"] * scale

        own["rc_basis"] = np.where(
            has_gbm & (credibility < 0.999),
            own["rc_basis"].astype(str) + "+ml_blend",
            own["rc_basis"],
        )
        n_blended = int((has_gbm & (credibility < 0.999)).sum())
        if has_gbm.any():
            denom = own.loc[has_gbm, "rc_p50_pre_blend"].replace(0, np.nan)
            avg_shift_pct = float(
                ((own.loc[has_gbm, "rc_p50"] / denom - 1.0).abs() * 100).mean()
            )
        else:
            avg_shift_pct = 0.0
        print(f"[rate card] ML-blended target price for {n_blended}/{len(own)} cells "
              f"(credibility < 1.0 -- thin or zero booking history); "
              f"mean |shift| vs. pure empirical target: {avg_shift_pct:.1f}%")
    else:
        print("[rate card] no GBM predictions supplied -- rate card is pure "
              "empirical/peer-fallback (pre-fix behavior).")

    # ---- commercial floor, applied LAST so nothing can undercut it -------- #
    # The band rescale above preserves ORDERING (floor < target < cap) but not
    # an absolute bound, and the ML blend can scale a cold-start band down far
    # enough that the floor lands below any price this business has ever
    # accepted. Measured before this clamp: 1,308 cells (2.0%) had a floor
    # under 9.89, the cheapest observed price_p10 anywhere in the network, and
    # 99.9% of them were cold-start cells priced ~100% by the model.
    #
    # Lift the WHOLE band by one factor rather than raising the floor alone --
    # that keeps floor < target < cap and keeps the band's relative width,
    # which is what the win model's price_ratio is defined against. Raising
    # only the floor could invert it against a low target.
    if CLAMP_FLOOR_TO_PEER_MINIMUM:
        peer_min = (grain[grain["has_history"]]
                    .groupby(peer_cols)["price_p10"].min()
                    .rename("peer_floor_min").reset_index())
        own = own.merge(peer_min, on=peer_cols, how="left")
        floor_min = own["peer_floor_min"]
        lift = np.where((floor_min.notna()) & (own["rc_p10"] > 0)
                        & (own["rc_p10"] < floor_min),
                        floor_min / own["rc_p10"].replace(0, np.nan), 1.0)
        lift = np.nan_to_num(lift, nan=1.0, posinf=1.0, neginf=1.0)
        n_lifted = int((lift > 1.0 + 1e-12).sum())
        for c in ("rc_p10", "rc_p50", "rc_p90"):
            own[c] = own[c] * lift
        own = own.drop(columns=["peer_floor_min"])
        if n_lifted:
            print(f"[rate card] lifted {n_lifted:,} band(s) to the peer-group floor "
                  f"-- their floor sat below the cheapest price comparable "
                  f"inventory has ever transacted at (mean lift "
                  f"{(lift[lift > 1.0].mean() - 1) * 100:.1f}%)")


    # screen-level rate card (mean of rc_p50 across that screen's 6 blocks) --
    # used for lost_leads, which have no time_block_id at all (real data gap,
    # established earlier this session).
    screen_level = own.groupby("screen_id")["rc_p50"].mean().rename("screen_rc_p50").reset_index()
    # ----------------------------------------------------------------------- #
    # One row per (screen_id, time_block_id). Not optional.
    # ----------------------------------------------------------------------- #
    # `pricing_grain_d3.csv` is keyed on (screen_id, time_block_id, DAY_TYPE) --
    # 133,956 rows for 66,978 cells. Nothing downstream carries day_type, so
    # every cell arrived twice with different weekday/weekend economics:
    # pricing_quotes_d3.csv had 133,956 rows and 41,902 cells quoted at TWO
    # DIFFERENT PRICES (e.g. ACS-SCR-000001 block 2 at both 69.32 and 67.81).
    # That is precisely the "same screen quoted at different prices by different
    # reps" problem the briefing asks us to solve, generated by our own engine.
    #
    # `quote()` already averaged duplicates, but only on the single-cell path;
    # `quote_grid()` -- the one that writes the CSV -- read this frame directly.
    # Collapsing here fixes both paths and any future consumer.
    key = ["screen_id", "time_block_id"]
    n_before = len(own)
    if own.groupby(key, observed=True).ngroups != n_before:
        num = own.select_dtypes(include=["number"]).columns.difference(key)
        agg = {c: "mean" for c in num}
        agg.update({c: "first" for c in own.columns
                    if c not in num and c not in key})
        own = own.groupby(key, as_index=False, observed=True).agg(agg)
        print(f"[rate card] collapsed {n_before:,} grain rows -> {len(own):,} cells "
              f"(one per screen x time_block). The grain is keyed on day_type but "
              f"nothing downstream carries it, so each cell appeared twice with "
              f"different prices.")
        print(f"[rate card]   NOTE: this AVERAGES weekday and weekend economics. "
              f"Weekend ridership is ~1/3 of weekday, so the right long-term fix is "
              f"to make day_type part of the quote key rather than average it away.")

    return {"cell": own, "screen": screen_level, "global_p50": global_p50}


def attach_price_ratio_and_weight(win_loss: pd.DataFrame, rate_card: dict) -> pd.DataFrame:
    """
    Robust to TWO possible win_loss_table_d3.csv schemas, since this
    session's build_training_table.py has produced both over time:

      (a) REAL schema actually on disk for this user (8 cols):
          client_id, city_id, won, offered_price, event_date,
          screen_id_ref, is_price_driven_loss, is_unmet_demand
          -- NO time_block_id, NO deal_id. event_date is already
          populated at build time (booked_date for won rows; the best
          available lost-lead date, possibly NaT, for lost rows).

      (b) a richer schema some test fixtures used (screen_id,
          time_block_id, deal_id present directly).

    Previous version of this function assumed (b) unconditionally and
    crashed with KeyError: 'screen_id' on the real (a) schema -- this
    is schema-detected now, not re-guessed.
    """
    df = win_loss.copy()

    # --- normalize the screen-id column name ---
    if "screen_id" not in df.columns and "screen_id_ref" in df.columns:
        df = df.rename(columns={"screen_id_ref": "screen_id"})
    if "screen_id" not in df.columns:
        raise KeyError(
            "win_loss has neither 'screen_id' nor 'screen_id_ref' -- "
            f"actual columns: {df.columns.tolist()}"
        )

    screen_rc = rate_card["screen"]
    has_time_block = "time_block_id" in df.columns

    if has_time_block:
        # richer schema: cell-level rate card for rows with a real
        # time_block_id, screen-level fallback otherwise.
        cell_rc = rate_card["cell"][["screen_id", "time_block_id", "rc_p50"]]
        df = df.merge(cell_rc, on=["screen_id", "time_block_id"], how="left")
        df = df.merge(screen_rc, on="screen_id", how="left")
        df["rate_card_price"] = df["rc_p50"].where(df["rc_p50"].notna(), df["screen_rc_p50"])
        df = df.drop(columns=["rc_p50", "screen_rc_p50"])
    else:
        # real schema: win_loss has no time_block_id at all (lost_leads
        # never had one, and the real build_win_loss_table() doesn't
        # carry it for won rows either) -- screen-level only.
        df = df.merge(screen_rc, on="screen_id", how="left")
        df["rate_card_price"] = df["screen_rc_p50"]
        df = df.drop(columns=["screen_rc_p50"])

    df["rate_card_price"] = df["rate_card_price"].fillna(rate_card["global_p50"])
    df["price_ratio"] = df["offered_price"] / df["rate_card_price"]

    # --- recency: use event_date directly if the real schema already
    #     has it populated; only fall back to a deal_id/lead_id re-join
    #     against the raw CSVs if the real event_date column is absent
    #     AND a deal_id key exists to re-derive it from ---
    if "event_date" not in df.columns and "deal_id" in df.columns:
        bookings_dates = pd.read_csv(DATA_DIR / "bookings.csv", usecols=["deal_id", "booked_date"])
        bookings_dates["booked_date"] = pd.to_datetime(bookings_dates["booked_date"], errors="coerce")
        bookings_dates = bookings_dates.drop_duplicates("deal_id")

        lost_dates = pd.read_csv(DATA_DIR / "lost_leads.csv", usecols=["lead_id", "lost_date"])
        lost_dates["lost_date"] = pd.to_datetime(lost_dates["lost_date"], errors="coerce")
        lost_dates = lost_dates.rename(columns={"lead_id": "deal_id"})

        won_mask = df["won"] == True  # noqa: E712
        df = df.merge(bookings_dates, on="deal_id", how="left")
        df = df.merge(lost_dates, on="deal_id", how="left")
        df["event_date"] = df["booked_date"].where(won_mask, df["lost_date"])
        df = df.drop(columns=["booked_date", "lost_date"])
    else:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    max_date = df["event_date"].max()
    if pd.isna(max_date):
        max_date = pd.Timestamp.today()
    df["age_days"] = (max_date - df["event_date"]).dt.days
    df["weight"] = df["age_days"].apply(recency_weight)

    n_missing_ratio = df["price_ratio"].isna().sum()
    n_missing_date = df["event_date"].isna().sum()
    print(f"[win/loss prep] price_ratio missing: {n_missing_ratio}/{len(df)}; "
          f"event_date missing: {n_missing_date}/{len(df)} (most recent event: {max_date.date()})")
    return df

#: Loss reasons that mean the PRICE was the problem. The other seven values in
#: lost_leads.loss_reason are timing, creative, inventory or ghosting, and
#: counting those as price losses over-states price sensitivity and gives away
#: margin. Only 44% of lost leads are price losses.
PRICE_LOSS_REASONS = ("price_too_high", "budget_mismatch", "went_with_competitor")


def fit_win_model(win_loss_prepped: pd.DataFrame) -> WinModel:
    """Fit the win curve, preferring the sample that can actually identify it.

    Order matters:

      1. `fit_from_price_gap` on lost_leads -- the identified comparison. Among
         deals we lost, was price the reason? Same population both sides, so
         selection cancels. n is small (~1,450) but balanced.
      2. Only if that is unavailable or refused, fall back to the
         bookings-vs-leads fit, which the base_ratio guard will almost
         certainly refuse -- correctly. A refused fit keeps the prior, which is
         a defensible number; an accepted implausible fit is not.
    """
    ll_path = DATA_DIR / "lost_leads.csv"
    if ll_path.exists():
        ll = pd.read_csv(ll_path)
        if {"price_gap_pct", "loss_reason"} <= set(ll.columns):
            price_driven = ll["loss_reason"].isin(PRICE_LOSS_REASONS)
            n_price = int(price_driven.sum())
            print(f"[win model] using the IDENTIFIED sample: {len(ll):,} lost leads, "
                  f"{n_price:,} price-driven vs {len(ll) - n_price:,} not "
                  f"({100 * n_price / max(len(ll), 1):.1f}% price) -- inside-losses "
                  f"comparison, so selection cancels")
            wm = (WinModel()
                  .fit_from_price_gap(ll["price_gap_pct"], price_driven)
                  .calibrate_to_reference())
            if wm.fitted:
                return wm
            print(f"[win model] identified fit was refused ({wm.reject_reason}) -- "
                  f"trying the bookings-vs-leads sample, which the guard will "
                  f"almost certainly also refuse")
        else:
            missing = {"price_gap_pct", "loss_reason"} - set(ll.columns)
            print(f"[win model] lost_leads.csv is missing {missing} -- cannot run the "
                  f"identified fit. NOTE: price_gap_pct is NOT carried on "
                  f"win_loss_table_d3.csv, it has to come from lost_leads.csv.")

    obs = list(zip(
        win_loss_prepped["price_ratio"],
        win_loss_prepped["won"],
        win_loss_prepped["weight"],
    ))
    return WinModel().fit(obs)


# --------------------------------------------------------------------------- #
# The D1 -> D3 handoff: say when a cell's audience is not measured
# --------------------------------------------------------------------------- #

SELL_EVIDENCED = "evidenced"        # measurable audience: quote and recommend
SELL_UNEVIDENCED = "unevidenced"    # no measurable audience: quote, warn, do not lead
SELL_NO_CAPACITY = "no_capacity"    # nothing left to sell


def apply_audience_gate(quotes: pd.DataFrame,
                        cube_path: Optional[Path] = None) -> pd.DataFrame:
    """Attach D1's audience verdict to every quote. Adds columns, removes none.

    FLAG, DO NOT REFUSE. Block 1 (00:00-04:00) has no scheduled transit trips
    network-wide, so D1 reports zero people. An earlier version of this gate
    refused to price those cells. That was wrong: block 1 has traded 8,544
    times to 336 clients, 302 of them repeat buyers, with zero cancellations
    and $18.0M of live forward-dated orders. What we know is that we cannot
    MEASURE its audience -- not that there is none. Refusing would break
    renewals for 302 buyers.

    The market has in fact already priced it: block 1 trades at $36.09 against
    a $73.80 network median, a 51% discount reached with no audience data at
    all, and across the blocks that DO have measurable audience price
    correlates 0.71 with it.

    So this adds `sell_state` / `sellable` / `recommendable` / `sell_reason` and
    leaves every price intact. Nothing here is a refusal; `recommendable` is
    what a downstream consumer should filter on when it needs to stand behind
    an impressions figure.

    Returns `quotes` unchanged (with the columns absent) if the cube is missing,
    so this can never be the thing that stops a run.
    """
    cube_path = cube_path or (OUT_DIR / "audience_cube.csv")
    if not cube_path.exists():
        print(f"[gate] {cube_path.name} not found -- quotes emitted WITHOUT an "
              f"audience verdict. Run build_d1_audience_profiles.py to enable it.")
        return quotes

    cube = pd.read_csv(cube_path, low_memory=False)
    need = {"screen_id", "time_block_id", "people"}
    if not need <= set(cube.columns):
        print(f"[gate] {cube_path.name} lacks {need - set(cube.columns)} -- skipping")
        return quotes

    # Weekday is what a rep sells against. Sum across day types would double it.
    if "day_type" in cube.columns:
        cube = cube[cube["day_type"] == "weekday"]
    aud = (cube.groupby(["screen_id", "time_block_id"], as_index=False)["people"]
           .sum().rename(columns={"people": "d1_people"}))

    out = quotes.merge(aud, on=["screen_id", "time_block_id"], how="left")
    people = out["d1_people"]

    evidenced = people.notna() & (people >= 1.0)
    out["sell_state"] = np.where(evidenced, SELL_EVIDENCED, SELL_UNEVIDENCED)
    out["sellable"] = True                 # only sold-out capacity is unsellable
    out["recommendable"] = evidenced
    out["sell_reason"] = np.where(
        evidenced,
        people.fillna(0).map(lambda v: f"{v:,.0f} people/day"),
        "no measurable audience -- no scheduled transit service in this block. "
        "Quotable, but we cannot stand behind an impressions number",
    )

    n_un = int((~evidenced).sum())
    print(f"[gate] audience from {cube_path.name}: "
          f"{len(out) - n_un:,} evidenced, {n_un:,} unevidenced "
          f"({100 * n_un / max(len(out), 1):.1f}%), 0 refused -- prices left intact")
    if n_un:
        by_block = out.loc[~evidenced, "time_block_id"].value_counts().sort_index()
        print(f"[gate]   unevidenced by block: {by_block.to_dict()}")
    return out


# --------------------------------------------------------------------------- #
# Relationship layer -- group rollup + empirical category effects
# --------------------------------------------------------------------------- #

@dataclass
class RelationshipTable:
    per_client: pd.DataFrame                       # one row per client_id
    category_effect: Dict[str, Dict[str, float]]   # {column: {value: shift}}
    global_mean_ratio: float


def build_relationship_table(win_loss_prepped: pd.DataFrame) -> RelationshipTable:
    client_facts = pd.read_csv(DATA_DIR / "client_facts.csv")
    client_facts["relationship_start_date"] = pd.to_datetime(
        client_facts["relationship_start_date"], errors="coerce")
    max_date = win_loss_prepped["event_date"].max()
    client_facts["tenure_months"] = (
        (max_date - client_facts["relationship_start_date"]).dt.days / 30.44
    ).clip(lower=0)

    client_facts["group_key"] = client_facts["company_name"].apply(derive_group_key)
    sample_groups = (
        client_facts.groupby("group_key")["company_name"]
        .apply(lambda s: list(s)[:3])
        .loc[lambda s: s.apply(len) > 1]
        .head(5)
    )
    print(f"[relationship] derived {client_facts['group_key'].nunique()} group_keys from "
          f"company_name (heuristic -- no real parent-company column exists). "
          f"Sample multi-brand groups found: {dict(sample_groups) if len(sample_groups) else 'none'}")

    # --- trailing spend, recomputed fresh from real bookings (NOT the dropped
    #     leaky line_item_value column -- this recomputes the same arithmetic
    #     independently as a CLIENT-level aggregate for relationship tiering,
    #     which is a legitimate use, unlike using it as a per-row price feature) ---
    bookings_raw = pd.read_csv(
        DATA_DIR / "bookings.csv",
        usecols=["client_id", "booked_date", "contracted_price_per_slot_per_day",
                 "slots_booked_per_day", "duration_days"],
    )
    bookings_raw["booked_date"] = pd.to_datetime(bookings_raw["booked_date"], errors="coerce")
    bookings_raw["deal_value"] = (
        bookings_raw["contracted_price_per_slot_per_day"]
        * bookings_raw["slots_booked_per_day"] * bookings_raw["duration_days"]
    )
    trailing_cutoff = pd.Timestamp(max_date) - pd.DateOffset(days=365)
    trailing = bookings_raw[bookings_raw["booked_date"] >= trailing_cutoff]
    trailing_spend = trailing.groupby("client_id")["deal_value"].sum().rename("trailing_12m_spend")

    client_facts = client_facts.merge(trailing_spend, on="client_id", how="left")
    client_facts["trailing_12m_spend"] = client_facts["trailing_12m_spend"].fillna(0.0)

    group_spend = client_facts.groupby("group_key")["trailing_12m_spend"].sum().rename("group_trailing_12m_spend")
    client_facts = client_facts.merge(group_spend, on="group_key", how="left")
    client_facts["effective_spend"] = np.maximum(
        client_facts["trailing_12m_spend"], client_facts["group_trailing_12m_spend"])

    # data-driven "well-established relationship" threshold: top tercile of
    # effective_spend, not a guessed dollar figure.
    spend_p67 = client_facts["effective_spend"].quantile(0.67)
    client_facts["high_relationship_spend"] = client_facts["effective_spend"] >= spend_p67

    # --- empirical category effects: mean price_ratio among WON deals, per
    #     category value, for each of the 4 categorical relationship columns.
    #     No assumed label semantics -- just "what did this category actually
    #     transact at, relative to the rate card" ---
    won = win_loss_prepped[win_loss_prepped["won"] == True]  # noqa: E712
    won_with_client = won.merge(
        client_facts[["client_id"] + RELATIONSHIP_CATEGORY_COLS], on="client_id", how="inner")
    global_mean_ratio = won_with_client["price_ratio"].mean()

    category_effect: Dict[str, Dict[str, float]] = {}
    for col in RELATIONSHIP_CATEGORY_COLS:
        grp = won_with_client.groupby(col)["price_ratio"].agg(["mean", "count"])
        shift = {}
        for value, row in grp.iterrows():
            # shrink toward 0 when a category has few observations, so a rare
            # value doesn't produce an overconfident shift
            reliability = min(1.0, row["count"] / 30.0)
            shift[value] = float((row["mean"] - global_mean_ratio) * reliability)
        category_effect[col] = shift
        print(f"[relationship] {col} empirical price_ratio shift vs. global mean "
              f"({global_mean_ratio:.3f}): "
              + ", ".join(f"{v}={s:+.3f} (n={int(grp.loc[v, 'count'])})" for v, s in shift.items()))

    return RelationshipTable(
        per_client=client_facts, category_effect=category_effect,
        global_mean_ratio=global_mean_ratio,
    )


def relationship_shift_for_client(rel: RelationshipTable, client_id: Optional[str]) -> Tuple[float, List[dict]]:
    """Total price_ratio shift for one client, plus the named components that
    produced it (for the explain() string)."""
    if client_id is None:
        return 0.0, []
    row = rel.per_client[rel.per_client["client_id"] == client_id]
    if row.empty:
        return 0.0, [{"name": "unknown_client", "family": "relationship", "shift": 0.0,
                       "reason": "no client_facts record -- treated as an anonymous prospect"}]
    row = row.iloc[0]
    components = []
    total = 0.0
    for col in RELATIONSHIP_CATEGORY_COLS:
        val = row[col]
        shift = rel.category_effect.get(col, {}).get(val, 0.0)
        if abs(shift) > 1e-6:
            components.append({
                "name": f"{col}={val}", "family": "relationship", "shift": shift,
                "reason": f"clients with {col}={val} have historically transacted at "
                          f"{rel.global_mean_ratio + shift:.1%} of rate card vs. "
                          f"{rel.global_mean_ratio:.1%} overall",
            })
            total += shift
    if row["high_relationship_spend"]:
        bonus = 0.03  # modest, explicit, data-triggered (top-tercile effective_spend)
        basis = "group" if row["group_trailing_12m_spend"] > row["trailing_12m_spend"] else "own"
        components.append({
            "name": "high_relationship_spend", "family": "relationship", "shift": -bonus,
            "reason": f"{basis} trailing-12m spend {row['effective_spend']:,.0f} is in the "
                      f"top third of clients -- tolerates a higher price before walking"
                      + (f" (group {row['group_key']})" if basis == "group" else ""),
        })
        total -= bonus
    return total, components


# --------------------------------------------------------------------------- #
# Quote engine
# --------------------------------------------------------------------------- #

@dataclass
class PriceQuote:
    screen_id: str
    time_block_id: int
    client_id: Optional[str]
    floor: float
    target: float
    cap: float
    recommended: float
    rate_card_basis: str
    p_win_at_recommended: float
    expected_revenue: float
    components: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    event_adjustment_pct: float = 0.0
    event_names: str = ""
    ridership_trend_pct: float = 0.0
    ridership_trend_note: str = ""

    def explain(self) -> str:
        lines = [
            f"Screen {self.screen_id}, block {self.time_block_id}"
            + (f", client {self.client_id}" if self.client_id else " (rate card, no client)"),
            f"Band: floor {self.floor:,.2f} / target {self.target:,.2f} / cap {self.cap:,.2f} "
            f"(basis: {self.rate_card_basis})",
            f"Recommended: {self.recommended:,.2f}  ->  win probability "
            f"{self.p_win_at_recommended:.0%}, expected revenue {self.expected_revenue:,.2f}",
        ]
        if self.event_adjustment_pct > 0:
            lines.append(
                f"  +{self.event_adjustment_pct:.1%} band shift from a genuine event overlap "
                f"during the requested dates -- {self.event_names or 'matched event(s)'}"
            )
        if self.ridership_trend_pct != 0.0:
            direction = "up" if self.ridership_trend_pct > 0 else "down"
            lines.append(
                f"  {self.ridership_trend_pct:+.1%} band shift from a statistically-significant "
                f"ridership trend ({direction}) for the requested dates -- {self.ridership_trend_note}"
            )
        if self.components:
            lines.append("Why this price:")
            for c in sorted(self.components, key=lambda x: -abs(x["shift"])):
                verb = "premium" if c["shift"] < 0 else "discount"
                lines.append(f"  {-c['shift']:+.1%}  {c['name']:<28} [{c['family']}] {verb} -- {c['reason']}")
        else:
            lines.append("Why this price: rate card only -- no relationship data for this client")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


class PricingEngineD3:
    def __init__(self, rate_card: dict, win_model: WinModel, relationship: RelationshipTable):
        self.rate_card = rate_card
        self.win_model = win_model
        self.relationship = relationship
        # .sort_index() is carried forward from commit 59d4470 (patch_sort_index.py)
        # and must be kept: quote() does .loc[(screen, block)] on this MultiIndex,
        # and pandas raises a PerformanceWarning -- and on a duplicated key returns
        # rows in arbitrary order -- against an unsorted index. The collapse step
        # in build_rate_card() below removes the duplicate keys this was papering
        # over, but sorting is still correct and still cheaper.
        self._cell = rate_card["cell"].set_index(["screen_id", "time_block_id"]).sort_index()
        self._event_tables: Optional[dict] = None      # lazy: loaded on first date-aware quote
        self._event_overlap_cache: Dict[Tuple[str, str], pd.DataFrame] = {}  # (start,end) -> df
        self._ridership_tables: Optional[dict] = None    # lazy: loaded on first date-aware quote
        self._ridership_models: Optional[pd.DataFrame] = None  # one row per (corridor_id, time_block_id)
        self._ridership_min_date = None
        self._screen_corridor_map: Optional[pd.DataFrame] = None  # screen_id -> corridor_id
        self._ridership_forecast_cache: Dict[Tuple[str, str], pd.DataFrame] = {}  # (start,end) -> df

    def _event_overlap_for_window(self, campaign_start, campaign_end) -> pd.DataFrame:
        """Lazily loads event_overlap tables once, then caches the computed
        overlap DataFrame per (start,end) window so quote_grid() scoring all
        66,978 cells for the same campaign only computes this once."""
        cs, ce = pd.Timestamp(campaign_start), pd.Timestamp(campaign_end)
        cache_key = (cs.isoformat(), ce.isoformat())
        if cache_key in self._event_overlap_cache:
            return self._event_overlap_cache[cache_key]
        if self._event_tables is None:
            self._event_tables = _evo.load_tables()
        overlap = _evo.compute_event_overlap(self._event_tables, cs, ce, verbose=True)
        self._event_overlap_cache[cache_key] = overlap
        return overlap

    def _ridership_forecast_for_window(self, campaign_start, campaign_end) -> pd.DataFrame:
        """Lazily fits the ridership trend models ONCE (one OLS per corridor
        x time_block -- the expensive part), then caches the per-window
        forecast the same way _event_overlap_for_window() caches event
        overlap, so quote_grid() scoring every cell for the same campaign
        only does this work once. Independent of, and additive on top of,
        the event overlap adjustment above -- see ridership_trend.py's
        module docstring for why these are two separate signals rather than
        one combined model."""
        cs, ce = pd.Timestamp(campaign_start), pd.Timestamp(campaign_end)
        cache_key = (cs.isoformat(), ce.isoformat())
        if cache_key in self._ridership_forecast_cache:
            return self._ridership_forecast_cache[cache_key]
        if self._ridership_tables is None:
            self._ridership_tables = _rtr.load_tables()
            self._ridership_models, self._ridership_min_date = _rtr.fit_ridership_trend_models(
                self._ridership_tables)
            self._screen_corridor_map = _rtr.build_screen_corridor_map(
                self._ridership_tables["screens"], self._ridership_tables["vehicles"],
                self._ridership_tables["route_stops"])
        forecast = _rtr.get_ridership_forecast_for_window(
            self._ridership_tables, self._ridership_models, self._ridership_min_date,
            self._screen_corridor_map, cs, ce, verbose=True)
        self._ridership_forecast_cache[cache_key] = forecast
        return forecast

    def quote(self, screen_id: str, time_block_id: int, client_id: Optional[str] = None,
              campaign_start=None, campaign_end=None) -> PriceQuote:
        """campaign_start/campaign_end (optional, YYYY-MM-DD strings or
        pd.Timestamp): when BOTH are given, the floor/target/cap band is
        shifted upward for any event genuinely overlapping THIS specific
        campaign window (see event_overlap.py) -- on top of, never
        double-counting, the static per-location average already reflected
        in the rate card via event_demand_boost_log. Omit both to get the
        pre-existing static-only behavior unchanged."""
        key = (screen_id, time_block_id)
        if key not in self._cell.index:
            raise KeyError(f"{key} not found in the screen x time_block grid")
        row = self._cell.loc[key]
        if isinstance(row, pd.DataFrame) and len(row) > 1:
            numeric_cols = row.select_dtypes(include=["number"]).columns
            averaged = row[numeric_cols].mean()
            first_row = row.iloc[0].copy()
            first_row[numeric_cols] = averaged
            row = first_row
        floor, target, cap = float(row["rc_p10"]), float(row["rc_p50"]), float(row["rc_p90"])
        if floor >= cap:  # degenerate band (e.g. a single-observation cell) -- widen slightly
            floor, cap = target * 0.85, target * 1.15

        event_pct, event_names = 0.0, ""
        if campaign_start is not None and campaign_end is not None:
            overlap = self._event_overlap_for_window(campaign_start, campaign_end)
            hit = overlap[(overlap["screen_id"] == screen_id) &
                          (overlap["time_block_id"] == time_block_id)]
            overlap_score = float(hit["overlap_score"].iloc[0]) if len(hit) else 0.0
            static_baseline = float(row.get("event_demand_boost_log", 0.0) or 0.0)
            floor, target, cap, event_pct = _evo.apply_event_overlap_to_band(
                floor, target, cap, overlap_score, static_baseline)
            if len(hit):
                event_names = str(hit["matched_event_names"].iloc[0])

        ridership_pct, ridership_note = 0.0, ""
        if campaign_start is not None and campaign_end is not None:
            r_forecast = self._ridership_forecast_for_window(campaign_start, campaign_end)
            r_hit = r_forecast[(r_forecast["screen_id"] == screen_id) &
                                (r_forecast["time_block_id"] == time_block_id)]
            if len(r_hit):
                multiplier = float(r_hit["ridership_multiplier"].iloc[0])
                trend_trusted = bool(r_hit["trend_trusted"].iloc[0])
                floor, target, cap, ridership_pct = _rtr.apply_ridership_trend_to_band(
                    floor, target, cap, multiplier)
                if ridership_pct != 0.0:
                    basis = "a statistically-significant" if trend_trusted else "a flat seasonal"
                    ridership_note = f"{basis} ridership pattern for this corridor/block/date-range"
            # screens with no resolvable corridor_id (location-mounted, no route_stops
            # match either) simply have no row in r_forecast -- multiplier stays at the
            # neutral default of 1.0, i.e. no adjustment, exactly as intended.

        shift, components = relationship_shift_for_client(self.relationship, client_id)

        lo, hi, steps = floor, cap, 60
        best_price, best_rev = target, -1.0
        for i in range(steps + 1):
            cand = lo + (hi - lo) * i / steps
            pw = self.win_model.p_win(cand / target if target > 0 else 1.0, shift)
            rev = cand * pw
            if rev > best_rev:
                best_price, best_rev = cand, rev

        recommended = min(max(best_price, floor), cap)
        pw_final = self.win_model.p_win(recommended / target if target > 0 else 1.0, shift)

        warnings = []
        if not self.win_model.fitted:
            warnings.append("win model is running on its prior (too little real signal to fit) "
                             "-- treat p_win/expected_revenue as illustrative, not calibrated")
        basis_str = str(row["rc_basis"])
        if basis_str.startswith("cold_start_peer_fallback"):
            warnings.append("this screen/block has no booking history -- band is "
                             "borrowed from peer screens in the same city/mount_type/screen_type")
        credibility = row.get("ml_blend_credibility", np.nan)
        if pd.notna(credibility) and credibility < 0.999:
            warnings.append(
                # Plain words: this warning is read by a rep in front of a
                # buyer, and "the GradientBoostingRegressor contributes 45% of
                # the target" is not a sentence anyone says out loud. The fact
                # is unchanged and the split is still exact.
                f"the typical price here is part history, part model -- this screen "
                f"and time block has thin or no booking record of its own, so "
                f"{credibility:.0%} of the price comes from what it and comparable "
                f"screens actually sold for, and {(1 - credibility):.0%} from our "
                f"trained price model"
            )

        return PriceQuote(
            screen_id=screen_id, time_block_id=int(time_block_id), client_id=client_id,
            floor=floor, target=target, cap=cap, recommended=recommended,
            rate_card_basis=str(row["rc_basis"]), p_win_at_recommended=pw_final,
            expected_revenue=recommended * pw_final, components=components, warnings=warnings,
            event_adjustment_pct=event_pct, event_names=event_names,
            ridership_trend_pct=ridership_pct, ridership_trend_note=ridership_note,
        )

    def quote_grid(self, client_id: Optional[str] = None,
                   campaign_start=None, campaign_end=None) -> pd.DataFrame:
        """Vectorized recommended price / p_win / expected revenue for every
        cell in the grid, for one client (or the anonymous rate card if
        client_id is None). Used to score the full 66,978-cell grid fast.

        campaign_start/campaign_end: same date-aware event adjustment as
        quote() (see event_overlap.py), applied vectorized across the whole
        grid via a single merge instead of per-row lookups."""
        cell = self.rate_card["cell"].copy()
        shift, _ = relationship_shift_for_client(self.relationship, client_id)

        floor = cell["rc_p10"].to_numpy()
        target = cell["rc_p50"].to_numpy()
        cap = cell["rc_p90"].to_numpy()
        degenerate = floor >= cap
        floor = np.where(degenerate, target * 0.85, floor)
        cap = np.where(degenerate, target * 1.15, cap)

        event_pct_arr = np.zeros(len(cell))
        event_names_arr = np.full(len(cell), "", dtype=object)
        if campaign_start is not None and campaign_end is not None:
            overlap = self._event_overlap_for_window(campaign_start, campaign_end)
            cell_idx = cell[["screen_id", "time_block_id"]].reset_index(drop=True)
            merged = cell_idx.merge(overlap, on=["screen_id", "time_block_id"], how="left")
            overlap_score = merged["overlap_score"].fillna(0.0).to_numpy()
            static_baseline = cell["event_demand_boost_log"].fillna(0.0).to_numpy() \
                if "event_demand_boost_log" in cell.columns else np.zeros(len(cell))
            delta = np.clip(overlap_score - static_baseline, 0.0, None)
            event_pct_arr = np.clip(delta * _evo.EVENT_PRICE_SHIFT_PER_UNIT,
                                     0.0, _evo.EVENT_PRICE_SHIFT_MAX)
            floor = np.where(event_pct_arr > 0, floor * (1 + event_pct_arr), floor)
            target = np.where(event_pct_arr > 0, target * (1 + event_pct_arr), target)
            cap = np.where(event_pct_arr > 0,
                            np.maximum(cap * (1 + event_pct_arr), floor * 1.05), cap)
            event_names_arr = merged["matched_event_names"].fillna("").to_numpy()
            n_boosted = int((event_pct_arr > 0).sum())
            print(f"[quote_grid] event-adjusted {n_boosted}/{len(cell)} cells for window "
                  f"[{pd.Timestamp(campaign_start).date()}, {pd.Timestamp(campaign_end).date()}]")

        ridership_pct_arr = np.zeros(len(cell))
        ridership_trusted_arr = np.zeros(len(cell), dtype=bool)
        if campaign_start is not None and campaign_end is not None:
            r_forecast = self._ridership_forecast_for_window(campaign_start, campaign_end)
            cell_idx = cell[["screen_id", "time_block_id"]].reset_index(drop=True)
            r_merged = cell_idx.merge(r_forecast, on=["screen_id", "time_block_id"], how="left")
            # screens with no resolvable corridor_id (location-mounted, no route_stops
            # match either) have no row in r_forecast -- multiplier defaults to 1.0
            # (i.e. pct_shift 0.0), same fallback convention as event overlap above.
            ridership_pct_arr = r_merged["ridership_pct_shift"].fillna(0.0).to_numpy()
            ridership_pct_arr = np.clip(ridership_pct_arr,
                                          -_rtr.RIDERSHIP_PRICE_SHIFT_MAX, _rtr.RIDERSHIP_PRICE_SHIFT_MAX)
            ridership_trusted_arr = r_merged["trend_trusted"].fillna(False).to_numpy()
            has_shift = ridership_pct_arr != 0.0
            new_floor = floor * (1 + ridership_pct_arr)
            new_target = target * (1 + ridership_pct_arr)
            new_cap = cap * (1 + ridership_pct_arr)
            new_cap = np.where(new_cap <= new_floor, new_floor * 1.05, new_cap)
            floor = np.where(has_shift, new_floor, floor)
            target = np.where(has_shift, new_target, target)
            cap = np.where(has_shift, new_cap, cap)
            n_up = int((ridership_pct_arr > 0).sum())
            n_down = int((ridership_pct_arr < 0).sum())
            print(f"[quote_grid] ridership-trend-adjusted {n_up} cells up, {n_down} cells down "
                  f"for window [{pd.Timestamp(campaign_start).date()}, "
                  f"{pd.Timestamp(campaign_end).date()}] (all other cells -- mostly "
                  f"location-mounted screens with no resolvable corridor -- unaffected)")

        steps = 60
        grid_frac = np.linspace(0.0, 1.0, steps + 1)  # (steps+1,)
        candidates = floor[:, None] + (cap - floor)[:, None] * grid_frac[None, :]  # (n, steps+1)
        ratios = candidates / np.where(target[:, None] > 0, target[:, None], 1.0)
        pw = self.win_model.p_win_vec(ratios, np.full_like(ratios, shift))
        revenue = candidates * pw

        best_idx = np.argmax(revenue, axis=1)
        rows = np.arange(len(cell))
        recommended = np.clip(candidates[rows, best_idx], floor, cap)
        pw_final = self.win_model.p_win_vec(
            recommended / np.where(target > 0, target, 1.0), np.full_like(recommended, shift))

        out = cell[["screen_id", "time_block_id", "rc_basis"]].copy()
        out["client_id"] = client_id
        out["floor"], out["target"], out["cap"] = floor, target, cap
        out["recommended"] = recommended
        out["p_win_at_recommended"] = pw_final
        out["expected_revenue"] = recommended * pw_final
        out["event_adjustment_pct"] = event_pct_arr
        out["event_names"] = event_names_arr
        out["ridership_trend_pct"] = ridership_pct_arr
        out["ridership_trend_trusted"] = ridership_trusted_arr
        if "ml_blend_credibility" in cell.columns:
            out["ml_blend_credibility"] = cell["ml_blend_credibility"].to_numpy()
            out["gbm_price_p50"] = cell["gbm_price_p50"].to_numpy()
        return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-start", default=None, help="YYYY-MM-DD (optional). When "
                    "given together with --campaign-end, every quote below applies BOTH "
                    "the event overlap adjustment (event_overlap.py) and the ridership "
                    "trend adjustment (ridership_trend.py) for that specific window -- "
                    "omit both for the pre-existing static-only behavior.")
    ap.add_argument("--campaign-end", default=None, help="YYYY-MM-DD (optional). See --campaign-start.")
    args = ap.parse_args()
    campaign_start, campaign_end = args.campaign_start, args.campaign_end
    if (campaign_start is None) != (campaign_end is None):
        raise SystemExit("--campaign-start and --campaign-end must be given together, or both omitted.")

    win_loss, grain = load_win_loss_and_grain()

    # --- NEW: actually load + use the GBM, instead of only describing it ---
    gbm_predictions = load_gbm_price_predictions()

    rate_card = build_rate_card(grain, gbm_predictions=gbm_predictions)
    win_loss_prepped = attach_price_ratio_and_weight(win_loss, rate_card)

    win_model = fit_win_model(win_loss_prepped)
    with open(OUT_DIR / "win_model_fit_report.json", "w") as f:
        json.dump(win_model.to_report(), f, indent=2)

    relationship = build_relationship_table(win_loss_prepped)
    relationship.per_client.to_csv(OUT_DIR / "client_relationship_features.csv", index=False)
    print(f"client_relationship_features.csv: {relationship.per_client.shape[0]} rows x "
          f"{relationship.per_client.shape[1]} cols")

    engine = PricingEngineD3(rate_card, win_model, relationship)

    if campaign_start is not None:
        print(f"\n[main] campaign window given ({campaign_start} -> {campaign_end}) -- "
              f"quote_grid() below will apply BOTH the event overlap adjustment AND the "
              f"ridership trend adjustment for these exact dates.")

    base_quotes = engine.quote_grid(client_id=None, campaign_start=campaign_start, campaign_end=campaign_end)

    # One price per cell, or say so loudly. A duplicate that DISAGREES is worse
    # than a crash: it is two defensible-looking answers to the same question.
    dups = base_quotes.duplicated(subset=["screen_id", "time_block_id"], keep=False)
    if dups.any():
        n_cells = base_quotes.groupby(["screen_id", "time_block_id"]).ngroups
        disagree = int((base_quotes.groupby(["screen_id", "time_block_id"])
                        ["recommended"].nunique() > 1).sum())
        raise SystemExit(
            f"[FATAL] pricing_quotes_d3 has {len(base_quotes):,} rows for "
            f"{n_cells:,} cells, and {disagree:,} of them carry two different "
            f"recommended prices. Refusing to write a quote file that answers "
            f"the same question twice. Check the rate-card collapse in "
            f"build_rate_card()."
        )

    # D1 -> D3 handoff. Additive: attaches the audience verdict, changes no price.
    base_quotes = apply_audience_gate(base_quotes)

    base_quotes.to_csv(OUT_DIR / "pricing_quotes_d3.csv", index=False)
    print(f"pricing_quotes_d3.csv: {base_quotes.shape[0]} rows x {base_quotes.shape[1]} cols "
          f"(anonymous rate-card quotes for every screen x time_block cell)")

    # The check that would have caught the 100%-at-cap defect on the run that
    # produced it. A recommendation pinned to an endpoint is not a
    # recommendation -- it means p_win is flat and the argmax is degenerate.
    if {"recommended", "cap", "floor"} <= set(base_quotes.columns):
        at_cap = float((base_quotes["recommended"] >= base_quotes["cap"] - 1e-6).mean())
        at_floor = float((base_quotes["recommended"] <= base_quotes["floor"] + 1e-6).mean())
        interior = 1.0 - at_cap - at_floor
        print(f"[sanity] recommended price: {at_cap:.1%} at cap, {at_floor:.1%} at floor, "
              f"{interior:.1%} strictly interior")
        if at_cap > 0.5 or at_floor > 0.5:
            print(f"[sanity] *** WARNING: {max(at_cap, at_floor):.1%} of quotes sit on a "
                  f"band endpoint. That is the signature of a flat p_win curve -- check "
                  f"win_model_fit_report.json for reject_reason before trusting these "
                  f"prices. ***")

    # --- worked examples: one screen WITH history, one WITHOUT (the case
    # the ML blend actually exists for) ---
    sample_cell = rate_card["cell"][rate_card["cell"]["has_history"]].iloc[0]
    sample_screen, sample_block = sample_cell["screen_id"], int(sample_cell["time_block_id"])
    print(f"\n--- worked example: screen {sample_screen}, block {sample_block} "
          f"(has real booking history) ---")
    print(engine.quote(sample_screen, sample_block, client_id=None,
                        campaign_start=campaign_start, campaign_end=campaign_end).explain())

    cold_start_rows = rate_card["cell"][~rate_card["cell"]["has_history"]]
    if len(cold_start_rows) > 0:
        cs_cell = cold_start_rows.iloc[0]
        cs_screen, cs_block = cs_cell["screen_id"], int(cs_cell["time_block_id"])
        print(f"\n--- worked example: screen {cs_screen}, block {cs_block} "
              f"(cold-start -- no booking history at all) ---")
        print(engine.quote(cs_screen, cs_block, client_id=None,
                            campaign_start=campaign_start, campaign_end=campaign_end).explain())

    sample_clients = relationship.per_client["client_id"].head(2).tolist()
    for cid in sample_clients:
        print(f"\n--- same cell (with history), client {cid} ---")
        print(engine.quote(sample_screen, sample_block, client_id=cid,
                            campaign_start=campaign_start, campaign_end=campaign_end).explain())

    print("\nDone. Saved: win_model_fit_report.json, client_relationship_features.csv, "
          "pricing_quotes_d3.csv (anonymous grid). Call PricingEngineD3.quote(screen_id, "
          "time_block_id, client_id) for a live, per-client, explainable quote. If the GBM "
          "model/model_card were found, its prediction is now actually blended into every "
          "cold-start cell's target price -- check the printed [rate card] line above and "
          "the ml_blend_credibility column in pricing_quotes_d3.csv to see where.")


if __name__ == "__main__":
    main()

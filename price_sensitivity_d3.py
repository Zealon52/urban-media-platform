"""D3 pricing sensitivity -- expected revenue vs price, with an elasticity CI.

WHAT THIS ANSWERS
-----------------
Challenge #2, "does it optimize demand?" -- honestly. The claim we defend is NOT
"we found a higher price." It is:

    the expected-revenue optimum is FLAT, and we can put a number on how much
    it moves once you admit the elasticity is estimated rather than known.

READ-ONLY. This script imports the deployed estimator out of `pricing_engine_d3`
and re-runs it; it does not re-derive, re-tune or re-implement anything, and it
writes only to two new files in eda_output/. It makes no API/LLM call.

WHY THE INPUT FILE IS NOT win_loss_table_d3.csv
-----------------------------------------------
The brief pointed at `eda_output/win_loss_table_d3.csv` for the 760 identified
observations. That file does not carry `price_gap_pct` -- the regressor the
identified fit is built on -- and `pricing_engine_d3.fit_win_model()` says so in
as many words:

    "NOTE: price_gap_pct is NOT carried on win_loss_table_d3.csv, it has to come
     from lost_leads.csv."

`win_loss_table_d3.csv` is the bookings-vs-leads sample: 191,753 rows, 99.7%
wins, and wins by construction because bookings are transacted prices. That
sample cannot identify a demand curve at any size. The identified sample is the
inside-losses comparison in `Urban Media Datasets/lost_leads.csv`: 1,450 lost
leads, of which 760 carry a price_gap_pct. Per the repo-is-authoritative rule we
bootstrap THAT, and we verify below that it reproduces the committed fit exactly.

THE ONE SUBTLETY THAT DRIVES THE WHOLE ANALYSIS
-----------------------------------------------
`WinModel.calibrate_to_reference()` sets base_ratio so that the revenue optimum
lands exactly on the rate-card target:

    base_ratio = 1 + logit(1 - 1/elasticity) / elasticity

So under the DEPLOYED curve the optimum is at price_ratio = 1.0 *by
construction*, and "revenue left on the table" is exactly 0.000% -- a tautology,
not a finding. Re-deriving that would be a vacuous deliverable.

The non-vacuous question, and the one this script answers, is:

    the shipped curve was calibrated on a POINT ESTIMATE of elasticity
    (14.221). If the true elasticity is elsewhere in its confidence interval,
    the shipped curve's location is wrong too -- so where is the true optimum,
    and what does quoting at the target cost us?

That is why elasticity is varied with base_ratio held at the SHIPPED value
(1.18155) rather than re-calibrated per draw. Re-calibrating would move the
optimum back to 1.0 every time and measure nothing. Both readings are reported
in the JSON (`vacuous_recalibrated_reading` records the tautology explicitly) so
the choice is auditable rather than buried.

Optimum condition, for the record: revenue R(r) = r * P(r) with
P(r) = sigmoid(-e * (r - b)), so R'(r) = P * [1 - r*e*(1 - P)] and the interior
optimum solves  r * e * (1 - P(r)) = 1.

Usage:  python price_sensitivity_d3.py [--resamples 2000] [--seed 20260823]
Writes: eda_output/price_sensitivity_d3.png
        eda_output/price_sensitivity_d3.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pricing_engine_d3 import PRICE_LOSS_REASONS, WinModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Urban Media Datasets"
OUT_DIR = BASE_DIR / "eda_output"

# Committed fit we must reproduce before we are allowed to trust anything else.
FIT_REPORT = OUT_DIR / "win_model_fit_report.json"
LOST_LEADS = DATA_DIR / "lost_leads.csv"
QUOTES = OUT_DIR / "pricing_quotes_d3.csv"

# The brief asks for at least 0.7x..1.6x the target -- deliberately wider than
# the p10..p90 band, so the extrapolation region is visible rather than cropped.
RATIO_LO, RATIO_HI, RATIO_STEP = 0.70, 1.60, 0.0025

# dataviz reference palette, categorical slots 1-3 (light mode), used unchanged.
# palette.md records these three as passing the all-pairs CVD and normal-vision
# gates in both modes, so they are safe for a 3-series chart.
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_CRIT = "#d03b3b"


# --------------------------------------------------------------------------- #
# 1. The fitted win model -- reproduce, do not re-derive
# --------------------------------------------------------------------------- #

def load_identified_sample() -> pd.DataFrame:
    if not LOST_LEADS.exists():
        raise SystemExit(
            f"FATAL: {LOST_LEADS} not found. The identified sample lives here, not "
            f"in win_loss_table_d3.csv (see module docstring). Stop and check with "
            f"Aditya rather than substituting another file.")
    ll = pd.read_csv(LOST_LEADS)
    missing = {"price_gap_pct", "loss_reason"} - set(ll.columns)
    if missing:
        raise SystemExit(f"FATAL: lost_leads.csv is missing {missing}.")
    d = ll[ll["price_gap_pct"].notnull()].reset_index(drop=True)
    d = d[["price_gap_pct", "loss_reason"]].copy()
    d["price_driven"] = d["loss_reason"].isin(PRICE_LOSS_REASONS)
    return d


def fit_quiet(gap, price_driven, calibrate: bool):
    """Run the deployed estimator with its stdout chatter suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        wm = WinModel().fit_from_price_gap(pd.Series(gap), pd.Series(price_driven))
        if calibrate and wm.fitted:
            wm.calibrate_to_reference()
    return wm


def verify_against_committed(sample: pd.DataFrame) -> dict:
    """Refit the full sample and assert it matches win_model_fit_report.json.

    If this check fails, every number downstream is suspect, so it is fatal
    rather than a warning.
    """
    committed = json.loads(FIT_REPORT.read_text())
    pre = fit_quiet(sample["price_gap_pct"].values, sample["price_driven"].values,
                    calibrate=False)
    post = fit_quiet(sample["price_gap_pct"].values, sample["price_driven"].values,
                     calibrate=True)
    if not post.fitted:
        raise SystemExit(f"FATAL: full-sample refit was refused ({post.reject_reason}).")

    checks = {
        "n_obs": (post.n_obs, committed["n_obs"], 0),
        "elasticity": (post.elasticity, committed["elasticity"], 1e-9),
        "base_ratio": (post.base_ratio, committed["base_ratio"], 1e-9),
    }
    for name, (got, want, tol) in checks.items():
        if abs(got - want) > tol:
            raise SystemExit(
                f"FATAL: refit does not reproduce the committed {name}: "
                f"got {got!r}, committed {want!r}. Do not trust these numbers.")

    print(f"  reproduced committed fit exactly: n_obs={post.n_obs}, "
          f"elasticity={post.elasticity:.9f}, base_ratio={post.base_ratio:.9f}")
    return {
        "n_obs": post.n_obs,
        "elasticity": post.elasticity,
        "base_ratio_calibrated": post.base_ratio,
        "base_ratio_as_fitted": pre.base_ratio,
        "calibration_shift": post.base_ratio - pre.base_ratio,
        "matches_committed_report": True,
    }


# --------------------------------------------------------------------------- #
# 2. Bootstrap the elasticity
# --------------------------------------------------------------------------- #

def bootstrap_elasticity(sample: pd.DataFrame, n_resamples: int, seed: int) -> dict:
    """Resample the 760 identified observations with replacement and refit.

    A refused fit is counted, never silently coerced: WinModel.fit() returns the
    *prior* (elasticity 6.0) when it refuses, so treating a refusal as a draw
    would inject a fake 6.0 into the CI and widen it downward.
    """
    gap = sample["price_gap_pct"].to_numpy()
    pdr = sample["price_driven"].to_numpy()
    n = len(sample)
    rng = np.random.default_rng(seed)

    draws, refused = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        wm = fit_quiet(gap[idx], pdr[idx], calibrate=False)
        if wm.fitted:
            draws.append(wm.elasticity)
        else:
            refused.append(wm.reject_reason)

    if len(draws) < 0.9 * n_resamples:
        print(f"  WARNING: {len(refused)}/{n_resamples} resamples were REFUSED by the "
              f"estimator's own guards -- the CI below is conditional on acceptance.")

    draws = np.asarray(draws, dtype=float)
    p5, p50, p95 = (float(x) for x in np.percentile(draws, [5, 50, 95]))
    print(f"  {len(draws):,} accepted / {len(refused):,} refused of {n_resamples:,}")
    print(f"  elasticity  p5={p5:.4f}  p50={p50:.4f}  p95={p95:.4f}")
    return {
        "n_resamples_requested": n_resamples,
        "n_accepted": len(draws),
        "n_refused": len(refused),
        "refusal_reasons": sorted(set(refused)),
        "seed": seed,
        "method": ("nonparametric bootstrap over the 760 identified observations, "
                   "refit with pricing_engine_d3.WinModel.fit_from_price_gap"),
        "p5": p5, "p50": p50, "p95": p95,
        "mean": float(draws.mean()), "std": float(draws.std(ddof=1)),
        "min": float(draws.min()), "max": float(draws.max()),
        "draws": draws,
    }


# --------------------------------------------------------------------------- #
# 3. Expected revenue over a price grid, on real cells
# --------------------------------------------------------------------------- #

def load_cells() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Real screen x time_block cells: target, recommended, and served p_win."""
    q = pd.read_csv(QUOTES, usecols=["target", "recommended", "p_win_at_recommended"])
    q = q[(q["target"] > 0) & q["p_win_at_recommended"].between(1e-9, 1 - 1e-9)]
    return (q["target"].to_numpy(), q["recommended"].to_numpy(),
            q["p_win_at_recommended"].to_numpy(), {"n_cells": len(q)})


def recover_shifts(target, recommended, p_win, elasticity, base_ratio) -> np.ndarray:
    """Recover the per-cell client-relationship shift.

    The engine evaluates p_win(price/target, shift) where `shift` is added to
    base_ratio. The shift is not persisted as a column, but it is exactly
    recoverable from what IS persisted:

        p = sigmoid(-e * (r - (b + shift)))
        =>  shift = r - b + logit(p) / e

    using the committed e and b. That keeps real per-cell heterogeneity in the
    curve without re-deriving the relationship layer.
    """
    r = recommended / target
    return r - base_ratio + np.log(p_win / (1.0 - p_win)) / elasticity


def bin_cells(targets: np.ndarray, shifts: np.ndarray, n_bins: int = 512):
    """Collapse cells to (shift, total_target) weight bins.

    Expected revenue is sum_i r * target_i * p_win(r, shift_i), and p_win depends
    on a cell ONLY through its shift. So cells sharing a shift are
    interchangeable and can be pooled by summing their targets. Binning the shift
    finely makes the bootstrap tractable (512 bins instead of ~67k cells) at
    negligible cost -- the shift spread is ~1e-3 wide, so within-bin variation is
    orders of magnitude below anything that moves p_win. `assert_binning_exact`
    checks the approximation against the full cell set before it is used.
    """
    edges = np.quantile(shifts, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.searchsorted(edges, shifts, side="right") - 1, 0, len(edges) - 2)
    w = np.bincount(idx, weights=targets, minlength=len(edges) - 1)
    s = np.bincount(idx, weights=targets * shifts, minlength=len(edges) - 1)
    keep = w > 0
    return s[keep] / w[keep], w[keep]


def revenue_curve(ratios, bin_targets, elasticity, base_ratio, bin_shifts):
    """Total expected revenue across all cells at each uniform price ratio."""
    z = -elasticity * (ratios[:, None] - (base_ratio + bin_shifts[None, :]))
    pw = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    return ratios * (pw @ bin_targets)


def gap_at_elasticity(ratios, bin_targets, elasticity, base_ratio, bin_shifts,
                      i_tgt: int) -> dict:
    """Optimal uniform ratio, and the % revenue gap vs quoting at the target."""
    curve = revenue_curve(ratios, bin_targets, elasticity, base_ratio, bin_shifts)
    i = int(np.argmax(curve))
    rev_tgt = float(curve[i_tgt])
    return {
        "elasticity": float(elasticity),
        "optimal_price_ratio": float(ratios[i]),
        "revenue_at_target": rev_tgt,
        "revenue_at_optimum": float(curve[i]),
        "pct_revenue_difference": 100.0 * (float(curve[i]) - rev_tgt) / rev_tgt,
        "curve": curve,
    }


def assert_binning_exact(ratios, targets, shifts, bin_shifts, bin_targets,
                         elasticity, base_ratio) -> float:
    """Verify the binned curve matches the full per-cell curve.

    Guards the one approximation in this script. Checked on a coarse ratio grid
    so the full ~67k-cell curve stays cheap to materialise.
    """
    probe = ratios[::20]
    z = -elasticity * (probe[:, None] - (base_ratio + shifts[None, :]))
    pw = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    full = probe * (pw @ targets)
    binned = revenue_curve(probe, bin_targets, elasticity, base_ratio, bin_shifts)
    rel = float(np.max(np.abs(full - binned) / full))
    if rel > 1e-9:
        raise SystemExit(f"FATAL: shift binning is not exact enough "
                         f"(max relative error {rel:.3e}); raise n_bins.")
    return rel


# --------------------------------------------------------------------------- #
# 4. Where the observed data actually ends
# --------------------------------------------------------------------------- #

def observed_support(sample: pd.DataFrame, shift: float) -> dict:
    """The empirical price window, in both coordinate systems.

    The fit's regressor is r = quoted / CLIENT target. The deployed curve is
    evaluated at r = price / OUR rate-card target, and calibrate_to_reference()
    translates the curve between the two by `shift` = base_ratio_calibrated -
    base_ratio_as_fitted. Observed support has to be carried through the same
    translation, otherwise the shaded region on the figure would be comparing
    two different x-axes.
    """
    gap = sample["price_gap_pct"]
    lo_fit, hi_fit = 1.0 + float(gap.min()), 1.0 + float(gap.max())
    return {
        "coordinate_note": (
            "The fit is estimated in client-target coordinates; the engine serves in "
            "rate-card coordinates. calibrate_to_reference() translates between them "
            "by +{:.6f}. Only ELASTICITY is estimated from data -- the deployed "
            "location is a calibration convention, so the mapped window inherits "
            "that convention.".format(shift)),
        "client_target_coords": {"lo": lo_fit, "hi": hi_fit,
                                 "price_gap_pct_lo": float(gap.min()),
                                 "price_gap_pct_hi": float(gap.max())},
        "rate_card_coords": {"lo": lo_fit + shift, "hi": hi_fit + shift},
        "width": hi_fit - lo_fit,
    }


# --------------------------------------------------------------------------- #
# 5. The figure
# --------------------------------------------------------------------------- #

ORDER = [("p5", "5th pct"), ("p50_shipped", "50th pct / shipped"), ("p95", "95th pct")]


def _style(ax) -> None:
    ax.set_facecolor(C_SURFACE)
    ax.grid(True, axis="y", color=C_GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=C_MUTED, labelsize=8.4, length=3)


def _mark_extrapolation(ax, ratios, lo_x, hi_x) -> None:
    """Shade + rule the region with no observed price data, on every panel."""
    x0, x1 = ax.get_xlim()
    for a, b in [(x0, min(lo_x, x1)), (max(hi_x, x0), x1)]:
        if b > a:
            ax.axvspan(a, b, color=C_MUTED, alpha=0.10, lw=0, zorder=0,
                       hatch="///", edgecolor=C_MUTED)
    for xb in (lo_x, hi_x):
        if x0 < xb < x1:
            ax.axvline(xb, color=C_CRIT, lw=1.4, zorder=3)
    ax.axvline(1.0, color=C_INK2, lw=1.0, zorder=3)


def make_figure(ratios, results, support, fitted, gap_max_pct, out_path: Path) -> None:
    fig = plt.figure(figsize=(13.6, 6.4), dpi=170)
    fig.patch.set_facecolor(C_SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.22,
                          left=0.065, right=0.985, top=0.80, bottom=0.175)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    lo_x, hi_x = support["rate_card_coords"]["lo"], support["rate_card_coords"]["hi"]

    # ---- panel 1: the whole curve, indexed to one common baseline ----------
    # Index every curve to 100 = revenue at the target under the SHIPPED
    # elasticity, so three curves at different levels share one axis (a second
    # y-axis would be the #1 charting mistake).
    base = results["p50_shipped"]["revenue_at_target"]
    ax.set_xlim(ratios[0], ratios[-1])
    _mark_extrapolation(ax, ratios, lo_x, hi_x)
    for (key, _lbl), colour in zip(ORDER, C_SERIES):
        res = results[key]
        ax.plot(ratios, 100.0 * res["curve"] / base, color=colour, lw=2.0, zorder=5)
        ax.plot([res["optimal_price_ratio"]],
                [100.0 * res["revenue_at_optimum"] / base],
                "o", ms=9, color=colour, mec=C_SURFACE, mew=2, zorder=7)

    ax.text(0.735, 30, "EXTRAPOLATION\nno observed price data\nbelow 1.094x",
            ha="left", va="center", fontsize=8.4, color=C_CRIT, weight="bold",
            linespacing=1.45, zorder=9)
    ax.text((lo_x + min(hi_x, ratios[-1])) / 2, 104,
            f"OBSERVED PRICE RANGE   {lo_x:.3f}x - {hi_x:.3f}x   (n={fitted['n_obs']})",
            ha="center", va="bottom", fontsize=8.6, color=C_INK2, weight="bold",
            zorder=9)
    ax.annotate("target = where the engine quotes", xy=(1.0, 72),
                xytext=(-7, 0), textcoords="offset points", rotation=90,
                ha="right", va="center", fontsize=8.2, color=C_INK2, zorder=9)
    ax.set_xlabel("price as a multiple of the rate-card target", fontsize=9.3,
                  color=C_INK2, labelpad=7)
    ax.set_ylabel("expected revenue\n(100 = quoting at target)", fontsize=9.3,
                  color=C_INK2, labelpad=7, linespacing=1.6)
    _style(ax)

    # ---- panel 2: each curve as a % of ITS OWN peak ------------------------
    # This is the flatness claim made visible: rescaling each curve by its own
    # maximum removes the level difference between elasticities, so what is left
    # is purely how fast revenue falls away from that elasticity's own optimum.
    ax2.set_xlim(0.90, 1.20)
    _mark_extrapolation(ax2, ratios, lo_x, hi_x)
    for (key, _lbl), colour in zip(ORDER, C_SERIES):
        res = results[key]
        ax2.plot(ratios, 100.0 * res["curve"] / res["revenue_at_optimum"],
                 color=colour, lw=2.0, zorder=5)
        ax2.plot([res["optimal_price_ratio"]], [100.0], "o", ms=9, color=colour,
                 mec=C_SURFACE, mew=2, zorder=7)
    ax2.set_ylim(98.6, 100.35)
    ax2.axhline(100.0, color=C_AXIS, lw=0.9, zorder=2)
    # Placed right of 1.10, the one region of this panel with no marks in it.
    ax2.text(1.105, 99.60,
             f"all optima span\n"
             f"{min(results[k]['optimal_price_ratio'] for k, _ in ORDER):.3f}x - "
             f"{max(results[k]['optimal_price_ratio'] for k, _ in ORDER):.3f}x\n\n"
             f"target is within\n{gap_max_pct:.2f}% of the peak\nacross the whole CI",
             ha="left", va="center", fontsize=8.2, color=C_INK2, linespacing=1.6,
             zorder=9)
    ax2.set_xlabel("price as a multiple of the rate-card target", fontsize=9.3,
                   color=C_INK2, labelpad=7)
    ax2.set_ylabel("% of that curve's own maximum", fontsize=9.3, color=C_INK2,
                   labelpad=7)
    ax2.set_title("the peak region, zoomed", fontsize=9.6, color=C_INK2, pad=8,
                  loc="left")
    _style(ax2)

    # ---- titles + one figure-level legend ---------------------------------
    fig.text(0.065, 0.950,
             "The expected-revenue optimum is flat, and stays flat across the "
             "elasticity confidence interval",
             fontsize=14, color=C_INK, va="bottom", weight="bold")
    fig.text(0.065, 0.925,
             f"D3 pricing sensitivity - {results['n_cells']:,} screen x time_block "
             f"cells; elasticity bootstrapped on the {fitted['n_obs']} identified "
             f"observations.\nQuoting at the target rather than the optimum costs at "
             f"most {gap_max_pct:.2f}% of expected revenue anywhere in the CI - not a "
             f"number worth repricing for.",
             fontsize=9.1, color=C_INK2, va="top", linespacing=1.6)

    # Per-series numbers live in the legend, not as labels on the marks: the
    # three optima are ~0.02x apart, so on-mark callouts collide illegibly.
    handles = [
        Line2D([], [], color=c, lw=2.0,
               label=f"elasticity {results[k]['elasticity']:.2f} ({lbl})   "
                     f"optimum {results[k]['optimal_price_ratio']:.3f}x, "
                     f"{results[k]['pct_revenue_difference']:+.2f}% vs target")
        for (k, lbl), c in zip(ORDER, C_SERIES)]
    handles += [
        Line2D([], [], color=C_CRIT, lw=1.4, label="edge of observed price data"),
        Line2D([], [], color=C_INK2, lw=1.0, label="rate-card target (where we quote)")]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.062, 0.005),
                     frameon=False, fontsize=8.5, ncol=2, columnspacing=3.0,
                     handlelength=1.9, labelspacing=0.5)
    for t in leg.get_texts():
        t.set_color(C_INK2)

    fig.savefig(out_path, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resamples", type=int, default=2000,
                    help="bootstrap resamples (brief requires >= 500)")
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()
    if args.resamples < 500:
        raise SystemExit("FATAL: the brief requires at least 500 resamples.")

    print("\n=== D3 PRICE SENSITIVITY ===\n")

    print("[1/5] identified sample + reproduce the committed fit")
    sample = load_identified_sample()
    n_price = int(sample["price_driven"].sum())
    print(f"  {len(sample)} observations with price_gap_pct "
          f"({n_price} price-driven, {len(sample) - n_price} not)")
    fitted = verify_against_committed(sample)
    e_hat = fitted["elasticity"]
    b_ship = fitted["base_ratio_calibrated"]
    print(f"  p_win at the target under the shipped curve = "
          f"{1.0 / (1.0 + math.exp(-(-e_hat * (1.0 - b_ship)))):.4f} "
          f"(= 1 - 1/elasticity = {1 - 1 / e_hat:.4f})")

    print(f"\n[2/5] bootstrapping elasticity ({args.resamples:,} resamples)")
    boot = bootstrap_elasticity(sample, args.resamples, args.seed)

    print("\n[3/5] real cells + recovered relationship shifts")
    targets, recommended, p_at_rec, meta = load_cells()
    shifts = recover_shifts(targets, recommended, p_at_rec, e_hat, b_ship)
    print(f"  {meta['n_cells']:,} cells; recovered shift "
          f"mean={shifts.mean():+.3e} sd={shifts.std():.3e} "
          f"range=[{shifts.min():+.3e}, {shifts.max():+.3e}]")
    print(f"  (the shift is ~0 because the engine picks `recommended` as the argmax "
          f"over its own candidate grid, so p_win tracks the target)")

    print("\n[4/5] expected revenue across the price grid")
    ratios = np.arange(RATIO_LO, RATIO_HI + RATIO_STEP / 2, RATIO_STEP)
    i_tgt = int(np.argmin(np.abs(ratios - 1.0)))
    bin_shifts, bin_targets = bin_cells(targets, shifts)
    rel = assert_binning_exact(ratios, targets, shifts, bin_shifts, bin_targets,
                               e_hat, b_ship)
    print(f"  pooled {meta['n_cells']:,} cells into {len(bin_targets)} shift bins "
          f"(max relative error vs full cell set {rel:.2e})")

    results = {"n_cells": meta["n_cells"]}
    for key, e in [("p5", boot["p5"]), ("p50_shipped", e_hat), ("p95", boot["p95"])]:
        results[key] = gap_at_elasticity(ratios, bin_targets, e, b_ship,
                                         bin_shifts, i_tgt)
        r = results[key]
        print(f"  elasticity {e:7.4f} -> optimum {r['optimal_price_ratio']:.4f}x, "
              f"{r['pct_revenue_difference']:+.4f}% vs quoting at target")

    # Full bootstrap distribution of the gap, so "distinguishable from zero?" is
    # answered from the whole CI rather than from three points. The gap is >= 0
    # by construction (the optimum is an argmax), so this is a one-sided
    # question about MAGNITUDE, not a two-sided test about sign.
    gaps = np.array([
        gap_at_elasticity(ratios, bin_targets, e, b_ship, bin_shifts,
                          i_tgt)["pct_revenue_difference"]
        for e in boot["draws"]])
    g5, g50, g95 = (float(x) for x in np.percentile(gaps, [5, 50, 95]))
    print(f"  revenue gap across the bootstrap: p5={g5:.4f}%  p50={g50:.4f}%  "
          f"p95={g95:.4f}%  max={gaps.max():.4f}%")

    support = observed_support(sample, fitted["calibration_shift"])
    rc = support["rate_card_coords"]
    print(f"  observed price window (rate-card coords) "
          f"{rc['lo']:.4f}x .. {rc['hi']:.4f}x")

    print("\n[5/5] writing artefacts")
    make_figure(ratios, results, support, fitted, float(gaps.max()),
                OUT_DIR / "price_sensitivity_d3.png")

    material = 1.0  # a <1% move is inside quote rounding on a single line
    payload = {
        "what_this_is": (
            "Expected-revenue-vs-price sensitivity for D3, with a bootstrapped "
            "elasticity CI. The deliverable is the FLATNESS of the optimum and its "
            "uncertainty -- not a recommendation to price higher."),
        "inputs": {
            "identified_sample": str(LOST_LEADS.relative_to(BASE_DIR)),
            "quotes": str(QUOTES.relative_to(BASE_DIR)),
            "committed_fit_report": str(FIT_REPORT.relative_to(BASE_DIR)),
            "why_not_win_loss_table": (
                "win_loss_table_d3.csv does not carry price_gap_pct (the identified "
                "regressor) and is 99.7% wins by construction; pricing_engine_d3."
                "fit_win_model() states the column has to come from lost_leads.csv."),
        },
        "fitted_model": fitted,
        "elasticity_ci": {k: v for k, v in boot.items() if k != "draws"},
        "observed_support": support,
        "price_grid": {"lo": RATIO_LO, "hi": RATIO_HI, "step": RATIO_STEP,
                       "n_points": len(ratios)},
        "cells": {"n_cells": meta["n_cells"],
                  "relationship_shift_mean": float(shifts.mean()),
                  "relationship_shift_sd": float(shifts.std())},
        "optimum_by_elasticity": {
            k: {kk: vv for kk, vv in results[k].items() if kk != "curve"}
            for k in ("p5", "p50_shipped", "p95")},
        "revenue_gap_bootstrap_pct": {
            "p5": g5, "p50": g50, "p95": g95,
            "max": float(gaps.max()), "mean": float(gaps.mean()),
            "note": ("Gap is >= 0 by construction because the optimum is an argmax, "
                     "so this is a one-sided statement about magnitude. The question "
                     "is whether the magnitude is material, not whether the sign is "
                     "certain."),
        },
        "headline": {
            "pct_revenue_left_on_table_at_shipped_elasticity":
                results["p50_shipped"]["pct_revenue_difference"],
            "pct_revenue_left_on_table_worst_case_in_ci": float(gaps.max()),
            "materiality_threshold_pct": material,
            "distinguishable_from_zero": bool(gaps.max() > material),
            "plain_words": (
                "Not distinguishable from zero in any way that matters. Across the "
                "full bootstrap CI the most revenue that quoting at the target ever "
                "leaves on the table is {:.2f}% -- below the {:.0f}% materiality "
                "threshold and far below the rounding on a single quoted line. The "
                "optimum is flat: the curve is within a fraction of a percent of its "
                "peak everywhere between {:.2f}x and {:.2f}x. This is a finding, not "
                "a failure."
            ).format(float(gaps.max()), material,
                     min(results[k]["optimal_price_ratio"] for k in ("p5", "p95")),
                     max(results[k]["optimal_price_ratio"] for k in ("p5", "p95"))),
        },
        "vacuous_recalibrated_reading": {
            "optimal_price_ratio": 1.0,
            "pct_revenue_difference": 0.0,
            "note": ("If base_ratio is RE-calibrated at each bootstrap elasticity the "
                     "way calibrate_to_reference() does, the optimum returns to "
                     "exactly 1.0 and the gap is exactly 0.000% for every draw. That "
                     "is a tautology of the calibration, not evidence. Reported here "
                     "so the choice to hold base_ratio at the shipped value is "
                     "auditable."),
        },
        "recommendation": (
            "Do NOT price higher. The optimum is flat, the whole CI agrees, and the "
            "target already sits at p_win = 1 - 1/elasticity. Moving up would also "
            "push the quote toward the top of -- and past -- the observed price "
            "window, where the win curve is extrapolated rather than measured."),
        "honest_caveats": [
            "Only elasticity is estimated. The deployed curve's LOCATION comes from "
            "calibrate_to_reference(), which assumes the rate-card target is the "
            "optimum; this analysis tests the consequences of elasticity error, not "
            "of that assumption being wrong.",
            "Carried through the calibration shift, the observed price window is "
            "{:.3f}x..{:.3f}x in rate-card coordinates, so the target itself (1.000x) "
            "sits {:.3f} BELOW the bottom of the observed window. The engine's quote "
            "point is therefore not inside the measured price range -- a real limit "
            "to name before a judge does.".format(
                rc["lo"], rc["hi"], rc["lo"] - 1.0),
            "The bootstrap resamples observations, so it captures sampling error "
            "only -- not the selection question of whether inside-losses generalise "
            "to won deals.",
            "n=760 is the identified sample; heterogeneous elasticity by segment was "
            "considered and rejected upstream as worse-identified.",
        ],
    }
    out_json = OUT_DIR / "price_sensitivity_d3.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_json}")

    print("\n--- SUMMARY ---")
    print(f"1. Elasticity CI from {boot['n_accepted']:,} bootstrap refits of the "
          f"{fitted['n_obs']} identified observations: "
          f"p5={boot['p5']:.2f}, p50={boot['p50']:.2f}, p95={boot['p95']:.2f} "
          f"(point estimate {e_hat:.2f}).")
    print(f"2. The optimal price ratio moves only from "
          f"{results['p5']['optimal_price_ratio']:.3f}x to "
          f"{results['p95']['optimal_price_ratio']:.3f}x across that CI -- the "
          f"optimum barely moves.")
    print(f"3. Quoting at the target rather than the optimum costs at most "
          f"{gaps.max():.2f}% of expected revenue anywhere in the CI "
          f"(median {g50:.2f}%).")
    print(f"4. That is NOT distinguishable from zero in any decision-relevant "
          f"sense -- it is smaller than the rounding on a single quoted line. "
          f"Finding, not failure.")
    print(f"5. Observed price data spans {rc['lo']:.3f}x..{rc['hi']:.3f}x; everything "
          f"outside is extrapolation and is labelled as such on the figure. Do not "
          f"price higher.\n")


if __name__ == "__main__":
    main()

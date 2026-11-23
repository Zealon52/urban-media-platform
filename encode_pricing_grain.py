"""
encode_pricing_grain.py
Encodes pricing_grain_d3.csv into a model-ready file: fills structural
and cold-start nulls (persisting the exact values used, so serving-time
predictions use the identical contract), one-hot encodes categoricals,
keeps time_block_id ordinal, and hard-fails if any numeric/bool feature
column still has NaNs afterward (except price_p10/price_p90, which are
target-adjacent and handled elsewhere, never used as model inputs).

Fixes in this version:
  - booking_count added to the structural-zero imputation plan (never-
    booked cells legitimately have 0 bookings, not an unknown median)
  - day_type added to the categorical one-hot encoding list (new grain
    dimension from build_training_table.py's day_type-aware rebuild)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "eda_output"
SRC = OUT_DIR / "pricing_grain_d3.csv"
DEST = OUT_DIR / "pricing_grain_d3_encoded.csv"
CONTRACT_PATH = OUT_DIR / "imputation_contract.json"

# Columns whose missingness is STRUCTURAL (a real, meaningful absence,
# not something to guess at with a median) -- filled with 0, optionally
# paired with a boolean flag so the model can distinguish
# "genuinely zero" from "not applicable to this screen".
STRUCTURAL_ZERO_IMPUTATION_PLAN = {
    "location_footfall_log": "has_footfall",
    "corridor_ridership_log": "has_ridership",
    "booking_count": None,  # has_history already captures this distinction
}

# Columns whose missingness reflects cold-start / thin-history gaps --
# filled with the median of has_history=True rows, and that exact value
# is persisted to imputation_contract.json so serving-time inference
# uses the identical fill value used at training time.
NUMERIC_IMPUTATION_PLAN = [
    "avg_lead_time_days",
    "bundle_share",
    "income_index",
    "population_density_per_sqkm",
    "daytime_population_multiplier",
]

CATEGORICAL_COLUMNS = [
    "mount_type",
    "screen_type",
    "screen_size",
    "position",
    "market_tier",
    "city_id",
    "day_type",
]

ORDINAL_COLUMNS = ["time_block_id"]

# columns that are correctly excluded from the NaN hard-check: they are
# the target and target-adjacent percentile columns, never used as
# model inputs, and are allowed to be missing for never-booked cells.
TARGET_ADJACENT_COLUMNS = {"price_p10", "price_p50", "price_p90"}

# pure ID / join-key columns that should never be one-hot encoded or
# treated as model features, but are also not an error if left as-is.
ID_JOIN_KEY_COLUMNS = {"screen_id", "location_id", "vehicle_id", "corridor_id", "zone_id"}

def main():
    df = pd.read_csv(SRC, low_memory=False)
    print(f"loaded pricing_grain_d3.csv: {df.shape}")

    # -- position nulls: fill with an explicit 'unknown' category, don't drop rows --
    n_pos_null = df["position"].isna().sum()
    df["position"] = df["position"].fillna("unknown")
    print(f"[position] filled {n_pos_null} nulls -> 'unknown' category")
    print()

    # -- some expected categorical columns (client_tier, bundle_affinity,
    #    negotiation_leverage) live in client_facts.csv, not pricing_grain_d3.csv.
    #    They are relationship-pricing inputs used later in pricing_engine_d3.py,
    #    not part of this grid -- skip them here rather than error. --
    expected_but_elsewhere = ["client_tier", "bundle_affinity", "negotiation_leverage"]
    missing_expected = [c for c in expected_but_elsewhere if c not in df.columns]
    if missing_expected:
        print(f"[warn] expected categorical columns not found in file, skipping: {missing_expected}")

    contract = {"structural_zero": {}, "median_fill": {}}

    # -- structural-zero imputation --
    print()
    for col, flag_col in STRUCTURAL_ZERO_IMPUTATION_PLAN.items():
        if col not in df.columns:
            print(f"[impute] expected column '{col}' not found -- skipping "
                  f"(check pricing_grain_d3.csv still has this column)")
            continue
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        if flag_col:
            df[flag_col] = df[col].notna().astype(int)
        df[col] = df[col].fillna(0)
        contract["structural_zero"][col] = {"flag_col": flag_col, "n_missing": int(n_missing)}
        flag_note = f"; added flag '{flag_col}'" if flag_col else ""
        print(f"[impute] {col}: {n_missing} missing ({pct:.1f}%) -> filled with 0 "
              f"(structural absence, mount_type branch){flag_note}")

    # -- cold-start median imputation, persisted for serving-time consistency --
    has_history_mask = df["has_history"] if "has_history" in df.columns else pd.Series([True] * len(df))
    for col in NUMERIC_IMPUTATION_PLAN:
        if col not in df.columns:
            print(f"[impute] expected column '{col}' not found -- skipping")
            continue
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        pct = n_missing / len(df) * 100
        median_val = df.loc[has_history_mask, col].median()
        if pd.isna(median_val):
            median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        contract["median_fill"][col] = float(median_val)
        print(f"[impute] {col}: {n_missing} missing ({pct:.1f}%) -> filled with {median_val:.4f} "
              f"(median of has_history=True rows, persisted)")

    CONTRACT_PATH.write_text(json.dumps(contract, indent=2))
    print()

    # -- one-hot encode categoricals, keep time_block_id ordinal --
    cat_cols_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    cols_before = df.shape[1]
    df_encoded = pd.get_dummies(df, columns=cat_cols_present, dtype="int")
    cols_after = df_encoded.shape[1]

    new_cols = [c for c in df_encoded.columns if c not in df.columns]
    print(f"encoded {len(cat_cols_present)} categorical columns: {cat_cols_present}")
    print(f"kept ordinal (not one-hot): {ORDINAL_COLUMNS}")
    print(f"columns before encoding: {cols_before}, after: {cols_after} (+{cols_after - cols_before})")
    print()
    print(f"new one-hot columns created ({len(new_cols)}):")
    for c in new_cols:
        print(f"  {c}")

    # -- surface any remaining string/object columns instead of silently
    #    letting them vanish inside model_selector.py's select_dtypes call --
    remaining_object_cols = [
        c for c in df_encoded.select_dtypes(include=["object"]).columns
    ]
    print()
    if remaining_object_cols:
        unexpected = [c for c in remaining_object_cols if c not in ID_JOIN_KEY_COLUMNS
                      and c not in ("nearest_daypart",)]
        if unexpected:
            print(f"[warn] additional string columns not explicitly handled: {unexpected}")
            print("       (these would be silently dropped by model_selector.py -- review before training)")
        print(f"remaining string/object columns (kept as-is, e.g. ID join keys): {remaining_object_cols}")

    # -- hard-stop: fail loudly if any numeric/bool FEATURE column still
    #    has NaNs, explicitly excluding the target-adjacent percentile columns --
    numeric_bool_cols = df_encoded.select_dtypes(include=["number", "bool"]).columns
    feature_cols = [c for c in numeric_bool_cols if c not in TARGET_ADJACENT_COLUMNS]
    nan_counts = df_encoded[feature_cols].isna().sum()
    still_missing = nan_counts[nan_counts > 0]
    if len(still_missing) > 0:
        raise RuntimeError(
            "NaNs remain in numeric/bool FEATURE columns after imputation -- this is exactly "
            "the bug this rewrite exists to fix. Columns still missing values:\n"
            f"{still_missing.to_dict()}\n"
            "Add these to NUMERIC_IMPUTATION_PLAN or STRUCTURAL_ZERO_IMPUTATION_PLAN before proceeding."
        )

    df_encoded.to_csv(DEST, index=False)
    print()
    print(f"saved {DEST}: {df_encoded.shape[0]} rows x {df_encoded.shape[1]} cols")

if __name__ == "__main__":
    main()

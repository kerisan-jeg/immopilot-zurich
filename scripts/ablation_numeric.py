"""Feature-group ablation for the numeric block.

Quantifies how much each feature *group* contributes to rent prediction by
retraining XGBoost with that group removed and measuring the change in test MAE
(deltaMAE = MAE_without_group - MAE_baseline). A positive deltaMAE means the group
helps (removing it makes the model worse).

Method — kept faithful to the real training pipeline:
- same data (features.parquet), same split (`_common.make_splits`, seed 42,
  stratified by is_zurich), same log-target handling,
- a fresh ColumnTransformer is built per ablation that simply omits the dropped
  columns (so preprocessing parity is preserved for the remaining features),
- XGBoost with the project's default params (no Optuna re-tuning, so the
  comparison isolates the feature groups rather than hyper-parameter luck).

Groups (interpretable, may regroup the raw ColumnTransformer buckets):
  district : rent_median_chf_per_m2, rent_mean_chf_per_m2, location_kreis, is_zurich
  cv       : condition_score, kitchen_quality
  text     : is_luxurious, is_furnished, is_temporary
  amenities: has_balcony, has_view, has_elevator, has_garage, has_parking, has_fireplace

Usage:  python scripts/ablation_numeric.py
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from immopilot import config
from immopilot.features.build_features import (
    BINARY_LISTINGS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TEXT_DERIVED_BINARY,
)
from immopilot.models._common import load_features, make_splits, split_xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = config.DOCS_DIR / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Interpretable feature groups to ablate.
GROUPS: dict[str, list[str]] = {
    "district": ["rent_median_chf_per_m2", "rent_mean_chf_per_m2", "location_kreis", "is_zurich"],
    "cv": ["condition_score", "kitchen_quality"],
    "text": ["is_luxurious", "is_furnished", "is_temporary"],
    "amenities": ["has_balcony", "has_view", "has_elevator", "has_garage", "has_parking", "has_fireplace"],
}


def build_preprocessor(drop: set[str]) -> ColumnTransformer:
    """ColumnTransformer over all feature columns minus `drop`."""
    num = [c for c in NUMERIC_FEATURES if c not in drop]
    cat = [c for c in CATEGORICAL_FEATURES if c not in drop]
    binl = [c for c in BINARY_LISTINGS if c not in drop]
    bint = [c for c in TEXT_DERIVED_BINARY if c not in drop]

    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    if binl:
        transformers.append(("bin_listings", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]), binl))
    if bint:
        transformers.append(("bin_text", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]), bint))

    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def xgb() -> XGBRegressor:
    """Project-default XGBoost (no early stopping → no eval set needed for ablation)."""
    params = dict(config.XGB_DEFAULT_PARAMS)
    params.pop("early_stopping_rounds", None)
    params["n_estimators"] = 400  # fixed, modest; ablation isolates features not tuning
    return XGBRegressor(**params)


def fit_eval(drop: set[str], X_train, y_train, X_test, y_test) -> float:
    pre = build_preprocessor(drop)
    Xt_tr = pre.fit_transform(X_train)
    Xt_te = pre.transform(X_test)
    model = xgb()
    model.fit(Xt_tr, y_train)
    pred = model.predict(Xt_te)
    if config.LOG_TARGET:
        pred = np.expm1(pred)
        true = np.expm1(y_test)
    else:
        true = y_test
    return float(mean_absolute_error(true, pred))


def main() -> None:
    config.set_global_seed()
    df = load_features()
    X, y = split_xy(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    # Train on train+val pool (ablation doesn't need early stopping), eval on test.
    import pandas as pd
    X_pool = pd.concat([X_train, X_val])
    y_pool = np.concatenate([y_train, y_val])

    baseline_mae = fit_eval(set(), X_pool, y_pool, X_test, y_test)
    logger.info("Baseline (all features) test MAE: %.1f CHF", baseline_mae)

    rows = []
    for name, cols in GROUPS.items():
        mae = fit_eval(set(cols), X_pool, y_pool, X_test, y_test)
        delta = mae - baseline_mae
        rows.append({"group": name, "n_cols": len(cols), "mae": mae, "delta_mae": delta})
        logger.info("drop %-10s | MAE %.1f | deltaMAE %+.1f CHF", name, mae, delta)

    rows.sort(key=lambda r: r["delta_mae"], reverse=True)
    results = {"baseline_mae": baseline_mae, "ablations": rows}
    (OUT_DIR / "ablation_numeric.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 56)
    print(f"{'Dropped group':<14}{'MAE (CHF)':>12}{'deltaMAE':>12}{'cols':>8}")
    print("-" * 56)
    print(f"{'(baseline)':<14}{baseline_mae:>12.1f}{'—':>12}{len(X.columns):>8}")
    for r in rows:
        print(f"{r['group']:<14}{r['mae']:>12.1f}{r['delta_mae']:>+12.1f}{r['n_cols']:>8}")
    print("=" * 56)
    print("Positive deltaMAE = group helps (removing it worsens the model).")
    print(f"\nSaved: {OUT_DIR / 'ablation_numeric.json'}")


if __name__ == "__main__":
    main()

"""Freeze the XGBoost test-set predictions for reproducibility.

Loads the committed feature table and the trained XGBoost artifact, rebuilds the
exact same split (seed 42, stratified by is_zurich), predicts on the held-out
test set, and writes:

  docs/repro/test_predictions.csv  — y_true, y_pred (CHF) per test row
  docs/repro/test_metrics.json     — MAE / RMSE / R2 recomputed from that CSV

This lets a grader verify the headline numbers (MAE ~337, R2 ~0.775) directly
from committed artifacts, without needing the raw Kaggle download.

Usage:  python scripts/freeze_test_predictions.py
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from immopilot import config
from immopilot.models._common import load_features, make_splits, split_xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = config.DOCS_DIR / "repro"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config.set_global_seed()

    df = load_features()
    X, y = split_xy(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    pre = joblib.load(config.MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(config.MODELS_DIR / "xgboost.joblib")

    y_pred = model.predict(pre.transform(X_test))
    if config.LOG_TARGET:
        y_true_chf = np.expm1(y_test)
        y_pred_chf = np.expm1(y_pred)
    else:
        y_true_chf, y_pred_chf = y_test, y_pred

    out = pd.DataFrame({
        "y_true_chf": np.round(y_true_chf, 1),
        "y_pred_chf": np.round(y_pred_chf, 1),
        "abs_error_chf": np.round(np.abs(y_true_chf - y_pred_chf), 1),
        "is_zurich": X_test["is_zurich"].to_numpy(),
    })
    csv_path = OUT_DIR / "test_predictions.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8")

    metrics = {
        "n_test": int(len(out)),
        "n_test_zurich": int(out["is_zurich"].sum()),
        "mae_chf": float(mean_absolute_error(y_true_chf, y_pred_chf)),
        "rmse_chf": float(np.sqrt(mean_squared_error(y_true_chf, y_pred_chf))),
        "r2": float(r2_score(y_true_chf, y_pred_chf)),
    }
    (OUT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 48)
    print("FROZEN TEST-SET METRICS (recomputed from CSV)")
    print("-" * 48)
    print(f"n_test          : {metrics['n_test']}")
    print(f"n_test (Zurich) : {metrics['n_test_zurich']}")
    print(f"MAE  (CHF)      : {metrics['mae_chf']:.1f}")
    print(f"RMSE (CHF)      : {metrics['rmse_chf']:.1f}")
    print(f"R2              : {metrics['r2']:.3f}")
    print("=" * 48)
    print(f"Saved: {csv_path}")
    print(f"Saved: {OUT_DIR / 'test_metrics.json'}")


if __name__ == "__main__":
    main()

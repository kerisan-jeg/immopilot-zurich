"""Shared training utilities: split, evaluation metrics, model registry."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from immopilot import config
from immopilot.features.build_features import (
    BINARY_LISTINGS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TEXT_DERIVED_BINARY,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_LISTINGS + TEXT_DERIVED_BINARY


@dataclass
class EvalResult:
    model_name: str
    test_mae: float
    test_rmse: float
    test_r2: float
    cv_mae_mean: float
    cv_mae_std: float


def load_features() -> pd.DataFrame:
    p = config.PROCESSED_DIR / "features.parquet"
    if not p.exists():
        raise FileNotFoundError("Run `make features` first.")
    return pd.read_parquet(p)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    y = df[config.NUMERIC_TARGET].to_numpy()
    if config.LOG_TARGET:
        y = np.log1p(y)
    X = df[FEATURE_COLUMNS].copy()
    return X, y


def make_splits(X: pd.DataFrame, y: np.ndarray):
    """Stratify by is_zurich (binary) — robust regardless of class imbalance."""
    strat_col = "is_zurich" if "is_zurich" in X.columns else None
    strat = X[strat_col] if strat_col else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=strat
    )
    strat_tr = X_train[strat_col] if strat_col else None
    rel_val = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=rel_val, random_state=config.SEED, stratify=strat_tr
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model_name: str, model, preprocessor, X_test, y_test, X_full, y_full) -> EvalResult:
    """Compute test metrics + cross-validated MAE on the union train+val."""
    y_test_pred = model.predict(preprocessor.transform(X_test))
    if config.LOG_TARGET:
        y_test_inv = np.expm1(y_test)
        y_pred_inv = np.expm1(y_test_pred)
    else:
        y_test_inv, y_pred_inv = y_test, y_test_pred

    mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
    rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    r2 = float(r2_score(y_test_inv, y_pred_inv))

    # Cross-validated MAE on training pool — use a fresh clone per fold
    from sklearn.base import clone

    kf = KFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.SEED)
    fold_maes: list[float] = []
    Xt = preprocessor.transform(X_full)
    for tr_idx, va_idx in kf.split(Xt):
        m = clone(model)
        m.fit(Xt[tr_idx], y_full[tr_idx])
        pred = m.predict(Xt[va_idx])
        if config.LOG_TARGET:
            pred = np.expm1(pred)
            true = np.expm1(y_full[va_idx])
        else:
            true = y_full[va_idx]
        fold_maes.append(mean_absolute_error(true, pred))

    return EvalResult(
        model_name=model_name,
        test_mae=mae,
        test_rmse=rmse,
        test_r2=r2,
        cv_mae_mean=float(np.mean(fold_maes)),
        cv_mae_std=float(np.std(fold_maes)),
    )


def save_model(model, name: str, result: EvalResult) -> Path:
    out = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, out)
    metrics_out = config.MODELS_DIR / f"{name}.metrics.json"
    metrics_out.write_text(json.dumps(asdict(result), indent=2))
    logger.info("Saved %s — test MAE %.1f CHF, R² %.3f", out.name, result.test_mae, result.test_r2)
    return out

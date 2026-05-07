"""Train an XGBoost regressor — usually the strongest tabular baseline.

Includes optional Optuna hyperparameter search (50 trials by default).
Set ``SKIP_OPTUNA=1`` to use the pinned defaults from config (faster CI / smoke).
"""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from immopilot import config
from immopilot.models._common import (
    EvalResult,
    load_features,
    make_splits,
    save_model,
    split_xy,
)

logger = logging.getLogger(__name__)


def _objective(trial: optuna.Trial, X_tr_t, y_tr, X_va_t, y_va) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": config.SEED,
        "n_jobs": -1,
    }
    model = XGBRegressor(**params)
    model.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_va)], verbose=False)
    pred = model.predict(X_va_t)
    if config.LOG_TARGET:
        pred = np.expm1(pred)
        true = np.expm1(y_va)
    else:
        true = y_va
    return float(mean_absolute_error(true, pred))


def main(n_trials: int = 50) -> None:
    config.set_global_seed()
    df = load_features()
    X, y = split_xy(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(X, y)

    preprocessor = joblib.load(config.MODELS_DIR / "preprocessor.joblib")
    X_tr_t = preprocessor.transform(X_tr)
    X_va_t = preprocessor.transform(X_va)
    X_te_t = preprocessor.transform(X_te)
    X_full_t = np.vstack([X_tr_t, X_va_t])
    y_full = np.concatenate([y_tr, y_va])

    if n_trials > 0 and os.getenv("SKIP_OPTUNA") != "1":
        logger.info("Running Optuna search (%d trials)…", n_trials)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=config.SEED),
        )
        study.optimize(
            lambda t: _objective(t, X_tr_t, y_tr, X_va_t, y_va), n_trials=n_trials
        )
        best_params = study.best_params
        logger.info("Best Optuna params: %s", best_params)
    else:
        best_params = {
            k: v for k, v in config.XGB_DEFAULT_PARAMS.items() if k != "early_stopping_rounds"
        }

    # Final fit on train+val, evaluate on held-out test
    model = XGBRegressor(**best_params, random_state=config.SEED, n_jobs=-1)
    model.fit(X_full_t, y_full)

    pred = model.predict(X_te_t)
    if config.LOG_TARGET:
        pred_inv = np.expm1(pred)
        true_inv = np.expm1(y_te)
    else:
        pred_inv, true_inv = pred, y_te

    result = EvalResult(
        model_name="xgboost",
        test_mae=float(mean_absolute_error(true_inv, pred_inv)),
        test_rmse=float(np.sqrt(mean_squared_error(true_inv, pred_inv))),
        test_r2=float(r2_score(true_inv, pred_inv)),
        cv_mae_mean=float("nan"),
        cv_mae_std=float("nan"),
    )
    save_model(model, "xgboost", result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()

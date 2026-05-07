"""Random Forest regressor."""

from __future__ import annotations

import logging

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from immopilot import config
from immopilot.models._common import EvalResult, load_features, make_splits, save_model, split_xy


def main() -> None:
    config.set_global_seed()
    df = load_features()
    X, y = split_xy(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(X, y)

    preprocessor = joblib.load(config.MODELS_DIR / "preprocessor.joblib")
    X_full_t = np.vstack([preprocessor.transform(X_tr), preprocessor.transform(X_va)])
    y_full = np.concatenate([y_tr, y_va])
    X_te_t = preprocessor.transform(X_te)

    model = RandomForestRegressor(
        n_estimators=500, max_depth=None, n_jobs=-1, random_state=config.SEED
    )
    model.fit(X_full_t, y_full)

    pred = model.predict(X_te_t)
    if config.LOG_TARGET:
        pred_inv = np.expm1(pred)
        true_inv = np.expm1(y_te)
    else:
        pred_inv, true_inv = pred, y_te

    result = EvalResult(
        model_name="random_forest",
        test_mae=float(mean_absolute_error(true_inv, pred_inv)),
        test_rmse=float(np.sqrt(mean_squared_error(true_inv, pred_inv))),
        test_r2=float(r2_score(true_inv, pred_inv)),
        cv_mae_mean=float("nan"),
        cv_mae_std=float("nan"),
    )
    save_model(model, "random_forest", result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()

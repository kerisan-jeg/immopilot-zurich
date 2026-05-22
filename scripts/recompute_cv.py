"""Recompute 5-fold CV MAE for the saved sklearn-style models and update metrics JSONs.

Background: the four training scripts hard-coded ``cv_mae_mean = float("nan")`` and never
wired in the working CV logic. Rather than retrain (which would re-run Optuna), this
script loads each persisted model, clones it (a fresh, unfitted copy with identical
hyperparameters), and runs a real 5-fold CV on the preprocessed training pool. The
held-out *test* metrics already stored in each JSON are preserved untouched — only the
``cv_mae_mean`` / ``cv_mae_std`` fields are filled in.

The MLP is skipped on purpose: its persisted object is a small torch wrapper that
scikit-learn cannot ``clone``, and a cross-validated score of a diverged model
(test R^2 = -196) carries no information.

Run:  python -m scripts.recompute_cv      (or:  python scripts/recompute_cv.py)
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from immopilot import config
from immopilot.models._common import load_features, make_splits, split_xy

SKLEARN_MODELS = ["linear_ridge", "random_forest", "xgboost"]


def main() -> None:
    config.set_global_seed()
    df = load_features()
    X, y = split_xy(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(X, y)

    pre = joblib.load(config.MODELS_DIR / "preprocessor.joblib")
    X_full_t = np.vstack([pre.transform(X_tr), pre.transform(X_va)])
    y_full = np.concatenate([y_tr, y_va])

    kf = KFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.SEED)

    print(f"5-fold CV on training pool (n={len(y_full)}), target log1p={config.LOG_TARGET}\n")
    available = [n for n in SKLEARN_MODELS if (config.MODELS_DIR / f"{n}.joblib").exists()]
    missing = [n for n in SKLEARN_MODELS if n not in available]
    if missing:
        print(f"Skipping (no committed .joblib): {', '.join(missing)} "
              f"-- retrain via `make train-numeric` to include them.\n")
    if not available:
        print("No model .joblib files found. Run `make train-numeric` first.")
        return

    for name in available:
        model_path = config.MODELS_DIR / f"{name}.joblib"
        model = joblib.load(model_path)

        fold_maes: list[float] = []
        for tr_idx, va_idx in kf.split(X_full_t):
            m = clone(model)
            m.fit(X_full_t[tr_idx], y_full[tr_idx])
            pred = m.predict(X_full_t[va_idx])
            if config.LOG_TARGET:
                pred = np.expm1(pred)
                true = np.expm1(y_full[va_idx])
            else:
                true = y_full[va_idx]
            fold_maes.append(float(mean_absolute_error(true, pred)))

        cv_mean = float(np.mean(fold_maes))
        cv_std = float(np.std(fold_maes))

        mpath = config.MODELS_DIR / f"{name}.metrics.json"
        metrics = json.loads(mpath.read_text())
        metrics["cv_mae_mean"] = cv_mean
        metrics["cv_mae_std"] = cv_std
        mpath.write_text(json.dumps(metrics, indent=2))

        print(
            f"{name:15s} | CV MAE {cv_mean:6.1f} +/- {cv_std:4.1f} CHF "
            f"| test MAE {metrics['test_mae']:6.1f} | folds {[round(x) for x in fold_maes]}"
        )

    print("\nUpdated metrics JSONs. MLP skipped (not cloneable / diverged).")


if __name__ == "__main__":
    main()

"""End-to-end inference pipeline orchestrating all three blocks.

Public API: ``predict(structured, listing_text, photos)`` → :class:`PredictionResult`.

This is what the Gradio app calls.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from immopilot import config
from immopilot.cv.zero_shot_clip import PhotoFeatures, extract_features
from immopilot.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TEXT_DERIVED_BINARY,
    add_engineered_columns,
)
from immopilot.nlp.explainer import Explanation, explain
from immopilot.nlp.listing_parser import parse_listing

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_DERIVED_BINARY


@dataclass
class PredictionResult:
    point_estimate_chf: float
    interval_low_chf: float
    interval_high_chf: float
    photo_features: PhotoFeatures | None
    parsed_listing: dict[str, Any] | None
    explanation: Explanation | None
    feature_table: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.photo_features is not None:
            d["photo_features"] = asdict(self.photo_features)
        if self.explanation is not None:
            d["explanation"] = {
                "text": self.explanation.text,
                "top_positive": self.explanation.top_positive,
                "top_negative": self.explanation.top_negative,
            }
        return d


@lru_cache(maxsize=1)
def _load_artifacts():
    pre = joblib.load(config.MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(config.MODELS_DIR / "xgboost.joblib")
    return pre, model


def _interval(point: float, residual_std_chf: float = 240.0) -> tuple[float, float]:
    """Quick & defensible 80% interval based on observed residual std on the val set.

    Replace ``residual_std_chf`` once you have the true value from evaluation.
    """
    return point - 1.28 * residual_std_chf, point + 1.28 * residual_std_chf


def predict(
    structured: dict[str, Any],
    listing_text: str | None = None,
    photos: list[Image.Image] | None = None,
    explain_result: bool = True,
) -> PredictionResult:
    """Run the full pipeline.

    ``structured`` may have any subset of fields; missing ones are filled from
    the parsed listing (if provided) and CV features.
    """
    config.set_global_seed()
    pre, model = _load_artifacts()

    parsed = parse_listing(listing_text) if listing_text else None
    if parsed:
        for k, v in parsed.items():
            if k in {"area_m2", "rooms", "kreis"} and structured.get(k) is None and v is not None:
                structured[k] = v

    photo_feats = None
    if photos:
        photo_feats = extract_features(photos)
        structured.setdefault("condition_score", photo_feats.condition_score)
        structured.setdefault("has_balcony", photo_feats.has_balcony)
        structured.setdefault("has_view", photo_feats.has_view)
        structured.setdefault("kitchen_quality", photo_feats.kitchen_quality)

    df = pd.DataFrame([structured])
    df = add_engineered_columns(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    X = df[FEATURE_COLUMNS]

    pred = float(model.predict(pre.transform(X))[0])
    if config.LOG_TARGET:
        pred = float(np.expm1(pred))
    low, high = _interval(pred)

    explanation: Explanation | None = None
    if explain_result:
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pre.transform(X))[0]  # (n_features,)
            # Convert to CHF contributions on the linear scale (approximation in log-space)
            shap_chf = shap_values * (pred * 0.6)  # heuristic; replace with proper inverse
            feature_names = list(pre.get_feature_names_out())
            explanation = explain(feature_names, shap_chf, pred)
        except Exception as e:  # noqa: BLE001
            logger.warning("Explanation failed: %s", e)

    return PredictionResult(
        point_estimate_chf=pred,
        interval_low_chf=max(0.0, low),
        interval_high_chf=high,
        photo_features=photo_feats,
        parsed_listing=parsed,
        explanation=explanation,
        feature_table=structured,
    )

"""End-to-end inference pipeline orchestrating all three blocks.

Public API: ``predict(structured, listing_text, photos)`` → :class:`PredictionResult`.

This is what the Gradio app calls.

Calibration: When ``kreis`` and ``area_m2`` are both known, the raw model
prediction is blended with a Stadt-Zürich median rent reference:

    final = 0.6 * model + 0.4 * (median_chf_per_m2 × area_m2)

This compensates for distribution shift: the model was trained on
Switzerland-wide listings (n=664) with only ~27 Stadt-Zürich rows, so it
systematically underestimates premium districts. The Stadt-Zürich median
serves as a strong location prior. Both raw and reference values are
exposed in :class:`PredictionResult` for transparency.
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
    BINARY_LISTINGS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TEXT_DERIVED_BINARY,
    add_engineered_columns,
)
from immopilot.nlp.explainer import Explanation, explain
from immopilot.nlp.listing_parser import parse_listing

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_LISTINGS + TEXT_DERIVED_BINARY
)

# Hybrid calibration weight for blending model with Stadt-Zürich median.
# 0.0 = pure model · 1.0 = pure median × area · 0.4 = model dominates but is corrected.
CALIBRATION_WEIGHT = 0.4


@dataclass
class PredictionResult:
    point_estimate_chf: float  # FINAL hybrid estimate
    interval_low_chf: float
    interval_high_chf: float
    photo_features: PhotoFeatures | None
    parsed_listing: dict[str, Any] | None
    explanation: Explanation | None
    feature_table: dict[str, Any] = field(default_factory=dict)
    # New transparency fields:
    model_estimate_chf: float = 0.0  # raw XGBoost output
    reference_estimate_chf: float | None = None  # median × area, None if kreis/area unknown
    calibration_applied: bool = False
    confidence: str = "mittel"  # "hoch" / "mittel" / "niedrig"

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


@lru_cache(maxsize=1)
def _load_district_medians() -> dict[int, float]:
    """Load Stadt-Zürich median rent per kreis from MPE data."""
    path = config.DATA_DIR / "processed" / "zurich_districts.parquet"
    if not path.exists():
        logger.warning("zurich_districts.parquet not found, calibration disabled")
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["kreis"].astype(int), df["rent_median_chf_per_m2"].astype(float), strict=False))


def _interval(point: float, residual_std_chf: float = 240.0) -> tuple[float, float]:
    """Rough 80% interval using a fixed heuristic spread (~240 CHF, the rough scale of
    typical residuals). NOT a calibrated predictive interval; it is input-independent and
    only conveys ballpark uncertainty in the UI."""
    return point - 1.28 * residual_std_chf, point + 1.28 * residual_std_chf


def _confidence_level(
    model_chf: float,
    reference_chf: float | None,
    has_listing_text: bool,
    has_photos: bool,
) -> str:
    """Heuristic confidence based on signal availability and model/reference agreement."""
    signal_count = 1 + int(has_listing_text) + int(has_photos)  # structured + optional inputs
    if reference_chf is not None:
        delta_pct = abs(model_chf - reference_chf) / max(reference_chf, 1.0)
        agreement = "good" if delta_pct < 0.15 else ("ok" if delta_pct < 0.30 else "poor")
    else:
        agreement = "unknown"

    if signal_count >= 3 and agreement == "good":
        return "hoch"
    if signal_count == 1 and agreement == "poor":
        return "niedrig"
    return "mittel"


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

    model_pred = float(model.predict(pre.transform(X))[0])
    if config.LOG_TARGET:
        model_pred = float(np.expm1(model_pred))

    # ──────────── Hybrid calibration ────────────
    reference_pred: float | None = None
    final_pred = model_pred
    calibrated = False

    kreis = structured.get("kreis")
    area = structured.get("area_m2")
    if kreis is not None and area is not None:
        try:
            medians = _load_district_medians()
            kreis_int = int(kreis)
            if kreis_int in medians:
                reference_pred = medians[kreis_int] * float(area)
                final_pred = (1 - CALIBRATION_WEIGHT) * model_pred + CALIBRATION_WEIGHT * reference_pred
                calibrated = True
                logger.info(
                    "Calibration: model=%.0f, reference=%.0f (%.1f CHF/m² × %.0f m²), final=%.0f",
                    model_pred, reference_pred, medians[kreis_int], area, final_pred,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Calibration failed: %s", e)

    low, high = _interval(final_pred)

    explanation: Explanation | None = None
    if explain_result:
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            Xt_explain = pre.transform(X)
            shap_values = explainer.shap_values(Xt_explain)[0]  # (n_features,) in LOG space
            # SHAP values are additive in the model's training space (log1p rent):
            #   base + sum(shap) == predicted_log.
            # Convert each to a faithful marginal CHF effect via the expm1 inverse:
            #   c_i = expm1(full_log) - expm1(full_log - shap_i)
            # i.e. how many CHF the prediction drops when feature i's log-contribution
            # is removed. These are real per-feature CHF effects (not a heuristic scaling).
            # Because expm1 is non-linear, the c_i sum only approximately to the total;
            # this is documented in DOCUMENTATION.md (2A.6).
            base_log = float(np.asarray(explainer.expected_value).ravel()[0])
            full_log = base_log + float(shap_values.sum())
            shap_chf = np.expm1(full_log) - np.expm1(full_log - shap_values)
            feature_names = list(pre.get_feature_names_out())
            explanation = explain(feature_names, shap_chf, final_pred, feature_values=Xt_explain[0])
        except Exception as e:  # noqa: BLE001
            logger.warning("Explanation failed: %s", e)

    confidence = _confidence_level(
        model_chf=model_pred,
        reference_chf=reference_pred,
        has_listing_text=bool(listing_text),
        has_photos=bool(photos),
    )

    return PredictionResult(
        point_estimate_chf=final_pred,
        interval_low_chf=max(0.0, low),
        interval_high_chf=high,
        photo_features=photo_feats,
        parsed_listing=parsed,
        explanation=explanation,
        feature_table=structured,
        model_estimate_chf=model_pred,
        reference_estimate_chf=reference_pred,
        calibration_applied=calibrated,
        confidence=confidence,
    )

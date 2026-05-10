"""Explain a numeric prediction in natural language.

Pipeline:
1. Use SHAP TreeExplainer on the chosen model.
2. Extract the top-k positive and negative contributors.
3. Pass them, plus the prediction, to the LLM for a short, faithful explanation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from immopilot.nlp.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    text: str
    top_positive: list[tuple[str, float]]
    top_negative: list[tuple[str, float]]


SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent, der Mietpreis-Vorhersagen für Wohnungen in Zürich auf Deutsch erklärt.

Regeln:
- Antworte IMMER auf Deutsch (Hochdeutsch).
- 2 bis 4 kurze Sätze.
- Beziehe dich nur auf die genannten Faktoren — erfinde nichts.
- Übersetze technische Feature-Namen in alltägliche Sprache:
  * area_m2 = Wohnfläche
  * rooms = Zimmerzahl
  * lat / lon = geografische Lage
  * year_built = Baujahr
  * building_age = Alter des Gebäudes
  * area_per_room = Quadratmeter pro Zimmer
  * rent_median_chf_per_m2 = Mietpreis-Niveau im Kreis
  * has_balcony, has_view, has_elevator, has_garage, has_parking, has_fireplace = Ausstattung
  * is_zurich = Lage in der Stadt Zürich
  * is_new_building = Neubau
  * location_kreis_X = Stadtkreis X
  * size_bucket_X = Wohnungsgrösse-Kategorie
- Sei sachlich und neutral, kein Werbe-Ton.
- Erkläre zuerst die wichtigsten preistreibenden Faktoren, dann die preisdrückenden."""

USER_TEMPLATE = """Das Modell schätzt eine monatliche Miete von CHF {pred:.0f}.

Faktoren, die den Preis NACH OBEN treiben:
{positive}

Faktoren, die den Preis NACH UNTEN drücken:
{negative}

Erkläre dem Nutzer auf Deutsch in 2-4 Sätzen, warum das Modell zu diesem Preis kommt."""


def _format_factors(factors: list[tuple[str, float]]) -> str:
    if not factors:
        return "  (none)"
    return "\n".join(f"  - {name}: contribution CHF {val:+.0f}" for name, val in factors)


def explain(
    feature_names: list[str],
    shap_values: np.ndarray,  # shape (n_features,)
    prediction: float,
    top_k: int = 3,
    provider: str | None = None,
) -> Explanation:
    pairs = list(zip(feature_names, shap_values.tolist()))
    pairs.sort(key=lambda p: p[1])
    negatives = [(n, v) for n, v in pairs if v < 0][:top_k]
    positives = sorted([(n, v) for n, v in pairs if v > 0], key=lambda p: -p[1])[:top_k]

    client = LLMClient(provider=provider)  # type: ignore[arg-type]
    resp = client.complete(
        system=SYSTEM_PROMPT,
        user=USER_TEMPLATE.format(
            pred=prediction,
            positive=_format_factors(positives),
            negative=_format_factors(negatives),
        ),
        max_tokens=300,
        temperature=0.3,
    )
    return Explanation(text=resp.text.strip(), top_positive=positives, top_negative=negatives)

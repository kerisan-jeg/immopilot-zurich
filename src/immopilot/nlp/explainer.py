"""Explain a numeric prediction in natural language.

Pipeline:
1. Use SHAP TreeExplainer on the chosen model (contributions in log space).
2. The caller converts them to faithful CHF effects (see pipeline.py).
3. Select the *active* contributors — for one-hot features only the category that
   actually applies to this flat is kept (absence contributions like "not 'other'"
   are dropped, since they confuse non-expert readers).
4. Translate technical feature names into readable German labels.
5. Pass the top-k positive and negative factors to the LLM for a short explanation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from immopilot.nlp.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    text: str
    top_positive: list[tuple[str, float]]
    top_negative: list[tuple[str, float]]


# ── Readable labels for one-hot / engineered feature columns ──────────────────
KREIS_NAMES = {
    "1": "Altstadt", "2": "Enge/Wollishofen", "3": "Wiedikon", "4": "Aussersihl",
    "5": "Industriequartier", "6": "Unterstrass/Oberstrass", "7": "Zürichberg",
    "8": "Riesbach/Seefeld", "9": "Altstetten", "10": "Höngg/Wipkingen",
    "11": "Zürich-Nord", "12": "Schwamendingen",
}
SIZE_NAMES = {"xs": "sehr klein", "s": "klein", "m": "mittel", "l": "gross", "xl": "sehr gross"}


def _humanize(name: str) -> str:
    """Turn a transformed feature name into a readable German label."""
    if name.startswith("location_kreis_"):
        k = name[len("location_kreis_"):]
        if k == "other":
            return "Lage ausserhalb der erfassten Stadtkreise"
        return f"Stadtkreis {k} ({KREIS_NAMES.get(k, k)})"
    if name.startswith("size_bucket_"):
        b = name[len("size_bucket_"):]
        return f"Wohnungsgrösse {SIZE_NAMES.get(b, b)}"
    return name


def _is_onehot(name: str) -> bool:
    return name.startswith("location_kreis_") or name.startswith("size_bucket_")


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
        return "  (keine)"
    return "\n".join(f"  - {name}: Beitrag CHF {val:+.0f}" for name, val in factors)


def explain(
    feature_names: list[str],
    shap_values: np.ndarray,  # CHF contributions, shape (n_features,)
    prediction: float,
    feature_values: np.ndarray | None = None,  # transformed feature values, same shape
    top_k: int = 3,
    provider: str | None = None,
) -> Explanation:
    """Build a faithful, readable explanation.

    If ``feature_values`` is given, one-hot columns that are inactive for this
    instance (value ≈ 0) are dropped — their SHAP "absence" contribution would
    otherwise read as e.g. "+501 CHF because the flat is NOT 'other'", which is
    technically valid but confusing. Active categories are relabelled readably.
    """
    n = len(feature_names)
    if feature_values is None:
        feature_values = np.ones(n)  # fall back: keep everything

    pairs: list[tuple[str, float]] = []
    for name, val, x in zip(feature_names, shap_values.tolist(), np.asarray(feature_values).tolist(), strict=False):
        if _is_onehot(name) and abs(x) < 0.5:
            continue  # inactive category for this flat → skip absence contribution
        pairs.append((_humanize(name), float(val)))

    positives = sorted([(nm, v) for nm, v in pairs if v > 0], key=lambda p: -p[1])[:top_k]
    negatives = sorted([(nm, v) for nm, v in pairs if v < 0], key=lambda p: p[1])[:top_k]

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

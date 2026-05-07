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


SYSTEM_PROMPT = """You explain rent predictions for Zurich apartments.
- Speak in the user's language.
- Use 2–4 short sentences.
- Only reference the factors provided. Do not invent factors.
- Be neutral and helpful, not promotional."""

USER_TEMPLATE = """The model predicts a monthly rent of CHF {pred:.0f}.

Factors that PUSHED THE PRICE UP:
{positive}

Factors that PULLED THE PRICE DOWN:
{negative}

Explain to the user, in plain language, why the model arrived at this price."""


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

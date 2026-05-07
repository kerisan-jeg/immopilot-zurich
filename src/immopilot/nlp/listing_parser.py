"""Parse a free-text apartment listing into structured features.

Uses an LLM with a JSON-schema-shaped prompt. We deliberately avoid provider-
specific function calling to keep the wrapper portable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from immopilot.nlp.llm_client import LLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You parse German/English apartment listings into JSON.
Output ONLY a JSON object with these keys (use null when unknown):
{
  "area_m2": number | null,
  "rooms": number | null,
  "kreis": int | null,             // Zurich district 1..12
  "rent_chf": number | null,       // monthly net rent in CHF
  "is_luxurious": boolean,
  "is_furnished": boolean,
  "is_temporary": boolean,
  "has_balcony": boolean,
  "has_view": boolean,
  "summary": string                // one-sentence neutral summary
}
No prose, no markdown fences — just the JSON object."""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def parse_listing(listing_text: str, provider: str | None = None) -> dict[str, Any]:
    client = LLMClient(provider=provider)  # type: ignore[arg-type]
    resp = client.complete(
        system=SYSTEM_PROMPT,
        user=listing_text.strip(),
        max_tokens=512,
        temperature=0.0,
    )
    try:
        return _extract_json(resp.text)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON; retrying with stricter instruction.")
        resp2 = client.complete(
            system=SYSTEM_PROMPT + "\nIMPORTANT: return ONLY the JSON, nothing else.",
            user=listing_text.strip(),
            max_tokens=512,
            temperature=0.0,
        )
        return _extract_json(resp2.text)

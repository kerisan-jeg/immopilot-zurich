"""Provider-agnostic LLM wrapper.

Why: the assignment rewards comparing approaches. By abstracting providers
behind one interface we can A/B test Anthropic vs OpenAI in the NLP block
without changing call sites.

Robustness: if no API key is configured (e.g. on a fresh Hugging Face Space
before the secret is set) or the API call fails, ``complete`` returns a clear
fallback message instead of raising, so the app degrades gracefully rather than
crashing the request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from immopilot import config

logger = logging.getLogger(__name__)

ProviderName = Literal["anthropic", "openai"]

_NO_KEY_MSG = (
    "_(LLM nicht konfiguriert: Es ist kein API-Schlüssel hinterlegt. "
    "Die Vorhersage funktioniert, aber Erklärungen und Q&A benötigen einen "
    "ANTHROPIC_API_KEY in den Space-Secrets.)_"
)


@dataclass
class LLMResponse:
    text: str
    provider: ProviderName
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient:
    """Thin façade over Anthropic and OpenAI."""

    def __init__(self, provider: ProviderName | None = None, model: str | None = None):
        self.provider: ProviderName = provider or config.LLM_PROVIDER  # type: ignore[assignment]
        if self.provider == "anthropic":
            self.model = model or config.ANTHROPIC_MODEL
        elif self.provider == "openai":
            self.model = model or config.OPENAI_MODEL
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _has_key(self) -> bool:
        if self.provider == "anthropic":
            return bool(config.ANTHROPIC_API_KEY)
        return bool(config.OPENAI_API_KEY)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        if not self._has_key():
            logger.warning("No API key for provider %s — returning fallback message.", self.provider)
            return LLMResponse(text=_NO_KEY_MSG, provider=self.provider, model=self.model)
        try:
            if self.provider == "anthropic":
                return self._complete_anthropic(system, user, max_tokens, temperature)
            return self._complete_openai(system, user, max_tokens, temperature)
        except Exception as e:  # noqa: BLE001
            logger.exception("LLM call failed: %s", e)
            return LLMResponse(
                text=f"_(LLM-Aufruf fehlgeschlagen: {type(e).__name__}. Bitte später erneut versuchen.)_",
                provider=self.provider,
                model=self.model,
            )

    # --- providers ---------------------------------------------------------
    def _complete_anthropic(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        import anthropic

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        return LLMResponse(
            text=text,
            provider="anthropic",
            model=self.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )

    def _complete_openai(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            provider="openai",
            model=self.model,
            input_tokens=completion.usage.prompt_tokens if completion.usage else None,
            output_tokens=completion.usage.completion_tokens if completion.usage else None,
        )

"""Provider-agnostic LLM wrapper.

Why: the assignment rewards comparing approaches. By abstracting providers
behind one interface we can A/B test Anthropic vs OpenAI in the NLP block
without changing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from immopilot import config

ProviderName = Literal["anthropic", "openai"]


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

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        if self.provider == "anthropic":
            return self._complete_anthropic(system, user, max_tokens, temperature)
        return self._complete_openai(system, user, max_tokens, temperature)

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

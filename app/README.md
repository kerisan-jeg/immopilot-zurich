---
title: ImmoPilot Zürich
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.1.0
app_file: app/app.py
pinned: true
license: mit
short_description: Multimodal apartment-rent assistant for Zurich
---

# ImmoPilot Zürich

Multimodaler Mietpreis-Assistent für Zürich. Predicts a fair market rent from
listing text, photos and structured input, explains *why*, and answers
neighborhood questions via RAG.

Source code & full documentation: <https://github.com/USER/immopilot-zurich>

## Required secrets

In the Space's **Settings → Variables and secrets**:

- `ANTHROPIC_API_KEY` — for the LLM (default provider)
- `OPENAI_API_KEY` *(optional)* — fallback provider

## What's loaded at runtime

The Space pulls trained model artifacts from the HF Hub model repo
`USER/immopilot-models` on cold start. Training does **not** happen in the
Space — it runs locally via `make train` in the source repo.

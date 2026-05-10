"""Gradio app for ImmoPilot Zurich. Entry point for the Hugging Face Space.

Three tabs:
  1. Bewertung   — structured input + listing text → predicted rent + explanation
  2. Foto-Analyse — upload photos → CV features
  3. Q&A          — RAG chat about Zurich neighborhoods
"""

import logging

import gradio as gr

from immopilot import config
from immopilot.inference.pipeline import predict
from immopilot.nlp.rag_pipeline import answer as rag_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ───────────────────── Tab 1: Bewertung ─────────────────────
def bewerten(area_m2, rooms, kreis, listing_text, photo1, photo2, photo3):
    structured = {
        "area_m2": area_m2 if area_m2 else None,
        "rooms": rooms if rooms else None,
        "kreis": int(kreis) if kreis else None,
    }
    photos = [p for p in (photo1, photo2, photo3) if p is not None]
    result = predict(
        structured=structured,
        listing_text=listing_text or None,
        photos=photos or None,
    )

    headline = (
        f"### Geschätzte Miete: **CHF {result.point_estimate_chf:,.0f} / Monat**\n\n"
        f"80%-Intervall: CHF {result.interval_low_chf:,.0f} – {result.interval_high_chf:,.0f}"
    )
    explanation_md = result.explanation.text if result.explanation else "_(keine Erklärung)_"

    cv_md = "_(keine Fotos)_"
    if result.photo_features:
        f = result.photo_features
        cv_md = (
            f"- **Zustand**: {f.condition_score:.2f} (1 = modern)\n"
            f"- **Balkon erkannt**: {'ja' if f.has_balcony else 'nein'}\n"
            f"- **Aussicht erkannt**: {'ja' if f.has_view else 'nein'}\n"
            f"- **Küchenqualität**: {f.kitchen_quality:.2f}"
        )
    parsed_md = "_(kein Inserat-Text)_"
    if result.parsed_listing:
        lines = [f"- **{k}**: {v}" for k, v in result.parsed_listing.items() if v is not None]
        parsed_md = "\n".join(lines) or "_(nichts extrahiert)_"

    return headline, explanation_md, cv_md, parsed_md


# ───────────────────── Tab 2: Foto-Analyse ─────────────────────
def foto_analyse(p1, p2, p3):
    from immopilot.cv.zero_shot_clip import extract_features as cv_extract

    real_photos = [p for p in (p1, p2, p3) if p is not None]
    if not real_photos:
        return "Bitte mindestens ein Foto hochladen."
    f = cv_extract(real_photos)
    return (
        f"### Erkannte Merkmale aus {len(real_photos)} Foto(s)\n\n"
        f"- **Zustand**: {f.condition_score:.2f} (1 = modern, 0 = renovierungsbedürftig)\n"
        f"- **Balkon**: {'ja' if f.has_balcony else 'nein'}\n"
        f"- **Aussicht**: {'ja' if f.has_view else 'nein'}\n"
        f"- **Küchenqualität**: {f.kitchen_quality:.2f}\n\n"
        f"_Diese Merkmale fliessen automatisch in die Mietpreis-Schätzung ein._"
    )


# ───────────────────── Tab 3: Q&A ─────────────────────
def qa(question):
    if not question.strip():
        return "Bitte eine Frage stellen.", ""
    a = rag_answer(question)
    sources_md = "\n".join(
        f"{i+1}. **{c.title}** — {c.url or c.source}  _(score {c.score:.2f})_"
        for i, c in enumerate(a.chunks)
    )
    return a.answer, sources_md


# ───────────────────── UI ─────────────────────
with gr.Blocks(title="ImmoPilot Zürich", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🏠 ImmoPilot Zürich
Multimodaler Mietpreis-Assistent. Nicht zur Rechtsberatung. Fotos werden nicht gespeichert.
"""
    )

    with gr.Tab("Bewertung"):
        with gr.Row():
            with gr.Column():
                area = gr.Number(label="Fläche (m²)", value=None)
                rooms = gr.Number(label="Zimmer", value=None)
                kreis = gr.Slider(1, 12, step=1, label="Kreis", value=6)
                listing_text = gr.Textbox(
                    label="Inserat-Text (optional)",
                    placeholder="Hier den Text eines Inserats einfügen…",
                    lines=6,
                )
                photo1 = gr.Image(label="Foto 1 (optional)", type="pil")
                photo2 = gr.Image(label="Foto 2 (optional)", type="pil")
                photo3 = gr.Image(label="Foto 3 (optional)", type="pil")
                btn = gr.Button("Bewerten", variant="primary")
            with gr.Column():
                headline = gr.Markdown()
                explanation = gr.Markdown(label="Erklärung")
                cv_block = gr.Markdown(label="Aus Fotos extrahiert")
                parsed_block = gr.Markdown(label="Aus Inserat extrahiert")
        btn.click(
            bewerten,
            inputs=[area, rooms, kreis, listing_text, photo1, photo2, photo3],
            outputs=[headline, explanation, cv_block, parsed_block],
        )

    with gr.Tab("Foto-Analyse"):
        gr.Markdown("Lade Fotos hoch — wir zeigen, welche Merkmale erkannt werden.")
        with gr.Row():
            p1 = gr.Image(type="pil")
            p2 = gr.Image(type="pil")
            p3 = gr.Image(type="pil")
        out = gr.Markdown()
        gr.Button("Analysieren").click(foto_analyse, inputs=[p1, p2, p3], outputs=out)

    with gr.Tab("Q&A Quartiere"):
        gr.Markdown("Frag mich etwas über die Zürcher Quartiere — Antworten basieren auf offiziellen Quellen.")
        question = gr.Textbox(label="Frage", lines=2)
        ans = gr.Markdown()
        sources = gr.Markdown(label="Quellen")
        gr.Button("Fragen").click(qa, inputs=question, outputs=[ans, sources])

    gr.Markdown(
        f"""
---
*Provider: {config.LLM_PROVIDER} · Modell: {config.ANTHROPIC_MODEL if config.LLM_PROVIDER=='anthropic' else config.OPENAI_MODEL}*
"""
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        show_api=False,  # workaround for gradio 5.1 schema-parsing bug on Windows
    )

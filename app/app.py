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
def bewerten(
    area_m2,
    rooms,
    kreis,
    year_built,
    has_balcony,
    has_view,
    has_elevator,
    has_parking,
    is_new_building,
    listing_text,
    photo1,
    photo2,
    photo3,
):
    structured = {
        "area_m2": area_m2 if area_m2 else None,
        "rooms": rooms if rooms else None,
        "kreis": int(kreis) if kreis else None,
        "year_built": int(year_built) if year_built else None,
        "has_balcony": int(bool(has_balcony)),
        "has_view": int(bool(has_view)),
        "has_elevator": int(bool(has_elevator)),
        "has_parking": int(bool(has_parking)),
        "is_new_building": int(bool(is_new_building)),
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


# ───────────────────── Beispiele für Quick-Test ─────────────────────
EXAMPLES = [
    # area, rooms, kreis, year, balc, view, elev, park, new, listing
    [85, 3.5, 8, 2010, True, True, True, False, False, ""],   # Seefeld 3.5er mit Seeblick
    [55, 2.5, 4, 1995, True, False, False, False, False, ""], # Kreis 4 typische Wohnung
    [120, 4.5, 7, 2020, True, True, True, True, True, ""],    # Zürichberg Neubau
    [42, 1.5, 1, 1900, False, False, False, False, False, ""], # Altstadt 1.5er
    [70, 3.0, 12, 1970, True, False, False, True, False, ""], # Schwamendingen Standard
]


# ───────────────────── UI ─────────────────────
with gr.Blocks(title="ImmoPilot Zürich", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🏠 ImmoPilot Zürich
**Multimodaler Mietpreis-Assistent** — kombiniert ML, Computer Vision und RAG.
Nicht zur Rechtsberatung. Fotos werden nicht gespeichert.
"""
    )

    with gr.Tab("📊 Bewertung"):
        gr.Markdown("Gib die Wohnungsdaten ein. Optional: Inserat-Text und/oder Fotos hochladen für genauere Schätzung.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🏢 Grunddaten")
                with gr.Row():
                    area = gr.Number(label="Fläche (m²)", value=85, minimum=15, maximum=400)
                    rooms = gr.Number(label="Zimmer", value=3.5, minimum=1, maximum=10, step=0.5)
                kreis = gr.Slider(1, 12, step=1, label="Stadtkreis", value=6)
                year_built = gr.Number(label="Baujahr", value=1990, minimum=1800, maximum=2026)

                gr.Markdown("### ✨ Ausstattung")
                with gr.Row():
                    has_balcony = gr.Checkbox(label="Balkon", value=False)
                    has_view = gr.Checkbox(label="Aussicht", value=False)
                with gr.Row():
                    has_elevator = gr.Checkbox(label="Lift", value=False)
                    has_parking = gr.Checkbox(label="Parkplatz", value=False)
                is_new_building = gr.Checkbox(label="Neubau (< 5 Jahre)", value=False)

                with gr.Accordion("📝 Inserat-Text (optional)", open=False):
                    listing_text = gr.Textbox(
                        label="",
                        placeholder="z.B. '3.5 Zimmer Wohnung in Wiedikon, 78m², Altbau mit Balkon...'",
                        lines=4,
                    )

                with gr.Accordion("📷 Fotos hochladen (optional)", open=False):
                    photo1 = gr.Image(label="Foto 1", type="pil")
                    photo2 = gr.Image(label="Foto 2", type="pil")
                    photo3 = gr.Image(label="Foto 3", type="pil")

                btn = gr.Button("🔍 Bewerten", variant="primary", size="lg")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[area, rooms, kreis, year_built, has_balcony, has_view, has_elevator, has_parking, is_new_building, listing_text],
                    label="💡 Beispiel-Wohnungen",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 💰 Vorhersage")
                headline = gr.Markdown()
                gr.Markdown("### 📋 Erklärung")
                explanation = gr.Markdown()
                gr.Markdown("### 📷 Aus Fotos extrahiert")
                cv_block = gr.Markdown()
                gr.Markdown("### 📝 Aus Inserat extrahiert")
                parsed_block = gr.Markdown()

        btn.click(
            bewerten,
            inputs=[area, rooms, kreis, year_built, has_balcony, has_view, has_elevator, has_parking, is_new_building, listing_text, photo1, photo2, photo3],
            outputs=[headline, explanation, cv_block, parsed_block],
        )

    with gr.Tab("📷 Foto-Analyse"):
        gr.Markdown("Lade Fotos hoch — wir zeigen, welche Merkmale erkannt werden (CLIP zero-shot).")
        with gr.Row():
            p1 = gr.Image(type="pil", label="Foto 1")
            p2 = gr.Image(type="pil", label="Foto 2")
            p3 = gr.Image(type="pil", label="Foto 3")
        out = gr.Markdown()
        gr.Button("🔍 Analysieren", variant="primary").click(foto_analyse, inputs=[p1, p2, p3], outputs=out)

    with gr.Tab("💬 Q&A Quartiere"):
        gr.Markdown(
            "Frag mich etwas über die Zürcher Quartiere — Antworten basieren auf offiziellen Quellen "
            "(Stadt Zürich, Wikipedia). Beispiele: *'Welcher Kreis ist am günstigsten?'* · "
            "*'Was ist die Goldküste?'* · *'Mietpreise im Seefeld?'*"
        )
        question = gr.Textbox(label="Frage", lines=2, placeholder="Was ist Kreis 8?")
        gr.Button("🔍 Fragen", variant="primary").click(qa, inputs=question, outputs=None)
        ans = gr.Markdown(label="Antwort")
        sources = gr.Markdown(label="Quellen")
        gr.Examples(
            examples=[
                "Welcher Kreis ist am günstigsten?",
                "Was ist die Goldküste?",
                "Mietpreise im Seefeld?",
                "Was ist Kreis 5 / Industriequartier?",
                "Welche Quartiere gehören zu Zürich Nord?",
            ],
            inputs=question,
            label="💡 Beispiel-Fragen",
        )
        question.submit(qa, inputs=question, outputs=[ans, sources])
        # Re-bind button to actual function with outputs (gradio quirk)
        ans_btn = gr.Button("🔍 Antworten holen", variant="secondary")
        ans_btn.click(qa, inputs=question, outputs=[ans, sources])

    gr.Markdown(
        f"""
---
**Stack:** XGBoost (rent prediction) · CLIP zero-shot (photo features) · FAISS + sentence-transformers (RAG) · {config.LLM_PROVIDER.title()} {config.ANTHROPIC_MODEL if config.LLM_PROVIDER=='anthropic' else config.OPENAI_MODEL} (explanation + Q&A)

*ZHAW SML — AI Applications · Spring 2026 · [GitHub Repository](https://github.com/kerisan-jeg/immopilot-zurich)*
"""
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )

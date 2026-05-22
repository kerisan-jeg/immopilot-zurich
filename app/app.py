"""Gradio app for ImmoPilot Zürich.

Three tabs:
  1. Bewertung    — structured input + listing text → predicted rent + explanation
  2. Foto-Analyse — upload photos → CV features
  3. Q&A           — RAG chat about Zurich neighborhoods

Calibration: predictions blend XGBoost output with a Stadt-Zürich median
rent reference. Both raw and reference values are shown for transparency.

UI: custom "Swiss editorial" theme — deep navy + brass accent, Fraunces display
font + Inter-free body, card surfaces with depth. Light and dark via CSS variables.
Logic is unchanged from the functional version; only presentation differs.
"""

import logging

import gradio as gr

from immopilot import config
from immopilot.inference.pipeline import predict
from immopilot.nlp.rag_pipeline import answer as rag_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


CONFIDENCE_BADGE = {
    "hoch": '<span class="ip-badge ip-badge-high">Hoch</span>',
    "mittel": '<span class="ip-badge ip-badge-mid">Mittel</span>',
    "niedrig": '<span class="ip-badge ip-badge-low">Niedrig</span>',
}


# ───────────────────────── Tab 1: Bewertung ─────────────────────────
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

    # Headline with hybrid + confidence — rendered as a hero result card (HTML)
    badge = CONFIDENCE_BADGE.get(result.confidence, CONFIDENCE_BADGE["mittel"])
    headline = (
        '<div class="ip-result-card">'
        '<div class="ip-result-label">Geschätzte Nettomiete</div>'
        f'<div class="ip-result-value">CHF {result.point_estimate_chf:,.0f}'
        '<span class="ip-result-unit"> / Monat</span></div>'
        '<div class="ip-result-meta">'
        f'<span>80%-Intervall&nbsp;&nbsp;CHF {result.interval_low_chf:,.0f} – {result.interval_high_chf:,.0f}</span>'
        f'<span class="ip-result-conf">Konfidenz {badge}</span>'
        '</div>'
        '</div>'
    )

    # Calibration breakdown (transparent)
    if result.calibration_applied and result.reference_estimate_chf is not None:
        delta_pct = (result.model_estimate_chf - result.reference_estimate_chf) / result.reference_estimate_chf * 100
        breakdown = (
            f"**Modell-Schätzung (XGBoost):** CHF {result.model_estimate_chf:,.0f}\n\n"
            f"**Median-Referenz (Stadt Zürich Kreis {structured['kreis']}):** CHF {result.reference_estimate_chf:,.0f}\n\n"
            f"_(Modell weicht um {delta_pct:+.0f}% vom Median ab — Hybrid-Schätzung mit 60/40-Gewichtung angewendet)_"
        )
    else:
        breakdown = f"**Modell-Schätzung:** CHF {result.model_estimate_chf:,.0f}  _(keine Median-Kalibrierung möglich — Kreis oder Fläche fehlt)_"

    explanation_md = result.explanation.text if result.explanation else "_(keine Erklärung)_"

    cv_md = "_(keine Fotos hochgeladen)_"
    if result.photo_features:
        f = result.photo_features
        cv_md = (
            f"- **Zustand**: {f.condition_score:.2f} _(1 = modern, 0 = renovierungsbedürftig)_\n"
            f"- **Balkon erkannt**: {'ja' if f.has_balcony else 'nein'}\n"
            f"- **Aussicht erkannt**: {'ja' if f.has_view else 'nein'}\n"
            f"- **Küchenqualität**: {f.kitchen_quality:.2f}"
        )
    parsed_md = "_(kein Inserat-Text eingegeben)_"
    if result.parsed_listing:
        lines = [f"- **{k}**: {v}" for k, v in result.parsed_listing.items() if v is not None]
        parsed_md = "\n".join(lines) or "_(nichts extrahiert)_"

    return headline, breakdown, explanation_md, cv_md, parsed_md


# ───────────────────────── Tab 2: Foto-Analyse ─────────────────────────
def foto_analyse(p1, p2, p3):
    from immopilot.cv.zero_shot_clip import extract_features as cv_extract

    real_photos = [p for p in (p1, p2, p3) if p is not None]
    if not real_photos:
        return "Bitte mindestens ein Foto hochladen."
    f = cv_extract(real_photos)
    return (
        f"### Erkannte Merkmale aus {len(real_photos)} Foto(s)\n\n"
        f"- **Zustand**: {f.condition_score:.2f} _(1 = modern, 0 = renovierungsbedürftig)_\n"
        f"- **Balkon**: {'ja' if f.has_balcony else 'nein'}\n"
        f"- **Aussicht**: {'ja' if f.has_view else 'nein'}\n"
        f"- **Küchenqualität**: {f.kitchen_quality:.2f}\n\n"
        f"_Diese Merkmale fliessen automatisch in die Mietpreis-Schätzung im Tab 'Bewertung' ein._"
    )


# ───────────────────────── Tab 3: Q&A ─────────────────────────
def qa(question):
    if not question.strip():
        return "_Bitte eine Frage stellen._", ""
    a = rag_answer(question)
    sources_md = "\n".join(
        f"{i+1}. **{c.title}** — [{c.url or c.source}]({c.url or '#'})  _(Relevanz: {c.score:.2f})_"
        for i, c in enumerate(a.chunks)
    )
    return a.answer, "**Quellen:**\n\n" + sources_md


# ───────────────────────── Realistische Beispiele ─────────────────────────
EXAMPLES = [
    # area, rooms, kreis, year, balc, view, elev, park, new, listing
    [85, 3.5, 8, 2010, True, True, True, False, False, ""],   # Seefeld 3.5er mit Seeblick
    [55, 2.5, 4, 1995, True, False, False, False, False, ""],
    [120, 4.5, 7, 2020, True, True, True, True, True, ""],    # Zürichberg Neubau
    [42, 1.5, 1, 1900, False, False, False, False, False, ""],
    [70, 3.0, 12, 1970, True, False, False, True, False, ""], # Schwamendingen Standard
]

EXAMPLE_LISTINGS = [
    [
        "3.5-Zimmer-Wohnung am Seefeld\n\nHelle, moderne 3.5-Zimmer-Wohnung mit 85m² in begehrter Seefeld-Lage (Kreis 8). "
        "Bezugsbereit ab sofort. Grosser Süd-Balkon mit teilweiser Seesicht, hochwertige Einbauküche, Parkettböden. "
        "Lift, Kellerabteil, Velokeller. ÖV (Tram 2/4) in 2 Min Fussdistanz. Baujahr 2010. Miete: 3'200 CHF inkl. NK."
    ],
    [
        "Charmante Altbauwohnung Wiedikon\n\n2.5 Zimmer, 65m², Kreis 3 (Alt-Wiedikon nähe Idaplatz). Schöner Stuck, "
        "hohe Decken, renoviertes Badezimmer. Kein Balkon. 2. OG ohne Lift. Baujahr 1905. Sehr ruhige Wohnstrasse. "
        "Miete: 1'850 CHF + 180 NK."
    ],
    [
        "Neubau-Familienwohnung Altstetten\n\n4.5 Zi., 105m², Kreis 9. Erstbezug 2024. Minergie-zertifiziert, "
        "Bodenheizung, Geschirrspüler, Steamer. 2 Nasszellen. Loggia 12m². Tiefgaragenplatz inklusive. "
        "Bahnhof Altstetten 5 Gehminuten. Miete: 2'950 CHF brutto."
    ],
]

EXAMPLE_QUESTIONS = [
    "Welcher Kreis ist am günstigsten?",
    "Was ist die Goldküste?",
    "Welche Tramlinien fahren nach Schwamendingen?",
    "Mietpreise im Seefeld?",
    "Was ist Kreis 5 / Industriequartier?",
    "Wo ist der Sitz von Google in Zürich?",
    "Welche Quartiere gehören zu Zürich Nord?",
]


# ───────────────────────── Theme + CSS ─────────────────────────
# A restrained Gradio theme; the real character comes from the CSS below.
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_radius="14px",
    button_large_radius="12px",
    button_primary_background_fill="linear-gradient(135deg, #1e2a4a 0%, #2d3e6b 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #25325a 0%, #364a80 100%)",
    button_primary_text_color="#f5e9d4",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

/* ---------- Design tokens ---------- */
:root {
    --ip-navy: #1e2a4a;
    --ip-navy-2: #2d3e6b;
    --ip-brass: #b08442;
    --ip-brass-soft: #c9a063;
    --ip-ink: #1a1f2e;
    --ip-paper: #fbfaf7;
    --ip-card: #ffffff;
    --ip-border: rgba(30, 42, 74, 0.10);
    --ip-shadow: 0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06);
    --ip-shadow-lg: 0 2px 4px rgba(16,24,40,.04), 0 18px 48px rgba(16,24,40,.12);
}
.dark {
    --ip-navy: #b9c6e8;
    --ip-navy-2: #8ea3d8;
    --ip-brass: #d4a85f;
    --ip-brass-soft: #e0bd82;
    --ip-ink: #e8ecf5;
    --ip-paper: #0c1018;
    --ip-card: #141a26;
    --ip-border: rgba(255,255,255,0.08);
    --ip-shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35);
    --ip-shadow-lg: 0 2px 4px rgba(0,0,0,.3), 0 18px 48px rgba(0,0,0,.5);
}

/* ---------- Global ---------- */
.gradio-container,
.gradio-container > .main,
.gradio-container .contain,
div.gradio-container {
    max-width: 1180px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 28px !important;
    padding-right: 28px !important;
    box-sizing: border-box !important;
}
.gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
}
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    font-family: 'Fraunces', Georgia, serif !important;
    letter-spacing: -0.01em;
}

/* ---------- Hero header ---------- */
#ip-hero {
    position: relative;
    border-radius: 20px;
    padding: 48px 44px 40px;
    margin-top: 12px;
    margin-bottom: 16px;
    background:
        radial-gradient(1200px 300px at 85% -20%, rgba(176,132,66,0.18), transparent 60%),
        linear-gradient(135deg, #1a2440 0%, #233156 55%, #2d3e6b 100%);
    box-shadow: var(--ip-shadow-lg);
    overflow: hidden;
}
#ip-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 22px 22px;
    pointer-events: none;
}
#ip-hero h1 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 2.7rem !important;
    line-height: 1.05 !important;
    font-weight: 700 !important;
    color: #fdfcfa !important;
    margin: 0 0 6px !important;
}
#ip-hero .ip-hero-accent { color: var(--ip-brass-soft); }
#ip-hero .ip-hero-sub {
    font-size: 1.02rem;
    color: rgba(245,243,238,0.82);
    font-weight: 500;
    margin-bottom: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 0.78rem;
}
#ip-hero .ip-hero-lede {
    font-size: 1.05rem;
    color: rgba(245,243,238,0.92);
    max-width: 640px;
    line-height: 1.5;
    margin-top: 12px;
}
#ip-hero .ip-hero-tags {
    margin-top: 20px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
#ip-hero .ip-tag {
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 5px 12px;
    border-radius: 999px;
    color: #f3ede0;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
    backdrop-filter: blur(4px);
}

/* ---------- Disclaimer ---------- */
#ip-disclaimer {
    border-left: 3px solid var(--ip-brass);
    background: rgba(176,132,66,0.07);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 14px 0 4px;
    font-size: 0.86rem;
    line-height: 1.5;
}
#ip-disclaimer strong { color: var(--ip-brass); }

/* ---------- Section headings (replaces emoji md headers) ---------- */
.ip-section {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 1.18rem !important;
    font-weight: 600 !important;
    color: var(--ip-navy) !important;
    margin: 4px 0 2px !important;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--ip-border);
}

/* ---------- Cards / blocks ---------- */
.ip-card {
    background: var(--ip-card) !important;
    border: 1px solid var(--ip-border) !important;
    border-radius: 16px !important;
    box-shadow: var(--ip-shadow);
    padding: 22px 22px 18px !important;
}

/* ---------- Result hero card (the "wow" moment) ---------- */
/* Always dark surface + light text in both themes for reliable contrast. */
.ip-result-card {
    background: linear-gradient(150deg, #1e2a4a 0%, #2d3e6b 100%);
    border-radius: 18px;
    padding: 26px 28px;
    box-shadow: var(--ip-shadow-lg);
    color: #fff;
    position: relative;
    overflow: hidden;
    animation: ip-rise .5s cubic-bezier(.2,.7,.2,1);
}
.ip-result-card::before {
    content: "";
    position: absolute; right: -40px; top: -40px;
    width: 180px; height: 180px; border-radius: 50%;
    background: radial-gradient(circle, rgba(201,160,99,0.35), transparent 70%);
}
.ip-result-label {
    font-size: 0.74rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: #e6c891; font-weight: 700; margin-bottom: 6px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.25);
}
.ip-result-value {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.6rem; font-weight: 700; line-height: 1; color: #fdfcfa;
}
.ip-result-unit { font-size: 1.1rem; font-weight: 500; color: rgba(253,252,250,0.7); }
.ip-result-meta {
    display: flex; flex-wrap: wrap; gap: 8px 22px; align-items: center;
    margin-top: 14px; font-size: 0.9rem; color: rgba(253,252,250,0.9);
}
.ip-result-conf { display: inline-flex; align-items: center; gap: 7px; }
.ip-badge {
    font-size: 0.74rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;
    letter-spacing: 0.02em;
}
.ip-badge-high { background: #d1f4e0; color: #0f7a47; }
.ip-badge-mid  { background: #fdecc8; color: #9a6b16; }
.ip-badge-low  { background: #fbd7d5; color: #b3261e; }
@keyframes ip-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

/* ---------- Tabs ---------- */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.96rem !important;
    letter-spacing: 0.01em;
}
.tab-nav button.selected {
    color: var(--ip-navy) !important;
    border-bottom-color: var(--ip-brass) !important;
}

/* ---------- Primary buttons ---------- */
button.primary, .primary button {
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 14px rgba(30,42,74,0.22) !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}
button.primary:hover, .primary button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(30,42,74,0.30) !important;
}

/* ---------- Footer ---------- */
#ip-footer {
    margin-top: 18px; padding-top: 18px;
    border-top: 1px solid var(--ip-border);
    font-size: 0.82rem; line-height: 1.6; color: var(--ip-ink); opacity: 0.78;
}
#ip-footer a { color: var(--ip-brass); font-weight: 600; }
"""


# ───────────────────────── UI ─────────────────────────
with gr.Blocks(title="ImmoPilot Zürich", theme=THEME, css=CSS) as demo:
    gr.HTML(
        """
<div id="ip-hero">
  <div class="ip-hero-sub">ZHAW SML · AI Applications</div>
  <h1>ImmoPilot <span class="ip-hero-accent">Zürich</span></h1>
  <div class="ip-hero-lede">
    Der multimodale Mietpreis-Assistent für Zürcher Wohnungen — schätzt die Nettomiete,
    liest Inserate und Fotos aus und beantwortet Fragen zu den Stadtkreisen.
  </div>
  <div class="ip-hero-tags">
    <span class="ip-tag">Machine Learning</span>
    <span class="ip-tag">Computer Vision</span>
    <span class="ip-tag">RAG / NLP</span>
  </div>
</div>
"""
    )
    gr.HTML(
        """
<div id="ip-disclaimer">
  <strong>Disclaimer</strong> · Studienprojekt (ZHAW SML, AI Applications). Keine Rechtsberatung,
  keine offizielle Bewertung. Prognose basiert auf einem ML-Modell mit begrenztem Trainingsdatensatz
  (siehe Limitations in der Dokumentation). Fotos werden ausschliesslich im Arbeitsspeicher verarbeitet
  und nicht gespeichert.
</div>
"""
    )

    with gr.Tab("Bewertung"):
        gr.Markdown(
            "Gib die Wohnungsdaten ein. **Optional**: Inserat-Text einfügen oder Fotos hochladen — "
            "beides verbessert die Schätzung. Nach 'Bewerten' erscheint eine Hybrid-Schätzung "
            "(Modell + Stadt-Zürich-Median) mit Erklärung."
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ip-card"):
                    gr.HTML('<div class="ip-section">Grunddaten</div>')
                    with gr.Row():
                        area = gr.Number(label="Fläche (m²)", value=85, minimum=15, maximum=400, info="Wohnfläche in Quadratmetern")
                        rooms = gr.Number(label="Zimmer", value=3.5, minimum=1, maximum=10, step=0.5, info="Anzahl Zimmer (z.B. 3.5)")
                    kreis = gr.Slider(1, 12, step=1, label="Stadtkreis", value=6, info="1 = Altstadt · 8 = Seefeld · 12 = Schwamendingen")
                    year_built = gr.Number(label="Baujahr", value=1990, minimum=1800, maximum=2026, info="Baujahr des Gebäudes (Schätzung ok)")

                with gr.Group(elem_classes="ip-card"):
                    gr.HTML('<div class="ip-section">Ausstattung</div>')
                    with gr.Row():
                        has_balcony = gr.Checkbox(label="Balkon", value=False)
                        has_view = gr.Checkbox(label="Aussicht (See/Berg)", value=False)
                    with gr.Row():
                        has_elevator = gr.Checkbox(label="Lift", value=False)
                        has_parking = gr.Checkbox(label="Parkplatz", value=False)
                    is_new_building = gr.Checkbox(label="Neubau (< 5 Jahre)", value=False)

                with gr.Accordion("Inserat-Text einfügen (optional)", open=False):
                    listing_text = gr.Textbox(
                        label="Inserat",
                        placeholder="z.B. '3.5-Zi.-Wohnung Seefeld, 85m², Balkon mit Seeblick, Baujahr 2010, Lift...'",
                        lines=5,
                        info="Der Text wird mit einem LLM-Parser ausgelesen und ergänzt fehlende Felder.",
                    )
                    gr.Examples(
                        examples=EXAMPLE_LISTINGS,
                        inputs=listing_text,
                        label="Beispiel-Inserate",
                    )

                with gr.Accordion("Fotos hochladen (optional)", open=False):
                    gr.Markdown("_CLIP zero-shot erkennt automatisch Zustand, Balkon, Aussicht, Küchenqualität._")
                    photo1 = gr.Image(label="Foto 1", type="pil")
                    photo2 = gr.Image(label="Foto 2", type="pil")
                    photo3 = gr.Image(label="Foto 3", type="pil")

                btn = gr.Button("Miete schätzen", variant="primary", size="lg")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[area, rooms, kreis, year_built, has_balcony, has_view, has_elevator, has_parking, is_new_building, listing_text],
                    label="Beispiel-Wohnungen (Klick = einsetzen)",
                )

            with gr.Column(scale=1):
                headline = gr.HTML()
                with gr.Accordion("Detail: Modell vs. Median-Referenz", open=False):
                    breakdown = gr.Markdown()
                with gr.Group(elem_classes="ip-card"):
                    gr.HTML('<div class="ip-section">Erklärung (SHAP + LLM)</div>')
                    explanation = gr.Markdown()
                with gr.Group(elem_classes="ip-card"):
                    gr.HTML('<div class="ip-section">Aus Fotos extrahiert</div>')
                    cv_block = gr.Markdown()
                with gr.Group(elem_classes="ip-card"):
                    gr.HTML('<div class="ip-section">Aus Inserat extrahiert</div>')
                    parsed_block = gr.Markdown()

        btn.click(
            bewerten,
            inputs=[area, rooms, kreis, year_built, has_balcony, has_view, has_elevator, has_parking, is_new_building, listing_text, photo1, photo2, photo3],
            outputs=[headline, breakdown, explanation, cv_block, parsed_block],
        )

    with gr.Tab("Foto-Analyse"):
        gr.Markdown(
            "Lade Fotos einer Wohnung hoch. Wir analysieren mit **CLIP zero-shot** "
            "(OpenAI) Zustand, Ausstattung und Aussicht — ohne Training auf Wohnungs-Fotos."
        )
        with gr.Row():
            p1 = gr.Image(type="pil", label="Foto 1")
            p2 = gr.Image(type="pil", label="Foto 2")
            p3 = gr.Image(type="pil", label="Foto 3")
        out = gr.Markdown()
        gr.Button("Fotos analysieren", variant="primary").click(foto_analyse, inputs=[p1, p2, p3], outputs=out)

    with gr.Tab("Q&A Quartiere"):
        gr.Markdown(
            "Frag mich etwas über die Zürcher Quartiere. Antworten basieren auf einem **RAG-System** "
            "mit Daten von **Wikipedia** und **Statistik Stadt Zürich** (Mietpreiserhebung 2024). "
            "Quellen werden bei jeder Antwort mitgeliefert."
        )
        question = gr.Textbox(label="Deine Frage", lines=2, placeholder="Was ist Kreis 8?")
        gr.Examples(
            examples=EXAMPLE_QUESTIONS,
            inputs=question,
            label="Beispiel-Fragen",
        )
        ans = gr.Markdown(label="Antwort")
        sources = gr.Markdown()
        with gr.Row():
            ask_btn = gr.Button("Frage stellen", variant="primary", size="lg")
        ask_btn.click(qa, inputs=question, outputs=[ans, sources])
        question.submit(qa, inputs=question, outputs=[ans, sources])

    gr.HTML(
        f"""
<div id="ip-footer">
  <strong>Stack</strong> · XGBoost (Mietprognose · MAE 337 CHF, R² 0.78) · CLIP zero-shot (Foto-Merkmale) ·
  FAISS + sentence-transformers (RAG Q&A) · {config.LLM_PROVIDER.title()} {config.ANTHROPIC_MODEL if config.LLM_PROVIDER=='anthropic' else config.OPENAI_MODEL} (Erklärung + Q&A)<br>
  <strong>Kalibrierung</strong> · Hybrid-Schätzung mit Stadt-Zürich-Median (60% Modell, 40% Median × Fläche)
  zur Korrektur des Distribution-Shift bei Premium-Lagen.<br>
  <span style="opacity:.7">ZHAW SML — AI Applications · Spring 2026 · <a href="https://github.com/kerisan-jeg/immopilot-zurich">GitHub Repository</a></span>
</div>
"""
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )

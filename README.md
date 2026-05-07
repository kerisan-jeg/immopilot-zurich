# 🏠 ImmoPilot Zürich

> **Multimodal apartment-rent assistant for the city of Zurich.**
> Predicts a fair market price from listing text, photos, and structured input,
> explains *why* with SHAP, and answers neighborhood questions via RAG.

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-yellow)](https://huggingface.co/spaces/USER/immopilot-zurich)
[![CI](https://github.com/USER/immopilot-zurich/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/immopilot-zurich/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## ✨ What it does

| Block | Capability | Models / Methods |
|---|---|---|
| 📊 **ML Numeric** | Predicts rent in CHF with confidence interval | Linear · Random Forest · XGBoost · MLP · SHAP |
| 📸 **Computer Vision** | Detects condition, balcony, view, kitchen quality from photos | CLIP zero-shot · fine-tuned ResNet50 |
| 💬 **NLP / RAG** | Parses free-text listings · explains predictions · answers Q&A about neighborhoods | LLM function-calling · FAISS · LangChain · multi-provider (Anthropic / OpenAI) |

Outputs of CV become **input features** for the numeric model. The numeric prediction
together with the RAG context becomes **input** for the LLM-generated explanation.
The blocks are not parallel — they form one coherent pipeline.

---

## 🏛️ Architecture

![Architecture](docs/architecture.png)

```
[Photos]  ─── CLIP + ResNet50 ──▶ {modern: 0.83, balcony: 1, view: 0}
                                              │
[Listing text] ── LLM parser ───▶ {area: 78, rooms: 3.5, kreis: 6}
                                              │
                                              ▼
[Combined features] ── XGBoost ──▶ CHF 2 840 [2 620 – 3 080]
                                              │
                                              ▼
[Prediction + RAG context] ── LLM ──▶ "Main drivers: kreis, area, condition…"
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/USER/immopilot-zurich.git
cd immopilot-zurich

# 1. Environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Secrets
cp .env.example .env
# → edit .env, add ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 3. Reproduce everything (data → features → train → index → app)
make reproduce

# Or step by step
make data        # download / scrape
make features    # build feature tables
make train       # train all numeric + CV models
make index       # build RAG vector store
make app         # launch Gradio app on http://localhost:7860
```

---

## 📁 Project Structure

```
immopilot-zurich/
├── README.md                  ← you are here
├── docs/
│   ├── DOCUMENTATION.md       ← full project documentation (graded)
│   ├── architecture.png
│   └── screenshots/
├── src/immopilot/             ← all reusable code
│   ├── data/                  ← loaders, scrapers, joiners
│   ├── features/              ← preprocessing pipelines
│   ├── models/                ← train_*.py + evaluate.py
│   ├── cv/                    ← CLIP + ResNet50
│   ├── nlp/                   ← RAG, parser, explainer
│   └── inference/             ← end-to-end pipeline
├── app/app.py                 ← Gradio UI (HF Space entry point)
├── notebooks/                 ← EDA, experiments, error analysis
├── tests/                     ← pytest smoke tests
├── scripts/                   ← one-off scripts
├── data/                      ← .gitignored except samples
└── models/                    ← .gitignored, pulled from HF Hub
```

---

## 📊 Results (filled in after evaluation)

| Model | MAE (CHF) | RMSE | R² | Notes |
|---|---:|---:|---:|---|
| Linear Regression | _tbd_ | _tbd_ | _tbd_ | baseline |
| Random Forest | _tbd_ | _tbd_ | _tbd_ | |
| XGBoost | _tbd_ | _tbd_ | _tbd_ | best so far |
| MLP (PyTorch) | _tbd_ | _tbd_ | _tbd_ | |

**CV — condition classifier**
| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| CLIP zero-shot | _tbd_ | _tbd_ |
| ResNet50 fine-tuned | _tbd_ | _tbd_ |

**RAG — qualitative + Ragas faithfulness / answer-relevance**: see `docs/DOCUMENTATION.md` §4.4.

---

## 📚 Documentation

Full project documentation following the assignment template:
**[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)**

- §1 Project Idea & Methodology
- §2 Data & Preprocessing
- §3 Modeling & Implementation
- §4 Evaluation & Analysis
- §5 Deployment
- §6 Execution Instructions

---

## 🧪 Reproducibility

- All random seeds pinned (`SEED = 42`, see `src/immopilot/config.py`)
- `requirements.txt` uses `==` pinning
- Data snapshot date logged in `data/raw/SNAPSHOT.md`
- `make reproduce` runs end-to-end on a clean machine
- CI runs lint + smoke tests on every push

---

## 👥 Course Context

Final project for **AI Applications**, ZHAW SML, Spring 2026.
Lecturers: Jasmin Heierli ([@jasminh](https://github.com/jasminh)) · Benjamin Kühnis ([@bkuehnis](https://github.com/bkuehnis)) — both added as collaborators.

## 📜 License

MIT — see [LICENSE](LICENSE).

# ImmoPilot Zürich — Project Documentation

> **Course**: AI Applications · ZHAW SML · Spring 2026
> **Author**: _your name_
> **Submission**: 07 June 2026, 18:00
> **Repository**: https://github.com/USER/immopilot-zurich
> **Live demo**: https://huggingface.co/spaces/USER/immopilot-zurich
> **Collaborators added**: @jasminh, @bkuehnis

---

## 1. Project Idea & Methodology

### 1.1 Problem Definition
Apartment seekers in Zurich face an opaque rental market: prices vary strongly by
neighborhood, condition, and amenities, but listings rarely make these factors
transparent. As a result, renters cannot easily judge whether a quoted price is
fair, and they have no quick way to interpret a listing in terms of objective
quality signals.

### 1.2 Use Case & Target Users
**Primary user**: a person actively searching for a flat in Zurich.
**Goal**: paste a listing or upload photos, get an *evidence-based* fair-price
estimate with an explanation, and ask follow-up questions about the neighborhood.

### 1.3 Objectives
1. **Numeric**: predict monthly rent in CHF with MAE ≤ CHF 250 and R² ≥ 0.75 on a held-out test set.
2. **CV**: classify apartment condition (modern / standard / needs renovation) with macro-F1 ≥ 0.70 on a held-out test set.
3. **NLP/RAG**: produce factually grounded answers about Zurich neighborhoods (Ragas faithfulness ≥ 0.80 on 20-question eval set).
4. **Integration**: CV outputs measurably improve the numeric model (ablation: ΔMAE > 5%).

### 1.4 Selected Blocks & Integration Concept
This project integrates **all three blocks**.

| Block | Role in the system |
|---|---|
| ML Numeric | Core decision component — predicts the rent. |
| CV | Provides additional features (condition, amenities) extracted from photos. |
| NLP / RAG | (a) Parses free-text listings into structured features. (b) Generates a natural-language explanation of the prediction. (c) Answers questions about Zurich neighborhoods grounded in official sources. |

**Integration mechanisms**:
- *Shared features*: CV outputs and NLP-parsed fields feed into the numeric model.
- *Decision logic*: numeric prediction + SHAP feature importances are passed to the LLM as structured context.
- *User interaction*: a single Gradio interface orchestrates all three blocks.

### 1.5 Scope, Assumptions, Out-of-Scope
**In scope**: Zurich city (Kreis 1–12), residential rentals, German/English listings.
**Assumptions**: listings are honest about square meters and rooms; photos depict the actual unit.
**Out of scope**: commercial rentals, rentals outside Zurich city, fraud detection, legal advice.

---

## 2. Data & Preprocessing

### 2.1 Data Sources Overview

| # | Source | Type | Size (approx.) | License | Used for |
|---|---|---|---:|---|---|
| 1 | Homegate / ImmoScout24 listings (scraped or Kaggle snapshot) | Tabular + text | ~5 000 listings | research use, fair use | numeric model target + features |
| 2 | Stadt Zürich Open Data (Statistik) | Tabular | 12 districts × ~10 features | CC BY 3.0 | district-level features |
| 3 | Curated apartment photos (own + CC-licensed sources) | Images | ~500 images, 3 classes | CC BY / own | CV training |
| 4 | stadt-zuerich.ch district pages + Wikipedia | Text | ~30 documents | CC BY-SA / public | RAG corpus |
| 5 | LLM API (Anthropic Claude / OpenAI GPT-4o-mini) | Service | — | per Terms | parsing + generation |

### 2.2 Numeric Data — Cleaning, Joining, Outlier Handling
- _Describe_: deduplication, currency parsing, area parsing (`m²` → float), rooms parsing (`3.5` → float).
- Outliers: rent < CHF 500 or rent > CHF 12 000 dropped. Justification: …
- Missing values: median imputation for area (n=…), mode for kreis (n=…).
- Join with district statistics on `kreis` (1:N).

### 2.3 Image Data — Collection, Licensing, Augmentation
- Collection process and source breakdown.
- Manual annotation protocol (3 classes × 2 annotators, Cohen κ = …).
- Augmentation: horizontal flip, color jitter, random resized crop. Examples in §4.3.

### 2.4 Text Data — Sources, Chunking, Embedding
- District pages + Wikipedia → cleaned to plain text.
- Chunking strategy: 512 tokens, 64 overlap. Justification: …
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (vs. `multilingual-e5-base` — comparison in §3.3).
- Vector store: FAISS, IndexFlatIP.

### 2.5 Feature Engineering & Annotations
- `area_per_room`, `is_luxurious` (regex on description), one-hot for `kreis`, log-transform on rent target.
- CV-derived features: `condition_score` (probit), `has_balcony`, `has_view`, `kitchen_quality`.

### 2.6 Exploratory Data Analysis — Key Findings

> Insert plots from `notebooks/01_eda_numeric.ipynb`.

- Rent distribution is right-skewed → log transform.
- Strong rent ↔ kreis correlation (Kreis 1, 6, 7 ≫ Kreis 11, 12).
- Outliers concentrated in luxurious / penthouse listings.
- Photo-feature distribution per condition class (CV-EDA in `04_cv_zero_shot.ipynb`).

---

## 3. Modeling & Implementation

### 3.1 ML Numeric — Model Comparison

| Model | Library | Hyperparams (final) | Why |
|---|---|---|---|
| Linear Regression | scikit-learn | — | baseline, interpretable |
| Random Forest | scikit-learn | n_estimators=500, max_depth=None | non-linear, low tuning effort |
| XGBoost | xgboost | best from Optuna (50 trials) | usually wins on tabular |
| MLP | PyTorch | 3×128, ReLU, Adam, early-stop | tests deep-model lift |

**Selection criterion**: lowest CV-MAE on 5-fold cross-validation, tie-break by R².

### 3.2 Computer Vision — Zero-Shot vs. Fine-Tuned
- **Zero-shot CLIP**: prompts like *"a photo of a modern, renovated apartment interior"* — see `src/immopilot/cv/zero_shot_clip.py`.
- **Fine-tuned ResNet50**: ImageNet pretrained backbone, last block + classifier head fine-tuned on 500 images, 80/20 split, 20 epochs, label smoothing, weighted cross-entropy.
- **Comparison**: see §4.3.

### 3.3 NLP / RAG — Prompt & Retrieval Strategies

| Variant | Retriever | Re-ranking | Top-k | LLM |
|---|---|---|---:|---|
| A | BM25 | none | 5 | Claude Haiku |
| B | MiniLM | none | 5 | Claude Haiku |
| C | MiniLM + Cohere rerank | yes | 5 (from 20) | Claude Haiku |
| D | Multilingual-e5 | yes | 5 | GPT-4o-mini |

Eval: see §4.4.

### 3.4 Integration Pipeline
See architecture diagram in README. Code: `src/immopilot/inference/pipeline.py`.

### 3.5 Iterations & Improvements (what did NOT work)
- _Tried but rejected_: training a CNN from scratch (overfitted on 500 images).
- _Tried but rejected_: cross-encoder reranking — latency too high for the Space.
- _Tried but rejected_: LightGBM — marginal gain over XGBoost, kept simpler stack.

### 3.6 Tech Stack & Repository Structure
See README.

---

## 4. Evaluation & Analysis

### 4.1 Evaluation Strategy
- **Numeric**: 80/10/10 train/val/test, stratified by `kreis`. 5-fold CV on train+val for model selection. Final reported metric: held-out test MAE/RMSE/R².
- **CV**: 80/20 train/test, stratified by class. Metrics: accuracy, macro-F1, confusion matrix.
- **RAG**: 20 hand-crafted Q&A pairs about Zurich districts. Metrics: retrieval hit-rate (top-5), Ragas faithfulness, answer relevance, manual qualitative review.

### 4.2 Numeric Block — Results

| Model | Test MAE | Test RMSE | Test R² | CV MAE (mean ± std) |
|---|---:|---:|---:|---|
| Linear | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| RF | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **XGBoost** | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| MLP | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Per-Kreis breakdown** (best model): table.

**SHAP top-5 features**: list with values and interpretation.

**Ablation — does CV add value?**
| Variant | Test MAE |
|---|---:|
| All features | _tbd_ |
| − CV features | _tbd_ |
| − district features | _tbd_ |
| − text-derived features | _tbd_ |

### 4.3 CV Block — Results

| Model | Accuracy | Macro-F1 | Per-class F1 (mod / std / reno) |
|---|---:|---:|---|
| CLIP zero-shot | _tbd_ | _tbd_ | _tbd_ |
| ResNet50 ft | _tbd_ | _tbd_ | _tbd_ |

Confusion matrices in `notebooks/05_cv_finetuning.ipynb`.

**Qualitative**: 6 example images with predicted vs. true class, including 3 success and 3 failure cases.

### 4.4 NLP / RAG Block — Results

| Variant | Hit-rate@5 | Faithfulness | Answer Relevance | Latency (p50) |
|---|---:|---:|---:|---:|
| A — BM25 + Haiku | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| B — MiniLM + Haiku | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| C — MiniLM + rerank + Haiku | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| D — e5 + GPT-4o-mini | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

Three qualitative examples with full prompt, retrieved chunks, and answer.

### 4.5 Error Analysis
- **Numeric**: 5 worst residuals — discuss why (e.g., luxurious furnishings unrepresented in features, atypical contracts, scraping artifacts).
- **CV**: confusion matrix hot-spots — what kind of images are confused.
- **RAG**: 3 cases where retrieval failed and how a future fix would address it.

### 4.6 Interpretation & Limitations
- Bias toward listings that *appear* on Homegate/ImmoScout24 — informal market is not represented.
- Photos may be staged → CV features systematically biased toward *modern*.
- Rent-control regulations and `Bestandsmieten` not modeled.

### 4.7 Ethical Considerations
- The feature `frg_pct` (foreigner share per kreis) correlates with rent. Including it as a feature risks reinforcing socio-economic segregation in predicted prices. We discuss the trade-off and report results both with and without this feature.
- Photos uploaded by users are not stored server-side — see `app/app.py`. Privacy notice shown in the UI.
- Predictions are *informational*, not a substitute for professional valuation. The UI states this explicitly.

---

## 5. Deployment

### 5.1 Live URL
**https://huggingface.co/spaces/USER/immopilot-zurich**

### 5.2 Architecture: Training vs. Inference Separation
- **Training (offline)**: `make train` runs locally, produces artifacts in `models/`. These are pushed to a Hugging Face model repo (`USER/immopilot-models`).
- **Inference (online)**: the Gradio Space pulls model artifacts from the model repo on startup; no training code runs in the Space.
- This guarantees fast cold-starts and clean separation of concerns.

### 5.3 Screenshots
- `docs/screenshots/01_valuation_tab.png` — input + prediction + explanation
- `docs/screenshots/02_photo_tab.png` — photo upload + detected features
- `docs/screenshots/03_qa_tab.png` — RAG Q&A with sources

### 5.4 Performance & Latency
- Cold start: ~30 s (CLIP + ResNet50 + FAISS load).
- Per-request: ~1.2 s (XGB + LLM call dominates).

---

## 6. Execution Instructions

### 6.1 Local Setup
```bash
git clone https://github.com/USER/immopilot-zurich.git
cd immopilot-zurich
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
```

### 6.2 Reproducing the Full Training Pipeline
```bash
make reproduce
```
which is equivalent to
```bash
make data        # downloads + scrapes data → data/raw
make features    # cleans, joins, engineers → data/processed
make train       # trains all numeric + CV models → models/
make index       # builds FAISS RAG index → models/rag/
```

### 6.3 Running the App Locally
```bash
make app         # http://localhost:7860
```

### 6.4 Running Tests
```bash
make test
```

### 6.5 Repository & Collaborators
- Repository: https://github.com/USER/immopilot-zurich
- Collaborators added: @jasminh, @bkuehnis (verify under Settings → Collaborators)

---

## Appendix A — Notebooks Index
| Notebook | Purpose |
|---|---|
| `01_eda_numeric.ipynb` | EDA of listings + district data |
| `02_feature_engineering.ipynb` | Feature pipeline development |
| `03_model_comparison.ipynb` | Numeric model selection & SHAP |
| `04_cv_zero_shot.ipynb` | CLIP zero-shot exploration |
| `05_cv_finetuning.ipynb` | ResNet50 fine-tuning + evaluation |
| `06_rag_evaluation.ipynb` | RAG variant comparison + Ragas |
| `07_error_analysis.ipynb` | Cross-block error analysis |

## Appendix B — Hyperparameter Search
_Document Optuna study results, search spaces, best params._

## Appendix C — Prompt Templates
_Full text of every prompt used._

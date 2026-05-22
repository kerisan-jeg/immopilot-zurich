# ImmoPilot Zürich — Project Documentation

> **Course**: AI Applications · ZHAW SML · Spring 2026
> **Author**: _<your name>_ (`kerisan-jeg`)
> **Submission**: 07 June 2026, 18:00
> **Repository**: https://github.com/kerisan-jeg/immopilot-zurich
> **Live demo**: https://huggingface.co/spaces/jegatker/immopilot-zurich
> **Collaborators added**: @jasminh (Jasmin Heierli), @bkuehnis (Benjamin Kühnis)

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
**Goal**: enter the key facts of a flat (or paste a listing / upload photos), obtain an
*evidence-based* fair-price estimate with a transparent explanation, and ask
follow-up questions about the neighborhood.

### 1.3 Objectives

The original objectives and their **actual outcome**:

1. **Numeric**: predict monthly rent in CHF. *Target*: MAE ≤ CHF 250, R² ≥ 0.75.
   *Achieved*: **MAE 337 CHF, R² 0.78** on a held-out test set — the R² target was met,
   the MAE target was not (discussed in §4.5).
2. **CV**: extract apartment quality signals (condition, balcony, view) from photos.
   *Achieved*: zero-shot CLIP feature extraction is implemented and integrated; a
   fine-tuned classifier is prepared in code but not yet trained (see §3.2, §6).
3. **NLP/RAG**: produce factually grounded answers about Zurich neighborhoods with
   source citations. *Achieved*: implemented and qualitatively validated; a formal
   quantitative eval is future work (§4.4).
4. **Integration**: a single interface orchestrates all three blocks, with CV- and
   NLP-derived features feeding the numeric model. *Achieved*.

### 1.4 Selected Blocks & Integration Concept

This project integrates **all three blocks**.

| Block | Role in the system |
|---|---|
| ML Numeric | Core decision component — predicts the rent. |
| CV | Provides additional features (condition, balcony, view, kitchen quality) extracted from photos. |
| NLP / RAG | (a) Parses free-text listings into structured features. (b) Generates a natural-language explanation of the prediction. (c) Answers questions about Zurich neighborhoods grounded in cited sources. |

**Integration mechanisms**:
- *Shared features*: CV outputs (`condition_score`, `has_balcony`, `has_view`,
  `kitchen_quality`) and NLP-parsed fields (`area_m2`, `rooms`, `kreis`) flow into the
  numeric feature vector.
- *Decision logic*: the numeric prediction plus its SHAP feature contributions are
  passed to the LLM as structured context, which produces a German-language explanation.
- *User interaction*: a single Gradio interface (`app/app.py`) orchestrates all three
  blocks across three tabs.

### 1.5 Scope, Assumptions, Out-of-Scope

**In scope**: Zurich city (Kreis 1–12), residential rentals, German/English listings.
**Assumptions**: listings are honest about square meters and rooms; photos depict the
actual unit.
**Out of scope**: commercial rentals, rentals outside Zurich city, fraud detection,
legal advice.

---

## 2. Data & Preprocessing

### 2.1 Data Sources Overview

| # | Source | Type | Size (actual) | License | Used for |
|---|---|---|---:|---|---|
| 1 | Swiss apartment-listings dataset (Kaggle snapshot) | Tabular + text | 664 listings after cleaning | research use | numeric model target + features |
| 2 | Statistik Stadt Zürich — Mietpreiserhebung (MPE) 2024 | Tabular | 12 districts × 4 fields | CC BY | district-level median/mean rent (`rent_median_chf_per_m2`) |
| 3 | Statistik Stadt Zürich — Wohndichte | Tabular | district-level | CC BY | district context |
| 4 | Wikipedia + Stadt-Zürich district knowledge | Text | 13 documents | CC BY-SA / public | RAG corpus |
| 5 | LLM API (Anthropic Claude Haiku) | Service | — | per Terms | listing parsing, explanation, Q&A |

> **Note on data realism**: the listing dataset is Switzerland-wide. Only **27 of 664
> rows** lie within the city of Zurich. This imbalance is the single most important
> property of the dataset and drives the central limitation discussed in §4.6.

### 2.2 Numeric Data — Cleaning, Joining, Outlier Handling

- **Parsing**: area (`m²` → float), rooms (`3.5` → float), rent normalized to monthly
  net CHF.
- **Outlier filter**: rows with monthly rent outside **CHF 500–12'000** are dropped
  (`basic_outlier_filter` in `data/load_listings.py`) to avoid scraping artifacts and
  non-residential entries dominating the loss; 664 rows remain after filtering.
- **Join**: listings joined with the Stadt-Zürich MPE table on `kreis` (many-to-one),
  attaching `rent_median_chf_per_m2` and `rent_mean_chf_per_m2` as district-level
  features.
- **Missing values**: handled inside the preprocessing pipeline
  (`models/preprocessor.joblib`) via imputation; the pipeline is persisted so that
  inference applies exactly the same transforms as training.
- **Target**: `rent_chf`, log-transformed with `np.log1p` for training
  (`config.LOG_TARGET = True`) and inverse-transformed with `np.expm1` for reporting.
- **Split**: 80/10/10 train/val/test, stratified by `is_zurich`, `random_state = 42`.

### 2.3 Image Data — Status

The CV block currently operates **zero-shot** (CLIP), so no labeled training-image
corpus was required for the deployed system. The fine-tuning path
(`src/immopilot/cv/train_classifier.py`) is implemented and ready, but a labeled
apartment-photo dataset has not yet been collected and the ResNet classifier has not
been trained. This is the main planned extension (§6).

### 2.4 Text Data — Sources, Chunking, Embedding

- **Corpus**: 13 Markdown documents — one per Kreis (1–12) plus a city overview —
  covering location, demographics, transport, education, gastronomy, green spaces,
  economy, rent levels, and character. Authored from Wikipedia and Stadt-Zürich sources.
- **Chunking**: one document → one chunk for the per-Kreis files; the longer overview
  document is split into two. Result: **14 chunks from 13 files**. The per-Kreis
  granularity keeps each chunk topically coherent.
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast,
  multilingual-capable enough for German district text).
- **Vector store**: FAISS `IndexFlatIP` (cosine similarity on normalized embeddings).
- Build script: `python -m immopilot.nlp.build_index`.

### 2.5 Feature Engineering & Annotations

The numeric model uses **25 features**:

- *Core*: `area_m2`, `rooms`, `area_per_room`, `lat`, `lon`.
- *Building*: `year_built`, `building_age`, `years_since_renovation`, `is_new_building`.
- *District (from MPE join)*: `rent_median_chf_per_m2`, `rent_mean_chf_per_m2`,
  `location_kreis`, `is_zurich`, `size_bucket`.
- *Amenities (binary)*: `has_balcony`, `has_view`, `has_elevator`, `has_garage`,
  `has_parking`, `has_fireplace`, `is_luxurious`, `is_furnished`, `is_temporary`.
- *CV-derived*: `condition_score`, `kitchen_quality`.

The regression target (monthly rent) is modeled on a log scale where configured
(`config.LOG_TARGET`), then inverse-transformed for reporting in CHF.

### 2.6 Exploratory Data Analysis — Key Findings

Full analysis with figures in `notebooks/01_eda_numeric.ipynb` (executed, 7 plots).
Key findings:

- **Right-skewed target**: raw rent skewness is **2.41** (long tail of expensive flats);
  the `log1p` transform makes the distribution roughly symmetric — this is why the
  models train on the log target (`config.LOG_TARGET = True`).
- **Area is the strongest continuous predictor** of rent; Zurich listings sit above the
  Switzerland-wide cloud at comparable sizes (early visual sign of the location premium).
- **Strong location effect**: the official MPE table shows a **64 % spread** from the
  most expensive Kreis (1 = 36.3 CHF/m²) to the cheapest (12 = 22.1 CHF/m²), with
  Kreis 8 (Seefeld, 31.8) second — consistent with the calibration prior.
- **Data imbalance (key caveat)**: only **27 / 664** rows are in the city of Zurich,
  visible in every location-based plot and the driver of the §4.6 limitation.
- **Amenities** (view, fireplace, garage, …) show measurable median-rent uplift,
  supporting their inclusion as features.

---

## 3. Modeling & Implementation

### 3.1 ML Numeric — Model Comparison

Four model families were trained and compared on an identical preprocessed feature
matrix and the same held-out test split.

| Model | Library | Configuration | Rationale |
|---|---|---|---|
| Linear (Ridge) | scikit-learn | L2-regularized | interpretable baseline |
| Random Forest | scikit-learn | ensemble of trees | non-linear, low tuning effort |
| **XGBoost** | xgboost | tuned with Optuna (50 trials, TPE, seed 42) | gradient boosting, usually best on tabular |
| MLP | scikit-learn / torch | small feed-forward net | tests whether a deep model helps |

**Selection criterion**: lowest test MAE, tie-broken by R². Results in §4.2.

**Data split**: 80 / 10 / 10 train / val / test (`TEST_SIZE = VAL_SIZE = 0.1`),
**stratified by `is_zurich`** so the 27 Zurich rows are proportionally represented in
each split, `random_state = 42`. **5-fold KFold cross-validation** on the training pool
(`scripts/recompute_cv.py`) reports MAE mean ± std alongside the held-out test metrics
(§4.2). The MLP is excluded from CV (its torch wrapper is not `clone`-able and a diverged
model's CV is uninformative).

### 3.2 Computer Vision — Zero-Shot (deployed) vs. Fine-Tuned (planned)

- **Zero-shot CLIP** (`src/immopilot/cv/zero_shot_clip.py`): the deployed approach.
  Apartment photos are scored against natural-language prompts (e.g. *"a photo of a
  modern, renovated apartment interior"* vs. *"…in need of renovation"*) to derive
  `condition_score`, `has_balcony`, `has_view`, and `kitchen_quality` **without any
  task-specific training**.
- **Fine-tuned ResNet50** (`src/immopilot/cv/train_classifier.py`): implemented but
  **not yet trained** — pending collection of a labeled apartment-photo dataset.
  Planned as the main extension (§6).

### 3.3 NLP / RAG — Implemented Strategy

The **deployed** configuration is a single, well-tuned pipeline (not the multi-variant
matrix originally sketched):

| Component | Choice |
|---|---|
| Retriever | dense — `all-MiniLM-L6-v2` embeddings over FAISS `IndexFlatIP` |
| Top-k | 5 |
| Re-ranking | none (corpus is small and topically partitioned) |
| Generator | Anthropic Claude Haiku |
| Grounding | answers cite retrieved chunks as `[Source N]` with title + URL |

Two LLM-backed NLP functions complete the block:
- **Listing parser** (`listing_parser.py`): free-text listing → structured fields.
- **Explainer** (`explainer.py`): SHAP contributions + prediction → German explanation,
  with a feature-name dictionary so the user sees "Wohnfläche", "Stadtkreis", etc.
  rather than raw column names.

### 3.4 Integration Pipeline

`src/immopilot/inference/pipeline.py` exposes a single `predict(structured,
listing_text, photos)` entry point that:
1. parses the optional listing text and fills missing structured fields;
2. extracts CV features from optional photos;
3. assembles the 25-feature vector, imputing absent columns;
4. runs the XGBoost model;
5. **calibrates** the result against the Stadt-Zürich median (§3.5);
6. computes SHAP contributions and asks the LLM for an explanation;
7. returns a structured `PredictionResult` (final estimate, interval, raw model
   value, reference value, confidence, explanation).

### 3.5 Hybrid Calibration (key methodological contribution)

Because the model is trained Switzerland-wide (27/664 Zurich rows), it **systematically
underestimates premium Zurich districts**. To correct this distribution shift at
inference time, when `kreis` and `area_m2` are known we blend:

```
reference = median_chf_per_m2(kreis) × area_m2
final = 0.6 × model_prediction + 0.4 × reference
```

The 60/40 weight keeps the model's multi-feature signal dominant while pulling premium
locations toward the official median. Both the raw model value and the reference are
shown in the UI for transparency. *Example*: a 85 m² flat in Kreis 8 (Seefeld) — raw
model CHF 2,007, median reference CHF 2,706, **calibrated CHF 2,287** (−26% model
deviation corrected).

### 3.6 Iterations & Improvements (what did NOT work)

- **MLP on tabular data**: catastrophic failure (test R² = −196) — a textbook example
  of deep nets underperforming tree ensembles on small tabular datasets
  (cf. Grinsztajn et al., *Why do tree-based models still outperform deep learning on
  tabular data?*, NeurIPS 2022). Kept in the comparison precisely because it is
  instructive.
- **Dependency stack**: a long chain of Gradio/pydantic/numpy conflicts on Windows was
  resolved by pinning a tested set of versions in `requirements.txt` and rebuilding the
  virtual environment from scratch (documented for reproducibility).

### 3.7 Tech Stack & Repository Structure

Python 3.12 · scikit-learn · XGBoost · PyTorch + CLIP · sentence-transformers · FAISS ·
LangChain · Anthropic SDK · Gradio 5.9.1. Pinned versions in `requirements.txt`. See
README for the directory tree.

---

## 4. Evaluation & Analysis

### 4.1 Evaluation Strategy

- **Numeric**: 80/10/10 split stratified by `is_zurich`; reported metrics are
  held-out **test** MAE / RMSE / R², complemented by **5-fold CV MAE** (mean ± std) on
  the training pool for the three sklearn-style models.
- **CV**: zero-shot, evaluated qualitatively on example images; no labeled test set yet.
- **RAG**: qualitative review on hand-written questions; formal metrics are future work.

### 4.2 Numeric Block — Results

**Actual held-out test results, with 5-fold cross-validated MAE on the training pool:**

| Model | Test MAE (CHF) | Test RMSE (CHF) | Test R² | CV MAE (mean ± std) |
|---|---:|---:|---:|---:|
| Linear (Ridge) | 428.0 | 707.5 | 0.721 | 418.8 ± 26.0 |
| Random Forest | 365.4 | 668.3 | 0.751 | 379.0 ± 18.2 |
| **XGBoost** ✅ | **337.4** | **634.6** | **0.775** | **335.0 ± 3.0** |
| MLP | 3870.1 | 18792.1 | −196.1 | — (diverged) |

XGBoost is the champion on all metrics. Three observations strengthen the model choice:

1. **Stability**: XGBoost's per-fold MAEs are 336/329/338/336/337 — a standard deviation
   of only **±3 CHF**, indicating a robust, non-overfit model.
2. **Test ≈ CV**: XGBoost's test MAE (337) almost equals its CV mean (335), so the single
   reported test split is representative rather than lucky.
3. **Consistent ranking**: XGBoost < RF < Linear holds in *both* CV and test, so the
   selection is well-grounded, not an artifact of one split.

The performance ordering (boosting > bagging > linear ≫ MLP) is exactly what tabular-ML
literature predicts.

**SHAP interpretation** (from the deployed explainer): the dominant positive drivers
are district level (`location_kreis`, `rent_median_chf_per_m2`) and `area_m2`; the
geographic coordinates (`lat`/`lon`) and `area_per_room` act as the main downward
adjustments. This is consistent with rent being primarily a function of *where* and
*how big*.

**Ablation — does CV / district / text add value?** — *planned* (§6); the pipeline is
structured to make feature-group ablations straightforward.

### 4.3 CV Block — Results

Zero-shot CLIP is evaluated qualitatively: on clear interior photos it separates modern
vs. dated interiors and detects balconies/views plausibly. A quantitative comparison
against a fine-tuned ResNet50 is pending data collection (§6).

### 4.4 NLP / RAG Block — Results

Qualitative evaluation on representative questions shows correct, well-grounded answers
with citations. Verified examples:

- *"Welche Tramlinien fahren nach Schwamendingen?"* → correctly returns Tram 7 & 9 plus
  the Glattalbahn, citing the Kreis 12 document.
- *"Was ist die Goldküste?"* → correctly explains it is the right-shore lake municipalities
  **outside** the city (not a Stadtkreis).
- *"Wo ist der Sitz von Google in Zürich?"* → correctly locates it near Kreis 5 /
  Sihlpost with Zürich-West context.

A formal quantitative eval (retrieval hit-rate, faithfulness on a 20-question set) is
future work (§6).

### 4.5 Error Analysis

- **MAE vs. target**: the CHF 337 MAE misses the CHF 250 objective. Root cause is the
  Zurich-data scarcity — the model cannot fully learn city-specific premia from 27 rows.
  The hybrid calibration (§3.5) mitigates this at inference but does not change the
  underlying training-data limitation.
- **MLP failure**: the negative R² indicates predictions worse than the mean baseline —
  the small, heterogeneous tabular set plus minimal tuning makes the MLP diverge.
- **Premium-district bias**: pre-calibration, Kreis 8 (Seefeld) is underestimated by
  ~26% relative to the official median — the motivating case for §3.5.

### 4.6 Interpretation & Limitations

- **Distribution shift (central limitation)**: training data is Switzerland-wide with
  only 27/664 Zurich rows. The model encodes a Swiss-average price surface, not a
  Zurich-specific one. Mitigation: hybrid median calibration. Proper fix: acquire a
  Zurich-specific listing dataset and retrain / reweight.
- **Photos may be staged** → zero-shot CV features biased toward "modern".
- **Rent control / `Bestandsmieten`** (sitting-tenant rents) are not modeled; the system
  estimates *advertised* market rent.
- **Metric variance**: 5-fold CV quantifies variance — XGBoost is very stable
  (±3 CHF), while Linear (±26) and RF (±18) vary more across folds. The reported test
  metrics come from a single stratified split, consistent with the CV means.

### 4.7 Ethical Considerations

- **Socio-economic features**: district-level statistics correlate with demographic
  composition. Using them risks encoding socio-economic segregation into predicted
  prices. The deployed feature set relies on rent statistics and physical attributes
  rather than demographic shares; any demographic feature should be reported with and
  without, and justified.
- **Privacy**: user-uploaded photos are processed in memory only and not stored
  server-side; the UI states this explicitly.
- **Informational, not valuation**: the UI carries a clear disclaimer that predictions
  are not professional valuations or legal advice.

---

## 5. Deployment

### 5.1 Status

**Live**: https://huggingface.co/spaces/jegatker/immopilot-zurich

The app is deployed as a Hugging Face Gradio Space (CPU basic, free tier) running
Python 3.12 with the pinned dependency set. All three tabs work end-to-end on the
hosted instance: rent prediction with calibration breakdown and German SHAP
explanation, photo feature extraction, and cited RAG Q&A. The `ANTHROPIC_API_KEY` is
configured as a Space secret. Locally it also runs via `python app/app.py`
(http://127.0.0.1:7860).

**Deployment notes** (documented for reproducibility):
- HF Spaces defaults to Python 3.13, for which `torch==2.4.1` / `numpy==1.26.4` have no
  wheels; pinned `python_version: "3.12"` in the Space README frontmatter to fix this.
- HF's infrastructure `datasets` package requires a newer `pyarrow`; bumped
  `pyarrow` to 19.0.1 to resolve a `pa.json_()` AttributeError at startup.
- A root-level `app.py` launcher puts `src/` on the path and runs `app/app.py`, so the
  Space entry-point convention is satisfied without restructuring the package.

### 5.2 Architecture: Training vs. Inference Separation

- **Training (offline)**: training scripts produce artifacts in `models/`
  (`xgboost.joblib`, `preprocessor.joblib`, RAG index). Not run at serving time.
- **Inference (online)**: the app loads persisted artifacts and serves predictions; no
  training occurs in the app. This guarantees fast startup and clean separation.

### 5.3 Reproducibility

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows PowerShell
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m immopilot.nlp.build_index     # build RAG index
python app\app.py                       # launch app
```

A global seed is set (`config.set_global_seed()`) for deterministic preprocessing.

### 5.4 Screenshots

To be added under `docs/screenshots/`:
- `01_valuation_tab.png` — input + prediction + calibration breakdown + explanation
- `02_photo_tab.png` — photo upload + detected features
- `03_qa_tab.png` — RAG Q&A with cited sources

---

## 6. Future Work / Remaining for Final Submission

Concrete, prioritized to-dos to reach full marks:

1. **CV fine-tuning**: collect ~100+ labeled apartment photos, train ResNet50 via
   `train_classifier.py`, report accuracy / macro-F1 / confusion matrix, and compare
   against zero-shot CLIP.
2. **Numeric ablation**: quantify the contribution of CV, district, and text-derived
   feature groups (ΔMAE) by retraining with each group removed.
3. **RAG quantitative eval**: build a ~20-question gold set; measure retrieval hit-rate
   and answer faithfulness.
4. **Screenshots**: capture the three tabs for the docs folder.

---

## Appendix — Verified Pipeline Facts

All key pipeline specifics have been confirmed against the committed code:

- [x] **Train/test split**: 80/10/10, stratified by `is_zurich`, seed 42
  (`_common.make_splits`).
- [x] **Log-target**: `LOG_TARGET = True` — `np.log1p` for training, `np.expm1` for
  reporting.
- [x] **5-fold CV**: computed and persisted via `scripts/recompute_cv.py`; results in §4.2.
- [x] **Outlier bounds**: rent kept within CHF 500–12'000
  (`data/load_listings.py::basic_outlier_filter`).
- [x] **XGBoost tuning**: Optuna, 50 trials, TPE sampler, seed 42 (`models/train_xgb.py`).

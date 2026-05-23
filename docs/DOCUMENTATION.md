# AI Applications Project Documentation

> Completed using the official project documentation template; structure unchanged.
> Code references link directly to the GitHub repository per the Documentation Hint.

## Project Metadata

- **Project title**: ImmoPilot Zürich — Multimodal Apartment Rent Assistant
- **Student**: kerisan-jeg (jegatker@students.zhaw.ch)
- **GitHub repository URL**: https://github.com/kerisan-jeg/immopilot-zurich
- **Deployment URL**: https://huggingface.co/spaces/jegatker/immopilot-zurich
- **Submission date**: 2026-06-07

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):

- **Primary block 1**: ML Numeric Data
- **Primary block 2**: NLP

The third block (Computer Vision) is implemented and documented separately as extra work
in Section 5 (Optional Bonus Evidence), with its full technical detail in Section 2C.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- **Problem statement**: Rental prices in the city of Zurich are opaque and vary widely by
  district, size, condition, and amenities. Prospective tenants and small landlords have no
  easy, transparent tool to estimate a fair net rent and understand *why* a flat costs what
  it does.
- **Goal**: A single web app that (a) estimates monthly net rent for a Zurich apartment from
  structured inputs, an optional listing text, and optional photos; (b) explains the estimate
  in plain German; and (c) answers questions about Zurich neighborhoods with cited sources.
- **Success criteria**: a working deployed app; rent prediction with R² > 0.75 on a held-out
  test set; a grounded, source-citing Q&A; transparent, honest treatment of the model's
  limitations (the dataset is Switzerland-wide with few Zurich rows).

### 1.2 Integration Logic

- **How the selected blocks interact**: The **NLP block feeds the ML block** (a listing-text
  parser extracts structured features — area, rooms, Kreis — that fill missing numeric inputs)
  and **explains the ML block** (a SHAP-based explainer turns the numeric model's feature
  contributions into a German natural-language explanation). A second NLP component (RAG Q&A)
  shares the same Zurich district knowledge base used by the ML calibration. The **CV block
  (bonus)** contributes derived photo features (`condition_score`, `kitchen_quality`) as
  additional numeric inputs at inference time (constant in the photo-less training data — see the
  CV caveat in 2A.4).
- **Data and output flow between blocks**:
  `listing text → (NLP parser) → structured features → ML model → rent estimate → (NLP explainer) → German explanation`;
  in parallel `photos → (CV) → condition features → ML model`; and independently
  `user question → (NLP RAG) → cited answer`.

Pipeline overview: see the `predict()` function in
[`pipeline.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/inference/pipeline.py#L127)
(orchestrates parsing → CV features → ML prediction → explanation).

---

## 2. Block Documentation

### 2A. ML Numeric Data (selected — primary)

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Swiss apartment-rental listings — Kaggle `fredeys/immoscout24-ch-switzerland-rental-property-dataset` (Switzerland-wide, pre-cleaned to Zurich-relevant rows) | Structured CSV | 664 rows × 33 cols (only 27 in city of Zurich) | Main training data (target = `rent_chf`) |
| 2 | Statistik Stadt Zürich — Mietpreiserhebung (MPE) 2024, median net rent per Kreis | Structured table | 12 districts | District features + hybrid-calibration prior |
| 3 | Stadt Zürich MPE 2024 — mean net rent per Kreis (second role of source 2) | Structured table | 12 districts | `rent_mean_chf_per_m2` feature (distinct from the median in source 2) |

The downloader (`load_zurich_open.py`) also fetches two further CC0 Stadt-Zürich
datasets (Bevölkerung/BEV, Wohndichte), but only the MPE table is currently joined into the
feature set (`load_district_features()` uses MPE only). BEV/Wohndichte are therefore *not*
counted as integrated data sources here — listing them would overstate the feature set.

**Delineation from the course material.** The semester used an apartment-price example for
Zurich (the `kuhs/apartment` demo, with a Kreis split and `luxurious`/`temporary`/`furnished`
features). This project deliberately does **not** reuse that data or setup: the numeric block
is built on a *different, Switzerland-wide* Kaggle dataset
(`fredeys/immoscout24-ch-switzerland-rental-property-dataset`), merged with a *different*
open-data source (Statistik Stadt Zürich MPE rent per Kreis — not the course's demographic
data). The forbidden course dataset ("apartments, canton of Zurich") is **not** used. Beyond
the numeric block, the NLP (RAG + LLM listing parser + SHAP explainer) and Computer-Vision
(CLIP / ResNet condition scoring) blocks are entirely new and have no counterpart in the
course example. Loaders:
[`load_listings.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/data/load_listings.py),
[`load_zurich_open.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/data/load_zurich_open.py).

#### 2A.2 Preprocessing and Features

- **Cleaning steps**: rent outliers filtered to CHF 500–12'000
  ([`load_listings.py::basic_outlier_filter`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/data/load_listings.py));
  rows without the target dropped; district table joined on `kreis` with dtype matching to
  avoid silent join misses.
- **Preprocessing steps**: a single sklearn `ColumnTransformer` guarantees train/inference
  parity — median imputation + standard scaling for numerics, most-frequent imputation +
  one-hot for categoricals, constant-0 imputation for binaries
  ([`build_features.py::make_preprocessor`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/features/build_features.py)).
  The target is log-transformed (`np.log1p`, inverse `np.expm1`) because the raw rent
  distribution is right-skewed (skewness 2.41 → roughly symmetric after log).
  *No preprocessing leakage*: the persisted preprocessor is fit on the **train+val pool only**,
  replicating the held-out test split (seed 42, stratified by `is_zurich`) before fitting, so
  imputation medians and scaler statistics never see the test rows
  ([`build_features.main`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/features/build_features.py#L230)).
  An earlier version fit the transformer on the full table; that was corrected and all metrics
  were re-frozen on the leakage-free preprocessor (the headline numbers in 2A.5 reflect this).
- **Feature engineering and selection**: 25 features in four groups — engineered
  (`area_per_room`, `building_age`, `years_since_renovation`, `size_bucket`), district
  (`rent_median/mean_chf_per_m2`, `location_kreis`, `is_zurich`), amenities (balcony, view,
  elevator, garage, parking, fireplace), and CV/text-derived
  (`condition_score`, `kitchen_quality`, `is_luxurious`, `is_furnished`, `is_temporary`).
  See [`build_features.py::add_engineered_columns`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/features/build_features.py#L78) (the engineering logic) and `make_preprocessor` (#L159).

**EDA key findings** (full notebook:
[`01_eda_numeric.ipynb`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/notebooks/01_eda_numeric.ipynb), 7 plots):
raw rent skewness **2.41** (motivates log target); strong location effect — a **64 % spread**
from the most expensive Kreis (1 = 36.3 CHF/m²) to the cheapest (12 = 22.1 CHF/m²); the key
data caveat — only **27 / 664** rows are in the city of Zurich.

#### 2A.3 Model Selection

- **Models tested**: Linear (Ridge), Random Forest, XGBoost, and a small MLP.
- **Why these models were chosen**: a standard tabular-ML ladder from linear baseline →
  bagging → boosting, plus a neural net to confirm the literature expectation that deep nets
  underperform on small tabular data. XGBoost was tuned with Optuna (50 trials, TPE sampler,
  seed 42; [`train_xgb.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/models/train_xgb.py)).

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | Ridge regression on all features | Linear (Ridge) | Test MAE 427.6 / R² 0.720 | — |
| 2 | Non-linear | Bagging trees | Random Forest | Test MAE 364.4 / R² 0.748 | −63.2 CHF MAE |
| 3 | Boosting + tuning | XGBoost, Optuna 50 trials | XGBoost (champion) | **Test MAE 308.6 / R² 0.797** | −55.8 CHF MAE |
| 4 | Deep net check | Small MLP | MLP | Test MAE 4148 / R² −243 (run diverged) | excluded — not a fair baseline |

Cross-validated MAE (5-fold, training pool): Linear 418.2 ± 25.3, RF 378.4 ± 18.2,
**XGBoost 325.2 ± 6.4**. XGBoost wins on every metric and has the lowest fold-to-fold spread,
and its test MAE (309) ≈ CV mean (325), so the test split is representative. (The exact CV std
is mildly platform-dependent; the robust point is the *relative* stability versus Linear/RF.)
CV recomputed via
[`recompute_cv.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/scripts/recompute_cv.py).

**Feature-group ablation** ([`ablation_numeric.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/scripts/ablation_numeric.py)):
dropping each group and retraining gives positive ΔMAE for text +17.1,
district +13.5, amenities +9.8, and a small cv +5.6 (ablation baseline 338.2 — this uses a fixed
untuned XGBoost with n_estimators=400 to isolate feature effects, so it is higher than the tuned
champion's 308.6; the ablation measures *feature-group* contribution, not the final model). These
values reproduce exactly in the pinned environment (XGBoost 2.1.1, seed 42); the *exact* ΔMAE
magnitudes are somewhat sensitive to the XGBoost build and BLAS thread count, so the robust
conclusion is narrower than the exact CHF deltas: **text is consistently the largest
contributor**, and **cv is consistently the smallest**; the middle groups (district, amenities)
are close together and their relative order is build-sensitive (a clean Linux run with different
BLAS threading can put district as low as +4 and below amenities). So the robust reading is "text
dominates, cv is marginal," not a precise ranking of the middle groups.

**Important caveat on the CV group.** The CV-derived columns `condition_score` and
`kitchen_quality` are **constant (0.5) across all 664 training rows**, because the Kaggle listings
carry no photos — only the live app supplies real photo scores at inference time. A zero-variance
feature cannot be split on, so the trained XGBoost learns nothing from these columns, and the small
cv +5.6 in the ablation is an artefact of XGBoost's column-subsampling (`colsample_bytree`) draw
changing when the columns are removed, **not** a genuine signal contribution. We therefore do *not*
claim that the CV modality improves the trained numeric model: the CV→ML wiring is real and active
at inference (a user's photos do produce non-default scores), but on the current photo-less training
data it carries no learned signal. Enriching the training rows with real per-listing CV scores is
the natural fix and is noted as future work.

**Hybrid calibration** (methodological contribution): because the model is trained on
Switzerland-wide data with only 27 Zurich rows, it underestimates premium districts. The
final estimate blends the model with a Stadt-Zürich median prior:
`final = 0.6 × model + 0.4 × (median_chf_per_m2 × area)`
(see the *Hybrid calibration* block in
[`pipeline.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/inference/pipeline.py#L167)).

#### 2A.5 Evaluation and Error Analysis

- **Metrics used**: MAE, RMSE, R² on a held-out 10 % test split (stratified by `is_zurich`,
  seed 42), plus 5-fold cross-validated MAE on the training pool.
- **Final results**: XGBoost — Test MAE **308.6 CHF**, RMSE 603.1, R² **0.797**, CV MAE
  325.2 ± 6.4. These numbers are reproducible from committed artifacts: the trained
  `models/xgboost.joblib` + `models/preprocessor.joblib` and the feature table are committed, so
  `scripts/freeze_test_predictions.py` regenerates the frozen predictions and metrics in
  [`docs/repro/`](https://github.com/kerisan-jeg/immopilot-zurich/tree/main/docs/repro)
  directly from the repo; per-model summaries in
  [`models/*.metrics.json`](https://github.com/kerisan-jeg/immopilot-zurich/tree/main/models).
  These are the **leakage-free** results: the preprocessor is fit on the train+val pool only
  (see 2A.2), so no test statistics leak into imputation/scaling. Note: the R² 0.797 is the
  **Optuna-tuned** model (50 trials); re-training with library-default hyper-parameters yields a
  lower R², so the tuned model itself is committed rather than only the features.
- **Test-set composition (important caveat)**: the test split has **67 rows, of which only 3
  are in the city of Zurich**. The headline R² therefore reflects Swiss-wide accuracy; it is
  *not* a robust measure of Zurich-specific accuracy, and the stated CHF 250 goal — framed
  around Zurich — cannot be claimed as met on this evidence. A Zurich-only metric would need a
  larger Zurich sample than this dataset provides.
- **Error patterns and likely causes**: the MAE misses the aspirational CHF 250 target; the
  root cause is Zurich-data scarcity (27 rows total) — the model learns a Swiss-average price
  surface, not a Zurich-specific one. This is mitigated (not solved) by the hybrid median
  calibration. The MLP was **not** a fair deep-learning baseline: its run diverged (test
  RMSE 20919 ≫ MAE 4148 indicates a few exploded predictions in log space, i.e. a training
  pathology — likely learning-rate / output-scaling), so its R² −243 reflects a broken run, not
  evidence that trees beat neural nets on tabular data. We report it for transparency but do not
  draw a DL-vs-trees conclusion from it; a fair comparison would require debugging the MLP
  (output clipping, LR schedule) and is left as future work.

#### 2A.6 Integration with Other Block(s)

- **Inputs received from other block(s)**: structured fields parsed from listing text by the
  NLP parser (area, rooms, Kreis) fill missing inputs; CV-derived `condition_score` and
  `kitchen_quality` enter as numeric features at inference time (these are constant 0.5 in the
  photo-less training data, so the trained model carries no learned signal for them — see the CV
  caveat in 2A.4; the wiring is active for real user photos but does not currently improve the
  fitted model).
- **Outputs provided to other block(s)**: the prediction and its SHAP feature contributions
  are passed to the NLP explainer, which renders the German explanation. The model predicts in
  log space (`log1p`), so raw SHAP values are log-space contributions. They are converted to
  faithful CHF effects via the inverse transform — for each feature,
  `c_i = expm1(full_log) − expm1(full_log − shap_i)`, the actual CHF change when that feature's
  contribution is removed
  ([`pipeline.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/inference/pipeline.py)).
  Because `expm1` is non-linear, the per-feature CHF effects sum only *approximately* to the
  total prediction (typically within a few percent); they are honest marginal effects, not an
  exact additive decomposition. For one-hot features (Kreis, size bucket) only the category
  actually active for the flat is shown, with a readable label, so the explanation never lists
  "absence" contributions.

### 2B. NLP (selected — primary)

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Wikipedia + Stadt-Zürich district descriptions (12 Kreis files + 1 overview) | Text (Markdown) | 13 documents → 14 chunks | RAG knowledge base |
| 2 | User listing text (free-form apartment ad) | Text (runtime input) | per request | Parsed into structured features |
| 3 | Model prediction + SHAP values (from ML block) | Structured (runtime) | per request | Input to the explanation generator |

RAG corpus: [`data/raw/rag_corpus/`](https://github.com/kerisan-jeg/immopilot-zurich/tree/main/data/raw/rag_corpus).

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing**: corpus split into 14 chunks (chunk size 512, overlap 64), embedded
  with `sentence-transformers/all-MiniLM-L6-v2`, indexed in FAISS `IndexFlatIP` (cosine via
  normalized vectors). Build: [`build_index.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/nlp/build_index.py).
- **Prompt design / retrieval setup**: top-k = 5 retrieval; the system prompt instructs the
  LLM to answer only from the provided context, cite sources inline as `[Source N]`, and reply
  in the user's language (the `SYSTEM_PROMPT` and its "answer only from context" rule,
  [`rag_pipeline.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/nlp/rag_pipeline.py#L36)).

#### 2B.3 Approach Selection

- **Approach used**: retrieval-augmented generation (dense retrieval + Anthropic Claude
  Haiku generator) for Q&A; an LLM parser for listing→features; an LLM explainer for
  SHAP→German text. Provider abstracted behind one interface
  ([`llm_client.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/nlp/llm_client.py)).
- **Alternatives considered**: re-ranking was evaluated but dropped (corpus is small and
  topically partitioned, so it added latency without measurable gain); OpenAI vs Anthropic
  is switchable via the same client interface.

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline RAG | top-k=5, no citation rule | MiniLM + FAISS + Claude | answers often uncited | — |
| 2 | Grounding | enforce `[Source N]` citations in prompt | same + citation instruction | citation rate → 100 % | +grounding |
| 3 | Quantify | 20-question gold set | retrieval-only + LLM check | Hit-Rate@5 85 %, MRR 0.504 | measured |

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy**: a hand-curated 20-question gold set, each tagged with the corpus
  file(s) that should answer it ([`rag_gold_set.json`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/scripts/rag_gold_set.json),
  [`eval_rag.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/scripts/eval_rag.py)).
- **Results**: **Hit-Rate@5 = 85 %** (17/20), **MRR = 0.504**, **citation rate = 100 %**
  ([`docs/rag_eval/rag_eval.json`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/docs/rag_eval/rag_eval.json)).
- **Error patterns and likely causes**: the 3 misses (Altstadt, highest-rent, nightlife) are
  broad questions that embed close to many similar Kreis documents — a known small-corpus
  retrieval effect. For two of them the retriever still returns the city overview, which
  contains the answer, so the generated answer is likely correct despite the strict
  file-level "miss". The citation check verifies presence of a marker, not semantic
  faithfulness (LLM-judge faithfulness is the natural extension).

#### 2B.6 Integration with Other Block(s)

- **Inputs received from other block(s)**: the ML block's prediction + SHAP contributions
  drive the German explanation; user listing text is the parser's input.
- **Outputs provided to other block(s)**: parsed structured fields (area, rooms, Kreis) fill
  missing ML inputs — a direct NLP→ML feature contribution.

### 2C. Computer Vision (bonus — documented as extra work)

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Hand-collected apartment interior photos (Pexels, iStock, listings) | Images | 89 (modern 32 / standard 29 / needs_renovation 28) | Fine-tuning + evaluation of the condition classifier |
| 2 | OpenAI CLIP prompt set (condition / balcony / view / kitchen) | Text prompts | 4 prompt groups | Zero-shot feature extraction (deployed) |

Not used during the semester (apartment interior images). Split 80/20 (seed 42) via
[`make_val_split.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/scripts/make_val_split.py)
→ 71 train / 18 val.

#### 2C.2 Preprocessing and Augmentation

- **Image preprocessing**: resize to 224 px, center-crop, ImageNet normalization
  ([`train_classifier.py`, lines 33-55](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/cv/train_classifier.py#L33-L55)).
- **Augmentation strategy**: random-resized-crop (scale 0.8–1.0), horizontal flip, colour
  jitter — chosen to combat overfitting on the small dataset.

#### 2C.3 Model Selection

- **Vision model(s) used**: (a) zero-shot CLIP `clip-vit-base-patch32` (deployed feature
  extractor); (b) fine-tuned ResNet50 (ImageNet-pretrained, only `layer4` + `fc` unfrozen)
  for the comparison.
- **Why these model(s) were chosen**: CLIP needs no training data and degrades gracefully on
  real uploads (good for deployment); ResNet50 transfer learning is the standard small-data
  fine-tuning baseline and provides the required model comparison.

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Zero-shot baseline | condition prompts, argmax | CLIP zero-shot | Acc 0.72 / macro-F1 0.66 | — |
| 2 | Fine-tune | ResNet50, frozen backbone, augmentation | ResNet50 | Acc 1.00 / macro-F1 1.00 (best epoch, val) | +0.34 F1 |

Note on the ResNet number: 1.00 is the **best-epoch** score on the 18-image validation
set — i.e. the epoch was selected *using the same set it is scored on*. The final-epoch
of the same run scored Acc 0.833 / macro-F1 0.836 (3 errors), recorded in
[`models/resnet50_condition.metrics.json`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/models/resnet50_condition.metrics.json).
Both numbers come from the same tiny set; neither is a clean held-out estimate (see 2C.5).

Evaluation on the shared 18-image val set ([`eval_cv.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/scripts/eval_cv.py),
confusion matrices in [`docs/cv_eval/`](https://github.com/kerisan-jeg/immopilot-zurich/tree/main/docs/cv_eval)).

#### 2C.5 Evaluation and Error Analysis

- **Metrics and/or visual checks**: accuracy, macro-F1, per-class report, confusion matrices.
- **Final results**: ResNet50 fine-tuned — **best-epoch** Acc/F1 **1.00**, **final-epoch**
  Acc 0.833 / macro-F1 0.836 (same run); CLIP zero-shot **Acc 0.72 / F1 0.66**.
  CLIP nails the extremes (`modern`, `needs_renovation`) but collapses on the fuzzy `standard`
  class (recall 0.17 → 5/6 pushed to "modern"); fine-tuning learns that boundary.
- **Why two ResNet numbers, and which to trust**: the training loop checkpoints the
  best-validation epoch (1.00) and `eval_cv.py` scores *that* checkpoint, so
  `docs/cv_eval/cv_comparison.json` shows 1.00; the model's own metrics file additionally
  recorded the final-epoch report (0.833). The honest reading is that **neither is a clean
  estimate**: the "best" epoch was selected on the very 18 images it is then scored on
  (selection-on-validation), and 18 images split 6/6/6 is far too small to separate true
  condition from stock-photo style. The most defensible single statement is *"ResNet
  fine-tuning clearly beats zero-shot CLIP on this small in-distribution set, somewhere in the
  0.83–1.00 range, but the result is not a reliable generalization estimate."*
- **Error patterns and limitations**: this is exactly why the **deployed** app keeps zero-shot
  CLIP, which needs no training data and degrades more gracefully on real uploads. A trustworthy
  ResNet claim would require a genuinely held-out, out-of-distribution test set of real listing
  photos — listed as future work.

#### 2C.6 Integration with Other Block(s)

- **Inputs received from other block(s)**: none (CV is upstream).
- **Outputs provided to other block(s)**: `condition_score`, `kitchen_quality`,
  `has_balcony`, `has_view` enter the ML model as additional numeric features
  ([`zero_shot_clip.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/cv/zero_shot_clip.py)).

---

## 3. Deployment

- **Deployment URL**: https://huggingface.co/spaces/jegatker/immopilot-zurich
- **Main user flow**: three tabs — (1) *Bewertung*: enter flat details (optionally paste a
  listing or upload photos) → calibrated rent estimate with confidence interval + German
  explanation; (2) *Foto-Analyse*: upload photos → detected features; (3) *Q&A Quartiere*:
  ask about any Kreis → cited answer.
- **Screenshot or short demo**: three tabs captured in
  [`docs/screenshots/`](https://github.com/kerisan-jeg/immopilot-zurich/tree/main/docs/screenshots) —
  `tab1_bewertung.png`, `tab2_foto.png`, `tab3_qa.png`.

**Training vs. inference separation**: training/eval scripts (`src/immopilot/models/`,
`scripts/`) run offline and persist artifacts (`xgboost.joblib`, `preprocessor.joblib`, RAG
index). At serving time the app only *loads* these artifacts and runs `predict()` /
`answer()` — no training on the server. Deployment is a Hugging Face Gradio Space (Python
3.12, CPU); the `ANTHROPIC_API_KEY` is a Space secret.

---

## 4. Execution Instructions

- **Environment setup**:
  ```bash
  python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- **Data setup**:
  ```bash
  python -m immopilot.data.load_listings        # build listings.parquet
  python -m immopilot.data.load_zurich_open     # build zurich_districts.parquet
  python -m immopilot.features.build_features   # build features.parquet + preprocessor
  ```
- **Training command(s)**:
  ```bash
  python -m immopilot.models.train_baseline     # Ridge
  python -m immopilot.models.train_rf           # Random Forest
  python -m immopilot.models.train_xgb          # XGBoost (champion, Optuna)
  python -m immopilot.nlp.build_index           # RAG index
  python -m immopilot.cv.train_classifier       # ResNet50 (bonus; needs data/images/)
  ```
- **Inference / run command(s)**:
  ```bash
  $env:PYTHONPATH = "src"     # PowerShell; bash: export PYTHONPATH=src
  python app/app.py           # → http://127.0.0.1:7860
  ```
- **Evaluation / reproduction**:
  ```bash
  python scripts/freeze_test_predictions.py  # verify test MAE 308.6 / R² 0.797 from artifacts
  python scripts/recompute_cv.py        # 5-fold CV MAE
  python scripts/ablation_numeric.py    # feature-group ablation
  python scripts/eval_cv.py             # ResNet vs CLIP
  python scripts/eval_rag.py            # RAG hit-rate / MRR / citations
  ```
  The committed `data/processed/features.parquet`, `models/*.metrics.json`, the trained
  `models/xgboost.joblib` + `models/preprocessor.joblib`, and `docs/repro/` let a grader verify
  the headline numbers directly with `python scripts/freeze_test_predictions.py` — no raw Kaggle
  download needed. (The larger artifacts — `random_forest.joblib`, `resnet50_condition.pt` — and
  the raw images are *not* committed for size/licensing reasons; retrain those via the training
  commands above.)
- **Reproducibility notes**: all seeds fixed to 42
  ([`config.py`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/src/immopilot/config.py));
  pinned dependencies in
  [`requirements.txt`](https://github.com/kerisan-jeg/immopilot-zurich/blob/main/requirements.txt).

---

## 5. Optional Bonus Evidence

- [x] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [x] A core section is done exceptionally well
- [x] Extended evaluation
- [x] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

**Evidence for selected bonus items:**

- **Third block (Computer Vision)**: fully implemented as a model comparison — fine-tuned
  ResNet50 vs. zero-shot CLIP on a self-collected 89-image dataset, with confusion matrices
  and an honest in-distribution caveat (Section 2C).
- **More than two data sources**: four integrated sources across the three blocks — Kaggle
  listings (numeric target + features), the Stadt-Zürich MPE table (district median/mean rent,
  used both as features and as the calibration prior), a 13-document RAG corpus (NLP), and a
  hand-collected image set (CV). Each is genuinely consumed by the system.
- **Core section done exceptionally well**: the hybrid Stadt-Zürich median calibration
  (Section 2A.4) is a non-trivial methodological response to the dataset's distribution shift.
- **Extended evaluation**: 5-fold cross-validation, a feature-group ablation (ΔMAE per group),
  the CV model comparison, and a 20-question RAG gold-set eval (hit-rate, MRR, citation rate).
- **Ethics / bias analysis**: district-level statistics correlate with socio-economic
  composition, so the model risks encoding segregation into predictions; the app frames
  outputs as non-binding estimates with a disclaimer, processes photos only in memory, and
  models advertised market rent (not protected sitting-tenant rents). The central honest
  limitation — Switzerland-wide training data with only 27 Zurich rows — is stated throughout
  rather than hidden.

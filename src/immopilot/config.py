"""Central configuration: paths, seeds, model identifiers, hyperparameters.

Single source of truth — never hard-code paths or seeds elsewhere.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────── Paths ───────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = ROOT / "models"
RAG_DIR = MODELS_DIR / "rag"
DOCS_DIR = ROOT / "docs"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, IMAGES_DIR, MODELS_DIR, RAG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ────────────────────────── Reproducibility ──────────────────
SEED = int(os.getenv("SEED", "42"))


def set_global_seed(seed: int = SEED) -> None:
    """Pin every relevant RNG. Call at the top of every entry-point script."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ─────────────────────────── LLM ─────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ─────────────────────── Embeddings / RAG ────────────────────
EMBEDDING_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_MULTILINGUAL = "intfloat/multilingual-e5-base"
RAG_CHUNK_SIZE = 512
RAG_CHUNK_OVERLAP = 64
RAG_TOP_K = 5

# ─────────────────────────── CV ──────────────────────────────
CLIP_MODEL = "openai/clip-vit-base-patch32"
RESNET_NUM_CLASSES = 3
CONDITION_LABELS = ["modern", "standard", "needs_renovation"]
CV_IMAGE_SIZE = 224

# ─────────────────────── Numeric model ───────────────────────
NUMERIC_TARGET = "rent_chf"
LOG_TARGET = True
TEST_SIZE = 0.10
VAL_SIZE = 0.10
N_SPLITS_CV = 5

XGB_DEFAULT_PARAMS: dict = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

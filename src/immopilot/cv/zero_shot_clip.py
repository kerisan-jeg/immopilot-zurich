"""Zero-shot photo feature extraction with CLIP.

For each uploaded photo we compute soft scores against curated prompts:
condition (modern / standard / needs_renovation), balcony, view, kitchen quality.
These scores feed the numeric model as additional features.

This is the *zero-shot* baseline; the fine-tuned ResNet50 in
``train_classifier.py`` is the comparison model required by the assignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from immopilot import config

logger = logging.getLogger(__name__)


PROMPT_BANK: dict[str, list[str]] = {
    "condition": [
        "a photo of a modern, freshly renovated apartment interior",
        "a photo of a standard, ordinary apartment interior",
        "a photo of an old apartment interior that needs renovation",
    ],
    "balcony": [
        "a photo of an apartment with a balcony or terrace",
        "a photo of an apartment without a balcony",
    ],
    "view": [
        "a photo from an apartment with a view of a lake or mountains",
        "a photo from an apartment with no notable view",
    ],
    "kitchen_quality": [
        "a photo of a high-end, modern kitchen",
        "a photo of a basic kitchen",
        "a photo of an outdated kitchen",
    ],
}


@dataclass
class PhotoFeatures:
    condition_score: float  # 0=needs reno, 1=modern
    has_balcony: int
    has_view: int
    kitchen_quality: float  # 0=outdated, 1=high-end


@lru_cache(maxsize=1)
def _load_clip():
    logger.info("Loading CLIP model %s …", config.CLIP_MODEL)
    model = CLIPModel.from_pretrained(config.CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    model.eval()
    return model, processor


def _scores(image: Image.Image, prompts: list[str]) -> list[float]:
    model, processor = _load_clip()
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    probs = out.logits_per_image.softmax(dim=-1)[0].tolist()
    return probs


def extract_features(images: list[Image.Image]) -> PhotoFeatures:
    """Aggregate over multiple images by averaging probabilities."""
    if not images:
        return PhotoFeatures(0.5, 0, 0, 0.5)

    cond, bal, vw, kit = [], [], [], []
    for img in images:
        c = _scores(img, PROMPT_BANK["condition"])
        cond.append(1.0 * c[0] + 0.5 * c[1] + 0.0 * c[2])
        b = _scores(img, PROMPT_BANK["balcony"])
        bal.append(b[0])
        v = _scores(img, PROMPT_BANK["view"])
        vw.append(v[0])
        k = _scores(img, PROMPT_BANK["kitchen_quality"])
        kit.append(1.0 * k[0] + 0.5 * k[1] + 0.0 * k[2])

    return PhotoFeatures(
        condition_score=float(sum(cond) / len(cond)),
        has_balcony=int((sum(bal) / len(bal)) > 0.5),
        has_view=int((sum(vw) / len(vw)) > 0.5),
        kitchen_quality=float(sum(kit) / len(kit)),
    )

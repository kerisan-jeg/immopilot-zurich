"""Compare the fine-tuned ResNet50 against the zero-shot CLIP baseline.

Both models are evaluated on the SAME validation split
(data/images/val/<class>/) for the 3-class apartment-condition task. We report
accuracy, macro-F1, a per-class classification report, and a confusion matrix
for each model, plus a side-by-side comparison. Confusion matrices are saved as
PNGs for the documentation.

Why this matters: the assignment asks us to compare a trained model against a
zero-shot baseline. ResNet50 is fine-tuned on our (small) labelled set; CLIP
classifies the same images zero-shot via the condition prompts already used in
the app. Evaluating both on identical data makes the comparison fair.

Usage:  python scripts/eval_cv.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torchvision import models, transforms

from immopilot import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

VAL_DIR = config.IMAGES_DIR / "val"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
OUT_DIR = config.DOCS_DIR / "cv_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────── data ────────────────────────────
def load_val_set() -> tuple[list[Path], list[str], list[str]]:
    """Return (image_paths, true_labels, class_names) from the val folder."""
    classes = sorted([d.name for d in VAL_DIR.iterdir() if d.is_dir()])
    paths: list[Path] = []
    labels: list[str] = []
    for cls in classes:
        for p in sorted((VAL_DIR / cls).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(p)
                labels.append(cls)
    return paths, labels, classes


# ──────────────────────────── ResNet ────────────────────────────
def eval_resnet(paths: list[Path], classes: list[str]) -> list[str]:
    ckpt_path = config.MODELS_DIR / "resnet50_condition.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_classes = ckpt["classes"]  # the order the model was trained with

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(model_classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize(int(config.CV_IMAGE_SIZE * 1.15)),
            transforms.CenterCrop(config.CV_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    preds: list[str] = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = tf(img).unsqueeze(0)
            logit = model(x)
            idx = int(logit.argmax(dim=-1).item())
            preds.append(model_classes[idx])
    return preds


# ──────────────────────────── CLIP zero-shot ────────────────────────────
def eval_clip(paths: list[Path]) -> list[str]:
    """Zero-shot 3-class prediction via the condition prompts (argmax)."""
    from immopilot.cv.zero_shot_clip import PROMPT_BANK, _scores

    # PROMPT_BANK["condition"] order is [modern, standard, needs_renovation]
    prompt_classes = ["modern", "standard", "needs_renovation"]
    preds: list[str] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        probs = _scores(img, PROMPT_BANK["condition"])
        idx = int(np.argmax(probs))
        preds.append(prompt_classes[idx])
    return preds


# ──────────────────────────── reporting ────────────────────────────
def plot_cm(cm: np.ndarray, classes: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def report_for(name: str, y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    rep = classification_report(y_true, y_pred, labels=classes, target_names=classes,
                                output_dict=True, zero_division=0)
    plot_cm(cm, classes, f"{name} — confusion matrix", OUT_DIR / f"cm_{name.lower().replace(' ', '_')}.png")
    logger.info("%s | accuracy %.3f | macro-F1 %.3f", name, acc, f1)
    return {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm.tolist(), "report": rep}


def main() -> None:
    paths, y_true, classes = load_val_set()
    logger.info("Val set: %d images across %d classes %s", len(paths), len(classes), classes)

    resnet_pred = eval_resnet(paths, classes)
    clip_pred = eval_clip(paths)

    results = {
        "n_val": len(paths),
        "classes": classes,
        "resnet50_finetuned": report_for("ResNet50 fine-tuned", y_true, resnet_pred, classes),
        "clip_zero_shot": report_for("CLIP zero-shot", y_true, clip_pred, classes),
    }

    out_json = OUT_DIR / "cv_comparison.json"
    out_json.write_text(json.dumps(results, indent=2))

    # Console summary table
    r = results["resnet50_finetuned"]
    c = results["clip_zero_shot"]
    print("\n" + "=" * 52)
    print(f"{'Model':<26}{'Accuracy':>12}{'macro-F1':>12}")
    print("-" * 52)
    print(f"{'ResNet50 (fine-tuned)':<26}{r['accuracy']:>12.3f}{r['macro_f1']:>12.3f}")
    print(f"{'CLIP (zero-shot)':<26}{c['accuracy']:>12.3f}{c['macro_f1']:>12.3f}")
    print("=" * 52)
    print(f"\nSaved: {out_json}")
    print(f"Confusion matrices: {OUT_DIR}/cm_resnet50_fine-tuned.png, cm_clip_zero-shot.png")


if __name__ == "__main__":
    main()

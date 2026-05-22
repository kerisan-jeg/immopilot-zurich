"""Fine-tune a ResNet50 on the apartment condition classification task.

Expected directory layout in ``data/images/``::

    data/images/
        train/
            modern/        *.jpg
            standard/      *.jpg
            needs_renovation/  *.jpg
        val/
            ... (same structure)

Targets the ``CONDITION_LABELS`` from config (3 classes).
"""

from __future__ import annotations

import json
import logging

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from immopilot import config

logger = logging.getLogger(__name__)


def _build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(config.CV_IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(config.CV_IMAGE_SIZE * 1.15)),
            transforms.CenterCrop(config.CV_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def _build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    # Freeze all but layer4 + fc — sane default for ~500 images
    for name, p in model.named_parameters():
        p.requires_grad = "layer4" in name or "fc" in name
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main(epochs: int = 20, batch_size: int = 32, lr: float = 1e-4) -> None:
    config.set_global_seed()
    images_root = config.IMAGES_DIR
    if not (images_root / "train").exists():
        raise FileNotFoundError(
            f"Expected training images at {images_root / 'train'}. "
            "Drop labeled folders before running."
        )

    train_tf, val_tf = _build_transforms()
    train_ds = datasets.ImageFolder(images_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(images_root / "val", transform=val_tf)

    # Sanity-check label order matches config
    if train_ds.classes != config.CONDITION_LABELS:
        logger.warning(
            "Label order mismatch. Got %s, expected %s. Override config or rename folders.",
            train_ds.classes,
            config.CONDITION_LABELS,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(num_classes=len(train_ds.classes)).to(device)

    # Class-balanced loss
    counts = torch.tensor(
        [(torch.tensor(train_ds.targets) == c).sum().item() for c in range(len(train_ds.classes))],
        dtype=torch.float,
    )
    weights = (counts.sum() / counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    best_f1 = 0.0
    best_state: dict | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                y_true.extend(y.numpy().tolist())
                y_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        f1 = f1_score(y_true, y_pred, average="macro")
        logger.info("epoch %2d | val macro-F1 %.3f", epoch, f1)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Persist model + metadata
    out = config.MODELS_DIR / "resnet50_condition.pt"
    torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, out)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=train_ds.classes, output_dict=True)
    (config.MODELS_DIR / "resnet50_condition.metrics.json").write_text(
        json.dumps({"best_macro_f1": best_f1, "report": report, "confusion_matrix": cm.tolist()}, indent=2)
    )
    logger.info("Saved %s — best macro-F1 %.3f", out.name, best_f1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()

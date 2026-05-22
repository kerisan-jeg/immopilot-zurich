"""Create a reproducible validation split for the CV condition dataset.

Moves ~20% of each class's images from data/images/train/<cls>/ to
data/images/val/<cls>/. Seed-fixed (42) so the split is reproducible and
documented. Run once after all training images are collected.

Usage:  python scripts/make_val_split.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

SEED = 42
VAL_FRACTION = 0.20
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "images" / "train"
VAL_DIR = ROOT / "data" / "images" / "val"


def main() -> None:
    random.seed(SEED)
    if not TRAIN_DIR.exists():
        raise SystemExit(f"Not found: {TRAIN_DIR}")

    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    print(f"Classes: {classes}\n")

    total_moved = 0
    for cls in classes:
        src = TRAIN_DIR / cls
        dst = VAL_DIR / cls
        dst.mkdir(parents=True, exist_ok=True)

        imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS])
        n_val = max(1, round(len(imgs) * VAL_FRACTION))

        # Deterministic shuffle, then take the first n_val for validation
        random.shuffle(imgs)
        val_imgs = imgs[:n_val]

        for p in val_imgs:
            shutil.move(str(p), str(dst / p.name))

        remaining = len(list(src.glob("*")))
        print(f"{cls:18s} | train {remaining:3d} | val {n_val:3d}  (was {len(imgs)})")
        total_moved += n_val

    print(f"\nMoved {total_moved} images to val/. Split complete.")


if __name__ == "__main__":
    main()

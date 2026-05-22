"""Annotate the CV evaluation artifacts so the 1.00 vs 0.833 numbers are
documented rather than contradictory. Adds a human-readable _note to each JSON.
No metric values change.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CMP = ROOT / "docs" / "cv_eval" / "cv_comparison.json"
MET = ROOT / "models" / "resnet50_condition.metrics.json"

CMP_NOTE = (
    "ResNet50 accuracy/macro_f1 = 1.00 is the BEST-EPOCH score on the 18-image "
    "validation set, evaluated on the same set used for epoch selection "
    "(selection-on-validation). It is optimistic, not a held-out estimate. The "
    "final epoch of the same run scored ~0.833 (see "
    "models/resnet50_condition.metrics.json). With only 18 images (6/6/6) the "
    "true generalization F1 is uncertain in the ~0.83-1.00 range. The deployed "
    "app uses zero-shot CLIP. See DOCUMENTATION.md section 2C.5."
)
MET_NOTE = (
    "best_macro_f1 (1.0) is the best validation epoch; the report/confusion_matrix "
    "below are the FINAL epoch of the same run (acc 0.833, macro-F1 0.836). Both are "
    "on the same 18-image set. See DOCUMENTATION.md 2C.5."
)

def annotate(path, note):
    data = json.loads(path.read_text(encoding="utf-8"))
    new = {"_note": note}
    new.update(data)
    path.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"annotated {path.name}")

annotate(CMP, CMP_NOTE)
annotate(MET, MET_NOTE)
print("done")

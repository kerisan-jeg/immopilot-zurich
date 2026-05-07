"""Small PyTorch MLP regressor — the deep-learning entry in the comparison.

Kept intentionally simple: 3 hidden layers, ReLU, Adam, early stopping.
Goal is fair comparison, not deep-learning glory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from immopilot import config
from immopilot.models._common import EvalResult, load_features, make_splits, save_model, split_xy

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x).squeeze(-1)


@dataclass
class _SkLearnLikeWrapper:
    """Make a torch model behave like sklearn so save_model and inference are uniform."""

    model: MLP

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(X, dtype=np.float32))
            out = self.model(t).cpu().numpy()
        return out


def _train(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def _eval(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += criterion(model(xb), yb).item() * xb.size(0)
    return total / len(loader.dataset)


def main(epochs: int = 80, batch_size: int = 256, patience: int = 10) -> None:
    config.set_global_seed()
    df = load_features()
    X, y = split_xy(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(X, y)

    preprocessor = joblib.load(config.MODELS_DIR / "preprocessor.joblib")
    X_tr_t = preprocessor.transform(X_tr).astype(np.float32)
    X_va_t = preprocessor.transform(X_va).astype(np.float32)
    X_te_t = preprocessor.transform(X_te).astype(np.float32)
    y_tr_t = y_tr.astype(np.float32)
    y_va_t = y_va.astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_features=X_tr_t.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_t), torch.from_numpy(y_tr_t)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va_t), torch.from_numpy(y_va_t)),
        batch_size=batch_size,
    )

    best_val = float("inf")
    epochs_no_improve = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        tr = _train(model, train_loader, optimizer, criterion, device)
        va = _eval(model, val_loader, criterion, device)
        logger.info("epoch %3d | train %.4f | val %.4f", epoch, tr, va)
        if va < best_val - 1e-4:
            best_val = va
            epochs_no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    wrapped = _SkLearnLikeWrapper(model.cpu())
    pred = wrapped.predict(X_te_t)
    if config.LOG_TARGET:
        pred_inv = np.expm1(pred)
        true_inv = np.expm1(y_te)
    else:
        pred_inv, true_inv = pred, y_te

    result = EvalResult(
        model_name="mlp",
        test_mae=float(mean_absolute_error(true_inv, pred_inv)),
        test_rmse=float(np.sqrt(mean_squared_error(true_inv, pred_inv))),
        test_r2=float(r2_score(true_inv, pred_inv)),
        cv_mae_mean=float("nan"),
        cv_mae_std=float("nan"),
    )
    save_model(wrapped, "mlp", result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()

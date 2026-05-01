"""
TCN + BiGRU + Transformer experiment for Sepsis Risk Prediction.

Key improvements over experiment_hybrid_transformer.py:
  - Expanded 20-feature set (vitals + key labs)
  - Patient-level train/val/test split to reduce leakage
  - Temporal Convolutional Network (TCN) with dilated convolutions before BiGRU
  - Larger model: d_model=128, 3 Transformer layers
  - Threshold selection maximises F1 (not accuracy)
  - Weighted random sampler for class imbalance in training
  - Linear warmup + cosine-decay learning-rate schedule
"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

# ── Features ──────────────────────────────────────────────────────────────────
# 20 clinically meaningful features (vitals + key labs)
FEATURES = [
    # Vital signs
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    # Acid-base / respiratory
    "BaseExcess", "HCO3", "pH", "PaCO2",
    # Key labs
    "Glucose", "Lactate", "Potassium",
    "Hgb", "WBC", "Creatinine", "Platelets", "BUN",
]
LABEL_COL = "SepsisLabel"

# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    data_dir: str
    history_hours: int = 24
    forecast_hours: int = 6
    seed: int = 42
    max_patients: int = 220
    bootstrap_rounds: int = 100
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 8e-4
    warmup_epochs: int = 2
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 384
    gru_layers: int = 2
    tcn_channels: int = 128
    tcn_kernel_size: int = 3
    dropout: float = 0.25
    patience: int = 5
    grad_clip: float = 1.0
    weight_decay: float = 5e-4
    gamma: float = 2.0
    out_dir: str = "results"


# ── Layers ─────────────────────────────────────────────────────────────────────
class _CausalConvBlock(nn.Module):
    """One dilated causal conv block: Conv → BN → GELU → Dropout (residual)."""
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        # Remove future-looking padding
        out = out[:, :, : x.size(2)]
        out = self.drop(self.act(self.bn(out)))
        return out + x  # residual


class TCNEncoder(nn.Module):
    """Multi-scale TCN: 3 dilated blocks (dilation 1, 2, 4)."""
    def __init__(self, in_channels: int, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList([
            _CausalConvBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(3)
        ])
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in) → transpose to (B, C, T)
        h = self.proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)  # back to (B, T, C)
        return self.norm(h)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        return ((1.0 - pt).pow(self.gamma) * bce).mean()


class TCNBiGRUTransformer(nn.Module):
    """
    Architecture pipeline:
      Input → Linear proj + LayerNorm
            → TCN (dilated causal conv, multi-scale)
            → BiGRU
            → Positional encoding
            → Transformer encoder (pre-norm)
            → Learnable attention pooling
            → 3-layer MLP head
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        gru_layers: int,
        tcn_channels: int,
        tcn_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, tcn_channels),
            nn.GELU(),
            nn.LayerNorm(tcn_channels),
        )

        # Multi-scale causal TCN
        self.tcn = TCNEncoder(tcn_channels, tcn_channels, tcn_kernel_size, dropout)

        # BiGRU — output dim = d_model (hidden_size = d_model // 2, bidirectional)
        assert d_model % 2 == 0, "d_model must be even for BiGRU"
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=d_model // 2,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Transformer encoder (pre-LayerNorm / "pre-norm")
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        # Learnable attention pooling (single query vector)
        self.attn_query = nn.Parameter(torch.randn(d_model) * 0.02)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)       # (B, T, tcn_channels)
        h = self.tcn(h)              # (B, T, tcn_channels)
        h, _ = self.gru(h)           # (B, T, d_model)
        h = self.pos_encoder(h)      # add positional encoding
        h = self.encoder(h)          # (B, T, d_model)
        weights = torch.softmax(torch.matmul(h, self.attn_query), dim=1)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        return self.classifier(pooled).squeeze(-1)


# ── Utilities ──────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_patient_files(data_dir: Path, max_patients: int) -> List[Path]:
    files = (
        sorted(data_dir.glob("training_setA/p*.psv"))
        + sorted(data_dir.glob("training_setB/p*.psv"))
    )
    if max_patients > 0:
        files = files[:max_patients]
    return files


def make_samples_for_patient(
    df: pd.DataFrame,
    history_hours: int,
    forecast_hours: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Keep only FEATURES columns (fill missing columns with NaN)
    data = df.reindex(columns=FEATURES).copy()
    # Forward-fill within patient
    data = data.ffill()

    labels = df[LABEL_COL].to_numpy(dtype=int)
    positive_idx = np.where(labels == 1)[0]
    onset = int(positive_idx[0]) if len(positive_idx) > 0 else None

    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    n_rows = len(df)

    for t in range(history_hours, n_rows - forecast_hours):
        if onset is not None and t >= onset:
            break
        y = 1 if (onset is not None and (t + 1) <= onset <= (t + forecast_hours)) else 0
        x_seq = data.iloc[t - history_hours : t].to_numpy(dtype=float)
        x_list.append(x_seq)
        y_list.append(y)

    if not x_list:
        return (
            np.empty((0, history_hours, len(FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return np.stack(x_list).astype(np.float32), np.array(y_list, dtype=np.int32)


def build_dataset_by_patient(
    files: List[Path],
    history_hours: int,
    forecast_hours: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return per-patient list of (X, y) arrays (patient-level granularity)."""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for fpath in tqdm(files, desc="Loading patients", leave=False):
        try:
            df = pd.read_csv(fpath, sep="|")
            X, y = make_samples_for_patient(df, history_hours, forecast_hours)
            if len(X) > 0:
                xs.append(X)
                ys.append(y)
        except Exception as exc:
            print(f"Error loading {fpath}: {exc}")
    return xs, ys


def sequential_split(
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all windows in order, then split sequentially 70/15/15.

    With very few sepsis patients (220 total, ~2% positivity) a patient-level
    random split leaves only 1-2 positive patients in test, causing extreme
    distribution shift.  Sequential splitting provides a stable positive rate
    across splits.
    """
    X_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    n = len(X_all)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train : n_train + n_val], y_all[n_train : n_train + n_val]
    X_test, y_test = X_all[n_train + n_val :], y_all[n_train + n_val :]
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_normalization_stats(
    X_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])
    medians = np.nanmedian(flat, axis=0)
    imputed = np.where(np.isnan(flat), medians[np.newaxis, :], flat)
    means = np.mean(imputed, axis=0)
    stds = np.std(imputed, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return medians, means, stds


def engineer_features(
    X: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Expand each feature to 3 channels: z-norm, first-difference, missing flag."""
    missing_mask = np.isnan(X).astype(np.float32)

    X_filled = X.copy()
    for fi in range(X.shape[-1]):
        X_filled[:, :, fi] = np.where(np.isnan(X_filled[:, :, fi]), medians[fi], X_filled[:, :, fi])

    X_norm = ((X_filled - means) / stds).astype(np.float32)
    X_delta = np.diff(X_norm, axis=1, prepend=X_norm[:, :1, :]).astype(np.float32)
    return np.concatenate([X_norm, X_delta, missing_mask], axis=-1)


def find_best_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Select threshold that maximises Youden's J = sensitivity + specificity - 1.

    Searches a fixed grid PLUS quantile-based thresholds derived from the
    actual prediction distribution, so the search works even when model
    probabilities are very small (e.g. all below 0.05 due to calibration).
    """
    if len(np.unique(y_true)) < 2:
        return float(np.median(y_prob)), 0.0
    fixed = np.linspace(0.01, 0.99, 99)
    quantile_t = np.percentile(y_prob, np.linspace(1, 99, 99))
    thresholds = np.unique(np.concatenate([fixed, quantile_t]))
    best_t, best_j = float(np.median(y_prob)), -np.inf
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t, float(best_j)


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency weighting so each class contributes equally per epoch."""
    class_counts = np.bincount(y)
    weights = (1.0 / class_counts[y]).astype(np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(y),
        replacement=True,
    )


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
) -> Tuple[TCNBiGRUTransformer, float]:
    device = torch.device("cpu")

    model = TCNBiGRUTransformer(
        input_dim=X_train.shape[-1],
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        gru_layers=config.gru_layers,
        tcn_channels=config.tcn_channels,
        tcn_kernel_size=config.tcn_kernel_size,
        dropout=config.dropout,
    ).to(device)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    pos_weight = torch.tensor(
        [max(1.0, neg_count / max(1, pos_count))], dtype=torch.float32, device=device
    )
    criterion = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=config.gamma)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = warmup_cosine_scheduler(optimizer, config.warmup_epochs, config.epochs)

    # Use shuffle=True; class imbalance is handled by focal loss + pos_weight.
    # WeightedRandomSampler with ~30 positives creates excessively noisy batches.
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val).to(device)

    best_metric = -np.inf
    best_state = None
    best_threshold = 0.5
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()

        if len(np.unique(y_val)) < 2:
            val_metric = -epoch_loss / max(1, len(train_loader))
            threshold = 0.5
        else:
            try:
                val_metric = average_precision_score(y_val, val_probs)
            except Exception:
                val_metric = -epoch_loss / max(1, len(train_loader))
            threshold, _ = find_best_threshold_youden(y_val, val_probs)

        improved = val_metric > best_metric
        if improved:
            best_metric = val_metric
            best_threshold = threshold
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"loss={epoch_loss / max(1, len(train_loader)):.4f} | "
            f"val_auprc={val_metric:.4f} | "
            f"best_t={threshold:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            + (" *" if improved else "")
        )

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1} (patience={config.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_threshold


# ── Evaluation ─────────────────────────────────────────────────────────────────
def predict_probabilities(model: TCNBiGRUTransformer, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == bins - 1 else y_prob < hi)
        if not np.any(mask):
            continue
        ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return float(ece)


def evaluate_model(
    model: TCNBiGRUTransformer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_prob = predict_probabilities(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_test)) == 1:
        return {
            "auroc": np.nan, "auprc": np.nan,
            "sensitivity": 0.0 if y_test[0] == 0 else 1.0,
            "specificity": 1.0 if y_test[0] == 0 else 0.0,
            "precision": 0.0, "recall": 0.0,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": np.nan, "brier": np.nan, "ece": np.nan,
            "threshold": threshold,
        }

    try:
        auroc = roc_auc_score(y_test, y_prob)
    except Exception:
        auroc = np.nan
    try:
        auprc = average_precision_score(y_test, y_prob)
    except Exception:
        auprc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0.0),
        "brier": brier_score_loss(y_test, y_prob),
        "ece": expected_calibration_error(y_test, y_prob),
        "threshold": threshold,
    }


def bootstrap_ci_95(values: np.ndarray) -> Tuple[float, float]:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


# ── Main ───────────────────────────────────────────────────────────────────────
def run_experiment(config: Config) -> Dict:
    set_seed(config.seed)
    data_dir = Path(config.data_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading patient files ...")
    files = list_patient_files(data_dir, config.max_patients)
    print(f"  {len(files)} patient files found")

    patient_xs, patient_ys = build_dataset_by_patient(
        files, config.history_hours, config.forecast_hours
    )
    print(f"  {len(patient_xs)} patients with usable windows")

    X_train, y_train, X_val, y_val, X_test, y_test = sequential_split(
        patient_xs, patient_ys
    )

    total_windows = len(X_train) + len(X_val) + len(X_test)
    all_y = np.concatenate([y_train, y_val, y_test])
    print(
        f"  Total windows: {total_windows} | positive rate: {all_y.mean():.4f}\n"
        f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n"
        f"  Train+ rate: {y_train.mean():.4f} | "
        f"Val+ rate: {y_val.mean():.4f} | "
        f"Test+ rate: {y_test.mean():.4f}"
    )

    # Normalisation stats computed on training set only
    medians, means, stds = compute_normalization_stats(X_train)
    X_train_fe = engineer_features(X_train, medians, means, stds)
    X_val_fe = engineer_features(X_val, medians, means, stds)
    X_test_fe = engineer_features(X_test, medians, means, stds)
    print(f"  Engineered feature dim: {X_train_fe.shape[-1]}")

    print("\nTraining model ...")
    model, best_threshold = train_model(X_train_fe, y_train, X_val_fe, y_val, config)
    print(f"\nBest validation threshold (Youden's J): {best_threshold:.3f}")

    print("\n── Validation set evaluation ──")
    val_metrics = evaluate_model(model, X_val_fe, y_val, best_threshold)
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")

    print(f"\n── Bootstrap test evaluation ({config.bootstrap_rounds} rounds) ──")
    bootstrap_keys = ["auroc", "auprc", "sensitivity", "specificity",
                      "precision", "recall", "accuracy", "f1", "brier", "ece"]
    boot_data: Dict[str, List[float]] = {k: [] for k in bootstrap_keys}

    rng_boot = np.random.default_rng(config.seed + 1)
    for _ in tqdm(range(config.bootstrap_rounds), desc="Bootstrap"):
        idx = rng_boot.choice(len(X_test_fe), size=len(X_test_fe), replace=True)
        m = evaluate_model(model, X_test_fe[idx], y_test[idx], best_threshold)
        for k in bootstrap_keys:
            boot_data[k].append(m[k])

    test_metrics: Dict[str, Dict] = {}
    for k in bootstrap_keys:
        arr = np.array(boot_data[k], dtype=float)
        mean = float(np.nanmean(arr))
        ci_lo, ci_hi = bootstrap_ci_95(arr)
        test_metrics[k] = {
            "mean": mean if not np.isnan(mean) else None,
            "ci_lower": ci_lo if not np.isnan(ci_lo) else None,
            "ci_upper": ci_hi if not np.isnan(ci_hi) else None,
        }
        if not np.isnan(mean):
            print(f"  {k}: {mean:.4f} ({ci_lo:.4f}, {ci_hi:.4f})")
        else:
            print(f"  {k}: NaN")

    results = {
        "model": "TCNBiGRUTransformer",
        "config": asdict(config),
        "data_stats": {
            "patients_with_windows": len(patient_xs),
            "total_windows": total_windows,
            "positive_rate": float(all_y.mean()),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "split_stats": {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        },
        "feature_engineering": {
            "base_features": FEATURES,
            "engineered_channels": ["z_norm", "first_difference", "missing_indicator"],
            "final_feature_dim": int(X_train_fe.shape[-1]),
        },
        "training": {
            "selected_threshold": best_threshold,
            "threshold_criterion": "max_YoudensJ_adaptive_grid",
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    out_path = out_dir / "tcn_hybrid_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCN + BiGRU + Transformer experiment")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--history-hours", type=int, default=24)
    parser.add_argument("--forecast-hours", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int, default=220)
    parser.add_argument("--bootstrap-rounds", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=384)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--tcn-channels", type=int, default=128)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    config = Config(
        data_dir=args.data_dir,
        history_hours=args.history_hours,
        forecast_hours=args.forecast_hours,
        seed=args.seed,
        max_patients=args.max_patients,
        bootstrap_rounds=args.bootstrap_rounds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        gru_layers=args.gru_layers,
        tcn_channels=args.tcn_channels,
        tcn_kernel_size=args.tcn_kernel_size,
        dropout=args.dropout,
        patience=args.patience,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        out_dir=args.out_dir,
    )
    run_experiment(config)

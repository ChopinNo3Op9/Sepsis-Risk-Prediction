"""
Hybrid BiGRU + Transformer experiment for Sepsis Risk Prediction.
Adds temporal feature engineering, focal loss, and validation threshold tuning.
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

FEATURES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2"]
LABEL_COL = "SepsisLabel"


@dataclass
class Config:
    data_dir: str
    history_hours: int
    forecast_hours: int
    seed: int
    max_patients: int
    bootstrap_rounds: int
    batch_size: int
    epochs: int
    learning_rate: float
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    gru_layers: int
    dropout: float
    patience: int
    grad_clip: float
    weight_decay: float
    gamma: float
    out_dir: str = "results"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
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
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal_factor = (1.0 - pt).pow(self.gamma)
        return (focal_factor * bce).mean()


class HybridSepsisClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        gru_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
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
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.attn_query = nn.Parameter(torch.randn(d_model) * 0.02)
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
        h = self.input_proj(x)
        h, _ = self.gru(h)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        weights = torch.softmax(torch.matmul(h, self.attn_query), dim=1)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_patient_files(data_dir: Path, max_patients: int) -> List[Path]:
    files = sorted(data_dir.glob("training_setA/p*.psv")) + sorted(data_dir.glob("training_setB/p*.psv"))
    if max_patients > 0:
        files = files[:max_patients]
    return files


def make_samples_for_patient(
    df: pd.DataFrame,
    history_hours: int,
    forecast_hours: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = df[FEATURES].copy()
    data = data.ffill()

    labels = df[LABEL_COL].to_numpy(dtype=int)
    positive_idx = np.where(labels == 1)[0]
    onset = int(positive_idx[0]) if len(positive_idx) > 0 else None

    x_list = []
    y_list = []
    n_rows = len(df)

    for t in range(history_hours, n_rows - forecast_hours):
        if onset is not None and t >= onset:
            break

        y = 0
        if onset is not None and (t + 1) <= onset <= (t + forecast_hours):
            y = 1

        x_seq = data.iloc[t - history_hours : t].to_numpy(dtype=float)
        x_list.append(x_seq)
        y_list.append(y)

    if not x_list:
        return np.empty((0, history_hours, len(FEATURES))), np.empty((0,), dtype=int)

    return np.stack(x_list), np.array(y_list, dtype=int)


def build_dataset(
    files: List[Path],
    history_hours: int,
    forecast_hours: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_all = []
    y_all = []

    for fpath in tqdm(files, desc="Loading patients", leave=False):
        try:
            df = pd.read_csv(fpath, sep="|")
            X, y = make_samples_for_patient(df, history_hours, forecast_hours)
            if len(X) > 0:
                x_all.append(X)
                y_all.append(y)
        except Exception as exc:
            print(f"Error loading {fpath}: {exc}")

    if not x_all:
        return np.empty((0, history_hours, len(FEATURES))), np.empty((0,), dtype=int)

    X = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y


def compute_normalization_stats(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])
    medians = np.nanmedian(flat, axis=0)
    imputed = np.where(np.isnan(flat), medians, flat)
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
    missing_mask = np.isnan(X).astype(np.float32)

    X_filled = X.copy()
    for feature_idx in range(X.shape[-1]):
        X_filled[:, :, feature_idx] = np.where(
            np.isnan(X_filled[:, :, feature_idx]),
            medians[feature_idx],
            X_filled[:, :, feature_idx],
        )

    X_norm = (X_filled - means) / stds
    X_delta = np.diff(X_norm, axis=1, prepend=X_norm[:, :1, :])

    stacked = np.concatenate([X_norm, X_delta, missing_mask], axis=-1)
    return stacked.astype(np.float32)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    thresholds = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_acc = -1.0
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # Prefer higher accuracy, then use F1 to break ties.
        if (acc > best_acc) or (np.isclose(acc, best_acc) and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_acc)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
) -> Tuple[HybridSepsisClassifier, float]:
    device = torch.device("cpu")
    model = HybridSepsisClassifier(
        input_dim=X_train.shape[-1],
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        gru_layers=config.gru_layers,
        dropout=config.dropout,
    )
    model.to(device)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    pos_weight = torch.tensor([max(1.0, neg_count / max(1, pos_count))], dtype=torch.float32, device=device)

    criterion = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=config.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(2, config.epochs))

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    best_metric = -np.inf
    best_state = None
    patience_counter = 0
    best_threshold = 0.5

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

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
            val_logits = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        if len(np.unique(y_val)) < 2:
            val_metric = -epoch_loss
            threshold = 0.5
        else:
            try:
                val_metric = average_precision_score(y_val, val_probs)
            except Exception:
                val_metric = -epoch_loss
            threshold, _ = find_best_threshold(y_val, val_probs)

        improved = val_metric > best_metric
        if improved:
            best_metric = val_metric
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"loss={epoch_loss / max(1, len(train_loader)):.4f} - "
            f"val_auprc={val_metric:.4f} - "
            f"best_t={threshold:.3f} - "
            f"lr={lr_now:.6f}"
            f"{' *' if improved else ''}"
        )

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1} (patience={config.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_threshold


def predict_probabilities(model: HybridSepsisClassifier, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i < bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def evaluate_model(
    model: HybridSepsisClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred_proba = predict_probabilities(model, X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)

    if len(np.unique(y_test)) == 1:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "sensitivity": 0.0 if y_test[0] == 0 else 1.0,
            "specificity": 1.0 if y_test[0] == 0 else 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 1.0 if y_test[0] == 0 else 0.0,
            "f1": np.nan,
            "brier": np.nan,
            "ece": np.nan,
            "threshold": threshold,
        }

    try:
        auroc = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_test, y_pred_proba)
    except Exception:
        auprc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0.0)
    brier = brier_score_loss(y_test, y_pred_proba)
    ece = expected_calibration_error(y_test, y_pred_proba)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "brier": brier,
        "ece": ece,
        "threshold": threshold,
    }


def bootstrap_ci_95(metric_values: np.ndarray) -> Tuple[float, float]:
    valid = metric_values[~np.isnan(metric_values)]
    if len(valid) == 0:
        return np.nan, np.nan
    return np.percentile(valid, 2.5), np.percentile(valid, 97.5)


def run_experiment(config: Config) -> Dict:
    set_seed(config.seed)

    data_dir = Path(config.data_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    files = list_patient_files(data_dir, config.max_patients)
    print(f"Processing {len(files)} patient files")

    X, y = build_dataset(files, config.history_hours, config.forecast_hours)
    print(f"Dataset shape: {X.shape}, Positive rate: {y.mean():.4f}")

    split_point_1 = int(0.7 * len(X))
    split_point_2 = int(0.85 * len(X))

    X_train = X[:split_point_1]
    y_train = y[:split_point_1]
    X_val = X[split_point_1:split_point_2]
    y_val = y[split_point_1:split_point_2]
    X_test = X[split_point_2:]
    y_test = y[split_point_2:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(
        f"Train+ rate: {y_train.mean():.4f}, Val+ rate: {y_val.mean():.4f}, Test+ rate: {y_test.mean():.4f}"
    )

    medians, means, stds = compute_normalization_stats(X_train)
    X_train_feat = engineer_features(X_train, medians, means, stds)
    X_val_feat = engineer_features(X_val, medians, means, stds)
    X_test_feat = engineer_features(X_test, medians, means, stds)

    model, best_threshold = train_model(X_train_feat, y_train, X_val_feat, y_val, config)

    print("\nValidation Set Evaluation:")
    val_metrics = evaluate_model(model, X_val_feat, y_val, best_threshold)
    for key, val in val_metrics.items():
        print(f"  {key}: {val:.4f}" if not np.isnan(val) else f"  {key}: NaN")

    print(f"\nBootstrap Test Evaluation ({config.bootstrap_rounds} rounds)...")
    bootstrap_metrics = {
        "auroc": [],
        "auprc": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "recall": [],
        "accuracy": [],
        "f1": [],
        "brier": [],
        "ece": [],
    }

    for _ in tqdm(range(config.bootstrap_rounds), desc="Bootstrap"):
        indices = np.random.choice(len(X_test_feat), size=len(X_test_feat), replace=True)
        X_test_boot = X_test_feat[indices]
        y_test_boot = y_test[indices]

        metrics = evaluate_model(model, X_test_boot, y_test_boot, best_threshold)
        for key in bootstrap_metrics:
            bootstrap_metrics[key].append(metrics[key])

    results = {
        "model": "HybridBiGRUTransformer",
        "config": asdict(config),
        "data_stats": {
            "total_windows": len(X),
            "positive_rate": float(y.mean()),
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
            "engineered_channels": ["normalized_values", "first_difference", "missing_indicator"],
            "final_feature_dim": int(X_train_feat.shape[-1]),
        },
        "training": {
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_layers": config.num_layers,
            "dim_feedforward": config.dim_feedforward,
            "gru_layers": config.gru_layers,
            "dropout": config.dropout,
            "patience": config.patience,
            "grad_clip": config.grad_clip,
            "weight_decay": config.weight_decay,
            "focal_gamma": config.gamma,
            "selected_threshold": best_threshold,
        },
        "validation_metrics": val_metrics,
        "test_metrics": {},
    }

    for metric_key, values in bootstrap_metrics.items():
        arr = np.array(values)
        mean = np.nanmean(arr)
        ci_low, ci_high = bootstrap_ci_95(arr)
        results["test_metrics"][metric_key] = {
            "mean": float(mean) if not np.isnan(mean) else None,
            "ci_lower": float(ci_low) if not np.isnan(ci_low) else None,
            "ci_upper": float(ci_high) if not np.isnan(ci_high) else None,
        }
        print(
            f"{metric_key}: {mean:.4f} ({ci_low:.4f}, {ci_high:.4f})"
            if not np.isnan(mean)
            else f"{metric_key}: NaN"
        )

    results_json = out_dir / "hybrid_transformer_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid BiGRU+Transformer experiment for Sepsis Risk Prediction")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--history-hours", type=int, default=24, help="History window in hours")
    parser.add_argument("--forecast-hours", type=int, default=6, help="Forecast horizon in hours")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-patients", type=int, default=220, help="Max patients to use (0=all)")
    parser.add_argument("--bootstrap-rounds", type=int, default=100, help="Bootstrap rounds")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=14, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.0008, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=96, help="Hidden dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer encoder layers")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--gru-layers", type=int, default=2, help="BiGRU layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW weight decay")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory")

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
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        patience=args.patience,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        out_dir=args.out_dir,
    )

    run_experiment(config)

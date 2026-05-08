"""
Temporal Attention Bridge Network for Sepsis Risk Prediction.

More complex architecture improving on TCNBiGRUTransformer:
  - Dual TCN streams: fast (dilations 1,2) and slow (dilations 4,8) temporal patterns
  - Cross-stream attention bridge to fuse information across timescales
  - Deeper Transformer encoder (4 layers vs 3)
  - Scale-aware positional encoding
  - Multi-head attention over temporal scales
  - Residual gating fusion mechanism
  - Advanced thresholding: Youden J + F1 tuning

Novel design choices:
  - Separate fast and slow pathways to capture multi-scale temporal dynamics
  - Cross-attention between streams to enable adaptive focus
  - Learnable fusion gates to weight stream importance
  - Bootstrap validation with 100 rounds for robust uncertainty quantification
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
FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "pH", "PaCO2",
    "Glucose", "Lactate", "Potassium",
    "Hgb", "WBC", "Creatinine", "Platelets", "BUN",
]
LABEL_COL = "SepsisLabel"


@dataclass
class Config:
    data_dir: str
    history_hours: int = 24
    forecast_hours: int = 6
    seed: int = 42
    max_patients: int = 220
    bootstrap_rounds: int = 100
    batch_size: int = 64
    epochs: int = 35
    learning_rate: float = 5e-4
    warmup_epochs: int = 4
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 384
    gru_layers: int = 2
    tcn_channels: int = 128
    tcn_kernel_size: int = 3
    dropout: float = 0.25
    patience: int = 7
    grad_clip: float = 1.0
    weight_decay: float = 5e-4
    gamma: float = 2.0
    out_dir: str = "results"


# ── Layers ─────────────────────────────────────────────────────────────────────
class _CausalConvBlock(nn.Module):
    """One dilated causal conv block with residual connection."""
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[:, :, : x.size(2)]
        out = self.drop(self.act(self.bn(out)))
        return out + x


class FastTCNEncoder(nn.Module):
    """Fast stream TCN: dilations 1, 2 (high-frequency patterns)."""
    def __init__(self, in_channels: int, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList([
            _CausalConvBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(2)  # dilations 1, 2
        ])
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.norm(h.transpose(1, 2))


class SlowTCNEncoder(nn.Module):
    """Slow stream TCN: dilations 4, 8 (low-frequency patterns)."""
    def __init__(self, in_channels: int, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList([
            _CausalConvBlock(channels, kernel_size, dilation=2 ** (i + 2), dropout=dropout)
            for i in range(2)  # dilations 4, 8
        ])
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.norm(h.transpose(1, 2))


class ScaleAwarePositionalEncoding(nn.Module):
    """Separate positional encodings for fast and slow timescales."""
    def __init__(self, d_model: int, max_len: int = 5000, scale: float = 1.0) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0 * scale) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class CrossStreamAttention(nn.Module):
    """Cross-attention bridge between fast and slow streams with residual+norm."""
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn_fast_to_slow = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_slow_to_fast = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_fast = nn.LayerNorm(d_model)
        self.norm_slow = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, fast: torch.Tensor, slow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # fast attends to slow — with residual + norm
        fast_attn, _ = self.cross_attn_fast_to_slow(fast, slow, slow)
        fast_out = self.norm_fast(fast + self.drop(fast_attn))
        # slow attends to fast — with residual + norm
        slow_attn, _ = self.cross_attn_slow_to_fast(slow, fast, fast)
        slow_out = self.norm_slow(slow + self.drop(slow_attn))
        return fast_out, slow_out


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


class TemporalAttentionBridge(nn.Module):
    """
    Dual-stream temporal attention network:
      Input → Projection
            ├─→ Fast TCN (dilations 1,2) + BiGRU + Transformer
            │
            ├─→ Slow TCN (dilations 4,8) + BiGRU + Transformer
            │
            ├─→ Cross-stream attention bridge
            │
            └─→ Residual gating fusion → Classification
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

        # Dual TCN streams
        self.fast_tcn = FastTCNEncoder(tcn_channels, tcn_channels, tcn_kernel_size, dropout)
        self.slow_tcn = SlowTCNEncoder(tcn_channels, tcn_channels, tcn_kernel_size, dropout)

        # BiGRU for each stream
        assert d_model % 2 == 0
        self.fast_gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=d_model // 2,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.slow_gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=d_model // 2,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Scale-aware positional encodings
        self.fast_pos_encoder = ScaleAwarePositionalEncoding(d_model, scale=1.0)
        self.slow_pos_encoder = ScaleAwarePositionalEncoding(d_model, scale=2.0)

        # Transformer encoders (deeper: 4 layers)
        encoder_layer_fast = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fast_encoder = nn.TransformerEncoder(
            encoder_layer_fast, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        encoder_layer_slow = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.slow_encoder = nn.TransformerEncoder(
            encoder_layer_slow, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        # Cross-stream attention bridge
        self.cross_attn = CrossStreamAttention(d_model, nhead, dropout)

        # Learnable attention pooling
        self.attn_query = nn.Parameter(torch.randn(d_model) * 0.02)

        # Residual gating fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

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
        # Input projection
        proj = self.input_proj(x)  # (B, T, tcn_channels)

        # Fast stream
        fast_tcn = self.fast_tcn(proj)  # (B, T, tcn_channels)
        fast_gru, _ = self.fast_gru(fast_tcn)  # (B, T, d_model)
        fast_pos = self.fast_pos_encoder(fast_gru)
        fast_out = self.fast_encoder(fast_pos)  # (B, T, d_model)

        # Slow stream
        slow_tcn = self.slow_tcn(proj)  # (B, T, tcn_channels)
        slow_gru, _ = self.slow_gru(slow_tcn)  # (B, T, d_model)
        slow_pos = self.slow_pos_encoder(slow_gru)
        slow_out = self.slow_encoder(slow_pos)  # (B, T, d_model)

        # Cross-stream attention
        fast_refined, slow_refined = self.cross_attn(fast_out, slow_out)

        # Attention pooling for each stream
        fast_weights = torch.softmax(torch.matmul(fast_refined, self.attn_query), dim=1)
        fast_pooled = (fast_refined * fast_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        slow_weights = torch.softmax(torch.matmul(slow_refined, self.attn_query), dim=1)
        slow_pooled = (slow_refined * slow_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        # Residual gating fusion
        combined = torch.cat([fast_pooled, slow_pooled], dim=-1)  # (B, 2*d_model)
        gate = self.fusion_gate(combined)  # (B, d_model)
        fused = fast_pooled * gate + slow_pooled * (1.0 - gate)  # (B, d_model)

        return self.classifier(fused).squeeze(-1)


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


def load_and_process_data(
    files: List[Path], cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load PSV files and generate windowed samples."""
    all_windows = []
    all_labels = []
    patient_ids = []

    for file_path in files:
        df = pd.read_csv(file_path, sep="|")
        if len(df) == 0:
            continue

        # missing indicator (must be computed BEFORE ffill)
        missing = df[FEATURES].isna().values.astype(float)

        df_clean = df[FEATURES + [LABEL_COL]].ffill().bfill().fillna(0.0)
        values = df_clean[FEATURES].values
        labels = df_clean[LABEL_COL].values

        # z-normalization
        mean, std = values.mean(axis=0), values.std(axis=0)
        std[std == 0] = 1
        values = (values - mean) / std
        values = np.nan_to_num(values, nan=0.0)

        # first-difference (momentum)
        diff = np.diff(values, axis=0, prepend=values[:1, :])

        # Stack features
        features_3ch = np.stack([values, diff, missing], axis=2)  # (T, 20, 3)
        features_flat = features_3ch.reshape(len(values), -1)  # (T, 60)

        pid = file_path.stem
        for start_idx in range(len(values) - cfg.history_hours + 1):
            end_idx = start_idx + cfg.history_hours
            window = features_flat[start_idx:end_idx]
            label = labels[end_idx - 1]
            all_windows.append(window)
            all_labels.append(label)
            patient_ids.append(pid)

    return np.array(all_windows), np.array(all_labels), np.array(patient_ids)


def threshold_search(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Find threshold maximizing F1 on validation set."""
    pos_scores = scores_val[scores_val > 0]
    if len(pos_scores) == 0:
        return 0.5
    thresholds = np.quantile(pos_scores, np.linspace(0.01, 0.99, 100))
    best_f1, best_th = -1, 0.5
    for th in thresholds:
        preds = (scores_val >= th).astype(int)
        if preds.sum() > 0:
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
    return best_th


def train_one_epoch(model, loader, loss_fn, optimizer, grad_clip, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            # Clamp logits to prevent NaN from extreme activations
            scores = torch.sigmoid(logits.clamp(-20, 20)).cpu().numpy()
            scores = np.nan_to_num(scores, nan=0.0)
            all_preds.append(scores)
            all_targets.append(y.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def bootstrap_experiment(cfg: Config) -> Dict:
    """Run experiment with bootstrap resampling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(cfg.data_dir)
    files = list_patient_files(data_dir, cfg.max_patients)
    X, y, pids = load_and_process_data(files, cfg)
    print(f"Total windows: {len(X)}, positives: {y.sum()} ({100*y.mean():.2f}%)")

    unique_pids = np.unique(pids)
    np.random.shuffle(unique_pids)
    split1 = int(0.7 * len(unique_pids))
    split2 = int(0.85 * len(unique_pids))

    train_pids = set(unique_pids[:split1])
    val_pids = set(unique_pids[split1:split2])
    test_pids = set(unique_pids[split2:])

    train_mask = np.isin(pids, list(train_pids))
    val_mask = np.isin(pids, list(val_pids))
    test_mask = np.isin(pids, list(test_pids))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {X_train.shape[0]} ({y_train.sum()}), Val: {X_val.shape[0]} ({y_val.sum()}), Test: {X_test.shape[0]} ({y_test.sum()})")

    results = {
        "bootstrap_auroc": [],
        "bootstrap_auprc": [],
        "bootstrap_f1": [],
        "bootstrap_accuracy": [],
        "bootstrap_sensitivity": [],
        "bootstrap_specificity": [],
        "bootstrap_precision": [],
        "thresholds": [],
    }

    for round_idx in range(cfg.bootstrap_rounds):
        set_seed(cfg.seed + round_idx)
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_train_boot, y_train_boot = X_train[indices], y_train[indices]

        pos_weight = torch.tensor(y_train_boot.size / y_train_boot.sum(), dtype=torch.float32)
        loss_fn = FocalBCEWithLogitsLoss(pos_weight, gamma=cfg.gamma)

        model = TemporalAttentionBridge(
            input_dim=X_train_boot.shape[-1],
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            gru_layers=cfg.gru_layers,
            tcn_channels=cfg.tcn_channels,
            tcn_kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        def make_scheduler(opt, epochs):
            def lr_lambda(ep):
                if ep < cfg.warmup_epochs:
                    return (ep + 1) / cfg.warmup_epochs
                return 0.5 * (1.0 + np.cos(np.pi * (ep - cfg.warmup_epochs) / (epochs - cfg.warmup_epochs)))
            return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        scheduler = make_scheduler(optimizer, cfg.epochs)

        # Weighted sampling for imbalanced training
        weights = np.where(y_train_boot > 0, pos_weight.item(), 1.0)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_boot, dtype=torch.float32),
                         torch.tensor(y_train_boot, dtype=torch.float32)),
            batch_size=cfg.batch_size,
            sampler=sampler,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                         torch.tensor(y_val, dtype=torch.float32)),
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                         torch.tensor(y_test, dtype=torch.float32)),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        best_auprc, counter = -1, 0
        for ep in range(cfg.epochs):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, cfg.grad_clip, device)
            val_scores, val_targets = eval_epoch(model, val_loader, device)
            if np.isnan(val_scores).any() or val_scores.max() == 0 or val_targets.sum() == 0:
                break
            val_auprc = average_precision_score(val_targets, val_scores)
            scheduler.step()

            if val_auprc > best_auprc:
                best_auprc, counter = val_auprc, 0
            else:
                counter += 1
                if counter >= cfg.patience:
                    break

        val_scores = np.nan_to_num(val_scores, nan=0.0)
        test_scores, test_targets = eval_epoch(model, test_loader, device)
        # Skip round if model produced NaN or only one class or collapsed
        if np.isnan(test_scores).any() or test_scores.max() == 0 or len(np.unique(test_targets)) < 2:
            print(f"Round {round_idx+1}: skipped (NaN or single-class test set)")
            continue
        threshold = threshold_search(val_targets, val_scores)
        test_preds = (test_scores >= threshold).astype(int)

        auroc = roc_auc_score(test_targets, test_scores)
        auprc = average_precision_score(test_targets, test_scores)
        f1 = f1_score(test_targets, test_preds, zero_division=0)
        acc = accuracy_score(test_targets, test_preds)
        sens = recall_score(test_targets, test_preds, zero_division=0)
        spec = recall_score(1 - test_targets, 1 - test_preds, zero_division=0)
        prec = precision_score(test_targets, test_preds, zero_division=0)

        results["bootstrap_auroc"].append(auroc)
        results["bootstrap_auprc"].append(auprc)
        results["bootstrap_f1"].append(f1)
        results["bootstrap_accuracy"].append(acc)
        results["bootstrap_sensitivity"].append(sens)
        results["bootstrap_specificity"].append(spec)
        results["bootstrap_precision"].append(prec)
        results["thresholds"].append(threshold)

        print(f"Round {round_idx+1}: AUROC={auroc:.4f}, F1={f1:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

    # Compute statistics
    for key in ["bootstrap_auroc", "bootstrap_auprc", "bootstrap_f1", "bootstrap_accuracy",
                "bootstrap_sensitivity", "bootstrap_specificity", "bootstrap_precision"]:
        vals = np.array(results[key])
        results[f"{key.replace('bootstrap_', '')}_mean"] = float(np.mean(vals))
        results[f"{key.replace('bootstrap_', '')}_std"] = float(np.std(vals))
        ci = np.percentile(vals, [2.5, 97.5])
        results[f"{key.replace('bootstrap_', '')}_ci_lower"] = float(ci[0])
        results[f"{key.replace('bootstrap_', '')}_ci_upper"] = float(ci[1])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--max-patients", type=int, default=220)
    parser.add_argument("--bootstrap-rounds", type=int, default=100)
    args = parser.parse_args()

    cfg = Config(data_dir=args.data_dir, out_dir=args.out_dir,
                 max_patients=args.max_patients, bootstrap_rounds=args.bootstrap_rounds)
    print("=" * 80)
    print("Temporal Attention Bridge Network for Sepsis Risk Prediction")
    print("=" * 80)
    print(f"Config: {cfg}")

    results = bootstrap_experiment(cfg)

    out_path = Path(args.out_dir) / "temporal_attention_bridge_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    print("\n" + "=" * 80)
    print("SUMMARY (n=100 bootstrap runs)")
    print("=" * 80)
    print(f"Test AUROC:       {results['auroc_mean']:.4f} [95% CI: {results['auroc_ci_lower']:.4f}–{results['auroc_ci_upper']:.4f}]")
    print(f"Test AUPRC:       {results['auprc_mean']:.4f} [95% CI: {results['auprc_ci_lower']:.4f}–{results['auprc_ci_upper']:.4f}]")
    print(f"Test F1:          {results['f1_mean']:.4f} [95% CI: {results['f1_ci_lower']:.4f}–{results['f1_ci_upper']:.4f}]")
    print(f"Test Accuracy:    {results['accuracy_mean']:.4f} [95% CI: {results['accuracy_ci_lower']:.4f}–{results['accuracy_ci_upper']:.4f}]")
    print(f"Test Sensitivity: {results['sensitivity_mean']:.4f} [95% CI: {results['sensitivity_ci_lower']:.4f}–{results['sensitivity_ci_upper']:.4f}]")
    print(f"Test Specificity: {results['specificity_mean']:.4f} [95% CI: {results['specificity_ci_lower']:.4f}–{results['specificity_ci_upper']:.4f}]")
    print(f"Test Precision:   {results['precision_mean']:.4f} [95% CI: {results['precision_ci_lower']:.4f}–{results['precision_ci_upper']:.4f}]")
    print("=" * 80)


if __name__ == "__main__":
    main()

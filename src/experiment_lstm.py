"""
LSTM experiment for Sepsis Risk Prediction.
Focuses on single model with proper bootstrapped evaluation.
"""

import argparse
import json
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
    hidden_size: int
    out_dir: str = "results"


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        return self.fc(outputs[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_patient_files(data_dir: Path, max_patients: int) -> List[Path]:
    """List patient files from both training_setA and training_setB."""
    files = sorted(data_dir.glob("training_setA/p*.psv")) + sorted(
        data_dir.glob("training_setB/p*.psv")
    )
    if max_patients > 0:
        files = files[:max_patients]
    return files


def make_samples_for_patient(
    df: pd.DataFrame,
    history_hours: int,
    forecast_hours: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window samples from a patient's time-series.

    Args:
        df: Patient dataframe with FEATURES and LABEL_COL columns
        history_hours: Historical window size in hours
        forecast_hours: Forecast horizon in hours

    Returns:
        X: Shape (n_samples, history_hours, n_features)
        y: Shape (n_samples,)
    """
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
    """Build dataset from all patient files."""
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
    """Compute train-set medians and z-score statistics for sequence normalization."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    medians = np.nanmedian(flat, axis=0)
    imputed = np.where(np.isnan(flat), medians, flat)
    means = np.mean(imputed, axis=0)
    stds = np.std(imputed, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return medians, means, stds


def transform_sequences(
    X: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Median-impute and z-score normalize sequence data using train statistics."""
    X_filled = X.copy()
    for feature_idx in range(X.shape[-1]):
        X_filled[:, :, feature_idx] = np.where(
            np.isnan(X_filled[:, :, feature_idx]),
            medians[feature_idx],
            X_filled[:, :, feature_idx],
        )
    X_norm = (X_filled - means) / stds
    return X_norm.astype(np.float32)


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
) -> LSTMClassifier:
    """Train an LSTM model and keep the epoch with best validation AUROC."""
    device = torch.device("cpu")
    model = LSTMClassifier(n_features=X_train.shape[-1], hidden_size=config.hidden_size)
    model.to(device)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    pos_weight = torch.tensor(
        [max(1.0, neg_count / max(1, pos_count))],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    best_metric = -np.inf
    best_state = None

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
            optimizer.step()
            epoch_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        if len(np.unique(y_val)) < 2:
            val_metric = -epoch_loss
        else:
            val_metric = roc_auc_score(y_val, val_probs)

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"train_loss={epoch_loss / max(1, len(train_loader)):.4f} - "
            f"val_metric={val_metric:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_probabilities(model: LSTMClassifier, X: np.ndarray) -> np.ndarray:
    """Predict sepsis probabilities for sequence inputs."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate_model(
    model: LSTMClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate LSTM model.
    Returns NaN for AUROC/AUPRC if test set is single-class.
    """
    y_pred_proba = predict_probabilities(model, X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    if len(np.unique(y_test)) == 1:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "sensitivity": 0.0 if y_test[0] == 0 else 1.0,
            "specificity": 1.0 if y_test[0] == 0 else 0.0,
            "f1": np.nan if y_test[0] == 0 else 0.0,
            "brier": np.nan,
            "threshold": 0.5,
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
    f1 = f1_score(y_test, y_pred, zero_division=0.0)
    brier = brier_score_loss(y_test, y_pred_proba)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
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
        "threshold": 0.5,
    }


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


def bootstrap_ci_95(metric_values: np.ndarray) -> Tuple[float, float]:
    """Calculate 95% confidence interval using percentile method."""
    valid = metric_values[~np.isnan(metric_values)]
    if len(valid) == 0:
        return np.nan, np.nan
    return np.percentile(valid, 2.5), np.percentile(valid, 97.5)


def run_experiment(config: Config) -> Dict:
    """Run LSTM experiment with bootstrap evaluation."""
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
    X_train_norm = transform_sequences(X_train, medians, means, stds)
    X_val_norm = transform_sequences(X_val, medians, means, stds)
    X_test_norm = transform_sequences(X_test, medians, means, stds)

    model = train_lstm_model(X_train_norm, y_train, X_val_norm, y_val, config)

    print("\nValidation Set Evaluation:")
    val_metrics = evaluate_model(model, X_val_norm, y_val)
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
        indices = np.random.choice(len(X_test_norm), size=len(X_test_norm), replace=True)
        X_test_boot = X_test_norm[indices]
        y_test_boot = y_test[indices]

        metrics = evaluate_model(model, X_test_boot, y_test_boot)
        for key in bootstrap_metrics:
            bootstrap_metrics[key].append(metrics[key])

    results = {
        "model": "LSTM",
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
        "training": {
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "hidden_size": config.hidden_size,
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

    results_json = out_dir / "lstm_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM only experiment for Sepsis Risk Prediction")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--history-hours", type=int, default=24, help="History window in hours")
    parser.add_argument("--forecast-hours", type=int, default=6, help="Forecast horizon in hours")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-patients", type=int, default=0, help="Max patients to use (0=all)")
    parser.add_argument("--bootstrap-rounds", type=int, default=100, help="Bootstrap rounds")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
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
        hidden_size=args.hidden_size,
        out_dir=args.out_dir,
    )

    run_experiment(config)
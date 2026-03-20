import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from xgboost import XGBClassifier

FEATURES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2"]
LABEL_COL = "SepsisLabel"


@dataclass
class Config:
    data_dir: str
    history_hours: int
    forecast_hours: int
    seed: int
    max_patients: int
    batch_size: int
    epochs: int
    learning_rate: float
    bootstrap_rounds: int


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


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
    data = data.ffill()  # retain causal ordering by not using future values to impute

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
) -> Dict[str, np.ndarray]:
    all_x = []
    all_y = []
    all_pid = []

    for file_path in tqdm(files, desc="Building windows"):
        df = pd.read_csv(file_path, sep="|")
        missing_cols = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
        if missing_cols:
            continue

        x, y = make_samples_for_patient(df, history_hours, forecast_hours)
        if len(y) == 0:
            continue

        pid = file_path.stem
        all_x.append(x)
        all_y.append(y)
        all_pid.extend([pid] * len(y))

    x_arr = np.concatenate(all_x, axis=0)
    y_arr = np.concatenate(all_y, axis=0)
    pid_arr = np.array(all_pid)

    return {"x": x_arr, "y": y_arr, "pid": pid_arr}


def split_by_patient(patient_ids: np.ndarray, seed: int) -> Tuple[set, set, set]:
    unique_patients = np.array(sorted(set(patient_ids.tolist())))
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.15, random_state=seed)
    train_patients, val_patients = train_test_split(train_patients, test_size=0.1765, random_state=seed)
    return set(train_patients.tolist()), set(val_patients.tolist()), set(test_patients.tolist())


def slice_split(dataset: Dict[str, np.ndarray], split_patients: set) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.array([pid in split_patients for pid in dataset["pid"]])
    return dataset["x"][mask], dataset["y"][mask]


def compute_train_statistics(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])
    medians = np.nanmedian(flat, axis=0)
    means = np.nanmean(np.where(np.isnan(flat), medians, flat), axis=0)
    stds = np.nanstd(np.where(np.isnan(flat), medians, flat), axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return medians, means, stds


def transform_sequences(x: np.ndarray, medians: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    x_filled = x.copy()
    for i in range(x.shape[-1]):
        x_filled[:, :, i] = np.where(np.isnan(x_filled[:, :, i]), medians[i], x_filled[:, :, i])
    x_norm = (x_filled - means) / stds
    return x_norm.astype(np.float32)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi if i < bins - 1 else y_prob <= hi)
        if not np.any(in_bin):
            continue
        acc = y_true[in_bin].mean()
        conf = y_prob[in_bin].mean()
        ece += np.abs(acc - conf) * in_bin.mean()
    return float(ece)


def sensitivity_specificity(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[float, float]:
    y_hat = (y_prob >= threshold).astype(int)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)
    idx = int(np.argmax(f1))
    if idx == 0 or idx > len(thresholds):
        return 0.5
    return float(thresholds[idx - 1])


def bootstrap_ci(metric_fn, y_true: np.ndarray, y_prob: np.ndarray, rounds: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        vals.append(metric_fn(y_b, p_b))
    if len(vals) < 5:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_probs(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float, rounds: int, seed: int) -> Dict:
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = float(np.mean((y_prob - y_true) ** 2))
    ece = expected_calibration_error(y_true, y_prob)
    sens, spec = sensitivity_specificity(y_true, y_prob, threshold)
    f1 = f1_score(y_true, (y_prob >= threshold).astype(int))

    auroc_ci = bootstrap_ci(roc_auc_score, y_true, y_prob, rounds, seed)
    auprc_ci = bootstrap_ci(average_precision_score, y_true, y_prob, rounds, seed)

    return {
        "model": name,
        "threshold": threshold,
        "auroc": float(auroc),
        "auroc_ci95": list(auroc_ci),
        "auprc": float(auprc),
        "auprc_ci95": list(auprc_ci),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "brier": float(brier),
        "ece": float(ece),
    }


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> Tuple[LSTMClassifier, float]:
    set_seed(seed)
    device = torch.device("cpu")

    model = LSTMClassifier(n_features=x_train.shape[-1])
    model.to(device)

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([max(1.0, neg / max(1, pos))], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_auc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(x_val).to(device)).cpu().numpy()
            val_probs = 1 / (1 + np.exp(-val_logits))
            if len(np.unique(y_val)) > 1:
                val_auc = roc_auc_score(y_val, val_probs)
            else:
                val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    threshold = best_threshold_by_f1(y_val, predict_lstm(model, x_val))
    return model, threshold


def predict_lstm(model: LSTMClassifier, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).numpy()
    return 1 / (1 + np.exp(-logits))


def markdown_report(config: Config, stats: Dict, metrics: List[Dict]) -> str:
    lines = []
    lines.append("# Experiment Log: Early Sepsis Forecasting")
    lines.append("")
    lines.append("## Dataset")
    lines.append("- Source: PhysioNet Challenge 2019 training sets A and B")
    lines.append(f"- Number of patient files used: {stats['n_patients']}")
    lines.append(f"- Total windows: {stats['n_windows_total']}")
    lines.append(f"- Positive window rate: {stats['positive_rate']:.4f}")
    lines.append("")
    lines.append("## Window Definition")
    lines.append(f"- History window: {config.history_hours} hours")
    lines.append(f"- Forecast horizon: {config.forecast_hours} hours")
    lines.append("- Label = 1 if first sepsis onset occurs within (t, t + horizon]")
    lines.append("")
    lines.append("## Features")
    lines.append("- Dynamic variables: " + ", ".join(FEATURES))
    lines.append("- Missingness handling: per-patient forward-fill, then train-set median imputation")
    lines.append("- Scaling: z-score using train-set means and standard deviations")
    lines.append("")
    lines.append("## Split Strategy")
    lines.append("- Patient-level split to prevent leakage")
    lines.append(f"- Train windows: {stats['n_train']}, positive rate={stats['train_pos_rate']:.4f}")
    lines.append(f"- Validation windows: {stats['n_val']}, positive rate={stats['val_pos_rate']:.4f}")
    lines.append(f"- Test windows: {stats['n_test']}, positive rate={stats['test_pos_rate']:.4f}")
    lines.append("")
    lines.append("## Models")
    lines.append("- Logistic Regression (class_weight=balanced)")
    lines.append("- XGBoost (scale_pos_weight from train data)")
    lines.append("- LSTM (1 layer, hidden size 64, weighted BCE)")
    lines.append("")
    lines.append("## Test Metrics")
    lines.append("| Model | AUROC (95% CI) | AUPRC (95% CI) | Sensitivity | Specificity | F1 | Brier | ECE | Threshold |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in metrics:
        lines.append(
            "| {model} | {auroc:.4f} ({auroc_lo:.4f}, {auroc_hi:.4f}) | "
            "{auprc:.4f} ({auprc_lo:.4f}, {auprc_hi:.4f}) | {sensitivity:.4f} | {specificity:.4f} | "
            "{f1:.4f} | {brier:.4f} | {ece:.4f} | {threshold:.3f} |".format(
                model=m["model"],
                auroc=m["auroc"],
                auroc_lo=m["auroc_ci95"][0],
                auroc_hi=m["auroc_ci95"][1],
                auprc=m["auprc"],
                auprc_lo=m["auprc_ci95"][0],
                auprc_hi=m["auprc_ci95"][1],
                sensitivity=m["sensitivity"],
                specificity=m["specificity"],
                f1=m["f1"],
                brier=m["brier"],
                ece=m["ece"],
                threshold=m["threshold"],
            )
        )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- Random seed: {config.seed}")
    lines.append(f"- Bootstrap rounds: {config.bootstrap_rounds}")
    lines.append(f"- LSTM epochs: {config.epochs}, batch_size: {config.batch_size}, learning_rate: {config.learning_rate}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This run is a baseline benchmark from a subset of PhysioNet files, suitable for iteration and method comparison.")
    lines.append("- To align fully with the proposal, next iterations should add explicit missingness masks and a TFT-style architecture.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sepsis forecasting experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--history-hours", type=int, default=24)
    parser.add_argument("--forecast-hours", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bootstrap-rounds", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = Config(
        data_dir=str(args.data_dir),
        history_hours=args.history_hours,
        forecast_hours=args.forecast_hours,
        seed=args.seed,
        max_patients=args.max_patients,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        bootstrap_rounds=args.bootstrap_rounds,
    )

    set_seed(config.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = list_patient_files(args.data_dir, config.max_patients)
    if not files:
        raise RuntimeError(f"No patient files found under {args.data_dir}. Run download script first.")

    dataset = build_dataset(files, config.history_hours, config.forecast_hours)
    train_patients, val_patients, test_patients = split_by_patient(dataset["pid"], config.seed)

    x_train, y_train = slice_split(dataset, train_patients)
    x_val, y_val = slice_split(dataset, val_patients)
    x_test, y_test = slice_split(dataset, test_patients)

    medians, means, stds = compute_train_statistics(x_train)
    x_train_n = transform_sequences(x_train, medians, means, stds)
    x_val_n = transform_sequences(x_val, medians, means, stds)
    x_test_n = transform_sequences(x_test, medians, means, stds)

    x_train_flat = x_train_n.reshape(x_train_n.shape[0], -1)
    x_val_flat = x_val_n.reshape(x_val_n.shape[0], -1)
    x_test_flat = x_test_n.reshape(x_test_n.shape[0], -1)

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = float(neg / max(1, pos))

    logistic = LogisticRegression(max_iter=1200, class_weight="balanced", n_jobs=-1)
    logistic.fit(x_train_flat, y_train)
    p_val_lr = logistic.predict_proba(x_val_flat)[:, 1]
    thr_lr = best_threshold_by_f1(y_val, p_val_lr)
    p_test_lr = logistic.predict_proba(x_test_flat)[:, 1]

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        reg_lambda=1.0,
        eval_metric="auc",
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
        random_state=config.seed,
    )
    xgb.fit(x_train_flat, y_train)
    p_val_xgb = xgb.predict_proba(x_val_flat)[:, 1]
    thr_xgb = best_threshold_by_f1(y_val, p_val_xgb)
    p_test_xgb = xgb.predict_proba(x_test_flat)[:, 1]

    lstm, thr_lstm = train_lstm(
        x_train_n,
        y_train,
        x_val_n,
        y_val,
        epochs=config.epochs,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    p_test_lstm = predict_lstm(lstm, x_test_n)

    metrics = [
        evaluate_probs("LogisticRegression", y_test, p_test_lr, thr_lr, config.bootstrap_rounds, config.seed),
        evaluate_probs("XGBoost", y_test, p_test_xgb, thr_xgb, config.bootstrap_rounds, config.seed),
        evaluate_probs("LSTM", y_test, p_test_lstm, thr_lstm, config.bootstrap_rounds, config.seed),
    ]

    stats = {
        "n_patients": len(set(dataset["pid"].tolist())),
        "n_windows_total": int(len(dataset["y"])),
        "positive_rate": float(dataset["y"].mean()),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_pos_rate": float(y_train.mean()),
        "val_pos_rate": float(y_val.mean()),
        "test_pos_rate": float(y_test.mean()),
    }

    output = {
        "config": asdict(config),
        "stats": stats,
        "metrics": metrics,
    }

    json_path = args.out_dir / "experiment_metrics.json"
    md_path = args.out_dir / "experiment_details.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(markdown_report(config, stats, metrics))

    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved detailed report: {md_path}")


if __name__ == "__main__":
    main()

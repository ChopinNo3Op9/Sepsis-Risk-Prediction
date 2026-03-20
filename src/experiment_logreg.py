"""
Logistic Regression experiment for Sepsis Risk Prediction.
Focuses on single model with proper bootstrapped evaluation.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    out_dir: str = "results"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


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
    data = data.ffill()  # Causal forward-fill

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
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    if not x_all:
        return np.empty((0, history_hours, len(FEATURES))), np.empty((0,), dtype=int)

    X = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten (n_samples, history_hours, n_features) to (n_samples, history_hours*n_features)."""
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def train_logreg_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[LogisticRegression, SimpleImputer, StandardScaler]:
    """Train logistic regression model with balanced class weights and imputation."""
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    model = LogisticRegression(
        max_iter=1200,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    return model, imputer, scaler


def evaluate_model(
    model: LogisticRegression,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate logistic regression model with bootstrap confidence intervals.
    Returns NaN for AUROC/AUPRC if test set is single-class.
    """
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    # Handle single-class case
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
    except:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_test, y_pred_proba)
    except:
        auprc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, y_pred, zero_division=0.0)
    brier = brier_score_loss(y_test, y_pred_proba)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "brier": brier,
        "threshold": 0.5,
    }


def bootstrap_ci_95(metric_values: np.ndarray) -> Tuple[float, float]:
    """Calculate 95% confidence interval using percentile method."""
    valid = metric_values[~np.isnan(metric_values)]
    if len(valid) == 0:
        return np.nan, np.nan
    return np.percentile(valid, 2.5), np.percentile(valid, 97.5)


def run_experiment(config: Config) -> Dict:
    """Run logistic regression experiment with bootstrap evaluation."""
    set_seed(config.seed)

    data_dir = Path(config.data_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    files = list_patient_files(data_dir, config.max_patients)
    print(f"Processing {len(files)} patient files")

    X, y = build_dataset(files, config.history_hours, config.forecast_hours)
    print(f"Dataset shape: {X.shape}, Positive rate: {y.mean():.4f}")

    # Flatten sequences for logistic regression
    X_flat = flatten_sequences(X)

    # Patient-level split (first pass: determine which patients to use)
    patient_indices = np.arange(len(files))
    train_idx, test_idx = train_test_split(
        patient_indices,
        test_size=0.2,
        random_state=config.seed,
    )

    split_point_1 = int(0.7 * len(X))
    split_point_2 = int(0.85 * len(X))

    X_train = X_flat[:split_point_1]
    y_train = y[:split_point_1]

    X_val = X_flat[split_point_1:split_point_2]
    y_val = y[split_point_1:split_point_2]

    X_test = X_flat[split_point_2:]
    y_test = y[split_point_2:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(
        f"Train+ rate: {y_train.mean():.4f}, Val+ rate: {y_val.mean():.4f}, Test+ rate: {y_test.mean():.4f}"
    )

    # Train on training set
    model, imputer, scaler = train_logreg_model(X_train, y_train)

    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
    val_metrics = evaluate_model(model, imputer, scaler, X_val, y_val)
    for key, val in val_metrics.items():
        print(f"  {key}: {val:.4f}" if not np.isnan(val) else f"  {key}: NaN")

    # Bootstrap evaluation on test set
    print(f"\nBootstrap Test Evaluation ({config.bootstrap_rounds} rounds)...")
    bootstrap_metrics = {}
    for metric_key in [
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "f1",
        "brier",
    ]:
        bootstrap_metrics[metric_key] = []

    for i in tqdm(range(config.bootstrap_rounds), desc="Bootstrap"):
        indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_test_boot = X_test[indices]
        y_test_boot = y_test[indices]

        metrics = evaluate_model(model, imputer, scaler, X_test_boot, y_test_boot)
        for key in bootstrap_metrics.keys():
            bootstrap_metrics[key].append(metrics[key])

    # Calculate CIs and means
    results = {
        "model": "LogisticRegression",
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
            f"{metric_key}: {mean:.4f} "
            f"({ci_low:.4f}, {ci_high:.4f})" if not np.isnan(mean) else f"{metric_key}: NaN"
        )

    # Save results
    results_json = out_dir / "logreg_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logistic Regression only experiment for Sepsis Risk Prediction"
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--history-hours", type=int, default=24, help="History window in hours")
    parser.add_argument("--forecast-hours", type=int, default=6, help="Forecast horizon in hours")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-patients", type=int, default=0, help="Max patients to use (0=all)")
    parser.add_argument("--bootstrap-rounds", type=int, default=100, help="Bootstrap rounds")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()
    config = Config(
        data_dir=args.data_dir,
        history_hours=args.history_hours,
        forecast_hours=args.forecast_hours,
        seed=args.seed,
        max_patients=args.max_patients,
        bootstrap_rounds=args.bootstrap_rounds,
        out_dir=args.out_dir,
    )

    results = run_experiment(config)

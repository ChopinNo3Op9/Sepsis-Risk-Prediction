"""
Gradient-Boosted Trees experiment for Sepsis Risk Prediction.
Focuses on single model with proper bootstrapped evaluation.
"""

import argparse
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
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


def train_gbt_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[XGBClassifier, SimpleImputer, float]:
    """Train gradient-boosted trees model with class imbalance weighting and imputation."""
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train_imputed, y_train)
    return model, imputer, scale_pos_weight


def tune_gbt_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    seed: int,
) -> Dict:
    """Tune hyperparameters on validation set using AUPRC as primary objective."""
    param_grid = {
        "n_estimators": [150, 250, 350],
        "learning_rate": [0.03, 0.05],
        "max_depth": [3, 4],
        "min_child_weight": [1, 3],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [1.0],
    }

    grid_keys = list(param_grid.keys())
    grid_values = [param_grid[k] for k in grid_keys]
    total_trials = int(np.prod([len(v) for v in grid_values]))
    print(f"\nHyperparameter tuning on validation set ({total_trials} trials)...")

    best_record = None
    all_trials = []
    for values in tqdm(itertools.product(*grid_values), total=total_trials, desc="Tuning"):
        params = dict(zip(grid_keys, values))
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        model.fit(X_train, y_train)

        y_val_prob = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob >= 0.5).astype(int)

        trial_auprc = average_precision_score(y_val, y_val_prob)
        trial_auroc = roc_auc_score(y_val, y_val_prob)
        trial_f1 = f1_score(y_val, y_val_pred, zero_division=0.0)
        trial_brier = brier_score_loss(y_val, y_val_prob)

        record = {
            "params": params,
            "val_auprc": float(trial_auprc),
            "val_auroc": float(trial_auroc),
            "val_f1": float(trial_f1),
            "val_brier": float(trial_brier),
        }
        all_trials.append(record)

        if best_record is None:
            best_record = record
        else:
            if (
                (record["val_auprc"] > best_record["val_auprc"])
                or (
                    np.isclose(record["val_auprc"], best_record["val_auprc"])
                    and (record["val_auroc"] > best_record["val_auroc"])
                )
                or (
                    np.isclose(record["val_auprc"], best_record["val_auprc"])
                    and np.isclose(record["val_auroc"], best_record["val_auroc"])
                    and (record["val_brier"] < best_record["val_brier"])
                )
            ):
                best_record = record

    return {
        "selection_metric": "val_auprc",
        "tie_breakers": ["val_auroc", "val_brier_lower"],
        "total_trials": total_trials,
        "best": best_record,
        "trials": all_trials,
    }


def evaluate_model(
    model: XGBClassifier,
    imputer: SimpleImputer,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate gradient-boosted trees model.
    Returns NaN for AUROC/AUPRC if test set is single-class.
    """
    X_test_imputed = imputer.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

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
    """Run gradient-boosted trees experiment with bootstrap evaluation."""
    set_seed(config.seed)

    data_dir = Path(config.data_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    files = list_patient_files(data_dir, config.max_patients)
    print(f"Processing {len(files)} patient files")

    X, y = build_dataset(files, config.history_hours, config.forecast_hours)
    print(f"Dataset shape: {X.shape}, Positive rate: {y.mean():.4f}")

    # Flatten sequences for tree model
    X_flat = flatten_sequences(X)

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

    # Train baseline preprocessing and infer class-weight scaling
    _, imputer, scale_pos_weight = train_gbt_model(X_train, y_train)
    X_train_imputed = imputer.transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # Tune model on validation set
    tuning_results = tune_gbt_hyperparameters(
        X_train_imputed,
        y_train,
        X_val_imputed,
        y_val,
        scale_pos_weight,
        config.seed,
    )
    best_params = tuning_results["best"]["params"]
    print("Best tuned parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Train final tuned model on training set
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=config.seed,
        n_jobs=-1,
        tree_method="hist",
        **best_params,
    )
    model.fit(X_train_imputed, y_train)

    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
    val_metrics = evaluate_model(model, imputer, X_val, y_val)
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

    for _ in tqdm(range(config.bootstrap_rounds), desc="Bootstrap"):
        indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_test_boot = X_test[indices]
        y_test_boot = y_test[indices]

        metrics = evaluate_model(model, imputer, X_test_boot, y_test_boot)
        for key in bootstrap_metrics.keys():
            bootstrap_metrics[key].append(metrics[key])

    # Calculate CIs and means
    results = {
        "model": "GradientBoostedTrees",
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
        "tuning": {
            "selection_metric": tuning_results["selection_metric"],
            "tie_breakers": tuning_results["tie_breakers"],
            "total_trials": tuning_results["total_trials"],
            "scale_pos_weight": scale_pos_weight,
            "best_trial": tuning_results["best"],
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
            f"{metric_key}: {mean:.4f} "
            f"({ci_low:.4f}, {ci_high:.4f})" if not np.isnan(mean) else f"{metric_key}: NaN"
        )

    # Save results
    results_json = out_dir / "gbt_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gradient-Boosted Trees only experiment for Sepsis Risk Prediction"
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
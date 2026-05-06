"""
Run multiple complex models for sepsis risk prediction and compare performance.

Models included:
  - XGBoost (gbtree)
  - XGBoost (dart)
  - ExtraTrees
  - RandomForest
  - HistGradientBoosting
  - MLP (deep fully-connected)
  - Soft voting ensemble (top 3 validation AUPRC models)
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "pH", "PaCO2", "Glucose", "Lactate", "Potassium",
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
    out_dir: str = "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
    data = df.reindex(columns=FEATURES).copy().ffill()
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


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def fit_imputer(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_val_i = imputer.transform(X_val)
    X_test_i = imputer.transform(X_test)
    return X_train_i, X_val_i, X_test_i, imputer


def find_best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    fixed = np.linspace(0.01, 0.99, 99)
    quantiles = np.percentile(y_prob, np.linspace(1, 99, 99))
    thresholds = np.unique(np.concatenate([fixed, quantiles]))

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0.0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) == 1:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "sensitivity": 0.0 if y_true[0] == 0 else 1.0,
            "specificity": 1.0 if y_true[0] == 0 else 0.0,
            "f1": np.nan if y_true[0] == 0 else 0.0,
            "brier": np.nan,
            "threshold": threshold,
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "sensitivity": float(tp / (tp + fn) if (tp + fn) > 0 else 0.0),
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
    }


def bootstrap_ci_95(values: np.ndarray) -> Tuple[float, float]:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def get_models(scale_pos_weight: float, seed: int) -> Dict[str, Pipeline]:
    model_dict: Dict[str, Pipeline] = {}

    model_dict["xgboost_gbtree"] = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                booster="gbtree",
                n_estimators=500,
                learning_rate=0.02,
                max_depth=5,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
    ])

    model_dict["xgboost_dart"] = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                booster="dart",
                n_estimators=600,
                learning_rate=0.015,
                max_depth=4,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                rate_drop=0.1,
                skip_drop=0.4,
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
    ])

    model_dict["extratrees"] = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            ExtraTreesClassifier(
                n_estimators=800,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ])

    model_dict["random_forest"] = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ])

    model_dict["hist_gradient_boosting"] = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_depth=6,
                max_iter=450,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=seed,
            ),
        ),
    ])

    model_dict["mlp_deep"] = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=2e-4,
                max_iter=120,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=seed,
            ),
        ),
    ])

    return model_dict


def run_experiment(config: Config) -> Dict:
    set_seed(config.seed)

    data_dir = Path(config.data_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    files = list_patient_files(data_dir, config.max_patients)
    print(f"Processing {len(files)} patient files")

    X, y = build_dataset(files, config.history_hours, config.forecast_hours)
    X_flat = flatten_sequences(X)

    n = len(X_flat)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_train = X_flat[:n_train]
    y_train = y[:n_train]
    X_val = X_flat[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    X_test = X_flat[n_train + n_val :]
    y_test = y[n_train + n_val :]

    X_train_i, X_val_i, X_test_i, _ = fit_imputer(X_train, X_val, X_test)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    scale_pos_weight = max(1.0, neg_count / max(1, pos_count))

    model_dict = get_models(scale_pos_weight=scale_pos_weight, seed=config.seed)
    validation_records = []
    trained_models: Dict[str, Pipeline] = {}
    thresholds: Dict[str, float] = {}

    print("Training complex models...")
    for name, model in model_dict.items():
        print(f"  - {name}")
        model.fit(X_train_i, y_train)
        val_prob = model.predict_proba(X_val_i)[:, 1]
        threshold, best_val_f1 = find_best_threshold_f1(y_val, val_prob)
        val_metrics = compute_metrics(y_val, val_prob, threshold)
        validation_records.append(
            {
                "model": name,
                "threshold": threshold,
                "val_best_f1_search": best_val_f1,
                "val_metrics": val_metrics,
            }
        )
        trained_models[name] = model
        thresholds[name] = threshold

    validation_records = sorted(
        validation_records,
        key=lambda r: (r["val_metrics"]["auprc"], r["val_metrics"]["auroc"]),
        reverse=True,
    )

    top3_names = [r["model"] for r in validation_records[:3]]
    print(f"Building soft-voting ensemble from: {top3_names}")

    def soft_vote_probability(split_x: np.ndarray) -> np.ndarray:
        probs = [trained_models[model_name].predict_proba(split_x)[:, 1] for model_name in top3_names]
        return np.mean(np.stack(probs, axis=0), axis=0)

    val_prob_ensemble = soft_vote_probability(X_val_i)
    ensemble_threshold, _ = find_best_threshold_f1(y_val, val_prob_ensemble)
    validation_records.append(
        {
            "model": "soft_voting_top3",
            "threshold": ensemble_threshold,
            "val_best_f1_search": float(f1_score(y_val, (val_prob_ensemble >= ensemble_threshold).astype(int), zero_division=0.0)),
            "val_metrics": compute_metrics(y_val, val_prob_ensemble, ensemble_threshold),
        }
    )

    test_results: Dict[str, Dict] = {}
    all_models_for_test = [r["model"] for r in validation_records]

    print("Evaluating on test set with bootstrap...")
    for model_name in all_models_for_test:
        if model_name == "soft_voting_top3":
            test_prob_full = soft_vote_probability(X_test_i)
            threshold = ensemble_threshold
        else:
            test_prob_full = trained_models[model_name].predict_proba(X_test_i)[:, 1]
            threshold = thresholds[model_name]

        base_metrics = compute_metrics(y_test, test_prob_full, threshold)

        boot = {
            "auroc": [],
            "auprc": [],
            "sensitivity": [],
            "specificity": [],
            "f1": [],
            "brier": [],
        }

        for _ in tqdm(range(config.bootstrap_rounds), desc=f"Bootstrap {model_name}", leave=False):
            idx = np.random.randint(0, len(y_test), size=len(y_test))
            y_bs = y_test[idx]
            p_bs = test_prob_full[idx]
            m_bs = compute_metrics(y_bs, p_bs, threshold)
            for key in boot:
                boot[key].append(m_bs[key])

        summary = {}
        for key, values in boot.items():
            arr = np.array(values, dtype=float)
            lo, hi = bootstrap_ci_95(arr)
            summary[key] = {
                "mean": float(np.nanmean(arr)),
                "ci_lower": lo,
                "ci_upper": hi,
            }

        test_results[model_name] = {
            "threshold": float(threshold),
            "base_test_metrics": base_metrics,
            "bootstrap_metrics": summary,
        }

    ranking = sorted(
        [
            {
                "model": name,
                "test_auroc_mean": metrics["bootstrap_metrics"]["auroc"]["mean"],
                "test_auprc_mean": metrics["bootstrap_metrics"]["auprc"]["mean"],
                "test_f1_mean": metrics["bootstrap_metrics"]["f1"]["mean"],
                "test_sensitivity_mean": metrics["bootstrap_metrics"]["sensitivity"]["mean"],
                "test_specificity_mean": metrics["bootstrap_metrics"]["specificity"]["mean"],
            }
            for name, metrics in test_results.items()
        ],
        key=lambda r: (r["test_auroc_mean"], r["test_auprc_mean"], r["test_f1_mean"]),
        reverse=True,
    )

    result = {
        "experiment": "complex_model_benchmark",
        "config": asdict(config),
        "features": FEATURES,
        "split_stats": {
            "train_size": int(len(X_train_i)),
            "val_size": int(len(X_val_i)),
            "test_size": int(len(X_test_i)),
        },
        "data_stats": {
            "total_windows": int(len(X_flat)),
            "positive_rate": float(y.mean()),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
            "scale_pos_weight": float(scale_pos_weight),
        },
        "validation_results": validation_records,
        "test_results": test_results,
        "ranking": ranking,
    }

    output_path = out_dir / "complex_models_results.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved results: {output_path}")
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Complex model benchmark for sepsis prediction")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--history-hours", type=int, default=24)
    parser.add_argument("--forecast-hours", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int, default=220)
    parser.add_argument("--bootstrap-rounds", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()
    return Config(
        data_dir=args.data_dir,
        history_hours=args.history_hours,
        forecast_hours=args.forecast_hours,
        seed=args.seed,
        max_patients=args.max_patients,
        bootstrap_rounds=args.bootstrap_rounds,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_experiment(cfg)
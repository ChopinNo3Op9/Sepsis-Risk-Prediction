# Experiment Log: Early Sepsis Forecasting

## Dataset
- Source: PhysioNet Challenge 2019 training sets A and B
- Number of patient files used: 182
- Total windows: 3618
- Positive window rate: 0.0205

## Window Definition
- History window: 24 hours
- Forecast horizon: 6 hours
- Label = 1 if first sepsis onset occurs within (t, t + horizon]

## Features
- Dynamic variables: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, FiO2
- Missingness handling: per-patient forward-fill, then train-set median imputation
- Scaling: z-score using train-set means and standard deviations

## Split Strategy
- Patient-level split to prevent leakage
- Train windows: 2466, positive rate=0.0178
- Validation windows: 627, positive rate=0.0287
- Test windows: 525, positive rate=0.0229

## Models
- Logistic Regression (class_weight=balanced)
- XGBoost (scale_pos_weight from train data)
- LSTM (1 layer, hidden size 64, weighted BCE)

## Test Metrics
| Model | AUROC (95% CI) | AUPRC (95% CI) | Sensitivity | Specificity | F1 | Brier | ECE | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.5175 (0.3794, 0.6820) | 0.0253 (0.0137, 0.0462) | 0.5000 | 0.5906 | 0.0526 | 0.0590 | 0.0654 | 0.000 |
| XGBoost | 0.4253 (0.3500, 0.5081) | 0.0201 (0.0121, 0.0324) | 0.0000 | 0.8226 | 0.0000 | 0.0229 | 0.0215 | 0.002 |
| LSTM | 0.7243 (0.6685, 0.7757) | 0.0408 (0.0247, 0.0639) | 0.0000 | 0.9162 | 0.0000 | 0.2429 | 0.4672 | 0.595 |

## Reproducibility
- Random seed: 42
- Bootstrap rounds: 100
- LSTM epochs: 8, batch_size: 128, learning_rate: 0.001

## Notes
- This run is a baseline benchmark from a subset of PhysioNet files, suitable for iteration and method comparison.
- To align fully with the proposal, next iterations should add explicit missingness masks and a TFT-style architecture.

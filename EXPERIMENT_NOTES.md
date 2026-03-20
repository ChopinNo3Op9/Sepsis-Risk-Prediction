# Sepsis Experiment Notes (Reproducible Log)

## Date
- 2026-03-19

## Objective
- Implement and run a proposal-aligned early sepsis forecasting baseline using ICU time-series data.
- Compare Logistic Regression, XGBoost, and LSTM on a fixed forecasting task.

## Data Source
- PhysioNet Challenge 2019 (public ICU challenge training data)
- URL root: https://physionet.org/files/challenge-2019/1.0.0/training/
- Folders used: training_setA and training_setB
- Downloaded subset:
  - training_setA: first 150 patient files
  - training_setB: first 150 patient files
- Local data path: data/raw

## Environment
- OS: Windows
- Python: 3.11.9
- Python executable:
  - ./python.exe 

## Installed Dependencies
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1
- xgboost==2.1.1
- torch==2.4.1
- tqdm==4.66.5
- requests==2.32.3

## Implemented Code
- Data downloader: src/download_physionet2019.py
- Experiment pipeline: src/experiment_sepsis.py
- Dependency file: requirements.txt

## Commands Executed
1. Install dependencies
   - python -m pip install -r requirements.txt
2. Download dataset subset
   - python src/download_physionet2019.py --out-dir data/raw --n-set-a 150 --n-set-b 150
3. Run experiments
   - python src/experiment_sepsis.py --data-dir data/raw --max-patients 300 --history-hours 24 --forecast-hours 6 --epochs 8 --bootstrap-rounds 100 --out-dir results

## Experimental Design
- Task: predict whether first sepsis onset happens within next 6 hours.
- History context: previous 24 hours.
- Features (dynamic): HR, O2Sat, Temp, SBP, MAP, DBP, Resp, FiO2.
- Label rule: y=1 if first onset in (t, t+6].
- Split: patient-level train/val/test (approx. 70/15/15).
- Missing-data handling:
  - Causal forward-fill per patient sequence.
  - Remaining missing values imputed with train-set median.
- Scaling: z-score using train-set mean/std.
- Class imbalance:
  - LogisticRegression class_weight=balanced.
  - XGBoost scale_pos_weight=neg/pos.
  - LSTM BCEWithLogitsLoss with pos_weight.

## Model Configurations
- Logistic Regression
  - max_iter=1200
  - class_weight=balanced
- XGBoost
  - n_estimators=250
  - max_depth=5
  - learning_rate=0.05
  - subsample=0.8
  - colsample_bytree=0.8
- LSTM
  - layers=1
  - hidden_size=64
  - optimizer=Adam
  - learning_rate=1e-3
  - epochs=8
  - batch_size=128

## Evaluation Metrics
- AUROC (with bootstrap 95% CI)
- AUPRC (with bootstrap 95% CI)
- Sensitivity
- Specificity
- F1
- Brier score
- Expected Calibration Error (ECE)

## Outputs Produced
- Detailed markdown report: results/experiment_details.md
- Machine-readable metrics: results/experiment_metrics.json

## Run Summary (Test Set)
- Data used in final windows:
  - Patients with valid windows: 182
  - Total windows: 3618
  - Positive rate: 0.0205
- Model performance:
  - LogisticRegression: AUROC 0.5175, AUPRC 0.0253
  - XGBoost: AUROC 0.4253, AUPRC 0.0201
  - LSTM: AUROC 0.7243, AUPRC 0.0408

## Notes and Limitations
- This run is a baseline proof-of-experiment that follows the proposal direction but uses a subset for practicality.
- Full proposal alignment should include:
  - Larger cohort coverage from full challenge files.
  - Missingness-mask features explicitly injected into the model.
  - Transformer/TFT implementation and comparison.
  - Additional subgroup analyses and interpretability outputs (e.g., SHAP/attention analysis).

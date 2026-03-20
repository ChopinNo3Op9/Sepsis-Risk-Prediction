# Sepsis Experiment Notes (Reproducible Log)

## Date
- 2026-03-19

## 1. Experiment Title/Research Objective
- Title: Reproducible Early Sepsis Risk Forecasting Benchmark on PhysioNet 2019 ICU Data
- Research Objective: Build and compare Logistic Regression, XGBoost, and LSTM in predicting first sepsis onset in the next 6 hours using 24-hour history, then evaluate generalization and calibration with bootstrapped metrics.

## 2. Data Source
- PhysioNet Challenge 2019 (public ICU challenge training data)
- URL root: https://physionet.org/files/challenge-2019/1.0.0/training/
- Folders used: training_setA and training_setB
- Downloaded subset:
  - training_setA: first 150 patient files
  - training_setB: first 150 patient files
- Local data path: data/raw

## 3. Research Background/Theoretical Basis
- Sepsis is an acute, life-threatening response to infection; early detection in ICU can improve outcomes.
- Physiological vitals and lab time-series in PhysioNet Challenge 2019 provide signal-rich, sparse-typed data ideal for ML forecasting.
- Theory: temporally-resolved modeling (e.g., LSTM) can capture dynamic state and early deterioration patterns; classical models provide robust baselines and interpretability.

## 4. Experimental Procedures
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

## 5. Results and Observation Records
- Data used in final windows:
  - Patients with valid windows: 182
  - Total windows: 3618
  - Positive rate: 0.0205
- Model performance:
  - LogisticRegression: AUROC 0.5175, AUPRC 0.0253
  - XGBoost: AUROC 0.4253, AUPRC 0.0201
  - LSTM: AUROC 0.7243, AUPRC 0.0408
- Observed overfitting warning with small sample; XGBoost underperformed due to class imbalance and small event sizes.
- Observed calibration needs improvement (ECE indicates probability miscalibration).

## 6. Analysis and Conclusion
- Conclusion: LSTM demonstrates the strongest signal extraction from temporal data in this setup; classical models are not competitive here.
- Limitations: reduced cohort size, single forecasting horizon, and partial preprocessing likely limit generality.
- Next steps: scale to full dataset, add advanced model (Transformer/TFT), incorporate explicit missingness indicators, perform subcohort and interpretability analysis, and optimize threshold for clinical utility.

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

## 2. Experiment Title/Research Objective
- Title: Reproducible Early Sepsis Risk Forecasting Benchmark on PhysioNet 2019 ICU Data
- Research Objective: Build and compare Logistic Regression, XGBoost, LSTM in predicting first sepsis onset in the next 6 hours using 24-hour history, then evaluate generalization and calibration with bootstrapped metrics.

## 3. Research Background/Theoretical Basis
- Sepsis is an acute, life-threatening response to infection; early detection in ICU can improve outcomes.
- Physiological vitals and lab time-series in PhysioNet Challenge 2019 provide signal-rich, sparse-typed data ideal for ML forecasting.
- Theory: temporally-resolved modeling (e.g., LSTM) can capture dynamic state and early deterioration patterns; classical models provide robust baselines and interpretability.

## 4. Experimental Procedures
- Data ingestion: parse patient PSV files from training_setA/B.
- Preprocess: resample hourly, forward-fill causal sequence, median impute, z-score normalization.
- Label generation: positive if first sepsis within 6 hours window following current time step.
- Train/val/test split at patient level (~70/15/15), maintain class distribution.
- Train three model families with defined hyperparameters and handle imbalance.
- Evaluate with AUROC/AUPRC plus sensitivity, specificity, F1, Brier, ECE, and bootstrapped confidence intervals.

## 5. Results and Observation Records
- Data windows: 182 patients, 3618 windows, positive rate 2.05%.
- Observed overfitting warning with small sample; XGBoost underperformed due to class imbalance and small event sizes.
- Best test performance in this run: LSTM AUROC 0.7243, AUPRC 0.0408, but still low absolute recall.
- Observed calibration needs improvement (ECE indicates probability miscalibration).

## 6. Analysis and Conclusion
- Conclusion: LSTM demonstrates the strongest signal extraction from temporal data in this setup; classical models are not competitive here.
- Limitations: reduced cohort size, single forecasting horizon, and partial preprocessing likely limit generality.
- Next steps: scale to full dataset, add advanced model (Transformer/TFT), incorporate explicit missingness indicators, perform subcohort and interpretability analysis, and optimize threshold for clinical utility.

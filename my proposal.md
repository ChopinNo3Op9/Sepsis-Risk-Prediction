<link rel="stylesheet" href="markdown.CSS">

# Proposal: Early Sepsis Risk Prediction from ICU Time Series

## 1. Background

Sepsis constitutes a leading cause of mortality in the intensive care unit (ICU), resulting from a dysregulated host response that culminates in organ dysfunction [1]. Clinical outcomes are critically dependent on the timeliness of therapeutic intervention; each hour of delay in administering effective antimicrobials correlates with a significant elevation in the risk of death [2]. Modern ICUs continuously monitor hundreds of variables; however, the early physiological signature of sepsis is often obscured by measurement noise, irregular sampling, and the high dimensionality of the data, which diminishes the efficacy of conventional detection techniques [3,4].

A variety of computational strategies have been proposed, including simple logistic regression and gradient‑boosted decision trees. These approaches generally do not exploit the full temporal context of a patient’s trajectory, as they either operate on static snapshots or depend on handcrafted summary statistics [5,6]. Consequently, models that perform adequately within a single institution frequently exhibit substantial degradation during external validation [7]. Recurrent neural networks (RNNs), notably long short‑term memory (LSTM) architectures, partially address sequence modeling and have been deployed for sepsis prediction [8,9]; nevertheless, they remain vulnerable to pervasive missing‑data patterns common in electronic health records [10]. More recently, transformer‑based architectures have been introduced in critical‑care forecasting due to their capacity for modeling long‑range dependencies and for handling irregularly sampled inputs [11].

This project proposes **TFTSepsis**, an attention‑based framework built upon the Temporal Fusion Transformer (TFT). The model is designed to capture long‑range dependencies, to integrate static and dynamic covariates, and to explicitly represent missingness; the ultimate aim is to generate earlier and more reliable sepsis alerts.

### Data Source

Analyses will utilize the publicly available **MIMIC‑IV** database, subject to obtaining credentialed access [5]. MIMIC‑IV contains deidentified health records from a large tertiary‑care hospital system, including:

- patient demographics, vital signs, laboratory results and interventions; 
- timestamped clinical trajectories suitable for early‑warning modelling; 
- a widely accepted benchmark for ICU prediction research.

The dataset comprises approximately 300 000 unique patients (50 000 with ICU admissions) and 431 000 hospital encounters (73 000 ICU stays). ICU patients are generally older and exhibit higher mortality than non‑ICU cohorts. Data span 2008–2019; all dates are uniformly shifted to preserve privacy while retaining temporal relationships [5].

## 2. Problem Statement

The primary task is to construct a prognostic model that, at any point during an ICU admission, provides the probability that the patient will satisfy the Sepsis‑3 criteria within a prespecified forecast horizon (e.g. six hours) [1]. Formulated as a supervised classification problem, the model must recognise temporal patterns indicative of impending sepsis from preceding patient history.

### Objectives

1. **Primary objective.** Construct and validate a temporally aware machine‑learning model capable of flagging ICU patients who will meet Sepsis‑3 criteria within the forecast window. This objective is motivated by evidence that timely recognition can materially affect survival [2]; therefore, anticipatory alerts possess demonstrable clinical utility.

2. **Comparative analysis.** Evaluate a sequence of model classes—logistic regression, gradient‑boosted trees, LSTMs, and TFTs—using identical cohort definitions and preprocessing pipelines, to determine whether additional modelling complexity yields statistically and clinically meaningful gains.

3. **Clinical relevance.** Quantify the lead‑time advantage achievable without generating an unacceptable false‑alarm burden. Interpret sensitivity and specificity in terms of downstream consequences rather than abstract percentages.

4. **Interpretability.** Investigate which predictors most strongly influence risk estimates and assess whether these variables concord with established sepsis pathophysiology. Anomalies will be scrutinised for potential confounding or data artefacts.

Predictors will encompass dynamic vital signs, laboratory measurements, administered therapies, and static patient descriptors (age, sex, comorbidity indices). The binary outcome denotes sepsis onset within the forecast horizon. Model training will utilise an appropriate loss function with class‑imbalance mitigation (e.g. weighting or focal loss [6]), and outputs will be interpreted probabilistically.

A rigorous contextualisation of results will be provided by:

- comparing improvements to ranges reported in systematic reviews and recent MIMIC‑based studies [3];
- verifying that high‑importance features concord with known physiological drivers such as hemodynamic instability;
- expressing statistical significance in terms of clinical impact (e.g. hours gained versus additional false positives);
- discussing any surprising model behaviours with reference to potential confounding or measurement artefacts.

### Methodological Rationale

The project adopts a hierarchical modelling strategy. TFT is selected as the principal method because it inherently accommodates mixed static/dynamic inputs, models long‑range dependencies via multi‑head attention, and provides explicit missingness masks—properties well matched to ICU time series [4]. Should TFT demonstrably improve discrimination, calibration, or lead time relative to simpler baselines, this would substantiate the value of attention‑based temporal modelling; if improvements are marginal, the analysis will still delineate the conditions under which simpler models suffice.

### Feature Selection

Features will be selected on the basis of clinical plausibility and data availability. Only variables that are routinely measured and exhibit reasonable prevalence within the cohort will be retained, thereby reducing overfitting risk and enhancing external validity. Inclusion criteria, missingness rates, and descriptive statistics will be reported to ensure transparency and reproducibility.

### Literature Context

Existing methods fall into several families. Logistic regression models, often employing recent measurements or aggregated summaries, are easy to deploy but overlook temporal complexity. Gradient‑boosted trees such as XGBoost typically require extensive feature engineering to approximate dynamics. RNNs and one‑dimensional convolutional networks process entire time series but remain vulnerable to irregular sampling. Transformer and TFT models are emerging for their ability to represent long‑range interactions and to manage missing data through masking [4]. The TFTSepsis pipeline will incorporate uncertainty‑aware preprocessing and class‑balanced training strategies as described in contemporary studies [4,6].

### Analysis Scope

To avoid selection bias, all eligible ICU admissions from MIMIC‑IV will be included after applying prespecified inclusion/exclusion criteria. Data splitting will occur at the patient level to eliminate temporal leakage; the held‑out test set will contain entirely unseen patients. Metrics (AUROC, AUPRC, sensitivity, specificity, calibration) will be calculated on the full test cohort with bootstrap confidence intervals to convey realistic uncertainty.

Stratified analyses will be prespecified across clinically relevant subgroups (e.g. age, ICU type, infection source, severity score) and will be treated as confirmatory secondary analyses. Error analyses will characterise false positives and false negatives by subgroup to detect potential biases.

## 3. Plan and Timeline

The following schedule outlines the proposed workflow:

| Week | Stage | Activities |
|--------|--------------------------|----------------------------------------------------------------------------|
| 12 | Proposal ;& Setup | Finalise protocol; secure ethics approval and MIMICIV credentials |
| 34 | Data Preparation | Define cohort; create sequence windows (e.g. 24h history, 6h forecast); handle missingness; normalise; patientlevel train/val/test split |
| 56 | Baseline Implementation | Build logistic regression, XGBoost, and LSTM baselines; verify performance |
| 79 | Model Development | Develop TFTSepsis with variable selection, multihead attention, static conditioning and missingness masks; train with classweighted or focal loss |
| 1011 | Evaluation & Analysis | Compute AUROC, AUPRC, sensitivity, specificity, F1, calibration (ECE, Brier); estimate leadtime gain; perform interpretability analyses (attention plots, SHAP) |
| 1213 | Reporting | Conduct ablations; finalise manuscript and presentation |

## 4. Expected Outcomes

The project is expected to deliver a reproducible early‑warning model that can be applied to MIMIC‑IV or similar datasets. Quantitatively, we aim for TFTSepsis to outperform baseline models on discrimination (AUROC, AUPRC) and to yield a measurable increase in lead time prior to sepsis onset. Predicted probabilities should exhibit acceptable calibration according to ECE and Brier score [7]. Interpretability will be provided via attention visualisations and SHAP attributions [8]. More broadly, the study will demonstrate whether transformer‑based architectures that explicitly address missing data can capture the complex temporal dynamics of ICU time series [4].

## References

[1] Singer, M., Deutschman, C. S., Seymour, C. W., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801-810.

[2] Kumar, A., Roberts, D., Wood, K. E., et al. (2006). Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. *Critical Care Medicine*, 34(6), 1589-1596.

[3] Fleuren, L. M., Klausch, T. L. T., Zwager, C. L., et al. (2020). Machine learning for the prediction of sepsis: a systematic review and meta-analysis of diagnostic test accuracy. *Intensive Care Medicine*, 46, 383-400.

[4] Mao, Q., Jay, M., Hoffman, J., et al. (2018). Multicentre validation of a sepsis prediction algorithm using only vital sign data in the emergency department, general ward and ICU. *BMJ Open*, 8(1), e017833.

[5] Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.

[6] Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10, 1.

[7] Shickel, B., Tighe, P. J., Bihorac, A., & Rashidi, P. (2018). Deep EHR: A survey of recent advances in deep learning techniques for electronic health record (EHR) analysis. *IEEE Journal of Biomedical and Health Informatics*, 22(5), 1589-1604.

[8] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2980-2988).

[9] Futoma, J., Hariharan, S., Heller, K., & Doshi-Velez, F. (2017). Learning to detect sepsis with a multitask Gaussian process RNN classifier. arXiv preprint arXiv:1706.04152.

[10] Kam, H. J., & Kim, H. Y. (2017). Learning representations for clinical time series prediction tasks. In Proceedings of the ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB), 81–90.

[11] Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports*, 8(1), 6085.

[12] Harutyunyan, H., Khachatrian, H., Kale, D. C., et al. (2019). Multitask learning and benchmarking with clinical time series data. *Scientific Data*, 6(1), 170.

[13] Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.

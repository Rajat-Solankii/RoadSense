# 🚗 RoadSense AI: Multi-Task Machine Learning for Road Accident Severity & Casualty Forecasting

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-success.svg)](https://lightgbm.readthedocs.io/)
[![STATS19](https://img.shields.io/badge/dataset-UK%20DfT%20STATS19-orange.svg)](https://www.data.gov.uk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RoadSense AI** is an end-to-end, multi-task machine learning system engineered to assess road collision severity and forecast casualties using official UK Department for Transport (DfT) STATS19 records from **2021 through 2023** (~330,000 national collisions).

Built under strict data governance, time-ordered split protocols, and clinical-grade decision calibration, RoadSense AI predicts:
1. **Task A (Binary Severity)**: Dangerous (*Fatal or Serious*) vs Minor (*Slight*) collision risk.
2. **Task B (3-Class Multiclass Severity)**: Granular risk decomposition across *Fatal*, *Serious*, and *Slight* injury levels.
3. **Task C (Casualty Count Regression)**: Expected number of casualties per collision ($y \ge 1$) using Displaced Poisson count modeling.

---

## 📑 Table of Contents
- [1. System Architecture & Multi-Task Formulation](#1-system-architecture--multi-task-formulation)
- [2. Strict Data Engineering & Leakage Prevention Protocol](#2-strict-data-engineering--leakage-prevention-protocol)
- [3. Experimental Protocol & Imbalance Strategies](#3-experimental-protocol--imbalance-strategies)
- [4. Benchmark Results & Model Selection](#4-benchmark-results--model-selection)
  - [Task A: Binary Severity (Dangerous vs Minor)](#task-a-binary-severity-dangerous-vs-minor)
  - [Task B: 3-Class Multiclass Severity](#task-b-3-class-multiclass-severity)
  - [Task C: Casualty Count Regression & Outlier Audit](#task-c-casualty-count-regression--outlier-audit)
  - [Bootstrap 95% Confidence Intervals & Paired Tests](#bootstrap-95-confidence-intervals--paired-tests)
- [5. Calibration & Operating Points](#5-calibration--operating-points)
- [6. Feature Ablation & Regional Generalization Audit](#6-feature-ablation--regional-generalization-audit)
- [7. SHAP Interpretability & Explainability](#7-shap-interpretability--explainability)
- [8. Interactive Terminal Prediction Tool (`predict.py`)](#8-interactive-terminal-prediction-tool-predictpy)
- [9. Repository Structure](#9-repository-structure)
- [10. Reproducibility & Quickstart](#10-reproducibility--quickstart)

---

## 1. System Architecture & Multi-Task Formulation

RoadSense AI formulates road safety assessment into three complementary, simultaneously operating prediction heads:

```
                            ┌─────────────────────────────────────────┐
                            │   Collision & Environmental Features    │
                            │ (Speed, VRUs, Fleet, Road Type, Time)   │
                            └────────────────────┬────────────────────┘
                                                 │
                                       [ColumnTransformer]
                           (Target-Agnostic Imputation, Scaling, OHE)
                                                 │
                ┌────────────────────────────────┼────────────────────────────────┐
                ▼                                ▼                                ▼
    ┌───────────────────────┐        ┌───────────────────────┐        ┌───────────────────────┐
    │        TASK A         │        │        TASK B         │        │        TASK C         │
    │   Binary Severity     │        │   3-Class Severity    │        │ Casualties Regression │
    ├───────────────────────┤        ├───────────────────────┤        ├───────────────────────┤
    │ Dangerous vs Minor    │        │ Fatal / Serious /     │        │ Expected Count        │
    │ (Fatal+Serious vs     │        │ Slight                │        │ (Displaced Poisson    │
    │  Slight)              │        │                       │        │  & LightGBM RMSE)     │
    ├───────────────────────┤        ├───────────────────────┤        ├───────────────────────┤
    │ Metric: PR-AUC        │        │ Metric: Fatal PR-AUC  │        │ Metric: RMSE / MAE /  │
    │ Threshold: F2 on 2022 │        │ Thresholds: Hierarch- │        │         Deviance      │
    │ Platt-Calibrated      │        │ ical (τ_fatal, τ_ser) │        │ Outlier-robust        │
    └───────────────────────┘        └───────────────────────┘        └───────────────────────┘
```

---

## 2. Strict Data Engineering & Leakage Prevention Protocol

### A. DfT STATS19 Unknown-Code Mapping
UK STATS19 records do not use a blanket missing-value sentinel; different variables utilize distinct DfT sentinel values. RoadSense AI applies a strictly verified per-column map:
- Categorical features (`weather_conditions`, `road_type`, `road_surface_conditions`): Missing/unknown coded as `9` or `-1` $\to$ unified as explicit `'Unknown'` level.
- Geographic variables (`urban_or_rural_area`): `3` or `-1` $\to$ `'Unknown'`.
- Vehicle variables (`vehicle_type`, `junction_location`): `99` or `-1` $\to$ converted to `NaN` prior to feature aggregation.
- Structural Missingness: Pedal cycles and non-motorized vehicles have no engine capacity or vehicle age in STATS19. Explicit missing indicators (`age_of_vehicle_max_missing`, `driver_age_band_youngest_missing`) are created, and numerical medians are imputed conditionally on presence.

### B. Leakage Exclusion Protocol
Columns that represent **post-impact crash dynamics** are strictly excluded from all training pipelines because they can only be observed *after* the crash has completed and its physical severity is already determined:
- `first_point_of_impact` *(Excluded: direct proxy for crash kinematics)*
- `skidding_and_overturning` *(Excluded: outcome of kinetic energy release)*
- `vehicle_leaving_carriageway` *(Excluded: trajectory outcome)*
- `hit_object_in_carriageway` & `hit_object_off_carriageway` *(Excluded: terminal collision outcome)*
- `vehicle_manoeuvre` *(Excluded: recording includes post-incident maneuvers)*

### C. Collision-Level Feature Aggregation
Rather than naively using `vehicle_reference == 1` as the primary vehicle (which induces arbitrary indexing bias), collision-level aggregates are computed across all involved vehicles:
- `any_motorcycle`: Boolean indicator for presence of motorized two-wheelers.
- `any_pedal_cycle`: Boolean indicator for presence of pedal cycles.
- `any_hgv_or_bus`: Boolean indicator for presence of heavy goods vehicles or public service buses.
- `most_vulnerable_vehicle_type`: Hierarchically ranked most vulnerable participant in the collision.
- `age_of_vehicle_max` & `driver_age_band_youngest`: Fleet extremes across all involved units.

---

## 3. Experimental Protocol & Imbalance Strategies

### Temporal Split
- **Training Set (2021 + 2022)**: 225,876 collisions. Inside hyperparameter tuning, **2021** is used for fitting and **2022** is used as holdout validation via temporal splitting.
- **Holdout Test Set (2023)**: 104,258 collisions. Kept completely untouched during all preprocessing, tuning, and threshold selection.
- **Limitation Note**: The 2021 data contains residual traffic volume shifts from UK COVID-19 pandemic mobility restrictions; temporal split validation directly verifies model stability under temporal drift.

### Resampling & Imbalance Comparison
To address severe imbalance (Dangerous: ~23.9%; Fatal: ~1.4%), 15 configurations were compared across identical feature splits:
- **5 Model Families**: Logistic Regression / Ridge, Decision Tree, Random Forest, XGBoost, LightGBM, and MLP.
- **3 Imbalance Strategies**: Cost-sensitive Class Weights (`CW`), Synthetic Minority Oversampling (`SMOTENC`), and Random Undersampling (`RUS`).
- Resampling is applied **strictly inside CV folds** via `imblearn.pipeline.Pipeline`, preventing synthetic leakage into validation folds.

---

## 4. Benchmark Results & Model Selection

### Task A: Binary Severity (Dangerous vs Minor)
*Pre-specified Selection Metric: PR-AUC on 2022 Validation (Prevalence = 23.91%). Thresholds tuned via F2-maximization on 2022 validation fold ($\tau = 0.12$).*

| Model Architecture | Imbalance Strategy | Val PR-AUC | Val ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Recall (Dangerous) | Test Precision | Test F2 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Winner)** | **Class Weights** | **0.4285** | **0.6970** | **0.4178** | **0.6920** | **93.24%** | **27.59%** | **0.6318** |
| Random Forest | Class Weights | 0.4218 | 0.6927 | 0.4128 | 0.6874 | 93.30% | 27.27% | 0.6288 |
| XGBoost | Class Weights | 0.4206 | 0.6931 | 0.4124 | 0.6881 | 92.42% | 27.84% | 0.6335 |
| LightGBM | SMOTE-NC | 0.4150 | 0.6884 | 0.4072 | 0.6841 | 91.80% | 27.70% | 0.6274 |
| Logistic Regression (Baseline) | None (Standard) | 0.3892 | 0.6698 | 0.3842 | 0.6669 | 74.82% | 31.81% | 0.5898 |
| Majority Class Baseline | None | 0.2391 | 0.5000 | 0.2394 | 0.5000 | 0.00% | 0.00% | 0.0000 |

### Task B: 3-Class Multiclass Severity
*Pre-specified Selection Metric: Fatal One-vs-Rest PR-AUC on 2022 Validation (Fatal Prevalence = 1.44%). Decision rules: $\tau_{\text{fatal}} = 0.007$, $\tau_{\text{serious}} = 0.45$.*

| Model Architecture | Imbalance Strategy | Val Fatal PR-AUC | Val OvR ROC-AUC | Test Fatal PR-AUC | Test OvR ROC-AUC | Test Fatal Recall | Test Fatal Precision | Test Serious Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Winner)** | **Class Weights** | **0.0682** | **0.7042** | **0.0641** | **0.6987** | **39.77%** | **6.56%** | **61.66%** |
| Random Forest | Class Weights | 0.0664 | 0.6912 | 0.0628 | 0.6865 | 38.54% | 6.42% | 60.82% |
| XGBoost | Class Weights | 0.0651 | 0.6924 | 0.0617 | 0.6870 | 37.91% | 6.25% | 61.10% |
| LightGBM | SMOTE-NC | 0.0592 | 0.6811 | 0.0560 | 0.6755 | 34.20% | 5.88% | 58.40% |
| Logistic Regression (Baseline) | Standard | 0.0385 | 0.6480 | 0.0362 | 0.6412 | 22.40% | 4.10% | 48.90% |

### Task C: Casualty Count Regression & Outlier Audit
*Holdout Test Set includes an extreme 70-casualty bus crash outlier in 2023. Models are evaluated WITH and WITHOUT the outlier.*

| Model Architecture | Objective / Formulation | All Test RMSE | Excl. Outlier RMSE | All Test MAE | All Test Deviance | Exact-Match % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (RMSE Winner)** | **MSE Loss** | **0.6848** | **0.6507** | **0.3660** | **0.2198** | 75.99% |
| **LightGBM (Poisson Displaced)**| **Displaced Count ($z=y-1$)**| 0.6854 | 0.6515 | 0.3702 | 0.2203 | **76.46%** |
| Random Forest | MSE Loss | 0.6864 | 0.6524 | 0.3658 | 0.2211 | 75.75% |
| Decision Tree | MSE Loss | 0.6898 | 0.6559 | 0.3684 | 0.2242 | 75.43% |
| Ridge Regression | L2 Penalized Linear | 0.6921 | 0.6586 | 0.3824 | 0.2268 | 76.31% |
| *Predict-1 Baseline* | Mode ($\hat{y} = 1$) | 0.7868 | 0.7572 | **0.2755** | 0.3440 | **81.48%** |
| *Predict-Mean Baseline* | Training Mean ($\hat{y} = 1.35$) | 0.7370 | 0.7056 | 0.4475 | 0.2743 | 81.48% |

> **Key Count Modeling Insight**: Because ~81.5% of crashes have exactly 1 casualty, the naive `predict-1` baseline wins on raw MAE and exact-match percentage. However, all ML models beat both naive baselines on **RMSE** (0.6848 vs 0.7868) and **Poisson Deviance** (0.2198 vs 0.3440), proving superior expected-count estimation for multi-vehicle, multi-casualty incidents.

### Bootstrap 95% Confidence Intervals & Paired Tests
Calculated over $B = 1,000$ stratified bootstrap resamples on the 104,258 holdout collisions:
- **Task A LightGBM-CW**:
  - PR-AUC: **0.4178** [0.4117, 0.4245]
  - ROC-AUC: **0.6920** [0.6883, 0.6958]
  - Dangerous Recall: **93.24%** [92.93%, 93.54%]
  - Dangerous Precision: **27.59%** [27.29%, 27.88%]
  - Dangerous F2: **0.6318** [0.6281, 0.6353]
- **Task B LightGBM-CW**:
  - Fatal PR-AUC: **0.0641** [0.0575, 0.0709]
  - Fatal Recall: **39.77%** [37.39%, 42.26%]
  - Fatal Precision: **6.56%** [6.04%, 7.07%]
  - Serious Recall: **61.66%** [61.05%, 62.26%]
- **Paired Bootstrap Superiority**:
  - Task A LightGBM vs Random Forest: $\Delta \text{PR-AUC} = +0.0050$ (95% CI: [+0.0031, +0.0070], $p < 0.001$).
  - Task B LightGBM vs Random Forest: $\Delta \text{Fatal PR-AUC} = +0.0013$ (95% CI: [+0.0002, +0.0026], $p = 0.024$).

---

## 5. Calibration & Operating Points

### Probability Calibration
Platt scaling (logistic sigmoid calibrator) was fit on validation probabilities after class weighting:
- **Task A Calibration**: Brier Score = 0.1718 vs No-Skill Reference = 0.1821 (**Brier Skill Score = +5.65%**).
- **Task B Fatal Calibration**: Brier Score = 0.0142 vs No-Skill Reference = 0.0144 (**Brier Skill Score = +1.38%**).

### Operating Points & Decile Lift Table
Policy triage points for transport authorities allocating emergency or enforcement resources:

| Operating Threshold | Population Flagged % | Captured Dangerous Collisions % | Capture Lift | Cumulative Precision |
| :--- | :---: | :---: | :---: | :---: |
| Top 10% Risk Cutoff ($\tau = 0.387$) | 10.0% | 19.3% | **1.93x** | 46.2% |
| Top 20% Risk Cutoff ($\tau = 0.324$) | 20.0% | 34.6% | **1.73x** | 41.4% |
| Top 30% Risk Cutoff ($\tau = 0.282$) | 30.0% | 47.9% | **1.60x** | 38.2% |
| Model F2 Optimal ($\tau = 0.120$) | 80.9% | 93.2% | **1.15x** | 27.6% |

---

## 6. Feature Ablation & Regional Generalization Audit

### Feature Bundle Ablation Study
Ablating coherent data groups shows the marginal impact on holdout test performance:
- **Vulnerability & Vehicles Ablated**: $\Delta \text{PR-AUC} = \mathbf{-0.0808}$ (95% CI: [-0.0857, -0.0755]) — **Largest statistical drop**.
- **Speed & Geometry Ablated**: $\Delta \text{PR-AUC} = \mathbf{-0.0299}$ (95% CI: [-0.0329, -0.0271]) — Highly significant.
- **Environmental & Surface Ablated**: $\Delta \text{PR-AUC} = -0.0079$ — Minor impact when vehicle types and road limits are known.
- **Temporal Dynamics Ablated**: $\Delta \text{PR-AUC} = -0.0022$ — Marginal contribution.

### Regional Generalization & Fairness Audit
Auditing across Urban vs Rural areas and the top 10 UK Police Forces:
- **Urban Collisions** (N = 70,312): Base Danger Rate = 21.43%, Model Recall = **90.35%**, PR-AUC = 0.3603, ROC-AUC = 0.6838.
- **Rural Collisions** (N = 33,934): Base Danger Rate = 29.15%, Model Recall = **97.65%**, PR-AUC = 0.4878, ROC-AUC = 0.6906.
- **Police Forces Ranking Consistency**: Across Metropolitan Police (London), West Midlands, West Yorkshire, Kent, Hampshire, Thames Valley, and Devon & Cornwall, ROC-AUC remains stable between **0.644 and 0.731**, demonstrating that risk discrimination does not degrade in regional deployment.

---

## 7. SHAP Interpretability & Explainability

RoadSense AI uses **TreeExplainer** SHAP values to explain predictions at both macro and micro levels:

### Global Attribution Drivers
1. **Most Vulnerable Vehicle Type**: Collisions involving motorcycles, pedal cyclists, or pedestrians show the largest positive push toward Dangerous/Fatal outcomes.
2. **Speed Limit (mph)**: Higher posted speed limits (50-70 mph) significantly elevate the log-odds of fatality due to higher impact kinetics.
3. **Number of Vehicles**: Strongest predictor for Task C (Casualty Count Regression).
4. **Driver Age Band Youngest**: Inexperience is heavily weighted in multi-vehicle dynamic risk.

### Local Waterfall Case Studies
Every prediction in RoadSense AI can be decomposed into an auditable waterfall explanation:
- **Critical Risk Collision**: Motorcycle on 60 mph rural single-carriageway at night $\to$ Calibrated Dangerous Prob: **74.2%**, Fatal Prob: **8.9%**.
- **Low Risk Collision**: Two passenger cars at 20 mph urban roundabout in daylight $\to$ Calibrated Dangerous Prob: **9.1%**, Minor outcome predicted with high confidence.

---

## 8. Interactive Terminal Prediction Tool (`predict.py`)

RoadSense AI provides a full-featured terminal CLI for immediate scenario risk assessment and batch CSV scoring:

### Interactive Mode
```bash
python predict.py
```
Prompts the user for collision variables (speed limit, weather, lighting, road type, junction, vehicle types, vehicle age) and prints formatted risk assessments:

```text
══════════════════════════════════════════════════════════════════════
   📊  PREDICTION RESULTS
══════════════════════════════════════════════════════════════════════

  ┌─ TASK A: Binary Severity Assessment
  │
  │  Prediction:  DANGEROUS
  │  Probability: 31.0% (threshold: 12%)
  │
  │  Risk Band:   🟡 ELEVATED
  │               Top 30% highest-risk — heightened monitoring
  │  Model:       LightGBM (Class_Weights)
  └──────────────────────────────────────────────────

  ┌─ TASK B: 3-Class Severity Breakdown
  │
  │  Prediction:       SERIOUS
  │
  │  Class Probabilities:
  │       Fatal: ███░░░░░░░░░░░░░░░░░░░░░░░░░░░ 12.6%
  │     Serious: ██████████████████░░░░░░░░░░░░ 60.8%
  │      Slight: ███████░░░░░░░░░░░░░░░░░░░░░░░ 26.6%
  │
  │  Calibrated Fatal:  0.51%
  │  Model:             LightGBM (Class_Weights)
  └──────────────────────────────────────────────────

  ┌─ TASK C: Expected Casualty Count
  │
  │  Predicted Casualties: 1
  │  Model:                LightGBM (RMSE)
  └──────────────────────────────────────────────────
```

### Batch Mode
Score an entire CSV dataset of collision scenarios:
```bash
python predict.py --batch my_scenarios.csv
```
Outputs `my_scenarios_predictions.csv` with predicted severity categories, calibrated probabilities, and risk band assignments.

---

## 9. Repository Structure

```text
RoadSense/
├── data/
│   ├── clean_roadsense.parquet          # Master cleaned dataset (2021-2023, 330k records)
│   └── raw/                             # Raw DfT STATS19 CSV archives
├── models/
│   ├── best_model_task_a.joblib         # Winning LightGBM (Class Weights) + Platt Calibrator
│   ├── best_model_task_b.joblib         # Winning LightGBM (Class Weights) + Multi-Calibrator
│   └── best_model_task_c.joblib         # Winning LightGBM (RMSE Regression)
├── notebooks/
│   ├── 01_data_cleaning.ipynb           # Stage 1 Data Extraction & Cleaning
│   ├── 02_eda.ipynb                     # Stage 2 Exploratory Data Analysis & Statistics
│   ├── 03_task_a_binary.ipynb           # Stage 3-5 Task A Modeling, Tuning & Diagnostics
│   ├── 04_task_b_multiclass.ipynb       # Stage 4-5 Task B 3-Class Modeling & Hierarchical Tuning
│   ├── 05_task_c_regression.ipynb       # Stage 4-5 Task C Casualty Count & Displaced Poisson
│   └── 06_interpretability_and_ablation.ipynb # Stage 6-7 Ablation, Regional Audit & SHAP
├── outputs/
│   ├── charts/                          # 27 publication-grade diagnostic plots & SHAP figures
│   └── tables/                          # 22 verified CSV result tables & benchmark comparisons
├── src/
│   ├── __init__.py
│   ├── data.py                          # Data ingestion, cleaning, sentinel mapping
│   ├── eda.py                           # Statistical chart generation
│   ├── features.py                      # ColumnTransformer preprocessor & feature lists
│   ├── baselines.py                     # Stage 3 baseline evaluation
│   ├── train.py                         # Full 15-model cross-validation trainer
│   ├── train_refinement.py              # Calibration, hierarchical tuning & operating points
│   ├── train_regression.py              # Task C regression trainer & displaced Poisson
│   ├── evaluate.py                      # Metrics, Platt calibrator, bootstrap routines
│   ├── ablation.py                      # Feature group ablation & regional fairness audit
│   ├── shap_analysis.py                 # TreeExplainer SHAP attribution pipeline
│   └── create_interpretability_notebook.py
├── predict.py                           # Production CLI interactive & batch prediction tool
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 10. Reproducibility & Quickstart

### Prerequisites
- Python 3.10 or 3.11
- Virtual environment tool (`venv` or `conda`)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/RoadSense.git
cd RoadSense

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Reproducing Analysis & Models
```bash
# 1. Clean raw data and build master parquet
python src/data.py

# 2. Run exploratory data analysis and generate EDA charts
python src/eda.py

# 3. Train all model configurations and select winners
python src/train.py

# 4. Train regression head and evaluate casualty count baselines
python src/train_regression.py

# 5. Run feature group ablation and regional generalization audit
python src/ablation.py

# 6. Compute SHAP explanations and export interpretability charts
python src/shap_analysis.py

# 7. Launch the interactive prediction terminal
python predict.py
```

---

## 🏛️ Acknowledgments & Citation
This project utilizes road safety data collected by police forces across Great Britain and published by the **Department for Transport (DfT)** under the Open Government Licence (OGL v3.0).

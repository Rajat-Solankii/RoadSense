# 🚗 RoadSense: Road Collision Severity & Casualty Prediction Model

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-success.svg)](https://lightgbm.readthedocs.io/)
[![Benchmark: MoRTH](https://img.shields.io/badge/Benchmark-MoRTH%20Official-orange.svg)](https://morth.nic.in/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RoadSense** is a machine learning prediction system engineered to assess **collision severity, outcome probabilities, and expected casualty counts** from road, vehicular, environmental, and behavioral variables. Grounded in empirical distribution benchmarks from the **Ministry of Road Transport and Highways (MoRTH)**, RoadSense delivers calibrated predictions via both a high-performance **CLI tool** and an interactive **Flask Web Dashboard**.

---

## 📑 Machine Learning Prediction Tasks

1. **Task A (Binary Severity Classification)**:
   - Evaluates whether a collision is **Dangerous** (*Fatal or Grievous Injury*) versus **Minor / Property Damage**.
   - Tuned with $F_2$-score threshold optimization on probability-calibrated holdouts.
2. **Task B (3-Class Severity Decomposition)**:
   - Multi-class breakdown across official classifications: **Fatal**, **Grievous Injury**, and **Minor / Property Damage**.
   - Outputs full probability distribution across all 3 severity classes.
3. **Task C (Casualty Count Regression)**:
   - Forecasts the expected number of casualties per collision ($y \ge 1$) using a Displaced Poisson count model ($z = y - 1$).

---

## ⚡ Quickstart: CLI Predictions

### 1. Instant Default Scenario Prediction
```bash
python predict.py --quick
```

### 2. Run Curated Scenario Presets (1 to 4)
```bash
python predict.py --preset 1
```

### 3. Custom Road, Speed & Vehicle Parameters
```bash
python predict.py --road "Expressway" --weather "Dense Winter Fog / Smog" --speed 120 --actual-speed 110 --vehicle "Car / Hatchback / Sedan" --other-vehicle "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)"
```

### 4. Interactive Console Mode
```bash
python predict.py
```

### 5. Batch CSV Audit
```bash
python predict.py --batch data/roadsense.csv
```

---

## 🌐 Interactive Web Dashboard

Launch the Flask web interface for interactive form predictions, preset scenario buttons, and batch CSV audits:

```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 📊 Model Performance on Holdout Benchmark

| Prediction Head | Target Formulation | Holdout Metric | Test Score |
| :--- | :--- | :--- | :---: |
| **Task A: Binary Severity** | Fatal/Grievous vs Minor | **PR-AUC**<br>**ROC-AUC**<br>**$F_2$-Score** | **0.9002**<br>**0.9424**<br>**0.8612** |
| **Task B: 3-Class Severity** | Fatal / Grievous / Minor | **Fatal PR-AUC**<br>**Fatal ROC-AUC** | **0.7860**<br>**0.9602** |
| **Task C: Casualty Count** | Displaced Poisson ($z = y - 1$) | **RMSE**<br>**MAE** | **1.3481**<br>**1.0465** |

---

## 📂 Repository Structure

```
RoadSense/
├── app.py                               # Flask prediction web server (REST API)
├── predict.py                           # CLI prediction tool (quick, presets, batch, flags)
├── requirements.txt                     # Dependencies (lightgbm, scikit-learn, flask, etc.)
├── pytest.ini                           # Pytest configuration
├── README.md                            # Project documentation
├── data/
│   ├── morth_official_statistics.csv    # Official MoRTH benchmark distribution statistics
│   ├── roadsense.parquet                # Calibrated collision dataset (60,000 records)
│   └── roadsense.csv                    # Sample CSV for inspection
├── models/
│   ├── model_task_a.joblib              # LightGBM-CW (Binary Severity) + Platt Calibrator
│   ├── model_task_b.joblib              # LightGBM Multiclass (Fatal / Grievous / Minor)
│   └── model_task_c.joblib              # LightGBM Displaced Poisson (Casualty Count)
├── src/
│   ├── __init__.py
│   ├── data.py                          # MoRTH calibrated dataset generation pipeline
│   ├── features.py                      # ColumnTransformer, feature lists, lookups
│   └── train.py                         # Multi-task training pipeline & threshold tuning
├── templates/
│   └── index.html                       # Web prediction dashboard layout
├── static/
│   ├── css/style.css                    # Responsive dark theme UI stylesheet
│   └── js/app.js                        # Frontend controller (reactive inference, presets, CSV batch)
├── tests/                               # Automated test suite
│   ├── conftest.py
│   ├── test_app.py                      # Flask REST API tests
│   ├── test_cli.py                      # CLI execution tests
│   ├── test_data.py                     # Dataset generation & loading tests
│   ├── test_features.py                 # Feature definitions & ColumnTransformer tests
│   └── test_predict.py                  # End-to-end model inference tests
└── outputs/
    └── tables/
        └── model_performance.csv        # Holdout test evaluation results
```

---

## 🔄 Retraining Models

To regenerate the benchmark dataset and re-train all models:
```bash
python -m src.train
```

---

## 🧪 Running Automated Tests

Run the full automated test suite:
```bash
pytest -v
```

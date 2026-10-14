# 🚗 RoadSense AI: Indian Road Safety & Disaster Mitigation System 🇮🇳

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-success.svg)](https://lightgbm.readthedocs.io/)
[![Database](https://img.shields.io/badge/Database-MoRTH%20%2F%20iRAD-orange.svg)](https://morth.nic.in/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RoadSense AI** is a multi-task machine learning and disaster prevention system engineered for **Indian roads, vehicular dynamics, and climatic conditions**. Grounded in official empirical statistics from the **Ministry of Road Transport and Highways (MoRTH)** and the **Integrated Road Accident Database (iRAD)**, RoadSense AI forecasts collision severity, expected casualties, and catastrophic risk escalation—while prescribing actionable directives to avert large-scale disasters.

---

## 📑 Core Capabilities

1. **Task A (Binary Severity Assessment)**: Dangerous (*Fatal or Grievous*) vs Minor collision risk.
2. **Task B (3-Class Severity Breakdown)**: Granular risk decomposition across *Fatal*, *Grievous Injury*, and *Minor / Property Damage Only* (IPC/MoRTH classification).
3. **Task C (Casualty Count Regression)**: Expected number of casualties per collision ($y \ge 1$) using Displaced Poisson count modeling.
4. **Disaster Risk Index (DRI)**: 0 to 100% systemic score assessing whether a collision will trigger mass-casualty secondary pileups (e.g., 20+ vehicle expressway fog collisions, gorge plunges).
5. **Actionable Emergency Protocol**:
   - Pre-trip / In-transit speed caps and visibility guidelines.
   - Post-crash scene shielding & immediate evacuation **behind W-beam metal crash barriers**.
   - Deployment of retro-reflective triangles **75–100m upstream**.
   - Immediate **Golden Hour** dispatch via **NHAI 1033** and National Emergency **112**.
   - Good Samaritan legal protection under Section 134A of the Motor Vehicles Act.

---

## 🏛️ Indian Database & Domain Integration

RoadSense AI incorporates empirical distribution tables and causal parameters from:
* **MoRTH Annual Reports ("Road Accidents in India" 2022 & 2023)**: Transport Research Wing (TRW), Ministry of Road Transport and Highways, Government of India.
* **iRAD (Integrated Road Accident Database)**: Developed by MoRTH in collaboration with IIT Madras and NIC.

### Official Benchmarks Grounded in the System:
* **Road Hierarchies**: Expressways (Yamuna, Samruddhi, NE-4, Mumbai-Pune), National Highways (NH: accounts for 36.2% of fatalities), State Highways (SH: 24.3%), Major District Roads (MDR), PMGSY Rural Roads, and Ghat Hairpin Terrains.
* **Vehicle Mix**: Two-Wheelers (~44.8% of all road deaths), Auto/E-Rickshaws (~4.6%), Pedestrians (~19.8%), Tractors with Trolleys, State/Private Buses, LCVs/Tempos, and Multi-Axle Freight Trucks.
* **Climatic Hazards**: Dense Northern Winter Fog/Smog (expressway pileups), Monsoon Waterlogging (two-wheeler aquaplaning and brake fade), Extreme Summer Heatwaves (>42°C concrete tire blowouts), and Dust Storms (Andhi).
* **Behavioral & Mindset Violations**:
  - Overspeeding (>68% of fatal crashes)
  - Wrong-side / Contraflow driving (~5.5% of deaths)
  - Overloading & Triple riding
  - Non-wearing of helmets (>70% of two-wheeler deaths) & seatbelts
  - Night high-beam glare on undivided carriageways
  - Circadian low fatigue (01:00 AM – 05:30 AM trucking window)

---

## ⚡ Quickstart: Instant Predictions

### 1. Instant Default Scenario (1 Second)
```bash
python predict.py --quick
```

### 2. High-Speed Dense Fog on an Expressway
```bash
python predict.py --road "Expressway" --weather "Dense Winter Fog / Smog" --speed 120 --actual-speed 110 --vehicle "Car / Hatchback / Sedan" --other-vehicle "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)"
```

### 3. Monsoon Ghat Hairpin with a Two-Wheeler & Contraflow Overtaking
```bash
python predict.py --road "Ghat / Hill Terrain Road" --weather "Heavy Monsoon / Waterlogged" --speed 40 --vehicle "Two-Wheeler (Motorcycle / Scooter)" --other-vehicle "Bus (State Transport / Private / School)" --wrong-side --no-helmet
```

### 4. Interactive Mode with Presets
```bash
python predict.py
```

---

## 🌐 Interactive Web Frontend (UI)

RoadSense AI includes an interactive web dashboard with **1-click clickable scenario buttons** (Yamuna Fog Pileup, Western Ghats Monsoon, NH-48 Midnight Wrong-Side, Bangalore Auto Overcrowding) and real-time risk assessment output.

### Launch Flask Web App (Recommended)
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser to interact with the full dashboard.

### Launch Streamlit App (Alternative)
```bash
streamlit run streamlit_app.py
```

---

## 📊 Model Performance on Indian Benchmark

| Prediction Head | Target Formulation | Holdout Metric | Test Score |
| :--- | :--- | :--- | :---: |
| **Task A: Binary Severity** | Fatal/Grievous vs Minor | **PR-AUC**<br>**ROC-AUC**<br>**$F_2$-Score** | **0.9002**<br>**0.9424**<br>**0.8612** |
| **Task B: 3-Class Severity** | Fatal / Grievous / Minor | **Fatal PR-AUC**<br>**Fatal ROC-AUC** | **0.7860**<br>**0.9602** |
| **Task C: Casualty Count** | Displaced Poisson ($z = y - 1$) | **RMSE**<br>**MAE** | **1.3481**<br>**1.0465** |

---

## 📂 Repository Structure

```
RoadSense/
├── data/
│   ├── morth_official_statistics.csv    # Official MoRTH 2022-2023 statistics
│   ├── roadsense.parquet                # Calibrated collision dataset (60,000 records)
│   └── roadsense.csv                    # Sample CSV for inspection
├── models/
│   ├── model_task_a.joblib              # LightGBM-CW (Binary Severity) + Platt Calibrator
│   ├── model_task_b.joblib              # LightGBM Multiclass (Fatal / Grievous / Minor)
│   └── model_task_c.joblib              # LightGBM Displaced Poisson (Casualty Count)
├── outputs/
│   └── tables/
│       └── model_performance.csv        # Holdout test evaluation results
├── src/
│   ├── __init__.py
│   ├── features.py                      # ColumnTransformer, feature lists, lookups
│   ├── data.py                          # MoRTH/iRAD calibrated dataset pipeline
│   ├── train.py                         # Multi-task training pipeline & threshold tuning
│   └── disaster_prevention.py           # Disaster Risk Index & Emergency mitigation engine
├── predict.py                           # Primary CLI tool (quick, flags, presets, batch)
├── requirements.txt                     # Dependencies (lightgbm, scikit-learn, etc.)
└── README.md                            # Project documentation
```

---

## 🔄 Retraining Models

To regenerate the dataset and re-train all models:
```bash
python -m src.train
```

---

## 🧪 Running Automated Tests

Run the full automated test suite covering feature engineering, data synthesis, disaster mitigation rules, model predictions, and CLI arguments:

```bash
pytest -v
```

---

## 🏛️ Acknowledgments & References
- **Ministry of Road Transport and Highways (MoRTH)**, Government of India: *"Road Accidents in India" (2022 & 2023 reports)*.
- **Integrated Road Accident Database (iRAD)**: Developed by MoRTH, IIT Madras, and NIC.
- **National Highways Authority of India (NHAI)**: 24x7 Emergency Incident Management (Helpline: **1033**).
- **Ministry of Home Affairs (MHA)**: Emergency Response Support System (Helpline: **112**).

"""
RoadSense AI: Generator for Notebooks 03 and 04
Generates:
  - notebooks/03_task_a_binary.ipynb
  - notebooks/04_task_b_multiclass.ipynb
"""

import json
import os

def create_notebook_03():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RoadSense AI: Multi-Task Machine Learning for Road Accident Severity\n",
                    "## Notebook 03: Task A — Binary Severity Classification (Dangerous vs Minor)\n",
                    "**Course Module**: Artificial Intelligence (CO5: Classification, Regression and Prediction Problems)\n",
                    "**Lead Team Member**: Member 2 (Binary Classification & Imbalance Specialist)\n",
                    "\n",
                    "---\n",
                    "\n",
                    "### Objectives & Rigorous Protocol:\n",
                    "1. **Binary Target Formulation**: Predict whether a crash is `Dangerous` (Fatal + Serious = 1) or `Minor` (Slight = 0).\n",
                    "2. **Strict Split Isolation**: Train on 2021, tune hyperparameters and decision thresholds on 2022 validation, refit on 2021+2022, and test once on 2023 holdout test.\n",
                    "3. **Imbalance Benchmark (15 Configurations)**: Compare Class Weights vs SMOTE vs Random Undersampling across 5 model architectures (Decision Tree, Random Forest, LightGBM, XGBoost, Small MLP).\n",
                    "4. **Probability Recalibration**: Correct probability distortions caused by resampling/weighting using Platt scaling on the validation fold.\n",
                    "5. **F2 Threshold Maximization**: Optimize decision threshold $\\tau^*$ on validation fold to weight recall twice as much as precision (and report the 80% recall benchmark).\n",
                    "6. **Validation-Only Selection**: Select the winning model purely by 2022 validation metrics.\n",
                    "7. **Leakage Safeguard**: Automatically check ROC-AUC <= 0.90 to guarantee zero data leakage."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import joblib\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from IPython.display import display, Image\n",
                    "\n",
                    "# Ensure project root is accessible\n",
                    "sys.path.append(os.path.abspath('..'))\n",
                    "from src.features import ALL_MODEL_FEATURES\n",
                    "from src.evaluate import evaluate_binary_classification\n",
                    "\n",
                    "print('Environment and libraries initialized.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Clean Dataset & Verify Split Integrity\n",
                    "We load the clean master dataset (`data/clean_roadsense.parquet`).\n",
                    "We verify the strict chronological split: 2021 (Train), 2022 (Validation), 2023 (Holdout Test)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_parquet('../data/clean_roadsense.parquet')\n",
                    "print(f'Total records: {len(df):,}')\n",
                    "print('Distribution by year:')\n",
                    "print(df['collision_year'].value_counts().sort_index())\n",
                    "\n",
                    "print('\\nTask A Binary Class Distribution (Dangerous vs Minor):')\n",
                    "class_counts = df['target_severity_binary'].value_counts()\n",
                    "class_props = df['target_severity_binary'].value_counts(normalize=True) * 100\n",
                    "display(pd.DataFrame({'Count': class_counts, 'Percentage (%)': class_props.round(2)}))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Validation Comparison Across 15 Configurations\n",
                    "All 15 model configurations were trained on 2021 and evaluated on 2022 validation with Platt recalibration and $F_2$ threshold optimization.\n",
                    "Winning model selection is determined **exclusively** by 2022 validation metrics."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "val_table = pd.read_csv('../outputs/tables/task_a_validation_comparison.csv')\n",
                    "print('Top 5 Task A Configurations Ranked by Validation Macro-F1:')\n",
                    "display(val_table.sort_values(by='macro_f1', ascending=False).head(5)[[\n",
                    "    'model', 'imbalance_strategy', 'train_time_sec', 'optimal_f2_threshold', \n",
                    "    'max_f2_score', 'macro_f1', 'dangerous_recall', 'dangerous_precision', 'roc_auc', 'leakage_warning'\n",
                    "]])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Probability Recalibration (Platt Scaling)\n",
                    "Resampling techniques (such as SMOTE and Random Undersampling) or synthetic class weighting distort the posterior probabilities away from real-world risk proportions.\n",
                    "We inspect the calibration curve before and after Platt scaling."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/charts/09_calibration_curve_task_a.png', width=750))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Winning Model Artifact & 2023 Holdout Test Evaluation\n",
                    "The winning configuration was refitted on the combined 2021+2022 dataset.\n",
                    "We now load the saved joblib model artifact (`models/best_model_task_a.joblib`) and test it on the untouched 2023 holdout test set."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model_artifact = joblib.load('../models/best_model_task_a.joblib')\n",
                    "print(f'Winning Model: {model_artifact[\"model_name\"]} ({model_artifact[\"imbalance_strategy\"]})')\n",
                    "print(f'Optimal Threshold: {model_artifact[\"optimal_threshold\"]}')\n",
                    "print(f'Tuned Hyperparameters: {model_artifact[\"hyperparameters\"]}')\n",
                    "\n",
                    "# Test on 2023\n",
                    "test_df = df[df['collision_year'] == 2023]\n",
                    "X_test = test_df[ALL_MODEL_FEATURES]\n",
                    "y_test = test_df['target_severity_binary'].values\n",
                    "\n",
                    "raw_probs = model_artifact['pipeline'].predict_proba(X_test)[:, 1]\n",
                    "calib_probs = model_artifact['calibrator'].predict_proba(raw_probs)\n",
                    "preds = (calib_probs >= model_artifact['optimal_threshold']).astype(int)\n",
                    "\n",
                    "test_metrics = evaluate_binary_classification(y_test, preds, calib_probs)\n",
                    "for k, v in test_metrics.items():\n",
                    "    if k != 'confusion_matrix':\n",
                    "        print(f'{k:<25}: {v}')\n",
                    "\n",
                    "print('\\n2023 Test Confusion Matrix:')\n",
                    "cm = np.array(test_metrics['confusion_matrix'])\n",
                    "cm_df = pd.DataFrame(cm, index=['Actual Minor', 'Actual Dangerous'], columns=['Pred Minor', 'Pred Dangerous'])\n",
                    "display(cm_df)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Post-Selection Reference Table (All 15 Models on 2023 Test)\n",
                    "For full transparency and academic audit, we present the performance of all 15 models on the 2023 test set.\n",
                    "**Crucial Academic Integrity Note**: None of these test results were used for model selection or hyperparameter tuning."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ref_table = pd.read_csv('../outputs/tables/task_a_post_selection_reference.csv')\n",
                    "display(ref_table[[\n",
                    "    'model', 'imbalance_strategy', 'applied_threshold', 'train_time_sec',\n",
                    "    'macro_f1', 'dangerous_recall', 'dangerous_precision', 'roc_auc', 'pr_auc', 'leakage_warning'\n",
                    "]].sort_values(by='macro_f1', ascending=False))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open("notebooks/03_task_a_binary.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("[+] Generated notebooks/03_task_a_binary.ipynb")


def create_notebook_04():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RoadSense AI: Multi-Task Machine Learning for Road Accident Severity\n",
                    "## Notebook 04: Task B — 3-Class Severity Classification (Fatal vs Serious vs Slight)\n",
                    "**Course Module**: Artificial Intelligence (CO5: Classification, Regression and Prediction Problems)\n",
                    "**Lead Team Member**: Member 3 (Multi-Class & Extreme Imbalance Specialist)\n",
                    "\n",
                    "---\n",
                    "\n",
                    "### Objectives & Rigorous Protocol:\n",
                    "1. **3-Class Formulation**: Predict exact severity level — `Fatal` (0, ~1.48%), `Serious` (1, ~21.86%), or `Slight` (2, ~76.66%).\n",
                    "2. **The Extreme Imbalance Challenge**: Unweighted baselines predict 0% Fatal recall. We compare Class Weights vs SMOTE vs Random Undersampling across 5 model architectures.\n",
                    "3. **One-vs-Rest (OvR) Fatal Decision Rule**: Standard argmax prediction suppresses the rare Fatal class. We apply an OvR threshold $\\tau_{\\text{fatal}}^*$ maximizing $F_2$ for the Fatal class on 2022 validation.\n",
                    "4. **Platt Probability Recalibration**: Recalibrate Fatal predicted probabilities on the validation fold.\n",
                    "5. **Validation-Only Selection**: Model selection is governed exclusively by 2022 validation metrics.\n",
                    "6. **Complete Reporting**: Report per-class precision, recall, and F1 for all three classes, plus the full 3x3 confusion matrix on 2023 holdout test."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import joblib\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from IPython.display import display, Image\n",
                    "\n",
                    "# Ensure project root is accessible\n",
                    "sys.path.append(os.path.abspath('..'))\n",
                    "from src.features import ALL_MODEL_FEATURES\n",
                    "from src.evaluate import evaluate_multiclass_classification\n",
                    "\n",
                    "print('Environment and libraries initialized.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Clean Dataset & Inspect 3-Class Severe Imbalance\n",
                    "We inspect the distribution across Fatal (0), Serious (1), and Slight (2)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_parquet('../data/clean_roadsense.parquet')\n",
                    "counts = df['target_severity_3class'].value_counts().sort_index()\n",
                    "props = (df['target_severity_3class'].value_counts(normalize=True).sort_index() * 100).round(2)\n",
                    "labels = ['0: Fatal', '1: Serious', '2: Slight']\n",
                    "display(pd.DataFrame({'Severity Class': labels, 'Count': counts.values, 'Percentage (%)': props.values}))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Validation Comparison Across 15 Configurations\n",
                    "We load the 2022 validation results table (`outputs/tables/task_b_validation_comparison.csv`).\n",
                    "Notice the dramatic improvement in Fatal recall when using OvR $F_2$ thresholding compared to standard argmax baselines."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "val_b_table = pd.read_csv('../outputs/tables/task_b_validation_comparison.csv')\n",
                    "print('Top 5 Task B Configurations Ranked by Validation Macro-F1:')\n",
                    "display(val_b_table.sort_values(by='macro_f1', ascending=False).head(5)[[\n",
                    "    'model', 'imbalance_strategy', 'train_time_sec', 'optimal_fatal_threshold', \n",
                    "    'fatal_f2_score', 'macro_f1', 'fatal_recall', 'fatal_precision', 'fatal_f1', \n",
                    "    'serious_recall', 'slight_recall', 'roc_auc_ovr', 'leakage_warning'\n",
                    "]])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Fatal Class Probability Recalibration Curve\n",
                    "We inspect the calibration curve for the Fatal class before and after Platt recalibration."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/charts/10_calibration_curve_task_b.png', width=750))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Winning Model Artifact & 2023 Holdout Test Evaluation\n",
                    "We evaluate the winning model on the 2023 holdout test set with complete per-class metrics and confusion matrix."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model_b_artifact = joblib.load('../models/best_model_task_b.joblib')\n",
                    "print(f'Winning Model: {model_b_artifact[\"model_name\"]} ({model_b_artifact[\"imbalance_strategy\"]})')\n",
                    "print(f'Optimal Fatal OvR Threshold: {model_b_artifact[\"optimal_fatal_threshold\"]}')\n",
                    "print(f'Tuned Hyperparameters: {model_b_artifact[\"hyperparameters\"]}')\n",
                    "\n",
                    "test_df = df[df['collision_year'] == 2023]\n",
                    "X_test = test_df[ALL_MODEL_FEATURES]\n",
                    "y_test = test_df['target_severity_3class'].values\n",
                    "\n",
                    "test_probs = model_b_artifact['pipeline'].predict_proba(X_test)\n",
                    "raw_fatal = test_probs[:, 0]\n",
                    "calib_fatal = model_b_artifact['fatal_calibrator'].predict_proba(raw_fatal)\n",
                    "\n",
                    "opt_fatal_t = model_b_artifact['optimal_fatal_threshold']\n",
                    "test_preds = []\n",
                    "for i in range(len(test_probs)):\n",
                    "    if calib_fatal[i] >= opt_fatal_t:\n",
                    "        test_preds.append(0)\n",
                    "    else:\n",
                    "        if test_probs[i, 1] >= test_probs[i, 2]:\n",
                    "            test_preds.append(1)\n",
                    "        else:\n",
                    "            test_preds.append(2)\n",
                    "test_preds = np.array(test_preds)\n",
                    "\n",
                    "test_metrics_b = evaluate_multiclass_classification(y_test, test_preds, test_probs)\n",
                    "\n",
                    "print('\\n--- 2023 Holdout Test Per-Class Performance ---')\n",
                    "per_class_summary = pd.DataFrame({\n",
                    "    'Class': ['Fatal (0)', 'Serious (1)', 'Slight (2)'],\n",
                    "    'Precision': [test_metrics_b['fatal_precision'], test_metrics_b['serious_precision'], test_metrics_b['slight_precision']],\n",
                    "    'Recall': [test_metrics_b['fatal_recall'], test_metrics_b['serious_recall'], test_metrics_b['slight_recall']],\n",
                    "    'F1-Score': [test_metrics_b['fatal_f1'], test_metrics_b['serious_f1'], test_metrics_b['slight_f1']]\n",
                    "})\n",
                    "display(per_class_summary)\n",
                    "\n",
                    "print(f'Macro-F1: {test_metrics_b[\"macro_f1\"]:.4f} | Overall Accuracy: {test_metrics_b[\"accuracy\"]:.4f} | OVR ROC-AUC: {test_metrics_b[\"roc_auc_ovr\"]:.4f}')\n",
                    "\n",
                    "print('\\n--- 2023 Holdout Test 3x3 Confusion Matrix ---')\n",
                    "cm = np.array(test_metrics_b['confusion_matrix'])\n",
                    "cm_df = pd.DataFrame(cm, index=['Actual Fatal', 'Actual Serious', 'Actual Slight'], columns=['Pred Fatal', 'Pred Serious', 'Pred Slight'])\n",
                    "display(cm_df)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Post-Selection Reference Table (All 15 Models on 2023 Test)\n",
                    "All 15 model configurations evaluated on 2023 test as post-selection reference."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ref_b_table = pd.read_csv('../outputs/tables/task_b_post_selection_reference.csv')\n",
                    "display(ref_b_table[[\n",
                    "    'model', 'imbalance_strategy', 'applied_fatal_threshold', 'train_time_sec',\n",
                    "    'macro_f1', 'fatal_recall', 'fatal_precision', 'fatal_f1',\n",
                    "    'serious_recall', 'slight_recall', 'roc_auc_ovr', 'leakage_warning'\n",
                    "]].sort_values(by='macro_f1', ascending=False))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open("notebooks/04_task_b_multiclass.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("[+] Generated notebooks/04_task_b_multiclass.ipynb")

if __name__ == "__main__":
    create_notebook_03()
    create_notebook_04()

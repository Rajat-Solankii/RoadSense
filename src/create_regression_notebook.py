"""
RoadSense AI: Generator for Notebook 05 (Task C Casualties Count Regression)
"""

import json
import os

def create_notebook_05():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RoadSense AI: Multi-Task Machine Learning for Road Accident Severity\n",
                    "## Notebook 05: Task C — Casualties Count Regression & Poisson Count Modeling\n",
                    "**Course Module**: Artificial Intelligence (CO5: Classification, Regression and Prediction Problems)\n",
                    "**Lead Team Member**: Member 4 (Count Data Modeling & Regression Specialist)\n",
                    "\n",
                    "---\n",
                    "\n",
                    "### Objectives & Rigorous Protocol:\n",
                    "1. **Count Modeling Challenge**: Casualties count is a strictly positive integer ($y \\ge 1$, since every collision in STATS19 involves at least 1 casualty). The distribution is heavily right-skewed with a mode of 1 (~79%), long tail, and an extreme 70-casualty bus crash outlier in the 2023 holdout test set.\n",
                    "2. **Baselines**: Evaluate against naive `predict-1` (mode) and `predict-mean` (training mean).\n",
                    "3. **Displaced Poisson Formulation**: Model displaced count $z = y - 1 \\ge 0$ with Poisson loss: $\\hat{y} = 1 + \\hat{\\lambda}$. This guarantees non-negative integer counts and heteroscedastic variance matching real-world crash count data.\n",
                    "4. **Model Comparison**: Ridge Regression, Decision Tree, Random Forest, LightGBM (RMSE), and LightGBM (Displaced Poisson).\n",
                    "5. **Robustness Audit**: Evaluate metrics WITH and WITHOUT the 2023 70-casualty extreme outlier to prove robustness of MAE and Poisson loss over MSE/RMSE.\n",
                    "6. **Exact-Match Accuracy**: Audit accuracy for exact-1, exact-2, and exact-3+ casualties."
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
                    "\n",
                    "print('Environment and libraries initialized.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Casualties Distribution & Outlier Inspection\n",
                    "We inspect the distribution across all 3 years and locate the 2023 70-casualty outlier."
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
                    "print('Summary statistics for number_of_casualties:')\n",
                    "print(df['target_casualties'].describe().round(3))\n",
                    "\n",
                    "print('\\nDistribution of Casualties Count:')\n",
                    "counts = df['target_casualties'].value_counts().head(5)\n",
                    "props = (df['target_casualties'].value_counts(normalize=True).head(5) * 100).round(2)\n",
                    "display(pd.DataFrame({'Count': counts, 'Percentage (%)': props}))\n",
                    "\n",
                    "outliers = df[df['target_casualties'] >= 20]\n",
                    "print(f'\\nCollisions with >= 20 casualties (extreme mass-casualty incidents): {len(outliers)}')\n",
                    "display(outliers[['collision_year', 'target_casualties', 'number_of_vehicles', 'any_hgv_or_bus', 'speed_limit']])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Validation Comparison (2022 Validation Fold)\n",
                    "We compare all models on the 2022 validation set."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "val_table = pd.read_csv('../outputs/tables/task_c_validation_comparison.csv')\n",
                    "display(val_table[['model', 'train_time_sec', 'mae', 'median_ae', 'rmse', 'poisson_deviance', 'exact_match_overall_pct', 'exact_match_1_pct']])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Holdout Test Set Comparison: Robustness With & Without the 70-Casualty Record\n",
                    "Notice how the single 70-casualty bus crash dramatically inflates RMSE across all models, while MAE, Poisson deviance, and exact match rates remain highly stable."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "test_table = pd.read_csv('../outputs/tables/task_c_regression_comparison.csv')\n",
                    "display(test_table[['model', 'eval_split', 'mae', 'rmse', 'poisson_deviance', 'exact_match_overall_pct', 'exact_match_1_pct', 'exact_match_2_pct', 'exact_match_3plus_pct']])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Visual Diagnostics: Actual vs Predicted Casualty Distribution\n",
                    "We inspect the diagnostic plot comparing empirical and predicted casualty distributions and error groupings."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/charts/13_task_c_regression_diagnostics.png', width=900))"
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
    with open("notebooks/05_task_c_regression.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("[+] Generated notebooks/05_task_c_regression.ipynb")

if __name__ == "__main__":
    create_notebook_05()

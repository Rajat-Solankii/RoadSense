"""
RoadSense AI: Generator for Notebook 06 (Interpretability, Ablation, and Governance)
"""

import json
import os

def create_notebook_06():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RoadSense AI: Multi-Task Machine Learning for Accident Severity and Casualties\n",
                    "## Notebook 06: Model Interpretability, Feature Ablation, and Regional Generalization Audit\n",
                    "**Course Module**: Artificial Intelligence (CO5: Classification, Regression and Prediction Problems)\n",
                    "**Lead Team Member**: Model Interpretability, Explainability & AI Governance Specialist\n",
                    "\n",
                    "---\n",
                    "\n",
                    "### Objectives & Rigorous Protocol:\n",
                    "1. **Feature Group Ablation Studies**: Systematically ablate coherent feature groups (*Speed & Geometry*, *Vulnerability & Vehicles*, *Environmental & Surface*, *Temporal Dynamics*) on the winning models to quantify delta PR-AUC and 95% bootstrap confidence intervals.\n",
                    "2. **Group Permutation Importance**: Evaluate the drop in holdout PR-AUC when shuffling entire feature bundles simultaneously, avoiding spurious single one-hot feature splitting artifacts.\n",
                    "3. **Regional Generalization & Equity Audit**: Benchmark model performance across Urban vs. Rural geographies and the top 10 UK Police Forces to detect geographic bias and operational blindspots.\n",
                    "4. **Global SHAP Feature Attribution**: Compute exact TreeExplainer SHAP values for Task A (Binary), Task B (Fatal / Serious / Slight), and Task C (Casualties) to reveal macro-level risk drivers via beeswarm and mean absolute SHAP bar charts.\n",
                    "5. **Local Waterfall Explanations & Counterfactuals**: Inspect individual case studies (highest-risk vs. median-risk vs. lowest-risk collisions) to provide auditable justification for emergency resource dispatch."
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
                    "CHARTS_DIR = '../outputs/charts'\n",
                    "TABLES_DIR = '../outputs/tables'\n",
                    "print('Environment and libraries initialized.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Feature Group Ablation Study\n",
                    "Ablating entire bundles of features allows us to quantify the marginal value of specific data categories.\n",
                    "Below we inspect the ablation results for Task A (Binary) and Task B (3-Class Fatal PR-AUC)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load Task A Ablation Table\n",
                    "df_abl_a = pd.read_csv(f'{TABLES_DIR}/task_a_ablation_study.csv')\n",
                    "print('Task A (Binary Severity) Feature Ablation Study:')\n",
                    "display(df_abl_a)\n",
                    "\n",
                    "# Load Task B Ablation Table\n",
                    "df_abl_b = pd.read_csv(f'{TABLES_DIR}/task_b_ablation_study.csv')\n",
                    "print('\\nTask B (Fatal Severity) Feature Ablation Study:')\n",
                    "display(df_abl_b)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Ablation Insights:\n",
                    "- **Vulnerability & Vehicles** is the single most critical feature group for Task A, causing a severe drop in PR-AUC of **-0.0808** (95% CI [-0.0857, -0.0755]). Without vehicle type and vulnerable road user flags, danger detection degrades dramatically.\n",
                    "- **Speed & Geometry** is the second most impactful bundle, reducing Task A PR-AUC by **-0.0299** and Task B Fatal PR-AUC by **-0.0188** (both statistically significant drops).\n",
                    "- **Environmental & Surface** and **Temporal Dynamics** show relatively minor or statistically indistinguishable drops when removed in the presence of vehicle and road geometry features."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Group Permutation Feature Importance\n",
                    "Group permutation importance shuffles feature bundles together, preserving within-bundle correlation while breaking the relationship with collision severity."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_perm = pd.read_csv(f'{TABLES_DIR}/group_permutation_importance.csv')\n",
                    "display(df_perm)\n",
                    "\n",
                    "display(Image(filename=f'{CHARTS_DIR}/14_group_permutation_importance.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Regional Generalization & AI Governance Audit\n",
                    "To ensure ethical and equitable AI deployment, we evaluate the models across Urban vs. Rural splits and across the top 10 UK Police Forces."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_reg = pd.read_csv(f'{TABLES_DIR}/regional_generalization_audit.csv')\n",
                    "display(df_reg)\n",
                    "\n",
                    "display(Image(filename=f'{CHARTS_DIR}/15_regional_generalization_audit.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Regional Audit Key Findings:\n",
                    "1. **Rural Collisions have Higher Baseline Danger**: Rural collisions have a 29.15% danger rate and 2.81% fatal rate (vs 21.43% and 0.81% in Urban areas). Consequently, Rural PR-AUC (0.4878) is higher due to higher prevalence, and model recall reaches **97.65%** for Dangerous and **60.27%** for Fatal.\n",
                    "2. **Urban Performance**: In urban areas (e.g. London Metropolitan Police), prevalence is lower and speeds are lower, yielding 90.8% Dangerous recall and 15.8% Fatal recall under the fixed decision threshold.\n",
                    "3. **Police Force Consistency**: ROC-AUC remains consistently robust across all police forces (0.64 - 0.73 for Task A, 0.69 - 0.83 for Task B), proving good ranking capability nationwide."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. SHAP Global Feature Attribution (Task A: Binary Severity)\n",
                    "We examine global feature attributions computed via TreeExplainer on the holdout test set."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_shap_a = pd.read_csv(f'{TABLES_DIR}/shap_importance_task_a.csv').head(15)\n",
                    "display(df_shap_a)\n",
                    "\n",
                    "print('Task A: Mean Absolute SHAP Importance Bar Chart:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_bar_task_a.png'))\n",
                    "\n",
                    "print('Task A: SHAP Beeswarm Distribution Plot:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_beeswarm_task_a.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Task A SHAP Dependence Plots (Top 5 Features)\n",
                    "These plots show how individual feature values influence the log-odds of a Dangerous collision."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for rank in range(1, 6):\n",
                    "    # Find matching dependence plot file\n",
                    "    matched = [f for f in os.listdir(CHARTS_DIR) if f.startswith(f'shap_dep_task_a_{rank}_')]\n",
                    "    if matched:\n",
                    "        print(f'Top {rank} Feature Dependence:')\n",
                    "        display(Image(filename=os.path.join(CHARTS_DIR, matched[0])))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. SHAP Global Feature Attribution (Task B: Fatal & Serious Classes)\n",
                    "For multiclass severity, we examine feature impact specifically on the rarest, highest-consequence class: **Fatal**."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_shap_b_fatal = pd.read_csv(f'{TABLES_DIR}/shap_importance_task_b_class0.csv').head(15)\n",
                    "display(df_shap_b_fatal)\n",
                    "\n",
                    "print('Task B (Fatal): Mean Absolute SHAP Importance Bar Chart:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_bar_task_b_fatal.png'))\n",
                    "\n",
                    "print('Task B (Fatal): SHAP Beeswarm Distribution Plot:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_beeswarm_task_b_fatal.png'))\n",
                    "\n",
                    "print('Task B (Serious): Mean Absolute SHAP Importance Bar Chart:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_bar_task_b_serious.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Local Waterfall Explanations (Case Studies)\n",
                    "Local waterfall plots decompose the model's prediction for an individual collision from the base rate to the final output."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Task A: Highest Risk Collision Waterfall:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_waterfall_task_a_highest_risk.png'))\n",
                    "\n",
                    "print('Task A: Median Risk Collision Waterfall:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_waterfall_task_a_median_risk.png'))\n",
                    "\n",
                    "print('Task A: Lowest Risk Collision Waterfall:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_waterfall_task_a_lowest_risk.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. SHAP Global Feature Attribution (Task C: Casualties Count)\n",
                    "Finally, we analyze the drivers of expected casualty counts."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_shap_c = pd.read_csv(f'{TABLES_DIR}/shap_importance_task_c.csv').head(15)\n",
                    "display(df_shap_c)\n",
                    "\n",
                    "print('Task C: Mean Absolute SHAP Importance Bar Chart:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_bar_task_c.png'))\n",
                    "\n",
                    "print('Task C: SHAP Beeswarm Distribution Plot:')\n",
                    "display(Image(filename=f'{CHARTS_DIR}/shap_beeswarm_task_c.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Summary & AI Governance Takeaways\n",
                    "1. **Physiological Vulnerability Dominates**: Across Task A and B, the presence of motorcycles and pedal cycles, along with high speed limits (50-70 mph), are the leading drivers of severe outcomes.\n",
                    "2. **Vehicle Count Drives Casualties**: For Task C, the number of vehicles involved and vehicle types (buses, multi-vehicle crashes) drive higher casualty counts.\n",
                    "3. **Interpretability Enhances Trust**: TreeExplainer SHAP waterfalls enable first responders and police planners to understand why a scenario received an Elevated or Critical risk classification.\n",
                    "4. **Deployment Readiness**: The models, calibrated thresholds, and explainability artifacts provide an end-to-end framework for evidence-based road safety intervention."
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
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = "notebooks/06_interpretability_and_ablation.ipynb"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[+] Successfully generated: {out_path}")

if __name__ == "__main__":
    create_notebook_06()

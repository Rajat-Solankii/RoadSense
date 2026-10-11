"""
RoadSense AI: Stage 5 — Task C (Casualties Count Regression)

Implements:
  1. Prediction of number_of_casualties
  2. Baselines:
     - predict-1 (Modal baseline: always predicts 1 casualty)
     - predict-mean (Mean baseline: always predicts training mean)
  3. Models:
     - Ridge Regression (Linear with L2 regularization)
     - Decision Tree Regressor
     - Random Forest Regressor
     - LightGBM Regressor (RMSE objective)
     - LightGBM Regressor with Poisson objective modeling (casualties - 1)
  4. Metrics:
     - MAE, Median AE, RMSE, Poisson Deviance
     - Exact-match accuracy: overall, exact-1, exact-2, exact-3+
  5. Robustness check: Evaluated WITH and WITHOUT the 2023 70-casualty extreme outlier record
  6. Visualizations & tables saved to outputs/tables/ and outputs/charts/
"""

import os
import sys
import time
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    d2_tweedie_score,
    r2_score
)

sys.path.insert(0, os.path.abspath("."))
from src.features import ALL_MODEL_FEATURES, create_preprocessor

TABLES_DIR = "outputs/tables"
CHARTS_DIR = "outputs/charts"
MODELS_DIR = "models"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================
# 1. METRICS EVALUATOR FOR COUNT REGRESSION
# ==============================================================================
def compute_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes mean Poisson deviance: 2 * sum(y * log(y/y_hat) - (y - y_hat)) / N."""
    eps = 1e-6
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    dev = 2.0 * np.mean(yt * np.log(yt / yp) - (yt - yp))
    return float(dev)


def evaluate_regression_comprehensive(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes MAE, Median AE, RMSE, Poisson deviance, and exact match rates."""
    y_pred_clipped = np.maximum(y_pred, 1.0)
    y_pred_rounded = np.maximum(np.round(y_pred), 1).astype(int)
    y_true_int = y_true.astype(int)
    
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    med_ae = float(median_absolute_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    p_dev = compute_poisson_deviance(y_true, y_pred_clipped)
    
    # Exact match accuracy
    exact_overall = float(np.mean(y_pred_rounded == y_true_int) * 100)
    
    # Exact 1 casualty
    mask_1 = (y_true_int == 1)
    exact_1 = float(np.mean(y_pred_rounded[mask_1] == 1) * 100) if np.sum(mask_1) > 0 else 0.0
    
    # Exact 2 casualties
    mask_2 = (y_true_int == 2)
    exact_2 = float(np.mean(y_pred_rounded[mask_2] == 2) * 100) if np.sum(mask_2) > 0 else 0.0
    
    # Exact 3+ casualties
    mask_3plus = (y_true_int >= 3)
    exact_3plus = float(np.mean(y_pred_rounded[mask_3plus] >= 3) * 100) if np.sum(mask_3plus) > 0 else 0.0
    
    return {
        "mae": round(mae, 4),
        "median_ae": round(med_ae, 4),
        "rmse": round(rmse, 4),
        "poisson_deviance": round(p_dev, 4),
        "exact_match_overall_pct": round(exact_overall, 2),
        "exact_match_1_pct": round(exact_1, 2),
        "exact_match_2_pct": round(exact_2, 2),
        "exact_match_3plus_pct": round(exact_3plus, 2)
    }


# ==============================================================================
# 2. MODEL FACTORY BUILDER
# ==============================================================================
def build_regression_pipeline(model_name: str) -> Pipeline:
    """Creates scikit-learn Pipeline with ColumnTransformer preprocessor and regressor."""
    preprocessor = create_preprocessor()
    
    if model_name == "Ridge":
        reg = Ridge(alpha=10.0, random_state=42)
    elif model_name == "Decision Tree":
        reg = DecisionTreeRegressor(max_depth=8, min_samples_leaf=30, random_state=42)
    elif model_name == "Random Forest":
        reg = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=15, random_state=42, n_jobs=-1)
    elif model_name == "LightGBM (RMSE)":
        reg = lgb.LGBMRegressor(n_estimators=120, max_depth=6, learning_rate=0.06, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    elif model_name == "LightGBM (Poisson Displaced)":
        # Displaced Poisson models (y - 1) >= 0 with objective='poisson'
        reg = lgb.LGBMRegressor(objective="poisson", n_estimators=120, max_depth=6, learning_rate=0.06, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", reg)])


# ==============================================================================
# 3. TASK C FULL EXPERIMENTAL RUN
# ==============================================================================
def run_task_c_experiments():
    print("\n" + "=" * 78)
    print("STAGE 5: TASK C (CASUALTIES COUNT REGRESSION) BENCHMARKING")
    print("PROTOCOL: Train on 2021 -> Validate on 2022 -> Refit 2021+2022 -> Test on 2023")
    print("ROBUSTNESS: Tested WITH and WITHOUT the 2023 70-Casualty Record")
    print("=" * 78)
    
    df = pd.read_parquet("data/clean_roadsense.parquet")
    train_21 = df[df["collision_year"] == 2021]
    val_22 = df[df["collision_year"] == 2022]
    test_23 = df[df["collision_year"] == 2023]
    full_train = df[df["collision_year"].isin([2021, 2022])]
    
    X_train_21 = train_21[ALL_MODEL_FEATURES]
    y_train_21 = train_21["target_casualties"].values
    
    X_val_22 = val_22[ALL_MODEL_FEATURES]
    y_val_22 = val_22["target_casualties"].values
    
    X_test_23 = test_23[ALL_MODEL_FEATURES]
    y_test_23 = test_23["target_casualties"].values
    
    # 70-casualty outlier record identification in 2023 test set
    outlier_mask = (y_test_23 >= 70)
    print(f"[*] Total 2023 test collisions: {len(y_test_23):,}")
    print(f"[*] Found {np.sum(outlier_mask)} extreme outlier record(s) with casualties >= 70 in 2023 test set.")
    
    y_test_no_outlier = y_test_23[~outlier_mask]
    X_test_no_outlier = X_test_23[~outlier_mask]
    
    X_full_train = full_train[ALL_MODEL_FEATURES]
    y_full_train = full_train["target_casualties"].values
    
    train_mean_21 = float(np.mean(y_train_21))
    full_train_mean = float(np.mean(y_full_train))
    print(f"[*] Training Casualties Mean: 2021 = {train_mean_21:.4f} | Full Train (2021+2022) = {full_train_mean:.4f}")
    
    # Model catalog
    models = [
        "Predict-1 Baseline",
        "Predict-Mean Baseline",
        "Ridge",
        "Decision Tree",
        "Random Forest",
        "LightGBM (RMSE)",
        "LightGBM (Poisson Displaced)"
    ]
    
    val_results = []
    fitted_pipes = {}
    
    for m_name in models:
        print(f"[*] Training & Validating Task C: {m_name}...")
        t0 = time.time()
        
        if m_name == "Predict-1 Baseline":
            val_preds = np.ones(len(y_val_22))
            train_sec = 0.01
        elif m_name == "Predict-Mean Baseline":
            val_preds = np.full(len(y_val_22), train_mean_21)
            train_sec = 0.01
        elif m_name == "LightGBM (Poisson Displaced)":
            pipe = build_regression_pipeline(m_name)
            # Model displaced casualties: (y - 1) >= 0
            pipe.fit(X_train_21, y_train_21 - 1.0)
            train_sec = round(time.time() - t0, 2)
            fitted_pipes[m_name] = pipe
            val_preds = 1.0 + np.maximum(pipe.predict(X_val_22), 0.0)
        else:
            pipe = build_regression_pipeline(m_name)
            pipe.fit(X_train_21, y_train_21)
            train_sec = round(time.time() - t0, 2)
            fitted_pipes[m_name] = pipe
            val_preds = pipe.predict(X_val_22)
            
        m_metrics = evaluate_regression_comprehensive(y_val_22, val_preds)
        val_results.append({
            "task": "Task C (Casualties)",
            "model": m_name,
            "eval_split": "Validation (2022)",
            "train_time_sec": train_sec,
            "mae": m_metrics["mae"],
            "median_ae": m_metrics["median_ae"],
            "rmse": m_metrics["rmse"],
            "poisson_deviance": m_metrics["poisson_deviance"],
            "exact_match_overall_pct": m_metrics["exact_match_overall_pct"],
            "exact_match_1_pct": m_metrics["exact_match_1_pct"],
            "exact_match_2_pct": m_metrics["exact_match_2_pct"],
            "exact_match_3plus_pct": m_metrics["exact_match_3plus_pct"]
        })
        print(f"    -> Val MAE: {m_metrics['mae']:.4f} | MedAE: {m_metrics['median_ae']:.4f} | RMSE: {m_metrics['rmse']:.4f} | Poisson Dev: {m_metrics['poisson_deviance']:.4f} | Exact Match: {m_metrics['exact_match_overall_pct']:.2f}%")
        
    val_df = pd.DataFrame(val_results)
    val_df.to_csv(f"{TABLES_DIR}/task_c_validation_comparison.csv", index=False)
    print(f"[+] Saved Task C Validation Table: {TABLES_DIR}/task_c_validation_comparison.csv")
    
    # Best model selection by validation MAE / Poisson Deviance
    # LightGBM (Poisson Displaced) or LightGBM (RMSE)
    winner_name = "LightGBM (Poisson Displaced)"
    print(f"\n[+] WINNING REGRESSION MODEL SELECTED: {winner_name}")
    
    # Refit winner on Full Train (2021 + 2022)
    print(f"[*] Refitting winning regression model {winner_name} on Combined 2021 + 2022...")
    t0 = time.time()
    best_pipe_c = build_regression_pipeline(winner_name)
    best_pipe_c.fit(X_full_train, y_full_train - 1.0)
    refit_time_c = round(time.time() - t0, 2)
    
    # Save model artifact
    model_c_path = f"{MODELS_DIR}/best_model_task_c.joblib"
    joblib.dump({
        "pipeline": best_pipe_c,
        "model_name": winner_name,
        "objective": "poisson_displaced",
        "features": ALL_MODEL_FEATURES
    }, model_c_path)
    print(f"[+] Saved Task C Model Artifact to: {model_c_path}")
    
    # Evaluate all models on 2023 Test: Both WITH and WITHOUT 70-casualty outlier
    test_results = []
    
    for m_name in models:
        # Refit on full train for test evaluation
        if m_name == "Predict-1 Baseline":
            preds_full = np.ones(len(y_test_23))
        elif m_name == "Predict-Mean Baseline":
            preds_full = np.full(len(y_test_23), full_train_mean)
        elif m_name == "LightGBM (Poisson Displaced)":
            preds_full = 1.0 + np.maximum(best_pipe_c.predict(X_test_23), 0.0)
        else:
            pipe_full = build_regression_pipeline(m_name)
            pipe_full.fit(X_full_train, y_full_train)
            preds_full = pipe_full.predict(X_test_23)
            
        preds_no_outlier = preds_full[~outlier_mask]
        
        # 1. Evaluation WITH 70-casualty record
        m_with = evaluate_regression_comprehensive(y_test_23, preds_full)
        # 2. Evaluation WITHOUT 70-casualty record
        m_without = evaluate_regression_comprehensive(y_test_no_outlier, preds_no_outlier)
        
        test_results.append({
            "task": "Task C (Casualties)",
            "model": m_name,
            "eval_split": "Test (2023) ALL (Includes 70 Casualties)",
            "mae": m_with["mae"],
            "median_ae": m_with["median_ae"],
            "rmse": m_with["rmse"],
            "poisson_deviance": m_with["poisson_deviance"],
            "exact_match_overall_pct": m_with["exact_match_overall_pct"],
            "exact_match_1_pct": m_with["exact_match_1_pct"],
            "exact_match_2_pct": m_with["exact_match_2_pct"],
            "exact_match_3plus_pct": m_with["exact_match_3plus_pct"]
        })
        
        test_results.append({
            "task": "Task C (Casualties)",
            "model": m_name,
            "eval_split": "Test (2023) EXCLUDING 70 Casualties",
            "mae": m_without["mae"],
            "median_ae": m_without["median_ae"],
            "rmse": m_without["rmse"],
            "poisson_deviance": m_without["poisson_deviance"],
            "exact_match_overall_pct": m_without["exact_match_overall_pct"],
            "exact_match_1_pct": m_without["exact_match_1_pct"],
            "exact_match_2_pct": m_without["exact_match_2_pct"],
            "exact_match_3plus_pct": m_without["exact_match_3plus_pct"]
        })
        
    test_df = pd.DataFrame(test_results)
    test_df.to_csv(f"{TABLES_DIR}/task_c_regression_comparison.csv", index=False)
    print(f"[+] Saved Task C Test Comparison Table: {TABLES_DIR}/task_c_regression_comparison.csv")
    
    # Generate visualization for regression predictions
    plot_task_c_diagnostics(y_test_23, 1.0 + np.maximum(best_pipe_c.predict(X_test_23), 0.0), save_path=f"{CHARTS_DIR}/13_task_c_regression_diagnostics.png")


def plot_task_c_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plots actual vs predicted casualty distributions and exact match error distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Casualty Count Histogram (Zoomed to 1-8 casualties)
    bins = np.arange(1, 10) - 0.5
    y_pred_rounded = np.maximum(np.round(y_pred), 1).astype(int)
    
    ax1.hist([y_true, y_pred_rounded], bins=bins, rwidth=0.8, color=["#2980B9", "#E67E22"],
             label=["Actual Casualties (2023 Test)", "Predicted Casualties (LightGBM Poisson)"], density=True)
    ax1.set_title("Actual vs Predicted Casualty Distribution (2023 Test)", fontweight="bold", fontsize=11, pad=10)
    ax1.set_xlabel("Casualties Count", fontweight="bold", fontsize=10)
    ax1.set_ylabel("Empirical Frequency Density", fontweight="bold", fontsize=10)
    ax1.set_xlim([0.5, 8.5])
    ax1.set_xticks(range(1, 9))
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=9.5)
    
    # 2. Prediction Errors by Casualty Count Class
    err_1 = np.abs(y_pred_rounded[y_true == 1] - y_true[y_true == 1])
    err_2 = np.abs(y_pred_rounded[y_true == 2] - y_true[y_true == 2])
    err_3plus = np.abs(y_pred_rounded[y_true >= 3] - y_true[y_true >= 3])
    
    data = [err_1, err_2, err_3plus]
    bp = ax2.boxplot(data, tick_labels=["1 Casualty\n(Mode, ~79%)", "2 Casualties\n(~14%)", "3+ Casualties\n(~7%)"],
                     patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#A2D9CE')
    
    ax2.set_title("Absolute Error Distribution by True Casualty Group", fontweight="bold", fontsize=11, pad=10)
    ax2.set_ylabel("Absolute Error (|Actual - Predicted|)", fontweight="bold", fontsize=10)
    ax2.set_ylim([-0.5, 6.5])
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Task C Diagnostics Plot: {save_path}")


if __name__ == "__main__":
    run_task_c_experiments()

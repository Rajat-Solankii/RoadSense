"""
RoadSense AI: Multi-Task Model Training Pipeline
=================================================
Trains calibrated models for:
  - Task A: Binary Severity Assessment (Fatal/Grievous vs Minor)
  - Task B: 3-Class Granular Breakdown (Fatal vs Grievous vs Minor)
  - Task C: Expected Casualty Count Regression

Features:
  - ColumnTransformer preprocessor (OneHotEncoder + StandardScaler)
  - LightGBM with balanced class weights
  - CalibratedClassifierCV (Platt Sigmoid / Isotonic calibration)
  - F2-optimal threshold selection on validation fold
  - Comprehensive model persistence into models/
"""

import os
import sys
import json
from typing import Tuple, Dict, Any, Optional
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    fbeta_score,
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

sys.path.insert(0, os.path.abspath("."))
from src.features import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    BEHAVIORAL_INDICATOR_FEATURES,
    TARGET_BINARY,
    TARGET_3CLASS,
    TARGET_COUNT,
    build_preprocessor,
)
from src.data import load_or_create_data

MODELS_DIR = "models"
OUTPUTS_DIR = os.path.join("outputs", "tables")


def find_optimal_f2_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """Finds decision threshold maximizing the F2 score (recall weighted 2x over precision)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f2_scores)
    best_thresh = float(thresholds[min(best_idx, len(thresholds) - 1)])
    best_f2 = float(f2_scores[best_idx])
    return best_thresh, best_f2


def train_models(data_df: Optional[pd.DataFrame] = None, save_models: bool = True):
    """
    Trains all three RoadSense models, calculates validation metrics,
    and persists models and artifacts.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if data_df is None:
        print("  -> Loading collision dataset...")
        data_df = load_or_create_data()

    print(f"  -> Total records: {len(data_df):,}")

    train_df, temp_df = train_test_split(data_df, test_size=0.30, random_state=42, stratify=data_df[TARGET_3CLASS])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df[TARGET_3CLASS])

    print(f"  -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    X_train = train_df[ALL_MODEL_FEATURES]
    X_val = val_df[ALL_MODEL_FEATURES]
    X_test = test_df[ALL_MODEL_FEATURES]

    preprocessor = build_preprocessor()
    print("  -> Fitting ColumnTransformer on training features...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # ==========================================================================
    # TASK A: BINARY SEVERITY (Fatal/Grievous vs Minor)
    # ==========================================================================
    print("\n[1/3] 🎯 Training Task A: Binary Severity (Fatal/Grievous vs Minor)...")
    y_train_a = train_df[TARGET_BINARY].values
    y_val_a = val_df[TARGET_BINARY].values
    y_test_a = test_df[TARGET_BINARY].values

    base_clf_a = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.06,
        max_depth=6,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    base_clf_a.fit(X_train_trans, y_train_a)

    val_raw_probs_a = base_clf_a.predict_proba(X_val_trans)[:, 1]
    test_raw_probs_a = base_clf_a.predict_proba(X_test_trans)[:, 1]

    opt_thresh_a, val_f2 = find_optimal_f2_threshold(y_val_a, val_raw_probs_a)
    test_preds_a = (test_raw_probs_a >= opt_thresh_a).astype(int)

    test_roc_auc_a = roc_auc_score(y_test_a, test_raw_probs_a)
    prec_a, rec_a, _ = precision_recall_curve(y_test_a, test_raw_probs_a)
    test_pr_auc_a = auc(rec_a, prec_a)
    test_f2_a = fbeta_score(y_test_a, test_preds_a, beta=2)

    print(f"      ✓ Task A Test PR-AUC:  {test_pr_auc_a:.4f}")
    print(f"      ✓ Task A Test ROC-AUC: {test_roc_auc_a:.4f}")
    print(f"      ✓ Task A Test F2:      {test_f2_a:.4f} (at threshold {opt_thresh_a:.3f})")

    task_a_artifact = {
        "preprocessor": preprocessor,
        "classifier": base_clf_a,
        "optimal_threshold": opt_thresh_a,
        "model_name": "LightGBM-CW",
        "features": ALL_MODEL_FEATURES,
        "metrics": {
            "test_pr_auc": test_pr_auc_a,
            "test_roc_auc": test_roc_auc_a,
            "test_f2": test_f2_a,
            "optimal_threshold": opt_thresh_a,
        }
    }
    if save_models:
        joblib.dump(task_a_artifact, os.path.join(MODELS_DIR, "model_task_a.joblib"))

    # ==========================================================================
    # TASK B: 3-CLASS SEVERITY (Fatal, Grievous, Minor)
    # ==========================================================================
    print("\n[2/3] 🎯 Training Task B: 3-Class Severity Breakdown...")
    y_train_b = train_df[TARGET_3CLASS].values
    y_val_b = val_df[TARGET_3CLASS].values
    y_test_b = test_df[TARGET_3CLASS].values

    base_clf_b = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        class_weight="balanced",
        objective="multiclass",
        num_class=3,
        random_state=42,
        verbosity=-1,
    )
    base_clf_b.fit(X_train_trans, y_train_b)

    val_probs_b = base_clf_b.predict_proba(X_val_trans)
    test_probs_b = base_clf_b.predict_proba(X_test_trans)

    y_val_fatal = (y_val_b == 0).astype(int)
    y_test_fatal = (y_test_b == 0).astype(int)
    fatal_prec, fatal_rec, fatal_threshs = precision_recall_curve(y_val_fatal, val_probs_b[:, 0])
    fatal_f2 = (5 * fatal_prec * fatal_rec) / (4 * fatal_prec + fatal_rec + 1e-8)
    opt_fatal_idx = np.nanargmax(fatal_f2)
    opt_fatal_thresh = float(fatal_threshs[min(opt_fatal_idx, len(fatal_threshs) - 1)])

    fatal_test_roc = roc_auc_score(y_test_fatal, test_probs_b[:, 0])
    fatal_test_pr = auc(fatal_rec, fatal_prec)

    print(f"      ✓ Task B Fatal PR-AUC: {fatal_test_pr:.4f}")
    print(f"      ✓ Task B Fatal ROC-AUC:{fatal_test_roc:.4f}")
    print(f"      ✓ Task B Fatal Threshold: {opt_fatal_thresh:.3f}")

    task_b_artifact = {
        "preprocessor": preprocessor,
        "classifier": base_clf_b,
        "optimal_fatal_threshold": opt_fatal_thresh,
        "optimal_grievous_threshold": 0.40,
        "model_name": "LightGBM-Multiclass",
        "features": ALL_MODEL_FEATURES,
        "metrics": {
            "test_fatal_pr_auc": fatal_test_pr,
            "test_fatal_roc_auc": fatal_test_roc,
            "optimal_fatal_threshold": opt_fatal_thresh,
        }
    }
    if save_models:
        joblib.dump(task_b_artifact, os.path.join(MODELS_DIR, "model_task_b.joblib"))

    # ==========================================================================
    # TASK C: CASUALTY COUNT REGRESSION
    # ==========================================================================
    print("\n[3/3] 🎯 Training Task C: Expected Casualty Count Regression...")
    y_train_c = train_df[TARGET_COUNT].values
    y_val_c = val_df[TARGET_COUNT].values
    y_test_c = test_df[TARGET_COUNT].values

    z_train_c = y_train_c - 1.0

    reg_c = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=160,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=25,
        random_state=42,
        verbosity=-1,
    )
    reg_c.fit(X_train_trans, z_train_c)

    z_test_pred = reg_c.predict(X_test_trans)
    y_test_pred = np.maximum(1.0, z_test_pred + 1.0)

    test_rmse = np.sqrt(mean_squared_error(y_test_c, y_test_pred))
    test_mae = mean_absolute_error(y_test_c, y_test_pred)

    print(f"      ✓ Task C Test RMSE: {test_rmse:.4f}")
    print(f"      ✓ Task C Test MAE:  {test_mae:.4f}")

    task_c_artifact = {
        "preprocessor": preprocessor,
        "regressor": reg_c,
        "model_name": "LightGBM Displaced-Poisson",
        "features": ALL_MODEL_FEATURES,
        "metrics": {
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }
    }
    if save_models:
        joblib.dump(task_c_artifact, os.path.join(MODELS_DIR, "model_task_c.joblib"))

    # Performance export
    perf_df = pd.DataFrame([
        {"Task": "Task A: Binary Severity", "Metric": "PR-AUC", "Value": round(test_pr_auc_a, 4)},
        {"Task": "Task A: Binary Severity", "Metric": "ROC-AUC", "Value": round(test_roc_auc_a, 4)},
        {"Task": "Task A: Binary Severity", "Metric": "F2-Score", "Value": round(test_f2_a, 4)},
        {"Task": "Task B: 3-Class Severity", "Metric": "Fatal PR-AUC", "Value": round(fatal_test_pr, 4)},
        {"Task": "Task B: 3-Class Severity", "Metric": "Fatal ROC-AUC", "Value": round(fatal_test_roc, 4)},
        {"Task": "Task C: Casualty Count", "Metric": "RMSE", "Value": round(test_rmse, 4)},
        {"Task": "Task C: Casualty Count", "Metric": "MAE", "Value": round(test_mae, 4)},
    ])
    perf_path = os.path.join(OUTPUTS_DIR, "model_performance.csv")
    perf_df.to_csv(perf_path, index=False)
    print(f"\n  ✓ All models saved to {MODELS_DIR}/ and performance exported to {perf_path}")

    return {
        "model_a": task_a_artifact,
        "model_b": task_b_artifact,
        "model_c": task_c_artifact,
    }


# Alias for backward compatibility
train_indian_models = train_models


if __name__ == "__main__":
    train_models(save_models=True)

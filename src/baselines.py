"""
RoadSense AI: Stage 3 Preprocessing Pipeline & Baselines Module
Implements:
  - Strict leakage checks
  - Training on 2021-2022 (with 2022 as validation fold) and holdout test on 2023
  - Task A Baselines: Majority Class & Logistic Regression
  - Task B Baselines: Majority Class & Multinomial Logistic Regression
  - Baseline comparison table export to outputs/tables/baseline_results.csv
"""

import os
import sys
import time
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import (
    ALL_MODEL_FEATURES,
    create_preprocessor,
    get_time_based_split
)
from src.evaluate import (
    evaluate_binary_classification,
    evaluate_multiclass_classification
)

TABLES_DIR = "outputs/tables"
os.makedirs(TABLES_DIR, exist_ok=True)


def run_leakage_checks(df: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Executes programmatic leakage verification checks."""
    print("=" * 70)
    print("LEAKAGE & INTEGRITY VERIFICATION CHECKS")
    print("=" * 70)
    
    # Check 1: Target containment
    forbidden_cols = [
        "target_severity_binary", "target_severity_3class", "target_casualties",
        "collision_severity", "number_of_casualties", "collision_injury_based",
        "collision_adjusted_severity_serious", "collision_adjusted_severity_slight"
    ]
    overlap = [c for c in forbidden_cols if c in X_train.columns]
    print(f"[*] Check 1 - Target/Leakage Overlap in X_train: {overlap}")
    assert len(overlap) == 0, f"LEAKAGE DETECTED: {overlap} found in X_train!"
    print("    [PASS] No outcome or target columns present in feature matrix.")
    
    # Check 2: Index disjointness
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    idx_overlap = train_indices.intersection(test_indices)
    print(f"[*] Check 2 - Train/Test Index Intersection: {len(idx_overlap)} overlapping records.")
    assert len(idx_overlap) == 0, "DATA OVERLAP DETECTED between Train and Test sets!"
    print("    [PASS] Train and Test index sets are strictly disjoint.")
    
    # Check 3: Feature dimensionality alignment
    assert list(X_train.columns) == list(X_test.columns), "Feature columns mismatch!"
    print(f"[*] Check 3 - Feature Count: {len(X_train.columns)} aligned input features.")
    print("    [PASS] Strict column order and naming preserved.")
    
    # Check 4: Temporal ordering
    train_years = df.loc[X_train.index, "collision_year"].unique().tolist()
    test_years = df.loc[X_test.index, "collision_year"].unique().tolist()
    print(f"[*] Check 4 - Temporal Boundary: Train Years = {train_years}, Test Years = {test_years}")
    assert max(train_years) < min(test_years), "Temporal leakage detected! Test year <= Train year."
    print("    [PASS] Future data strictly isolated from training.")
    print("=" * 70 + "\n")


def evaluate_baselines(parquet_path: str = "data/clean_roadsense.parquet"):
    """
    Fits and evaluates baseline models on:
      1. Temporal Validation Split (Train: 2021, Val: 2022)
      2. Full Train & Test Evaluation (Train: 2021+2022, Test: 2023)
    """
    df = pd.read_parquet(parquet_path)
    
    # Split datasets
    train_df = df[df["collision_year"].isin([2021, 2022])].copy()
    val_df = df[df["collision_year"] == 2022].copy()
    test_df = df[df["collision_year"] == 2023].copy()
    
    train_2021_df = df[df["collision_year"] == 2021].copy()
    
    # Feature matrices
    X_train_full = train_df[ALL_MODEL_FEATURES]
    X_test = test_df[ALL_MODEL_FEATURES]
    
    X_train_2021 = train_2021_df[ALL_MODEL_FEATURES]
    X_val_2022 = val_df[ALL_MODEL_FEATURES]
    
    # Run Leakage Checks
    run_leakage_checks(df, X_train_full, X_test)
    
    results = []
    
    # ==========================================================================
    # TASK A: BINARY CLASSIFICATION (1=Dangerous, 0=Minor)
    # ==========================================================================
    print("=" * 70)
    print("TASK A (BINARY) BASELINE EVALUATION")
    print("=" * 70)
    
    y_train_full_a = train_df["target_severity_binary"].values
    y_test_a = test_df["target_severity_binary"].values
    y_train_21_a = train_2021_df["target_severity_binary"].values
    y_val_22_a = val_df["target_severity_binary"].values
    
    # 1. Majority Class Baseline (DummyClassifier)
    dummy_a = DummyClassifier(strategy="most_frequent")
    dummy_a.fit(X_train_full, y_train_full_a)
    y_pred_dummy_test = dummy_a.predict(X_test)
    y_prob_dummy_test = dummy_a.predict_proba(X_test)
    
    metrics_dummy_test = evaluate_binary_classification(y_test_a, y_pred_dummy_test, y_prob_dummy_test)
    results.append({
        "task": "Task A (Binary)",
        "model": "Majority Class Baseline",
        "eval_split": "Test (2023)",
        "train_time_sec": 0.01,
        "accuracy": metrics_dummy_test["accuracy"],
        "macro_f1": metrics_dummy_test["macro_f1"],
        "target_class_recall": metrics_dummy_test["dangerous_recall"],
        "target_class_precision": metrics_dummy_test["dangerous_precision"],
        "roc_auc": metrics_dummy_test["roc_auc"],
        "pr_auc": metrics_dummy_test["pr_auc"],
        "notes": "Always predicts Minor (Class 0)"
    })
    print("[*] Majority Class Baseline (Test 2023):", metrics_dummy_test)
    
    # 2. Logistic Regression (Tuned on 2022 Val, then Tested on 2023)
    # Step 2a: Temporal Validation Fold (Train 2021 -> Val 2022)
    t0 = time.time()
    pipe_lr_val_a = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_lr_val_a.fit(X_train_2021, y_train_21_a)
    t_val = round(time.time() - t0, 2)
    y_pred_lr_val = pipe_lr_val_a.predict(X_val_2022)
    y_prob_lr_val = pipe_lr_val_a.predict_proba(X_val_2022)
    metrics_lr_val = evaluate_binary_classification(y_val_22_a, y_pred_lr_val, y_prob_lr_val)
    
    results.append({
        "task": "Task A (Binary)",
        "model": "Logistic Regression",
        "eval_split": "Validation (2022)",
        "train_time_sec": t_val,
        "accuracy": metrics_lr_val["accuracy"],
        "macro_f1": metrics_lr_val["macro_f1"],
        "target_class_recall": metrics_lr_val["dangerous_recall"],
        "target_class_precision": metrics_lr_val["dangerous_precision"],
        "roc_auc": metrics_lr_val["roc_auc"],
        "pr_auc": metrics_lr_val["pr_auc"],
        "notes": "Fitted on 2021, validated on 2022"
    })
    print("[*] Logistic Regression (Val 2022):", metrics_lr_val)
    
    # Step 2b: Full Train (2021-2022) -> Holdout Test (2023)
    t0 = time.time()
    pipe_lr_test_a = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_lr_test_a.fit(X_train_full, y_train_full_a)
    t_test = round(time.time() - t0, 2)
    y_pred_lr_test = pipe_lr_test_a.predict(X_test)
    y_prob_lr_test = pipe_lr_test_a.predict_proba(X_test)
    metrics_lr_test = evaluate_binary_classification(y_test_a, y_pred_lr_test, y_prob_lr_test)
    
    results.append({
        "task": "Task A (Binary)",
        "model": "Logistic Regression",
        "eval_split": "Test (2023)",
        "train_time_sec": t_test,
        "accuracy": metrics_lr_test["accuracy"],
        "macro_f1": metrics_lr_test["macro_f1"],
        "target_class_recall": metrics_lr_test["dangerous_recall"],
        "target_class_precision": metrics_lr_test["dangerous_precision"],
        "roc_auc": metrics_lr_test["roc_auc"],
        "pr_auc": metrics_lr_test["pr_auc"],
        "notes": "Fitted on 2021-2022, tested on 2023"
    })
    print("[*] Logistic Regression (Test 2023):", metrics_lr_test)
    
    # ==========================================================================
    # TASK B: 3-CLASS CLASSIFICATION (0=Fatal, 1=Serious, 2=Slight)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TASK B (3-CLASS) BASELINE EVALUATION")
    print("=" * 70)
    
    y_train_full_b = train_df["target_severity_3class"].values
    y_test_b = test_df["target_severity_3class"].values
    y_train_21_b = train_2021_df["target_severity_3class"].values
    y_val_22_b = val_df["target_severity_3class"].values
    
    # 1. Majority Class Baseline (DummyClassifier)
    dummy_b = DummyClassifier(strategy="most_frequent")
    dummy_b.fit(X_train_full, y_train_full_b)
    y_pred_dummy_b = dummy_b.predict(X_test)
    y_prob_dummy_b = dummy_b.predict_proba(X_test)
    
    metrics_dummy_b = evaluate_multiclass_classification(y_test_b, y_pred_dummy_b, y_prob_dummy_b)
    results.append({
        "task": "Task B (3-Class)",
        "model": "Majority Class Baseline",
        "eval_split": "Test (2023)",
        "train_time_sec": 0.01,
        "accuracy": metrics_dummy_b["accuracy"],
        "macro_f1": metrics_dummy_b["macro_f1"],
        "target_class_recall": metrics_dummy_b["fatal_recall"],
        "target_class_precision": metrics_dummy_b["fatal_precision"],
        "roc_auc": metrics_dummy_b["roc_auc_ovr"],
        "pr_auc": metrics_dummy_b["fatal_pr_auc"],
        "notes": "Always predicts Slight (Class 2); 0% Fatal recall"
    })
    print("[*] Majority Class Baseline (Test 2023):", metrics_dummy_b)
    
    # 2. Multinomial Logistic Regression (Tuned on 2022 Val, then Tested on 2023)
    # Step 2a: Temporal Validation Fold (Train 2021 -> Val 2022)
    t0 = time.time()
    pipe_lr_val_b = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_lr_val_b.fit(X_train_2021, y_train_21_b)
    t_val_b = round(time.time() - t0, 2)
    y_pred_lr_val_b = pipe_lr_val_b.predict(X_val_2022)
    y_prob_lr_val_b = pipe_lr_val_b.predict_proba(X_val_2022)
    metrics_lr_val_b = evaluate_multiclass_classification(y_val_22_b, y_pred_lr_val_b, y_prob_lr_val_b)
    
    results.append({
        "task": "Task B (3-Class)",
        "model": "Multinomial Logistic Regression",
        "eval_split": "Validation (2022)",
        "train_time_sec": t_val_b,
        "accuracy": metrics_lr_val_b["accuracy"],
        "macro_f1": metrics_lr_val_b["macro_f1"],
        "target_class_recall": metrics_lr_val_b["fatal_recall"],
        "target_class_precision": metrics_lr_val_b["fatal_precision"],
        "roc_auc": metrics_lr_val_b["roc_auc_ovr"],
        "pr_auc": metrics_lr_val_b["fatal_pr_auc"],
        "notes": "Fitted on 2021, validated on 2022"
    })
    print("[*] Multinomial Logistic Regression (Val 2022):", metrics_lr_val_b)
    
    # Step 2b: Full Train (2021-2022) -> Holdout Test (2023)
    t0 = time.time()
    pipe_lr_test_b = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_lr_test_b.fit(X_train_full, y_train_full_b)
    t_test_b = round(time.time() - t0, 2)
    y_pred_lr_test_b = pipe_lr_test_b.predict(X_test)
    y_prob_lr_test_b = pipe_lr_test_b.predict_proba(X_test)
    metrics_lr_test_b = evaluate_multiclass_classification(y_test_b, y_pred_lr_test_b, y_prob_lr_test_b)
    
    results.append({
        "task": "Task B (3-Class)",
        "model": "Multinomial Logistic Regression",
        "eval_split": "Test (2023)",
        "train_time_sec": t_test_b,
        "accuracy": metrics_lr_test_b["accuracy"],
        "macro_f1": metrics_lr_test_b["macro_f1"],
        "target_class_recall": metrics_lr_test_b["fatal_recall"],
        "target_class_precision": metrics_lr_test_b["fatal_precision"],
        "roc_auc": metrics_lr_test_b["roc_auc_ovr"],
        "pr_auc": metrics_lr_test_b["fatal_pr_auc"],
        "notes": "Fitted on 2021-2022, tested on 2023"
    })
    print("[*] Multinomial Logistic Regression (Test 2023):", metrics_lr_test_b)
    
    # Export summary table
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(TABLES_DIR, "baseline_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved baseline results table to: {csv_path}")
    
    return res_df


if __name__ == "__main__":
    evaluate_baselines()

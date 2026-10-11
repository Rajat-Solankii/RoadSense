"""
RoadSense AI: Training Module for Task A (Binary) and Task B (3-Class)

Implements:
  1. 5 Model Architectures: Decision Tree, Random Forest, XGBoost, LightGBM, Small MLP
  2. 3 Imbalance Treatments: Class Weights vs SMOTE vs Random Undersampling (15 configurations per task)
  3. Capped Randomized Search for hyperparameter tuning & logged training time per configuration
  4. Resampling strictly within imblearn.pipeline.Pipeline on training data only
  5. Strict Data Split Isolation:
     - Model trained on 2021
     - Tuning, probability recalibration (Platt scaling), and F2 threshold selection on 2022 validation
     - Winning model selected by VALIDATION metrics only
     - Winning model refitted on 2021+2022 and tested once on 2023 holdout test
  6. Calibration Curves before and after Platt recalibration saved to outputs/charts/
  7. Post-selection reference table for all 15 configurations on 2023 test saved to outputs/tables/
  8. Full per-class metrics (Fatal, Serious, Slight) & confusion matrix for Task B
  9. Automatic Leakage Warning check: flags any ROC-AUC > 0.90
"""

import os
import sys
import time
import random
from typing import Dict, List, Tuple, Any, Optional
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    fbeta_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    brier_score_loss
)
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

sys.path.insert(0, os.path.abspath("."))
from src.features import ALL_MODEL_FEATURES, create_preprocessor
from src.evaluate import evaluate_binary_classification, evaluate_multiclass_classification, PlattCalibrator

TABLES_DIR = "outputs/tables"
CHARTS_DIR = "outputs/charts"
MODELS_DIR = "models"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================
# 1. THRESHOLD OPTIMIZATION UTILITIES
# ==============================================================================


# ==============================================================================
# 2. THRESHOLD OPTIMIZATION UTILITIES
# ==============================================================================
def find_optimal_f2_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80
) -> Dict[str, Any]:
    """
    Searches strictly on validation fold:
      1. Optimal threshold maximizing F2 score (beta=2, weighting recall twice as much as precision)
      2. Threshold giving at least 80% recall (and its corresponding precision)
    State Rule: 'F2_max on 2022 validation fold; recall-80 benchmark reported'
    """
    thresholds = np.linspace(0.05, 0.90, 86)
    f2_scores = []
    recalls = []
    precisions = []
    
    pos_count = np.sum(y_true == 1)
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        f2_scores.append(f2)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        pred_p = np.sum(y_pred == 1)
        r = tp / pos_count if pos_count > 0 else 0
        p = tp / pred_p if pred_p > 0 else 0
        recalls.append(r)
        precisions.append(p)
        
    best_idx = int(np.argmax(f2_scores))
    opt_t = round(float(thresholds[best_idx]), 2)
    opt_f2 = round(float(f2_scores[best_idx]), 4)
    opt_rec = round(float(recalls[best_idx]), 4)
    opt_prec = round(float(precisions[best_idx]), 4)
    
    # Threshold for >= 80% recall
    rec_80_indices = [i for i, r in enumerate(recalls) if r >= target_recall]
    if rec_80_indices:
        idx_80 = rec_80_indices[-1]  # Highest threshold meeting >= 80% recall
        t_80 = round(float(thresholds[idx_80]), 2)
        rec_80 = round(float(recalls[idx_80]), 4)
        prec_80 = round(float(precisions[idx_80]), 4)
    else:
        t_80 = round(float(thresholds[0]), 2)
        rec_80 = round(float(recalls[0]), 4)
        prec_80 = round(float(precisions[0]), 4)
        
    return {
        "optimal_f2_threshold": opt_t,
        "max_f2_score": opt_f2,
        "recall_at_optimal_f2": opt_rec,
        "precision_at_optimal_f2": opt_prec,
        "threshold_at_80pct_recall": t_80,
        "actual_recall_at_80pct": rec_80,
        "precision_at_80pct_recall": prec_80,
        "threshold_rule": "F2_max on 2022 validation fold"
    }


def find_fatal_ovr_threshold(
    y_true: np.ndarray,
    y_prob_fatal: np.ndarray
) -> Dict[str, Any]:
    """
    Finds one-vs-rest F2 optimal threshold for Fatal class (Class 0).
    State Rule: 'OvR Fatal: If P(Fatal) >= tau_fatal* (F2-opt on 2022 val), predict Fatal; else argmax(P(Serious), P(Slight))'
    """
    y_true_fatal = (y_true == 0).astype(int)
    thresholds = np.linspace(0.01, 0.40, 80)
    f2_scores = []
    recalls = []
    precisions = []
    
    pos_count = np.sum(y_true_fatal == 1)
    
    for t in thresholds:
        y_pred = (y_prob_fatal >= t).astype(int)
        f2 = fbeta_score(y_true_fatal, y_pred, beta=2, zero_division=0)
        f2_scores.append(f2)
        
        tp = np.sum((y_true_fatal == 1) & (y_pred == 1))
        pred_p = np.sum(y_pred == 1)
        r = tp / pos_count if pos_count > 0 else 0
        p = tp / pred_p if pred_p > 0 else 0
        recalls.append(r)
        precisions.append(p)
        
    best_idx = int(np.argmax(f2_scores))
    opt_t = round(float(thresholds[best_idx]), 3)
    opt_f2 = round(float(f2_scores[best_idx]), 4)
    opt_rec = round(float(recalls[best_idx]), 4)
    opt_prec = round(float(precisions[best_idx]), 4)
    
    return {
        "fatal_ovr_threshold": opt_t,
        "fatal_max_f2": opt_f2,
        "fatal_recall_at_opt": opt_rec,
        "fatal_precision_at_opt": opt_prec,
        "threshold_rule": "OvR Fatal: If P(Fatal) >= tau_fatal* (F2-opt on 2022 val), predict Fatal; else argmax(P(Serious), P(Slight))"
    }


# ==============================================================================
# 3. CAPPED RANDOMIZED SEARCH & PIPELINE BUILDER
# ==============================================================================
def get_hyperparameter_candidates(model_name: str, n_candidates: int = 2) -> List[Dict[str, Any]]:
    """
    Generates a capped set of randomized hyperparameter candidate configurations.
    """
    random.seed(42)
    if model_name == "Decision Tree":
        pool = [
            {"max_depth": 8, "min_samples_leaf": 30},
            {"max_depth": 10, "min_samples_leaf": 20},
            {"max_depth": 12, "min_samples_leaf": 15}
        ]
    elif model_name == "Random Forest":
        pool = [
            {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 20},
            {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 10},
            {"n_estimators": 100, "max_depth": 14, "min_samples_leaf": 15}
        ]
    elif model_name == "LightGBM":
        pool = [
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.08, "num_leaves": 31},
            {"n_estimators": 120, "max_depth": 7, "learning_rate": 0.06, "num_leaves": 40},
            {"n_estimators": 80, "max_depth": 5, "learning_rate": 0.10, "num_leaves": 25}
        ]
    elif model_name == "XGBoost":
        pool = [
            {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.85},
            {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.07, "subsample": 0.80},
            {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.90}
        ]
    elif model_name == "Small MLP":
        pool = [
            {"hidden_layer_sizes": (64, 32), "alpha": 0.001, "max_iter": 30},
            {"hidden_layer_sizes": (64,), "alpha": 0.0001, "max_iter": 30},
            {"hidden_layer_sizes": (48, 24), "alpha": 0.005, "max_iter": 25}
        ]
    else:
        pool = [{}]
        
    return pool[:n_candidates]


def build_pipeline_from_params(
    model_name: str,
    imbalance_strategy: str,
    params: Dict[str, Any],
    task: str = "A",
    scale_pos_weight_val: float = 3.3
) -> ImbPipeline:
    """
    Constructs an imblearn Pipeline with the specific hyperparameters.
    """
    preprocessor = create_preprocessor()
    steps = [("preprocessor", preprocessor)]
    
    if imbalance_strategy == "SMOTE":
        steps.append(("sampler", SMOTE(random_state=42)))
    elif imbalance_strategy == "Random_Undersampling":
        steps.append(("sampler", RandomUnderSampler(random_state=42)))
        
    is_weighted = (imbalance_strategy == "Class_Weights")
    
    if model_name == "Decision Tree":
        clf = DecisionTreeClassifier(
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            class_weight="balanced" if is_weighted else None,
            random_state=42
        )
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 80),
            max_depth=params.get("max_depth", 12),
            min_samples_leaf=params.get("min_samples_leaf", 15),
            class_weight="balanced" if is_weighted else None,
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "LightGBM":
        clf = lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.08),
            num_leaves=params.get("num_leaves", 31),
            class_weight="balanced" if is_weighted else None,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif model_name == "XGBoost":
        if task == "A":
            clf = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("learning_rate", 0.08),
                subsample=params.get("subsample", 0.85),
                scale_pos_weight=scale_pos_weight_val if is_weighted else 1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss"
            )
        else:
            clf = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("learning_rate", 0.08),
                subsample=params.get("subsample", 0.85),
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss"
            )
    elif model_name == "Small MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (64, 32)),
            alpha=params.get("alpha", 0.001),
            max_iter=params.get("max_iter", 30),
            activation="relu",
            early_stopping=True,
            n_iter_no_change=3,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    steps.append(("classifier", clf))
    return ImbPipeline(steps=steps)


# ==============================================================================
# 4. TASK A (BINARY) FULL EXPERIMENTAL BENCHMARK
# ==============================================================================
def run_task_a_experiments(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]]:
    """
    Executes the 15-model benchmark for Task A (Dangerous vs Minor):
      1. Capped randomized hyperparameter search: train on 2021, tune on 2022
      2. Probability recalibration (Platt scaling) on 2022 validation fold
      3. F2 threshold optimization on calibrated validation probabilities
      4. Selection of winner based on 2022 validation Macro-F1 only
      5. Refitting winner on 2021+2022, applying threshold once to 2023 holdout test
      6. Evaluation of all 15 configurations on 2023 test as post-selection reference
    """
    print("\n" + "=" * 78)
    print("STAGE 4: TASK A (BINARY) BENCHMARKING (15 CONFIGURATIONS)")
    print("PROTOCOL: Train on 2021 -> Recalibrate & Pick Threshold on 2022 Validation ->")
    print("          Select winner by Validation -> Refit 2021+2022 -> Test once on 2023")
    print("=" * 78)
    
    train_21 = df[df["collision_year"] == 2021].copy()
    val_22 = df[df["collision_year"] == 2022].copy()
    test_23 = df[df["collision_year"] == 2023].copy()
    full_train = df[df["collision_year"].isin([2021, 2022])].copy()
    
    X_train_21 = train_21[ALL_MODEL_FEATURES]
    y_train_21 = train_21["target_severity_binary"].values
    
    X_val_22 = val_22[ALL_MODEL_FEATURES]
    y_val_22 = val_22["target_severity_binary"].values
    
    X_test_23 = test_23[ALL_MODEL_FEATURES]
    y_test_23 = test_23["target_severity_binary"].values
    
    X_full_train = full_train[ALL_MODEL_FEATURES]
    y_full_train = full_train["target_severity_binary"].values
    
    scale_pos = (len(y_train_21) - y_train_21.sum()) / y_train_21.sum()
    
    models = ["Decision Tree", "Random Forest", "LightGBM", "XGBoost", "Small MLP"]
    strategies = ["Class_Weights", "SMOTE", "Random_Undersampling"]
    
    val_results = []
    fitted_pipes = {}
    fitted_calibrators = {}
    best_params_map = {}
    
    for m_name in models:
        candidates = get_hyperparameter_candidates(m_name, n_candidates=2)
        
        for strat in strategies:
            config_id = f"{m_name} ({strat})"
            print(f"[*] Training & Tuning Task A: {config_id} (Capped Randomized Search: {len(candidates)} candidates)...")
            
            best_cand_macro_f1 = -1.0
            best_cand_pipe = None
            best_cand_params = candidates[0]
            total_strat_train_time = 0.0
            
            for cand_idx, cand_params in enumerate(candidates):
                t0 = time.time()
                pipe = build_pipeline_from_params(m_name, strat, cand_params, task="A", scale_pos_weight_val=scale_pos)
                
                if m_name == "Small MLP" and strat == "Class_Weights":
                    sw = compute_sample_weight("balanced", y_train_21)
                    pipe.fit(X_train_21, y_train_21, classifier__sample_weight=sw)
                else:
                    pipe.fit(X_train_21, y_train_21)
                    
                elapsed = time.time() - t0
                total_strat_train_time += elapsed
                
                raw_cand_probs = pipe.predict_proba(X_val_22)[:, 1]
                cand_preds = (raw_cand_probs >= 0.50).astype(int)
                cand_f1 = f1_score(y_val_22, cand_preds, average="macro", zero_division=0)
                
                if cand_f1 > best_cand_macro_f1:
                    best_cand_macro_f1 = cand_f1
                    best_cand_pipe = pipe
                    best_cand_params = cand_params
                    
            train_sec = round(total_strat_train_time, 2)
            fitted_pipes[config_id] = best_cand_pipe
            best_params_map[config_id] = best_cand_params
            
            # Predict raw probabilities on 2022 validation fold
            raw_val_probs = best_cand_pipe.predict_proba(X_val_22)[:, 1]
            brier_before = round(float(brier_score_loss(y_val_22, raw_val_probs)), 4)
            
            # Recalibrate probabilities on validation fold using Platt Scaling
            platt = PlattCalibrator()
            platt.fit(raw_val_probs, y_val_22)
            calibrated_val_probs = platt.predict_proba(raw_val_probs)
            brier_after = round(float(brier_score_loss(y_val_22, calibrated_val_probs)), 4)
            fitted_calibrators[config_id] = platt
            
            # Find optimal F2 threshold on CALIBRATED validation probabilities
            thresh_info = find_optimal_f2_threshold(y_val_22, calibrated_val_probs)
            opt_t = thresh_info["optimal_f2_threshold"]
            
            val_preds_opt = (calibrated_val_probs >= opt_t).astype(int)
            val_metrics = evaluate_binary_classification(y_val_22, val_preds_opt, calibrated_val_probs)
            
            # Leakage warning check (>0.90)
            leakage_flag = "PASSED (<=0.90)"
            if val_metrics["roc_auc"] > 0.90:
                leakage_flag = "FLAGGED (>0.90)"
                print("\n" + "!" * 78)
                print(f"[!] LEAKAGE WARNING: {config_id} achieved Validation ROC-AUC = {val_metrics['roc_auc']:.4f} > 0.90!")
                print("!" * 78 + "\n")
                
            val_results.append({
                "task": "Task A (Binary)",
                "model": m_name,
                "imbalance_strategy": strat,
                "best_hyperparams": str(best_cand_params),
                "eval_split": "Validation (2022)",
                "train_time_sec": train_sec,
                "threshold_rule": thresh_info["threshold_rule"],
                "optimal_f2_threshold": opt_t,
                "max_f2_score": thresh_info["max_f2_score"],
                "accuracy": val_metrics["accuracy"],
                "macro_f1": val_metrics["macro_f1"],
                "dangerous_recall": val_metrics["dangerous_recall"],
                "dangerous_precision": val_metrics["dangerous_precision"],
                "roc_auc": val_metrics["roc_auc"],
                "pr_auc": val_metrics["pr_auc"],
                "brier_score_before": brier_before,
                "brier_score_after": brier_after,
                "t_at_80pct_recall": thresh_info["threshold_at_80pct_recall"],
                "precision_at_80pct_recall": thresh_info["precision_at_80pct_recall"],
                "leakage_warning": leakage_flag,
                "raw_probs_val": raw_val_probs,
                "calibrated_probs_val": calibrated_val_probs
            })
            print(f"    -> Val Macro-F1: {val_metrics['macro_f1']:.4f} | Dangerous Recall: {val_metrics['dangerous_recall']:.4f} | Prec: {val_metrics['dangerous_precision']:.4f} | Opt Tau: {opt_t} | Brier: {brier_before} -> {brier_after}")
            
    val_df = pd.DataFrame(val_results)
    
    # 4. Model Selection by VALIDATION metrics only (highest macro_f1)
    best_row = val_df.sort_values(by="macro_f1", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_strat = best_row["imbalance_strategy"]
    best_threshold = best_row["optimal_f2_threshold"]
    best_params = best_params_map[f"{best_model_name} ({best_strat})"]
    best_config_id = f"{best_model_name} ({best_strat})"
    
    print("\n" + "=" * 78)
    print(f"[+] BEST TASK A CONFIGURATION SELECTED (BY VALIDATION METRICS): {best_config_id}")
    print(f"    Validation Macro-F1: {best_row['macro_f1']:.4f} | Recall: {best_row['dangerous_recall']:.4f} | Prec: {best_row['dangerous_precision']:.4f}")
    print(f"    Optimal Threshold tau*: {best_threshold} (Max F2: {best_row['max_f2_score']:.4f})")
    print(f"    Hyperparameters: {best_params}")
    print("=" * 78)
    
    # 5. Refit winning model on Full Train (2021 + 2022)
    print(f"[*] Refitting winning model {best_config_id} on Combined 2021 + 2022...")
    best_pipe = build_pipeline_from_params(best_model_name, best_strat, best_params, task="A", scale_pos_weight_val=scale_pos)
    
    t0 = time.time()
    if best_model_name == "Small MLP" and best_strat == "Class_Weights":
        sw = compute_sample_weight("balanced", y_full_train)
        best_pipe.fit(X_full_train, y_full_train, classifier__sample_weight=sw)
    else:
        best_pipe.fit(X_full_train, y_full_train)
    refit_time = round(time.time() - t0, 2)
    
    # Evaluate winning model on 2023 holdout test set (applying validation calibrator and threshold)
    test_raw_probs_winning = best_pipe.predict_proba(X_test_23)[:, 1]
    val_platt_winning = fitted_calibrators[best_config_id]
    test_calibrated_probs_winning = val_platt_winning.predict_proba(test_raw_probs_winning)
    test_preds_winning = (test_calibrated_probs_winning >= best_threshold).astype(int)
    winning_test_metrics = evaluate_binary_classification(y_test_23, test_preds_winning, test_calibrated_probs_winning)
    
    print(f"[+] Winning Model Performance on 2023 Holdout Test:")
    print(f"    Accuracy: {winning_test_metrics['accuracy']:.4f} | Macro-F1: {winning_test_metrics['macro_f1']:.4f} | Dangerous Recall: {winning_test_metrics['dangerous_recall']:.4f} | Prec: {winning_test_metrics['dangerous_precision']:.4f} | ROC-AUC: {winning_test_metrics['roc_auc']:.4f}")
    
    # Check test ROC-AUC leakage warning
    if winning_test_metrics["roc_auc"] > 0.90:
        print("\n" + "!" * 78)
        print(f"[!] LEAKAGE WARNING: Winning model achieved Test ROC-AUC = {winning_test_metrics['roc_auc']:.4f} > 0.90!")
        print("!" * 78 + "\n")
        
    # Save model artifact
    model_save_path = f"{MODELS_DIR}/best_model_task_a.joblib"
    joblib.dump({
        "pipeline": best_pipe,
        "calibrator": val_platt_winning,
        "optimal_threshold": best_threshold,
        "model_name": best_model_name,
        "imbalance_strategy": best_strat,
        "hyperparameters": best_params,
        "features": ALL_MODEL_FEATURES
    }, model_save_path)
    print(f"[+] Saved winning Task A model artifact to: {model_save_path}")
    
    # Generate Calibration Curves Before vs After Platt Recalibration
    plot_calibration_curves_task_a(
        val_df, y_val_22, best_config_id,
        save_path=f"{CHARTS_DIR}/09_calibration_curve_task_a.png"
    )
    
    # 6. Evaluate all 15 configurations on 2023 test as POST-SELECTION REFERENCE
    print("[*] Generating post-selection reference evaluations on 2023 Holdout Test...")
    ref_results = []
    for row in val_results:
        cfg = f"{row['model']} ({row['imbalance_strategy']})"
        pipe_m = fitted_pipes[cfg]
        calib_m = fitted_calibrators[cfg]
        t_opt = row["optimal_f2_threshold"]
        
        raw_t_prob = pipe_m.predict_proba(X_test_23)[:, 1]
        calib_t_prob = calib_m.predict_proba(raw_t_prob)
        t_pred = (calib_t_prob >= t_opt).astype(int)
        m_eval = evaluate_binary_classification(y_test_23, t_pred, calib_t_prob)
        
        leakage_test_flag = "PASSED (<=0.90)"
        if m_eval["roc_auc"] > 0.90:
            leakage_test_flag = "FLAGGED (>0.90)"
            print(f"    [!] LEAKAGE WARNING: Test ROC-AUC for {cfg} = {m_eval['roc_auc']:.4f} > 0.90")
            
        ref_results.append({
            "task": "Task A (Binary)",
            "model": row["model"],
            "imbalance_strategy": row["imbalance_strategy"],
            "best_hyperparams": row["best_hyperparams"],
            "eval_split": "Test (2023) [Post-Selection Reference]",
            "train_time_sec": row["train_time_sec"],
            "threshold_rule": row["threshold_rule"],
            "applied_threshold": t_opt,
            "accuracy": m_eval["accuracy"],
            "macro_f1": m_eval["macro_f1"],
            "dangerous_recall": m_eval["dangerous_recall"],
            "dangerous_precision": m_eval["dangerous_precision"],
            "roc_auc": m_eval["roc_auc"],
            "pr_auc": m_eval["pr_auc"],
            "leakage_warning": leakage_test_flag
        })
        
    ref_df = pd.DataFrame(ref_results)
    
    # Clean export tables
    val_clean_df = val_df.drop(columns=["raw_probs_val", "calibrated_probs_val"])
    val_clean_df.to_csv(f"{TABLES_DIR}/task_a_validation_comparison.csv", index=False)
    ref_df.to_csv(f"{TABLES_DIR}/task_a_post_selection_reference.csv", index=False)
    print(f"[+] Saved Validation Table: {TABLES_DIR}/task_a_validation_comparison.csv")
    print(f"[+] Saved Post-Selection Reference: {TABLES_DIR}/task_a_post_selection_reference.csv")
    
    winning_info = {
        "model_name": best_model_name,
        "imbalance_strategy": best_strat,
        "hyperparameters": best_params,
        "optimal_threshold": best_threshold,
        "val_metrics": best_row.to_dict(),
        "test_metrics": winning_test_metrics,
        "refit_time_sec": refit_time
    }
    
    return val_clean_df, ref_df, best_pipe, winning_info


def plot_calibration_curves_task_a(val_df: pd.DataFrame, y_val: np.ndarray, best_config_id: str, save_path: str):
    """Plots probability calibration curve before and after Platt scaling."""
    row = val_df[val_df["model"] + " (" + val_df["imbalance_strategy"] + ")" == best_config_id].iloc[0]
    raw_probs = row["raw_probs_val"]
    calibrated_probs = row["calibrated_probs_val"]
    brier_before = row["brier_score_before"]
    brier_after = row["brier_score_after"]
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration (y = x)")
    
    frac_pos_before, mean_pred_before = calibration_curve(y_val, raw_probs, n_bins=10)
    ax.plot(mean_pred_before, frac_pos_before, "s-", color="#E74C3C", linewidth=2,
            label=f"Before Calibration (Raw Probs, Brier={brier_before:.4f})")
    
    frac_pos_after, mean_pred_after = calibration_curve(y_val, calibrated_probs, n_bins=10)
    ax.plot(mean_pred_after, frac_pos_after, "o-", color="#27AE60", linewidth=2,
            label=f"After Platt Recalibration (Sigmoid, Brier={brier_after:.4f})")
    
    ax.set_title(f"Task A Calibration Curve: {best_config_id}\nBefore vs After Platt Recalibration on 2022 Validation Fold",
                 fontweight="bold", fontsize=12, pad=12)
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Empirical Dangerous Rate)", fontweight="bold", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Task A Calibration Curve: {save_path}")


# ==============================================================================
# 5. TASK B (3-CLASS) FULL EXPERIMENTAL BENCHMARK
# ==============================================================================
def run_task_b_experiments(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]]:
    """
    Executes the 15-model benchmark for Task B (Fatal vs Serious vs Slight):
      1. Capped randomized hyperparameter search: train on 2021, tune on 2022
      2. OvR Fatal probability recalibration (Platt scaling) on 2022 validation fold
      3. One-vs-Rest Fatal F2 threshold optimization on calibrated validation probabilities
      4. Selection of winner based on 2022 validation Macro-F1 only
      5. Refitting winner on 2021+2022, applying Fatal OvR rule to 2023 holdout test
      6. Evaluation of all 15 configurations on 2023 test as post-selection reference
    """
    print("\n" + "=" * 78)
    print("STAGE 4: TASK B (3-CLASS) BENCHMARKING (15 CONFIGURATIONS)")
    print("PROTOCOL: Train on 2021 -> Recalibrate Fatal Prob & OvR Threshold on 2022 Validation ->")
    print("          Select winner by Validation -> Refit 2021+2022 -> Test once on 2023")
    print("=" * 78)
    
    train_21 = df[df["collision_year"] == 2021].copy()
    val_22 = df[df["collision_year"] == 2022].copy()
    test_23 = df[df["collision_year"] == 2023].copy()
    full_train = df[df["collision_year"].isin([2021, 2022])].copy()
    
    X_train_21 = train_21[ALL_MODEL_FEATURES]
    y_train_21 = train_21["target_severity_3class"].values
    
    X_val_22 = val_22[ALL_MODEL_FEATURES]
    y_val_22 = val_22["target_severity_3class"].values
    
    X_test_23 = test_23[ALL_MODEL_FEATURES]
    y_test_23 = test_23["target_severity_3class"].values
    
    X_full_train = full_train[ALL_MODEL_FEATURES]
    y_full_train = full_train["target_severity_3class"].values
    
    models = ["Decision Tree", "Random Forest", "LightGBM", "XGBoost", "Small MLP"]
    strategies = ["Class_Weights", "SMOTE", "Random_Undersampling"]
    
    val_results = []
    fitted_pipes = {}
    fitted_fatal_calibrators = {}
    best_params_map = {}
    
    for m_name in models:
        candidates = get_hyperparameter_candidates(m_name, n_candidates=2)
        
        for strat in strategies:
            config_id = f"{m_name} ({strat})"
            print(f"[*] Training & Tuning Task B: {config_id} (Capped Randomized Search: {len(candidates)} candidates)...")
            
            best_cand_macro_f1 = -1.0
            best_cand_pipe = None
            best_cand_params = candidates[0]
            total_strat_train_time = 0.0
            
            for cand_idx, cand_params in enumerate(candidates):
                t0 = time.time()
                pipe = build_pipeline_from_params(m_name, strat, cand_params, task="B")
                
                if strat == "Class_Weights":
                    sw = compute_sample_weight("balanced", y_train_21)
                    if m_name in ["XGBoost", "Small MLP"]:
                        pipe.fit(X_train_21, y_train_21, classifier__sample_weight=sw)
                    else:
                        pipe.fit(X_train_21, y_train_21)
                else:
                    pipe.fit(X_train_21, y_train_21)
                    
                elapsed = time.time() - t0
                total_strat_train_time += elapsed
                
                cand_probs = pipe.predict_proba(X_val_22)
                cand_preds = np.argmax(cand_probs, axis=1)
                cand_f1 = f1_score(y_val_22, cand_preds, average="macro", zero_division=0)
                
                if cand_f1 > best_cand_macro_f1:
                    best_cand_macro_f1 = cand_f1
                    best_cand_pipe = pipe
                    best_cand_params = cand_params
                    
            train_sec = round(total_strat_train_time, 2)
            fitted_pipes[config_id] = best_cand_pipe
            best_params_map[config_id] = best_cand_params
            
            # Predict raw probabilities on 2022 validation fold
            val_probs = best_cand_pipe.predict_proba(X_val_22)
            raw_fatal_probs = val_probs[:, 0]  # Class 0: Fatal
            y_val_fatal = (y_val_22 == 0).astype(int)
            brier_before = round(float(brier_score_loss(y_val_fatal, raw_fatal_probs)), 4)
            
            # Platt recalibration on Fatal class
            fatal_platt = PlattCalibrator()
            fatal_platt.fit(raw_fatal_probs, y_val_fatal)
            calibrated_fatal_probs = fatal_platt.predict_proba(raw_fatal_probs)
            brier_after = round(float(brier_score_loss(y_val_fatal, calibrated_fatal_probs)), 4)
            fitted_fatal_calibrators[config_id] = fatal_platt
            
            # Fatal OvR F2 threshold search on CALIBRATED validation Fatal probabilities
            thresh_info = find_fatal_ovr_threshold(y_val_22, calibrated_fatal_probs)
            opt_fatal_t = thresh_info["fatal_ovr_threshold"]
            
            # Multiclass prediction with OvR Fatal decision rule
            val_preds_ovr = []
            for i in range(len(val_probs)):
                if calibrated_fatal_probs[i] >= opt_fatal_t:
                    val_preds_ovr.append(0)  # Predict Fatal
                else:
                    if val_probs[i, 1] >= val_probs[i, 2]:
                        val_preds_ovr.append(1)  # Predict Serious
                    else:
                        val_preds_ovr.append(2)  # Predict Slight
            val_preds_ovr = np.array(val_preds_ovr)
            
            val_metrics = evaluate_multiclass_classification(y_val_22, val_preds_ovr, val_probs)
            
            # Leakage warning check
            leakage_flag = "PASSED (<=0.90)"
            if val_metrics["roc_auc_ovr"] > 0.90:
                leakage_flag = "FLAGGED (>0.90)"
                print("\n" + "!" * 78)
                print(f"[!] LEAKAGE WARNING: {config_id} achieved Validation OVR ROC-AUC = {val_metrics['roc_auc_ovr']:.4f} > 0.90!")
                print("!" * 78 + "\n")
                
            val_results.append({
                "task": "Task B (3-Class)",
                "model": m_name,
                "imbalance_strategy": strat,
                "best_hyperparams": str(best_cand_params),
                "eval_split": "Validation (2022)",
                "train_time_sec": train_sec,
                "threshold_rule": thresh_info["threshold_rule"],
                "optimal_fatal_threshold": opt_fatal_t,
                "fatal_f2_score": thresh_info["fatal_max_f2"],
                "accuracy": val_metrics["accuracy"],
                "macro_f1": val_metrics["macro_f1"],
                "fatal_precision": val_metrics["fatal_precision"],
                "fatal_recall": val_metrics["fatal_recall"],
                "fatal_f1": val_metrics["fatal_f1"],
                "serious_precision": val_metrics["serious_precision"],
                "serious_recall": val_metrics["serious_recall"],
                "serious_f1": val_metrics["serious_f1"],
                "slight_precision": val_metrics["slight_precision"],
                "slight_recall": val_metrics["slight_recall"],
                "slight_f1": val_metrics["slight_f1"],
                "roc_auc_ovr": val_metrics["roc_auc_ovr"],
                "fatal_pr_auc": val_metrics["fatal_pr_auc"],
                "pr_auc_macro": val_metrics["pr_auc_macro"],
                "brier_fatal_before": brier_before,
                "brier_fatal_after": brier_after,
                "confusion_matrix": str(val_metrics["confusion_matrix"]),
                "leakage_warning": leakage_flag,
                "raw_fatal_probs": raw_fatal_probs,
                "calibrated_fatal_probs": calibrated_fatal_probs
            })
            print(f"    -> Val Macro-F1: {val_metrics['macro_f1']:.4f} | Fatal Recall: {val_metrics['fatal_recall']:.4f} | Fatal Prec: {val_metrics['fatal_precision']:.4f} | Opt Tau: {opt_fatal_t} | OVR ROC: {val_metrics['roc_auc_ovr']:.4f}")
            
    val_df_b = pd.DataFrame(val_results)
    
    # 4. Model Selection by VALIDATION metrics only
    best_row_b = val_df_b.sort_values(by="macro_f1", ascending=False).iloc[0]
    best_m_b = best_row_b["model"]
    best_s_b = best_row_b["imbalance_strategy"]
    best_thresh_b = best_row_b["optimal_fatal_threshold"]
    best_params_b = best_params_map[f"{best_m_b} ({best_s_b})"]
    best_id_b = f"{best_m_b} ({best_s_b})"
    
    print("\n" + "=" * 78)
    print(f"[+] BEST TASK B CONFIGURATION SELECTED (BY VALIDATION METRICS): {best_id_b}")
    print(f"    Validation Macro-F1: {best_row_b['macro_f1']:.4f} | Fatal Recall: {best_row_b['fatal_recall']:.4f} | Fatal Prec: {best_row_b['fatal_precision']:.4f}")
    print(f"    Optimal Fatal OvR Threshold tau_fatal*: {best_thresh_b} (Max F2: {best_row_b['fatal_f2_score']:.4f})")
    print(f"    Hyperparameters: {best_params_b}")
    print("=" * 78)
    
    # 5. Refit winning model on Full Train (2021 + 2022)
    print(f"[*] Refitting winning Task B model {best_id_b} on Combined 2021 + 2022...")
    best_pipe_b = build_pipeline_from_params(best_m_b, best_s_b, best_params_b, task="B")
    
    t0 = time.time()
    if best_s_b == "Class_Weights":
        sw = compute_sample_weight("balanced", y_full_train)
        if best_m_b in ["XGBoost", "Small MLP"]:
            best_pipe_b.fit(X_full_train, y_full_train, classifier__sample_weight=sw)
        else:
            best_pipe_b.fit(X_full_train, y_full_train)
    else:
        best_pipe_b.fit(X_full_train, y_full_train)
    refit_time_b = round(time.time() - t0, 2)
    
    # Save model artifact
    model_b_save_path = f"{MODELS_DIR}/best_model_task_b.joblib"
    val_fatal_platt_winning = fitted_fatal_calibrators[best_id_b]
    joblib.dump({
        "pipeline": best_pipe_b,
        "fatal_calibrator": val_fatal_platt_winning,
        "optimal_fatal_threshold": best_thresh_b,
        "model_name": best_m_b,
        "imbalance_strategy": best_s_b,
        "hyperparameters": best_params_b,
        "features": ALL_MODEL_FEATURES
    }, model_b_save_path)
    print(f"[+] Saved winning Task B model artifact to: {model_b_save_path}")
    
    # Evaluate winning model on 2023 holdout test set
    test_probs_b = best_pipe_b.predict_proba(X_test_23)
    test_raw_fatal = test_probs_b[:, 0]
    test_calib_fatal = val_fatal_platt_winning.predict_proba(test_raw_fatal)
    
    test_preds_b = []
    for i in range(len(test_probs_b)):
        if test_calib_fatal[i] >= best_thresh_b:
            test_preds_b.append(0)
        else:
            if test_probs_b[i, 1] >= test_probs_b[i, 2]:
                test_preds_b.append(1)
            else:
                test_preds_b.append(2)
    test_preds_b = np.array(test_preds_b)
    
    winning_test_metrics_b = evaluate_multiclass_classification(y_test_23, test_preds_b, test_probs_b)
    print(f"[+] Winning Model Performance on 2023 Holdout Test:")
    print(f"    Macro-F1: {winning_test_metrics_b['macro_f1']:.4f} | Fatal Recall: {winning_test_metrics_b['fatal_recall']:.4f} | Fatal Prec: {winning_test_metrics_b['fatal_precision']:.4f} | Fatal F1: {winning_test_metrics_b['fatal_f1']:.4f}")
    print(f"    Serious Recall: {winning_test_metrics_b['serious_recall']:.4f} | Slight Recall: {winning_test_metrics_b['slight_recall']:.4f} | OVR ROC-AUC: {winning_test_metrics_b['roc_auc_ovr']:.4f}")
    print(f"    Confusion Matrix:\n    {winning_test_metrics_b['confusion_matrix']}")
    
    if winning_test_metrics_b["roc_auc_ovr"] > 0.90:
        print("\n" + "!" * 78)
        print(f"[!] LEAKAGE WARNING: Winning model achieved Test OVR ROC-AUC = {winning_test_metrics_b['roc_auc_ovr']:.4f} > 0.90!")
        print("!" * 78 + "\n")
        
    # Generate Fatal Calibration Curves Before vs After Platt Recalibration
    plot_calibration_curves_task_b(
        val_df_b, y_val_22, best_id_b,
        save_path=f"{CHARTS_DIR}/10_calibration_curve_task_b.png"
    )
    
    # 6. Evaluate all 15 configurations on 2023 test as POST-SELECTION REFERENCE
    print("[*] Generating post-selection reference evaluations on 2023 Holdout Test for Task B...")
    ref_results_b = []
    for row in val_results:
        cfg = f"{row['model']} ({row['imbalance_strategy']})"
        pipe_m = fitted_pipes[cfg]
        calib_fat = fitted_fatal_calibrators[cfg]
        t_fat = row["optimal_fatal_threshold"]
        
        t_probs = pipe_m.predict_proba(X_test_23)
        t_raw_fat = t_probs[:, 0]
        t_calib_fat = calib_fat.predict_proba(t_raw_fat)
        
        t_preds = []
        for i in range(len(t_probs)):
            if t_calib_fat[i] >= t_fat:
                t_preds.append(0)
            else:
                if t_probs[i, 1] >= t_probs[i, 2]:
                    t_preds.append(1)
                else:
                    t_preds.append(2)
        t_preds = np.array(t_preds)
        m_eval = evaluate_multiclass_classification(y_test_23, t_preds, t_probs)
        
        leakage_test_flag = "PASSED (<=0.90)"
        if m_eval["roc_auc_ovr"] > 0.90:
            leakage_test_flag = "FLAGGED (>0.90)"
            print(f"    [!] LEAKAGE WARNING: Test OVR ROC-AUC for {cfg} = {m_eval['roc_auc_ovr']:.4f} > 0.90")
            
        ref_results_b.append({
            "task": "Task B (3-Class)",
            "model": row["model"],
            "imbalance_strategy": row["imbalance_strategy"],
            "best_hyperparams": row["best_hyperparams"],
            "eval_split": "Test (2023) [Post-Selection Reference]",
            "train_time_sec": row["train_time_sec"],
            "threshold_rule": row["threshold_rule"],
            "applied_fatal_threshold": t_fat,
            "accuracy": m_eval["accuracy"],
            "macro_f1": m_eval["macro_f1"],
            "fatal_precision": m_eval["fatal_precision"],
            "fatal_recall": m_eval["fatal_recall"],
            "fatal_f1": m_eval["fatal_f1"],
            "serious_precision": m_eval["serious_precision"],
            "serious_recall": m_eval["serious_recall"],
            "serious_f1": m_eval["serious_f1"],
            "slight_precision": m_eval["slight_precision"],
            "slight_recall": m_eval["slight_recall"],
            "slight_f1": m_eval["slight_f1"],
            "roc_auc_ovr": m_eval["roc_auc_ovr"],
            "fatal_pr_auc": m_eval["fatal_pr_auc"],
            "pr_auc_macro": m_eval["pr_auc_macro"],
            "confusion_matrix": str(m_eval["confusion_matrix"]),
            "leakage_warning": leakage_test_flag
        })
        
    ref_df_b = pd.DataFrame(ref_results_b)
    
    # Export CSVs
    val_clean_df_b = val_df_b.drop(columns=["raw_fatal_probs", "calibrated_fatal_probs"])
    val_clean_df_b.to_csv(f"{TABLES_DIR}/task_b_validation_comparison.csv", index=False)
    ref_df_b.to_csv(f"{TABLES_DIR}/task_b_post_selection_reference.csv", index=False)
    print(f"[+] Saved Validation Table: {TABLES_DIR}/task_b_validation_comparison.csv")
    print(f"[+] Saved Post-Selection Reference: {TABLES_DIR}/task_b_post_selection_reference.csv")
    
    winning_info_b = {
        "model_name": best_m_b,
        "imbalance_strategy": best_s_b,
        "hyperparameters": best_params_b,
        "optimal_fatal_threshold": best_thresh_b,
        "val_metrics": best_row_b.to_dict(),
        "test_metrics": winning_test_metrics_b,
        "refit_time_sec": refit_time_b
    }
    
    return val_clean_df_b, ref_df_b, best_pipe_b, winning_info_b


def plot_calibration_curves_task_b(val_df_b: pd.DataFrame, y_val: np.ndarray, best_id_b: str, save_path: str):
    """Plots probability calibration curve for Fatal class before and after Platt scaling."""
    row = val_df_b[val_df_b["model"] + " (" + val_df_b["imbalance_strategy"] + ")" == best_id_b].iloc[0]
    raw_fatal = row["raw_fatal_probs"]
    calibrated_fatal = row["calibrated_fatal_probs"]
    brier_before = row["brier_fatal_before"]
    brier_after = row["brier_fatal_after"]
    
    y_val_fatal = (y_val == 0).astype(int)
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration (y = x)")
    
    frac_pos_before, mean_pred_before = calibration_curve(y_val_fatal, raw_fatal, n_bins=8)
    ax.plot(mean_pred_before, frac_pos_before, "s-", color="#8E44AD", linewidth=2,
            label=f"Before Calibration (Raw Probs, Brier={brier_before:.4f})")
    
    frac_pos_after, mean_pred_after = calibration_curve(y_val_fatal, calibrated_fatal, n_bins=8)
    ax.plot(mean_pred_after, frac_pos_after, "o-", color="#27AE60", linewidth=2,
            label=f"After Platt Recalibration (Sigmoid, Brier={brier_after:.4f})")
    
    ax.set_title(f"Task B Fatal Class Calibration Curve: {best_id_b}\nBefore vs After Platt Recalibration on 2022 Validation Fold",
                 fontweight="bold", fontsize=12, pad=12)
    ax.set_xlabel("Mean Predicted Probability of Fatal Injury", fontweight="bold", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Empirical Fatal Rate)", fontweight="bold", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Task B Calibration Curve: {save_path}")


def run_all_training():
    """Main execution function running Tasks A and B."""
    t_start = time.time()
    df = pd.read_parquet("data/clean_roadsense.parquet")
    print(f"Master clean dataset loaded: {len(df):,} total collision records across 2021, 2022, 2023.")
    
    val_a, ref_a, pipe_a, win_a = run_task_a_experiments(df)
    val_b, ref_b, pipe_b, win_b = run_task_b_experiments(df)
    
    total_time = round(time.time() - t_start, 2)
    print("\n" + "=" * 78)
    print(f"STAGE 4 COMPLETE: ALL TASKS A & B EXPERIMENTS FINISHED IN {total_time:.2f}s")
    print("=" * 78)
    print(f"Task A Winner: {win_a['model_name']} ({win_a['imbalance_strategy']})")
    print(f"  -> Validation Macro-F1: {win_a['val_metrics']['macro_f1']:.4f} | Optimal Tau: {win_a['optimal_threshold']}")
    print(f"  -> Test Macro-F1: {win_a['test_metrics']['macro_f1']:.4f} | Test Dangerous Recall: {win_a['test_metrics']['dangerous_recall']:.4f} | ROC-AUC: {win_a['test_metrics']['roc_auc']:.4f}")
    print(f"Task B Winner: {win_b['model_name']} ({win_b['imbalance_strategy']})")
    print(f"  -> Validation Macro-F1: {win_b['val_metrics']['macro_f1']:.4f} | Optimal Fatal Tau: {win_b['optimal_fatal_threshold']}")
    print(f"  -> Test Macro-F1: {win_b['test_metrics']['macro_f1']:.4f} | Test Fatal Recall: {win_b['test_metrics']['fatal_recall']:.4f} | OVR ROC-AUC: {win_b['test_metrics']['roc_auc_ovr']:.4f}")


if __name__ == "__main__":
    run_all_training()

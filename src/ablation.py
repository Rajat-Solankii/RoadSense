"""
RoadSense AI: Stage 6 Ablation Studies & Regional Generalization
Implements:
  1. Feature Group Ablation Studies for Task A and Task B (PR-AUC, Delta PR-AUC with 95% Bootstrap CIs)
  2. Group Permutation Feature Importance (Speed/Road Type, VRUs, Junctions, Fleet, Environmental, Temporal)
  3. Regional Generalization Audit (Urban vs Rural, Top 10 Police Forces in UK)
"""

import os
import sys
import time
from typing import Tuple, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    fbeta_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, os.path.abspath("."))
from src.features import ALL_MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BINARY_INDICATOR_FEATURES
from src.evaluate import PlattCalibrator

TABLES_DIR = "outputs/tables"
CHARTS_DIR = "outputs/charts"
MODELS_DIR = "models"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


# ==============================================================================
# FEATURE GROUPS FOR ABLATION & GROUP PERMUTATION
# ==============================================================================
FEATURE_BUNDLES = {
    "Speed & Geometry": [
        "speed_limit", "road_type", "first_road_class", "urban_or_rural_area",
        "junction_detail", "junction_control"
    ],
    "Vulnerability & Vehicles": [
        "any_motorcycle", "any_pedal_cycle", "any_hgv_or_bus", "most_vulnerable_vehicle_type",
        "number_of_vehicles", "age_of_vehicle_max", "age_of_vehicle_max_missing",
        "driver_age_band_youngest", "driver_age_band_youngest_missing"
    ],
    "Environmental & Surface": [
        "weather_conditions", "light_conditions", "road_surface_conditions",
        "trunk_road_flag", "carriageway_hazards"
    ],
    "Temporal Dynamics": [
        "hour", "day_of_week", "month"
    ]
}

GROUP_PERMUTATION_CLUSTERS = {
    "Speed & Road Geometry": ["speed_limit", "road_type", "first_road_class", "urban_or_rural_area"],
    "Vulnerable Road Users": ["any_motorcycle", "any_pedal_cycle", "most_vulnerable_vehicle_type"],
    "Vehicle Fleet & Age": ["number_of_vehicles", "any_hgv_or_bus", "age_of_vehicle_max", "age_of_vehicle_max_missing", "driver_age_band_youngest", "driver_age_band_youngest_missing"],
    "Junction Characteristics": ["junction_detail", "junction_control"],
    "Weather & Road Surface": ["weather_conditions", "light_conditions", "road_surface_conditions", "carriageway_hazards", "trunk_road_flag"],
    "Time & Seasonality": ["hour", "day_of_week", "month"]
}


def _clean_and_stringify(X):
    df = pd.DataFrame(X).copy()
    return df.fillna("Unknown").astype(str)

def create_custom_preprocessor(active_features: list) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer tailored to a subset of features."""
    from sklearn.preprocessing import FunctionTransformer
    active_cat = [f for f in active_features if f in CATEGORICAL_FEATURES]
    active_num = [f for f in active_features if f in NUMERICAL_FEATURES]
    active_bin = [f for f in active_features if f in BINARY_INDICATOR_FEATURES]
    
    transformers = []
    if active_num:
        num_pipe = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipe, active_num))
        
    if active_cat:
        cat_pipe = ImbPipeline([
            ("clean_str", FunctionTransformer(_clean_and_stringify)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipe, active_cat))
        
    if active_bin:
        transformers.append(("bin", "passthrough", active_bin))
        
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ==============================================================================
# 1. FEATURE ABLATION EXPERIMENTS (WITH BOOTSTRAP CIs)
# ==============================================================================
def run_feature_ablation_study(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits ablated models on 2021+2022 and tests on 2023 holdout test set.
    Measures PR-AUC drop and bootstrap 95% CIs.
    """
    print("\n" + "=" * 78)
    print("STAGE 6.1: FEATURE ABLATION STUDY (TASK A & TASK B)")
    print("=" * 78)
    
    train_df = df[df["collision_year"].isin([2021, 2022])].copy()
    test_df = df[df["collision_year"] == 2023].copy()
    
    y_train_a = train_df["target_severity_binary"].values
    y_test_a = test_df["target_severity_binary"].values
    
    y_train_b = train_df["target_severity_3class"].values
    y_test_b = test_df["target_severity_3class"].values
    y_test_fatal = (y_test_b == 0).astype(int)
    
    # ------------------ Task A Ablation ------------------
    # Baseline Full Model
    pre_full = create_custom_preprocessor(ALL_MODEL_FEATURES)
    clf_a_full = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.08, num_leaves=31, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
    pipe_a_full = ImbPipeline([("preprocessor", pre_full), ("classifier", clf_a_full)])
    pipe_a_full.fit(train_df[ALL_MODEL_FEATURES], y_train_a)
    probs_a_full = pipe_a_full.predict_proba(test_df[ALL_MODEL_FEATURES])[:, 1]
    base_pr_auc_a = average_precision_score(y_test_a, probs_a_full)
    base_roc_auc_a = roc_auc_score(y_test_a, probs_a_full)
    
    print(f"[+] Task A Full Model Baseline PR-AUC: {base_pr_auc_a:.4f} | ROC-AUC: {base_roc_auc_a:.4f}")
    
    ablation_rows_a = [{
        "Task": "Task A (Binary)",
        "Ablated_Bundle": "None (Full Model Baseline)",
        "Features_Removed_Count": 0,
        "Remaining_Features_Count": len(ALL_MODEL_FEATURES),
        "Test_PR_AUC": round(base_pr_auc_a, 4),
        "Delta_PR_AUC": 0.0000,
        "CI_95_Lower": 0.0000,
        "CI_95_Upper": 0.0000,
        "Test_ROC_AUC": round(base_roc_auc_a, 4),
        "Stat_Significant_Drop": "N/A"
    }]
    
    np.random.seed(42)
    n_boot = 500
    n_test = len(y_test_a)
    
    for bundle_name, bundle_cols in FEATURE_BUNDLES.items():
        rem_features = [f for f in ALL_MODEL_FEATURES if f not in bundle_cols]
        print(f"[*] Training Task A Ablated: Dropped '{bundle_name}' ({len(bundle_cols)} features removed)...")
        
        pre_abl = create_custom_preprocessor(rem_features)
        clf_abl = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.08, num_leaves=31, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
        pipe_abl = ImbPipeline([("preprocessor", pre_abl), ("classifier", clf_abl)])
        pipe_abl.fit(train_df[rem_features], y_train_a)
        probs_abl = pipe_abl.predict_proba(test_df[rem_features])[:, 1]
        
        abl_pr_auc = average_precision_score(y_test_a, probs_abl)
        abl_roc_auc = roc_auc_score(y_test_a, probs_abl)
        
        # Paired bootstrap for Delta PR-AUC
        deltas = []
        for _ in range(n_boot):
            b_idx = np.random.randint(0, n_test, n_test)
            pr_full = average_precision_score(y_test_a[b_idx], probs_a_full[b_idx])
            pr_abl = average_precision_score(y_test_a[b_idx], probs_abl[b_idx])
            deltas.append(pr_abl - pr_full)
            
        mean_delta = float(np.mean(deltas))
        ci_lo = float(np.percentile(deltas, 2.5))
        ci_hi = float(np.percentile(deltas, 97.5))
        sig_drop = "Yes (Performance Dropped)" if ci_hi < 0 else "No (Within Error Margin)"
        
        ablation_rows_a.append({
            "Task": "Task A (Binary)",
            "Ablated_Bundle": bundle_name,
            "Features_Removed_Count": len(bundle_cols),
            "Remaining_Features_Count": len(rem_features),
            "Test_PR_AUC": round(abl_pr_auc, 4),
            "Delta_PR_AUC": round(mean_delta, 4),
            "CI_95_Lower": round(ci_lo, 4),
            "CI_95_Upper": round(ci_hi, 4),
            "Test_ROC_AUC": round(abl_roc_auc, 4),
            "Stat_Significant_Drop": sig_drop
        })
        print(f"    -> Delta PR-AUC: {mean_delta:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}] ({sig_drop})")
        
    df_abl_a = pd.DataFrame(ablation_rows_a)
    df_abl_a.to_csv(f"{TABLES_DIR}/task_a_ablation_study.csv", index=False)
    
    # ------------------ Task B Ablation ------------------
    clf_b_full = lgb.LGBMClassifier(n_estimators=120, max_depth=7, learning_rate=0.06, num_leaves=40, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
    pipe_b_full = ImbPipeline([("preprocessor", pre_full), ("classifier", clf_b_full)])
    pipe_b_full.fit(train_df[ALL_MODEL_FEATURES], y_train_b)
    probs_b_full_fat = pipe_b_full.predict_proba(test_df[ALL_MODEL_FEATURES])[:, 0]
    base_fatal_pr_auc = average_precision_score(y_test_fatal, probs_b_full_fat)
    
    print(f"\n[+] Task B Full Model Baseline Fatal PR-AUC: {base_fatal_pr_auc:.4f}")
    
    ablation_rows_b = [{
        "Task": "Task B (3-Class)",
        "Ablated_Bundle": "None (Full Model Baseline)",
        "Features_Removed_Count": 0,
        "Remaining_Features_Count": len(ALL_MODEL_FEATURES),
        "Test_Fatal_PR_AUC": round(base_fatal_pr_auc, 4),
        "Delta_Fatal_PR_AUC": 0.0000,
        "CI_95_Lower": 0.0000,
        "CI_95_Upper": 0.0000,
        "Stat_Significant_Drop": "N/A"
    }]
    
    for bundle_name, bundle_cols in FEATURE_BUNDLES.items():
        rem_features = [f for f in ALL_MODEL_FEATURES if f not in bundle_cols]
        print(f"[*] Training Task B Ablated: Dropped '{bundle_name}' ({len(bundle_cols)} features removed)...")
        
        pre_abl = create_custom_preprocessor(rem_features)
        clf_abl = lgb.LGBMClassifier(n_estimators=120, max_depth=7, learning_rate=0.06, num_leaves=40, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
        pipe_abl = ImbPipeline([("preprocessor", pre_abl), ("classifier", clf_abl)])
        pipe_abl.fit(train_df[rem_features], y_train_b)
        probs_abl_fat = pipe_abl.predict_proba(test_df[rem_features])[:, 0]
        
        abl_fatal_pr = average_precision_score(y_test_fatal, probs_abl_fat)
        
        deltas = []
        for _ in range(n_boot):
            b_idx = np.random.randint(0, n_test, n_test)
            pr_full = average_precision_score(y_test_fatal[b_idx], probs_b_full_fat[b_idx])
            pr_abl = average_precision_score(y_test_fatal[b_idx], probs_abl_fat[b_idx])
            deltas.append(pr_abl - pr_full)
            
        mean_delta = float(np.mean(deltas))
        ci_lo = float(np.percentile(deltas, 2.5))
        ci_hi = float(np.percentile(deltas, 97.5))
        sig_drop = "Yes (Performance Dropped)" if ci_hi < 0 else "No (Within Error Margin)"
        
        ablation_rows_b.append({
            "Task": "Task B (3-Class)",
            "Ablated_Bundle": bundle_name,
            "Features_Removed_Count": len(bundle_cols),
            "Remaining_Features_Count": len(rem_features),
            "Test_Fatal_PR_AUC": round(abl_fatal_pr, 4),
            "Delta_Fatal_PR_AUC": round(mean_delta, 4),
            "CI_95_Lower": round(ci_lo, 4),
            "CI_95_Upper": round(ci_hi, 4),
            "Stat_Significant_Drop": sig_drop
        })
        print(f"    -> Delta Fatal PR-AUC: {mean_delta:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}] ({sig_drop})")
        
    df_abl_b = pd.DataFrame(ablation_rows_b)
    df_abl_b.to_csv(f"{TABLES_DIR}/task_b_ablation_study.csv", index=False)
    
    return df_abl_a, df_abl_b


# ==============================================================================
# 2. GROUP PERMUTATION FEATURE IMPORTANCE
# ==============================================================================
def run_group_permutation_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Group Permutation Feature Importance on the 2023 holdout test set.
    Permutes correlated groups together to avoid breaking physical correlation structure.
    """
    print("\n" + "=" * 78)
    print("STAGE 6.2: GROUP PERMUTATION FEATURE IMPORTANCE (TASK A & B)")
    print("=" * 78)
    
    test_df = df[df["collision_year"] == 2023].copy()
    y_test_a = test_df["target_severity_binary"].values
    y_test_fatal = (test_df["target_severity_3class"].values == 0).astype(int)
    
    # Load winning models
    art_a = joblib.load(f"{MODELS_DIR}/best_model_task_a.joblib")
    art_b = joblib.load(f"{MODELS_DIR}/best_model_task_b.joblib")
    
    pipe_a = art_a["pipeline"]
    pipe_b = art_b["pipeline"]
    
    # Baseline scores
    base_probs_a = pipe_a.predict_proba(test_df[ALL_MODEL_FEATURES])[:, 1]
    base_pr_a = average_precision_score(y_test_a, base_probs_a)
    
    base_probs_b = pipe_b.predict_proba(test_df[ALL_MODEL_FEATURES])[:, 0]
    base_pr_b = average_precision_score(y_test_fatal, base_probs_b)
    
    print(f"[+] Baseline Test PR-AUC: Task A = {base_pr_a:.4f} | Task B Fatal = {base_pr_b:.4f}")
    
    perm_rows = []
    np.random.seed(42)
    n_repeats = 5
    
    for cluster_name, cluster_cols in GROUP_PERMUTATION_CLUSTERS.items():
        drops_a = []
        drops_b = []
        
        for _ in range(n_repeats):
            X_perm = test_df[ALL_MODEL_FEATURES].copy()
            # Shuffle rows of the group simultaneously
            perm_indices = np.random.permutation(len(X_perm))
            for col in cluster_cols:
                X_perm[col] = X_perm[col].iloc[perm_indices].values
                
            probs_perm_a = pipe_a.predict_proba(X_perm)[:, 1]
            pr_perm_a = average_precision_score(y_test_a, probs_perm_a)
            drops_a.append(base_pr_a - pr_perm_a)
            
            probs_perm_b = pipe_b.predict_proba(X_perm)[:, 0]
            pr_perm_b = average_precision_score(y_test_fatal, probs_perm_b)
            drops_b.append(base_pr_b - pr_perm_b)
            
        mean_drop_a = float(np.mean(drops_a))
        mean_drop_b = float(np.mean(drops_b))
        
        perm_rows.append({
            "Feature_Group": cluster_name,
            "Features_Included": ", ".join(cluster_cols),
            "Num_Features": len(cluster_cols),
            "Task_A_PR_AUC_Drop": round(mean_drop_a, 4),
            "Task_A_Relative_Importance_Pct": round(max(0, mean_drop_a / base_pr_a) * 100, 2),
            "Task_B_Fatal_PR_AUC_Drop": round(mean_drop_b, 4),
            "Task_B_Relative_Importance_Pct": round(max(0, mean_drop_b / base_pr_b) * 100, 2)
        })
        print(f"    [{cluster_name:<28}] Task A Drop: {mean_drop_a:.4f} | Task B Fatal Drop: {mean_drop_b:.4f}")
        
    df_perm = pd.DataFrame(perm_rows).sort_values(by="Task_A_PR_AUC_Drop", ascending=False).reset_index(drop=True)
    df_perm.to_csv(f"{TABLES_DIR}/group_permutation_importance.csv", index=False)
    
    # Plot Group Permutation Importance Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    y_pos = np.arange(len(df_perm))
    ax1.barh(y_pos, df_perm["Task_A_PR_AUC_Drop"], color="#1B365D", edgecolor="black")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_perm["Feature_Group"], fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("PR-AUC Drop upon Group Permutation", fontweight="bold")
    ax1.set_title("Task A (Dangerous Severity)\nGroup Permutation Feature Importance", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    df_perm_b = df_perm.sort_values(by="Task_B_Fatal_PR_AUC_Drop", ascending=False).reset_index(drop=True)
    y_pos_b = np.arange(len(df_perm_b))
    ax2.barh(y_pos_b, df_perm_b["Task_B_Fatal_PR_AUC_Drop"], color="#8B0000", edgecolor="black")
    ax2.set_yticks(y_pos_b)
    ax2.set_yticklabels(df_perm_b["Feature_Group"], fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Fatal PR-AUC Drop upon Group Permutation", fontweight="bold")
    ax2.set_title("Task B (Fatal Severity)\nGroup Permutation Feature Importance", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/14_group_permutation_importance.png", dpi=300)
    plt.close()
    print(f"[+] Saved Group Permutation Chart to {CHARTS_DIR}/14_group_permutation_importance.png")
    
    return df_perm


# ==============================================================================
# 3. REGIONAL GENERALIZATION AUDIT
# ==============================================================================
def run_regional_generalization_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audits model generalization across UK geography:
      1. Urban vs Rural environments
      2. Top 10 Police Force Jurisdictions
    """
    print("\n" + "=" * 78)
    print("STAGE 6.3: REGIONAL & GEOGRAPHIC GENERALIZATION AUDIT")
    print("=" * 78)
    
    test_df = df[df["collision_year"] == 2023].copy()
    
    art_a = joblib.load(f"{MODELS_DIR}/best_model_task_a.joblib")
    art_b = joblib.load(f"{MODELS_DIR}/best_model_task_b.joblib")
    pipe_a = art_a["pipeline"]
    calib_a = art_a["calibrator"]
    pipe_b = art_b["pipeline"]
    calib_b = art_b["fatal_calibrator"]
    
    raw_probs_a = pipe_a.predict_proba(test_df[ALL_MODEL_FEATURES])[:, 1]
    cal_probs_a = calib_a.predict_proba(raw_probs_a)
    
    raw_probs_b = pipe_b.predict_proba(test_df[ALL_MODEL_FEATURES])[:, 0]
    cal_probs_b = calib_b.predict_proba(raw_probs_b)
    
    test_df["pred_cal_a"] = cal_probs_a
    test_df["pred_cal_fatal"] = cal_probs_b
    
    # 1. Urban vs Rural
    reg_rows = []
    for ur_val, ur_label in [(1, "Urban"), (2, "Rural")]:
        sub = test_df[test_df["urban_or_rural_area"] == ur_val]
        n_sub = len(sub)
        y_a = sub["target_severity_binary"].values
        y_fat = (sub["target_severity_3class"].values == 0).astype(int)
        
        pr_a = average_precision_score(y_a, sub["pred_cal_a"])
        roc_a = roc_auc_score(y_a, sub["pred_cal_a"])
        rec_a = recall_score(y_a, (sub["pred_cal_a"] >= 0.12).astype(int), zero_division=0)
        
        pr_b = average_precision_score(y_fat, sub["pred_cal_fatal"])
        roc_b = roc_auc_score(y_fat, sub["pred_cal_fatal"])
        rec_b = recall_score(y_fat, (sub["pred_cal_fatal"] >= 0.04).astype(int), zero_division=0)
        
        reg_rows.append({
            "Region_Category": "Area Type",
            "Region_Name": ur_label,
            "Police_Force_Code": "N/A",
            "Collisions_Count": n_sub,
            "Dangerous_Prevalence": round(float(np.mean(y_a)), 4),
            "Task_A_PR_AUC": round(pr_a, 4),
            "Task_A_ROC_AUC": round(roc_a, 4),
            "Task_A_Recall_at_Tau": round(rec_a * 100, 2),
            "Fatal_Prevalence": round(float(np.mean(y_fat)), 4),
            "Task_B_Fatal_PR_AUC": round(pr_b, 4),
            "Task_B_Fatal_ROC_AUC": round(roc_b, 4),
            "Task_B_Fatal_Recall_at_Tau": round(rec_b * 100, 2)
        })
        
    # 2. Top 10 Police Forces
    top_forces = test_df["police_force"].value_counts().head(10).index.tolist()
    force_names = {
        1: "Metropolitan Police (London)",
        20: "West Midlands",
        6: "Greater Manchester",
        43: "Thames Valley",
        50: "Devon & Cornwall",
        13: "West Yorkshire",
        52: "Avon & Somerset",
        46: "Kent",
        42: "Essex",
        44: "Hampshire"
    }
    
    for pf in top_forces:
        sub = test_df[test_df["police_force"] == pf]
        n_sub = len(sub)
        y_a = sub["target_severity_binary"].values
        y_fat = (sub["target_severity_3class"].values == 0).astype(int)
        
        pr_a = average_precision_score(y_a, sub["pred_cal_a"])
        roc_a = roc_auc_score(y_a, sub["pred_cal_a"])
        rec_a = recall_score(y_a, (sub["pred_cal_a"] >= 0.12).astype(int), zero_division=0)
        
        pr_b = average_precision_score(y_fat, sub["pred_cal_fatal"])
        roc_b = roc_auc_score(y_fat, sub["pred_cal_fatal"])
        rec_b = recall_score(y_fat, (sub["pred_cal_fatal"] >= 0.04).astype(int), zero_division=0)
        
        reg_rows.append({
            "Region_Category": "Police Force",
            "Region_Name": force_names.get(pf, f"Police Force {pf}"),
            "Police_Force_Code": str(pf),
            "Collisions_Count": n_sub,
            "Dangerous_Prevalence": round(float(np.mean(y_a)), 4),
            "Task_A_PR_AUC": round(pr_a, 4),
            "Task_A_ROC_AUC": round(roc_a, 4),
            "Task_A_Recall_at_Tau": round(rec_a * 100, 2),
            "Fatal_Prevalence": round(float(np.mean(y_fat)), 4),
            "Task_B_Fatal_PR_AUC": round(pr_b, 4),
            "Task_B_Fatal_ROC_AUC": round(roc_b, 4),
            "Task_B_Fatal_Recall_at_Tau": round(rec_b * 100, 2)
        })
        
    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(f"{TABLES_DIR}/regional_generalization_audit.csv", index=False)
    
    # Plot Regional Generalization
    sub_pf = df_reg[df_reg["Region_Category"] == "Police Force"]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sub_pf))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, sub_pf["Task_A_ROC_AUC"], width, label="Task A ROC-AUC", color="#1B365D")
    rects2 = ax1.bar(x + width/2, sub_pf["Task_A_PR_AUC"], width, label="Task A PR-AUC", color="#5B92E5")
    
    ax1.set_ylabel("Metric Score", fontweight="bold")
    ax1.set_title("Regional Generalization Audit Across Top 10 UK Police Forces\nTask A Performance Consistency on 2023 Holdout Test", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub_pf["Region_Name"], rotation=30, ha="right", fontweight="bold", fontsize=9)
    ax1.set_ylim([0.0, 0.85])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/15_regional_generalization_audit.png", dpi=300)
    plt.close()
    print(f"[+] Saved Regional Generalization Chart to {CHARTS_DIR}/15_regional_generalization_audit.png")
    
    return df_reg


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("[*] Loading dataset for Stage 6...")
    df_all = pd.read_parquet("data/clean_roadsense.parquet")
    
    # 1. Feature Ablation Study
    df_abl_a, df_abl_b = run_feature_ablation_study(df_all)
    
    # 2. Group Permutation Feature Importance
    df_perm = run_group_permutation_importance(df_all)
    
    # 3. Regional Generalization Audit
    df_reg = run_regional_generalization_audit(df_all)
    
    print("\n" + "=" * 78)
    print("[+] STAGE 6 COMPLETE: All tables and charts saved successfully.")
    print("=" * 78)

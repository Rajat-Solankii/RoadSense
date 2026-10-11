"""
RoadSense AI: Stage 7 SHAP Feature Attribution
Implements:
  1. TreeExplainer-based SHAP values for winning LightGBM models (Task A & B)
  2. Global attribution: bar plot (mean |SHAP|) + beeswarm dot plot
  3. Waterfall plot for a sample collision (highest-risk, median-risk, lowest-risk)
  4. SHAP dependence plots for top-5 features
  5. Saved SHAP values to outputs/tables/ for downstream use
  6. Feature-name mapping from transformed ColumnTransformer back to human-readable names

SHAP value formats from LightGBM TreeExplainer (SHAP >= 0.42):
  - Binary classifier:   ndarray (n_samples, n_features), expected_value is scalar
  - Multiclass (K=3):    ndarray (n_samples, n_features, K), expected_value is array of K
  - Regressor:           ndarray (n_samples, n_features), expected_value is scalar
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.abspath("."))
from src.features import ALL_MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BINARY_INDICATOR_FEATURES

TABLES_DIR = "outputs/tables"
CHARTS_DIR = "outputs/charts"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Suppress verbose LightGBM/SHAP logs
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# 1. FEATURE NAME RECOVERY FROM ColumnTransformer
# ==============================================================================
def get_transformed_feature_names(pipeline) -> list:
    """
    Extracts human-readable feature names from the fitted ColumnTransformer
    inside the imblearn Pipeline.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []

    for trans_name, transformer, columns in preprocessor.transformers_:
        if trans_name == "num":
            feature_names.extend(columns)
        elif trans_name == "cat":
            encoder = transformer.named_steps["encoder"]
            cat_names = encoder.get_feature_names_out(columns)
            feature_names.extend(cat_names)
        elif trans_name == "bin":
            feature_names.extend(columns)

    return list(feature_names)


def shorten_feature_name(name: str) -> str:
    """Shortens OneHotEncoder feature names for readability in plots."""
    replacements = {
        "weather_conditions_": "Weather: ",
        "road_type_": "Road: ",
        "light_conditions_": "Light: ",
        "road_surface_conditions_": "Surface: ",
        "junction_detail_": "Junction: ",
        "urban_or_rural_area_": "Area: ",
        "first_road_class_": "RdClass: ",
        "junction_control_": "JnCtrl: ",
        "trunk_road_flag_": "Trunk: ",
        "carriageway_hazards_": "Hazard: ",
        "most_vulnerable_vehicle_type_": "Vuln: ",
    }
    for prefix, short in replacements.items():
        if name.startswith(prefix):
            return short + name[len(prefix):]
    return name


# ==============================================================================
# 2. SHAP VALUE SHAPE HELPERS
# ==============================================================================
def _extract_class_shap(sv: np.ndarray, expected_value, class_index: int = None):
    """
    Normalizes SHAP value extraction across different TreeExplainer output formats.
    
    Returns:
        sv_2d: 2D array (n_samples, n_features) for the requested class
        base_value: scalar base value (expected_value) for the class
    """
    if sv.ndim == 3:
        # Multiclass: shape (n_samples, n_features, n_classes)
        n_classes = sv.shape[2]
        ci = class_index if class_index is not None else 0
        sv_2d = sv[:, :, ci]
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = float(expected_value[ci])
        else:
            base_value = float(expected_value)
        return sv_2d, base_value
    elif sv.ndim == 2:
        # Binary classifier or regressor: shape (n_samples, n_features)
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = float(expected_value[0] if len(expected_value) == 1 else expected_value)
        else:
            base_value = float(expected_value)
        return sv, base_value
    elif isinstance(sv, list):
        # Legacy list-of-arrays format (unlikely with modern SHAP, but handle it)
        ci = class_index if class_index is not None else (1 if len(sv) == 2 else 0)
        sv_2d = sv[ci]
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = float(expected_value[ci])
        else:
            base_value = float(expected_value)
        return sv_2d, base_value
    else:
        raise ValueError(f"Unexpected SHAP values shape: {sv.shape if hasattr(sv, 'shape') else type(sv)}")


# ==============================================================================
# 3. SHAP COMPUTATION
# ==============================================================================
def compute_shap_values(model_artifact: dict, X_test: pd.DataFrame, task: str = "A",
                        max_samples: int = 5000) -> dict:
    """
    Computes SHAP values using TreeExplainer on the LightGBM classifier/regressor.
    """
    pipeline = model_artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Transform features
    X_transformed = preprocessor.transform(X_test)
    feature_names = get_transformed_feature_names(pipeline)

    # Subsample for speed if needed
    if len(X_transformed) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_transformed), size=max_samples, replace=False)
        X_transformed = X_transformed[idx]
    else:
        idx = np.arange(len(X_transformed))

    # Extract the raw model
    if task in ("A", "B"):
        model = pipeline.named_steps["classifier"]
    else:
        model = pipeline.named_steps["regressor"]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # Convert list to ndarray if needed
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)  # -> (n, f, K)

    print(f"  SHAP values shape: {shap_values.shape}, expected_value: {explainer.expected_value}")

    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": feature_names,
        "X_transformed": X_transformed,
        "sample_idx": idx,
    }


# ==============================================================================
# 4. GLOBAL ATTRIBUTION PLOTS
# ==============================================================================
def plot_global_shap_bar(shap_result: dict, task: str, top_n: int = 20,
                         class_index: int = None):
    """
    Bar plot of mean absolute SHAP values (global feature importance).
    """
    sv = shap_result["shap_values"]
    names = [shorten_feature_name(n) for n in shap_result["feature_names"]]

    sv_2d, _ = _extract_class_shap(sv, shap_result["expected_value"], class_index)
    vals = np.abs(sv_2d).mean(axis=0)

    # Title and suffix
    if sv.ndim == 3 and class_index is not None:
        class_label = {0: "Fatal", 1: "Serious", 2: "Slight"}.get(class_index, str(class_index))
        title = f"Task {task}: Mean |SHAP| — {class_label} Class"
        suffix = f"_{class_label.lower()}"
    else:
        title = f"Task {task}: Mean |SHAP| — Global Feature Importance"
        suffix = ""

    # Limit top_n to available features
    n_feats = min(top_n, len(vals))
    order = np.argsort(vals)[::-1][:n_feats]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(n_feats)
    ax.barh(y_pos, vals[order][::-1], color="#4C72B0", edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in order][::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    save_path = os.path.join(CHARTS_DIR, f"shap_bar_task_{task.lower()}{suffix}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved: {save_path}")
    return save_path


def plot_global_shap_beeswarm(shap_result: dict, task: str, top_n: int = 20,
                               class_index: int = None):
    """
    Beeswarm (dot) plot showing feature value impact distribution.
    """
    sv = shap_result["shap_values"]
    names = [shorten_feature_name(n) for n in shap_result["feature_names"]]
    X_disp = shap_result["X_transformed"]

    sv_2d, base_val = _extract_class_shap(sv, shap_result["expected_value"], class_index)

    # Title
    if sv.ndim == 3 and class_index is not None:
        class_label = {0: "Fatal", 1: "Serious", 2: "Slight"}.get(class_index, str(class_index))
        title = f"Task {task}: SHAP Beeswarm — {class_label}"
        suffix = f"_{class_label.lower()}"
    else:
        title = f"Task {task}: SHAP Beeswarm"
        suffix = ""

    # Build a shap.Explanation object
    explanation = shap.Explanation(
        values=sv_2d,
        base_values=np.full(sv_2d.shape[0], base_val),
        data=X_disp,
        feature_names=names
    )

    fig = plt.figure(figsize=(11, 9))
    shap.plots.beeswarm(explanation, max_display=top_n, show=False)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(CHARTS_DIR, f"shap_beeswarm_task_{task.lower()}{suffix}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved: {save_path}")
    return save_path


# ==============================================================================
# 5. WATERFALL PLOTS (SAMPLE EXPLANATIONS)
# ==============================================================================
def plot_waterfall_samples(shap_result: dict, task: str, class_index: int = None):
    """
    Generates waterfall plots for 3 sample collisions:
      - Highest risk (highest SHAP sum for positive/fatal class)
      - Median risk
      - Lowest risk
    """
    sv = shap_result["shap_values"]
    names = [shorten_feature_name(n) for n in shap_result["feature_names"]]

    sv_2d, base_val = _extract_class_shap(sv, shap_result["expected_value"], class_index)

    suffix = ""
    if sv.ndim == 3 and class_index is not None:
        class_label = {0: "Fatal", 1: "Serious", 2: "Slight"}.get(class_index, str(class_index))
        suffix = f"_{class_label.lower()}"

    # Find interesting samples
    sums = sv_2d.sum(axis=1)
    highest_idx = np.argmax(sums)
    lowest_idx = np.argmin(sums)
    median_idx = np.argsort(sums)[len(sums) // 2]

    samples = [
        ("highest_risk", highest_idx),
        ("median_risk", median_idx),
        ("lowest_risk", lowest_idx),
    ]

    saved = []
    for label, idx in samples:
        explanation = shap.Explanation(
            values=sv_2d[idx],
            base_values=float(base_val),
            data=shap_result["X_transformed"][idx],
            feature_names=names
        )

        fig = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(f"Task {task}: Waterfall — {label.replace('_', ' ').title()}{suffix}",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(CHARTS_DIR, f"shap_waterfall_task_{task.lower()}_{label}{suffix}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(save_path)
        print(f"  [+] Saved: {save_path}")

    return saved


# ==============================================================================
# 6. DEPENDENCE PLOTS (TOP-5 FEATURES)
# ==============================================================================
def plot_shap_dependence(shap_result: dict, task: str, top_n: int = 5,
                         class_index: int = None):
    """
    SHAP dependence plots for the top-N most important features.
    """
    sv = shap_result["shap_values"]
    names = shap_result["feature_names"]

    sv_2d, _ = _extract_class_shap(sv, shap_result["expected_value"], class_index)

    importance = np.abs(sv_2d).mean(axis=0)
    top_indices = np.argsort(importance)[::-1][:top_n]

    saved = []
    for rank, feat_idx in enumerate(top_indices):
        feat_name = names[feat_idx]
        short_name = shorten_feature_name(feat_name)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feat_idx,
            sv_2d,
            shap_result["X_transformed"],
            feature_names=[shorten_feature_name(n) for n in names],
            ax=ax,
            show=False
        )
        class_suffix = ""
        if class_index is not None:
            class_suffix = f" ({['Fatal','Serious','Slight'][class_index]})"
        ax.set_title(f"Task {task}{class_suffix}: SHAP Dependence — {short_name}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        safe_name = feat_name.replace("/", "_").replace(" ", "_")[:30]
        suffix = f"_class{class_index}" if class_index is not None else ""
        save_path = os.path.join(CHARTS_DIR, f"shap_dep_task_{task.lower()}_{rank+1}_{safe_name}{suffix}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(save_path)
        print(f"  [+] Saved: {save_path}")

    return saved


# ==============================================================================
# 7. SAVE GLOBAL SHAP IMPORTANCE TABLE
# ==============================================================================
def save_shap_importance_table(shap_result: dict, task: str, class_index: int = None):
    """
    Saves mean |SHAP| per feature as a CSV table for downstream analysis.
    """
    sv = shap_result["shap_values"]
    names = shap_result["feature_names"]

    sv_2d, _ = _extract_class_shap(sv, shap_result["expected_value"], class_index)
    vals = np.abs(sv_2d).mean(axis=0)

    suffix = ""
    if sv.ndim == 3 and class_index is not None:
        suffix = f"_class{class_index}"

    df = pd.DataFrame({
        "feature": names,
        "feature_short": [shorten_feature_name(n) for n in names],
        "mean_abs_shap": np.round(vals, 6)
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.index.name = "rank"
    df.index = df.index + 1

    save_path = os.path.join(TABLES_DIR, f"shap_importance_task_{task.lower()}{suffix}.csv")
    df.to_csv(save_path)
    print(f"  [+] Saved: {save_path}")
    return save_path


# ==============================================================================
# 8. MASTER RUNNER
# ==============================================================================
def run_shap_analysis():
    """
    Executes complete SHAP attribution analysis for Task A, Task B, and Task C.
    """
    print("=" * 78)
    print("STAGE 7: SHAP FEATURE ATTRIBUTION ANALYSIS")
    print("=" * 78)

    # Load test data
    df = pd.read_parquet("data/clean_roadsense.parquet")
    test_23 = df[df["collision_year"] == 2023].copy()
    X_test = test_23[ALL_MODEL_FEATURES]
    print(f"[*] Loaded 2023 test set: {len(X_test):,} collisions")

    # ==================== TASK A ====================
    print("\n" + "-" * 70)
    print("TASK A: Binary Severity (Dangerous vs Minor)")
    print("-" * 70)

    model_a = joblib.load("models/best_model_task_a.joblib")
    print(f"  Model: {model_a['model_name']} ({model_a['imbalance_strategy']})")

    shap_a = compute_shap_values(model_a, X_test, task="A", max_samples=5000)

    # Binary LightGBM: shap_values is 2D (n_samples, n_features)
    # No class_index needed
    print("  Computing global attribution plots...")
    plot_global_shap_bar(shap_a, task="A")
    plot_global_shap_beeswarm(shap_a, task="A")

    print("  Computing waterfall explanations...")
    plot_waterfall_samples(shap_a, task="A")

    print("  Computing dependence plots...")
    plot_shap_dependence(shap_a, task="A", top_n=5)

    print("  Saving importance table...")
    save_shap_importance_table(shap_a, task="A")

    # ==================== TASK B ====================
    print("\n" + "-" * 70)
    print("TASK B: 3-Class Severity (Fatal / Serious / Slight)")
    print("-" * 70)

    model_b = joblib.load("models/best_model_task_b.joblib")
    print(f"  Model: {model_b['model_name']} ({model_b['imbalance_strategy']})")

    shap_b = compute_shap_values(model_b, X_test, task="B", max_samples=5000)

    # Fatal class (0) — most policy-relevant
    print("  Computing Fatal class attribution...")
    plot_global_shap_bar(shap_b, task="B", class_index=0)
    plot_global_shap_beeswarm(shap_b, task="B", class_index=0)
    plot_waterfall_samples(shap_b, task="B", class_index=0)
    plot_shap_dependence(shap_b, task="B", top_n=5, class_index=0)
    save_shap_importance_table(shap_b, task="B", class_index=0)

    # Serious class (1) — for supplementary analysis
    print("  Computing Serious class attribution...")
    plot_global_shap_bar(shap_b, task="B", class_index=1)
    save_shap_importance_table(shap_b, task="B", class_index=1)

    # ==================== TASK C ====================
    print("\n" + "-" * 70)
    print("TASK C: Casualty Count Regression")
    print("-" * 70)

    model_c = joblib.load("models/best_model_task_c.joblib")
    print(f"  Model: {model_c['model_name']}")

    shap_c = compute_shap_values(model_c, X_test, task="C", max_samples=5000)

    print("  Computing global attribution plots...")
    plot_global_shap_bar(shap_c, task="C")
    plot_global_shap_beeswarm(shap_c, task="C")

    print("  Computing waterfall explanations...")
    plot_waterfall_samples(shap_c, task="C")

    print("  Computing dependence plots...")
    plot_shap_dependence(shap_c, task="C", top_n=5)

    print("  Saving importance table...")
    save_shap_importance_table(shap_c, task="C")

    print("\n" + "=" * 78)
    print("STAGE 7 COMPLETE: All SHAP attribution charts and tables saved.")
    print("=" * 78)


if __name__ == "__main__":
    run_shap_analysis()

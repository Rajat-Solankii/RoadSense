"""
RoadSense AI: Exploratory Data Analysis (EDA) Module
Responsible for:
  - Computing bivariate risk rates (Dangerous Rate = Fatal + Serious %) across core & extra features
  - Generating publication-quality charts saved to outputs/charts/
  - Evaluating distribution shifts across 2021 (COVID), 2022, and 2023
  - Documenting casualty count distribution and extreme outliers (70 casualties bus crash)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configure aesthetic visual defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.autolayout": True
})

CHARTS_DIR = "outputs/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Value mappings for readable labels in plots
WEATHER_LABELS = {
    1: "Fine (no wind)",
    2: "Rain (no wind)",
    3: "Snow (no wind)",
    4: "Fine + high winds",
    5: "Rain + high winds",
    6: "Snow + high winds",
    7: "Fog / mist",
    8: "Other",
    "Unknown": "Unknown"
}

ROAD_TYPE_LABELS = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    "Unknown": "Unknown"
}

LIGHT_LABELS = {
    1: "Daylight",
    4: "Darkness (lights lit)",
    5: "Darkness (lights unlit)",
    6: "Darkness (no lighting)",
    7: "Darkness (unknown lighting)",
    "Unknown": "Unknown"
}

VULN_LABELS = {
    "pedal_cycle": "Pedal Cycle",
    "motorcycle": "Motorcycle",
    "car_taxi": "Car / Taxi",
    "van": "Van / Light Goods",
    "hgv_or_bus": "Bus / HGV",
    "other_unknown": "Other / Unknown"
}


def plot_class_distributions(df: pd.DataFrame, save_path: str = f"{CHARTS_DIR}/01_class_distribution.png"):
    """Chart 1: Class balance for Task A (Binary) and Task B (3-class)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Task A: Binary
    bin_counts = df["target_severity_binary"].value_counts().sort_index()
    bin_pcts = (bin_counts / len(df)) * 100
    labels_a = ["Minor (Slight)", "Dangerous (Fatal + Serious)"]
    colors_a = ["#4A90E2", "#D0021B"]
    
    bars_a = axes[0].bar(labels_a, bin_counts, color=colors_a, width=0.55, edgecolor="black", alpha=0.85)
    axes[0].set_title("Task A: Binary Severity Distribution", fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Collisions")
    axes[0].set_ylim(0, max(bin_counts) * 1.15)
    for bar, count, pct in zip(bars_a, bin_counts, bin_pcts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3000,
                     f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    # Task B: 3-Class
    mc_counts = df["target_severity_3class"].value_counts().sort_index()
    mc_pcts = (mc_counts / len(df)) * 100
    labels_b = ["Fatal (Class 0)", "Serious (Class 1)", "Slight (Class 2)"]
    colors_b = ["#9013FE", "#F5A623", "#4A90E2"]
    
    bars_b = axes[1].bar(labels_b, mc_counts, color=colors_b, width=0.55, edgecolor="black", alpha=0.85)
    axes[1].set_title("Task B: 3-Class Severity Distribution (Severe Imbalance)", fontweight="bold", pad=12)
    axes[1].set_ylabel("Number of Collisions")
    axes[1].set_ylim(0, max(mc_counts) * 1.15)
    for bar, count, pct in zip(bars_b, mc_counts, mc_pcts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3000,
                     f"{count:,}\n({pct:.2f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_danger_by_speed_limit(df: pd.DataFrame, save_path: str = f"{CHARTS_DIR}/02_danger_rate_by_speed_limit.png"):
    """Chart 2: Danger rate & Fatal rate by posted speed limit."""
    grp = df.groupby("speed_limit").agg(
        total=("target_severity_binary", "count"),
        dangerous=("target_severity_binary", "sum"),
        fatal=("target_severity_3class", lambda x: (x == 0).sum())
    ).reset_index()
    
    grp["danger_rate"] = (grp["dangerous"] / grp["total"]) * 100
    grp["fatal_rate"] = (grp["fatal"] / grp["total"]) * 100
    
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(grp))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, grp["danger_rate"], width, label="Dangerous Rate (Fatal+Serious %)", color="#D0021B", alpha=0.85, edgecolor="black")
    bars2 = ax1.bar(x + width/2, grp["fatal_rate"] * 5, width, label="Fatal Rate % (Scale ×5 for visibility)", color="#9013FE", alpha=0.85, edgecolor="black")
    
    ax1.set_xlabel("Posted Speed Limit (mph)")
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Accident Severity Escapes Exponentially with Posted Speed Limit", fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(s)} mph\n(n={t:,})" for s, t in zip(grp["speed_limit"], grp["total"])])
    ax1.legend(loc="upper left")
    
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.6, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, actual_f in zip(bars2, grp["fatal_rate"]):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.6, f"({actual_f:.2f}%)", ha="center", va="bottom", fontsize=8, color="#551A8B")
        
    ax1.set_ylim(0, max(grp["danger_rate"]) * 1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_danger_by_weather(df: pd.DataFrame, save_path: str = f"{CHARTS_DIR}/03_danger_rate_by_weather.png"):
    """Chart 3: Danger rate across weather conditions."""
    clean_w = df["weather_conditions"].fillna("Unknown")
    grp = df.assign(weather_clean=clean_w).groupby("weather_clean").agg(
        total=("target_severity_binary", "count"),
        dangerous=("target_severity_binary", "sum")
    ).reset_index()
    grp["danger_rate"] = (grp["dangerous"] / grp["total"]) * 100
    grp["weather_label"] = grp["weather_clean"].map(lambda c: WEATHER_LABELS.get(c, str(c)))
    grp = grp[grp["total"] > 100].sort_values(by="danger_rate", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(grp["weather_label"], grp["danger_rate"], color="#4A90E2", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Dangerous Rate (Fatal + Serious %)")
    ax.set_title("Accident Danger Rate Across Weather Conditions", fontweight="bold", pad=12)
    ax.set_xlim(0, max(grp["danger_rate"]) * 1.2)
    
    for bar, rate, tot in zip(bars, grp["danger_rate"], grp["total"]):
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height()/2, f"{rate:.1f}% (n={tot:,})", ha="left", va="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_danger_by_road_type(df: pd.DataFrame, save_path: str = f"{CHARTS_DIR}/04_danger_rate_by_road_type.png"):
    """Chart 4: Danger rate by road type."""
    clean_r = df["road_type"].fillna("Unknown")
    grp = df.assign(road_clean=clean_r).groupby("road_clean").agg(
        total=("target_severity_binary", "count"),
        dangerous=("target_severity_binary", "sum")
    ).reset_index()
    grp["danger_rate"] = (grp["dangerous"] / grp["total"]) * 100
    grp["road_label"] = grp["road_clean"].map(lambda c: ROAD_TYPE_LABELS.get(c, str(c)))
    grp = grp.sort_values(by="danger_rate", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(grp["road_label"], grp["danger_rate"], color="#F5A623", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Dangerous Rate (Fatal + Serious %)")
    ax.set_title("Accident Danger Rate by Road Type Layout", fontweight="bold", pad=12)
    ax.set_xlim(0, max(grp["danger_rate"]) * 1.25)
    
    for bar, rate, tot in zip(bars, grp["danger_rate"], grp["total"]):
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height()/2, f"{rate:.1f}% (n={tot:,})", ha="left", va="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_danger_by_vehicle_type(df: pd.DataFrame, save_path: str = f"{CHARTS_DIR}/05_danger_rate_by_vehicle_type.png"):
    """Chart 5: Danger rate across most vulnerable vehicle involved."""
    grp = df.groupby("most_vulnerable_vehicle_type").agg(
        total=("target_severity_binary", "count"),
        dangerous=("target_severity_binary", "sum"),
        fatal=("target_severity_3class", lambda x: (x == 0).sum())
    ).reset_index()
    grp["danger_rate"] = (grp["dangerous"] / grp["total"]) * 100
    grp["fatal_rate"] = (grp["fatal"] / grp["total"]) * 100
    grp["label"] = grp["most_vulnerable_vehicle_type"].map(VULN_LABELS)
    grp = grp.sort_values(by="danger_rate", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(grp["label"], grp["danger_rate"], color="#D0021B", edgecolor="black", alpha=0.85, width=0.55)
    ax.set_ylabel("Dangerous Rate (%)")
    ax.set_title("Vulnerable Road Users (Motorcycles & Pedal Cycles) Suffer Severe Outcomes", fontweight="bold", pad=12)
    ax.set_ylim(0, max(grp["danger_rate"]) * 1.2)
    
    for bar, rate, fat, tot in zip(bars, grp["danger_rate"], grp["fatal_rate"], grp["total"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                f"{rate:.1f}%\n(Fatal: {fat:.1f}%)\n[n={tot:,}]", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_danger_by_hour(df: pd.DataFrame, save_path: str = f"{CHARTS_DIR}/06_danger_rate_by_hour.png"):
    """Chart 6: Collision Volume vs Danger Rate by Hour of Day."""
    grp = df.groupby("hour").agg(
        total=("target_severity_binary", "count"),
        dangerous=("target_severity_binary", "sum")
    ).reset_index()
    grp["danger_rate"] = (grp["dangerous"] / grp["total"]) * 100
    
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()
    
    # Volume bars
    ax1.bar(grp["hour"], grp["total"], color="#BDC3C7", alpha=0.6, label="Accident Frequency (Volume)", width=0.65)
    ax1.set_xlabel("Hour of Day (0 - 23)")
    ax1.set_ylabel("Total Accident Count", color="#555555")
    ax1.set_xticks(range(0, 24))
    
    # Danger rate line
    line = ax2.plot(grp["hour"], grp["danger_rate"], color="#D0021B", marker="o", linewidth=2.5, label="Dangerous Rate (%)")
    ax2.set_ylabel("Dangerous Rate (%)", color="#D0021B", fontweight="bold")
    ax2.set_ylim(15, 38)
    
    ax1.set_title("Collision Volume vs Danger Rate: Night Hours Peak in Lethality", fontweight="bold", pad=12)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_danger_by_light_condition(df: pd.DataFrame, save_path: str = f"{CHARTS_DIR}/07_danger_rate_by_light_condition.png"):
    """Chart 7: Danger rate by lighting conditions."""
    clean_l = df["light_conditions"].fillna("Unknown")
    grp = df.assign(light_clean=clean_l).groupby("light_clean").agg(
        total=("target_severity_binary", "count"),
        dangerous=("target_severity_binary", "sum")
    ).reset_index()
    grp["danger_rate"] = (grp["dangerous"] / grp["total"]) * 100
    grp["label"] = grp["light_clean"].map(lambda c: LIGHT_LABELS.get(c, str(c)))
    grp = grp.sort_values(by="danger_rate", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(grp["label"], grp["danger_rate"], color="#34495E", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Dangerous Rate (Fatal + Serious %)")
    ax.set_title("Accident Danger Rate by Environmental Light Conditions", fontweight="bold", pad=12)
    ax.set_xlim(0, max(grp["danger_rate"]) * 1.25)
    
    for bar, rate, tot in zip(bars, grp["danger_rate"], grp["total"]):
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height()/2, f"{rate:.1f}% (n={tot:,})", ha="left", va="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_casualties_distribution(df: pd.DataFrame, save_path: str = f"{CHARTS_DIR}/08_casualties_distribution_and_outlier.png"):
    """Chart 8: Casualties Distribution & Extreme Outlier (70 casualties)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Zoomed distribution (1 to 6 casualties)
    sub = df[df["target_casualties"] <= 6]
    cas_counts = sub["target_casualties"].value_counts().sort_index()
    cas_pcts = (cas_counts / len(df)) * 100
    
    bars = axes[0].bar(cas_counts.index, cas_counts.values, color="#27AE60", edgecolor="black", alpha=0.85, width=0.6)
    axes[0].set_title("Casualty Count Distribution (Truncated at 6 for Scale)", fontweight="bold", pad=12)
    axes[0].set_xlabel("Number of Casualties")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(range(1, 7))
    for bar, count, pct in zip(bars, cas_counts.values, cas_pcts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
                     f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    # Full distribution boxplot showing extreme outlier
    sns.boxplot(x=df["target_casualties"], ax=axes[1], color="#E74C3C")
    axes[1].set_title("Casualty Boxplot Highlighting 2023 Outlier (n=70 Casualties)", fontweight="bold", pad=12)
    axes[1].set_xlabel("Number of Casualties")
    axes[1].annotate("Extreme Outlier: 70 Casualties (2023 Bus Crash)\nKept in Test Set for Heavy Tail Evaluation",
                     xy=(70, 0), xytext=(35, -0.25),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                     fontsize=9.5, fontweight="bold", bbox=dict(boxstyle="round", facecolor="#FFEBEE", edgecolor="#E74C3C"))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {save_path}")


def run_all_eda(parquet_path: str = "data/clean_roadsense.parquet"):
    """Runs all 8 EDA chart generation pipelines."""
    print("=" * 70)
    print("ROADSENSE AI: STAGE 2 EDA INITIATING")
    print("=" * 70)
    df = pd.read_parquet(parquet_path)
    print(f"Loaded master clean dataset: {len(df):,} records.")
    
    plot_class_distributions(df)
    plot_danger_by_speed_limit(df)
    plot_danger_by_weather(df)
    plot_danger_by_road_type(df)
    plot_danger_by_vehicle_type(df)
    plot_danger_by_hour(df)
    plot_danger_by_light_condition(df)
    plot_casualties_distribution(df)
    print("=" * 70)
    print("[+] All 8 EDA charts successfully generated and saved to:", CHARTS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    run_all_eda()

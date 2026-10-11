"""
RoadSense AI: Interactive Terminal Prediction Demo
==================================================
Allows a user to describe a road collision scenario and receive:
  - Task A: Binary risk assessment (Dangerous vs Minor) with calibrated probability
  - Task B: 3-Class severity breakdown (Fatal / Serious / Slight)
  - Task C: Expected casualty count
  - Risk band classification (Low / Moderate / Elevated / High / Critical)
  - SHAP-based explanation of the top contributing factors

Usage:
  python predict.py              # Interactive mode
  python predict.py --batch FILE # Batch mode from CSV
"""

import os
import sys
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath("."))
from src.features import (
    ALL_MODEL_FEATURES,
    USER_VEHICLE_TYPES,
    USER_SPEED_LIMITS,
    USER_WEATHER_OPTIONS,
    USER_ROAD_TYPE_OPTIONS,
    map_user_vehicle_input,
)

warnings.filterwarnings("ignore")

MODELS_DIR = "models"

# ==============================================================================
# RISK BAND DEFINITIONS
# Derived from the operating_points_comparison.csv thresholds:
#   Top 10% threshold ≈ 0.387 → Critical
#   Top 20% threshold ≈ 0.324 → High  
#   Top 30% threshold ≈ 0.282 → Elevated
#   Model's F2 threshold ≈ 0.12 → Moderate (below this = Low)
# ==============================================================================
RISK_BANDS = [
    {"label": "CRITICAL",  "min_prob": 0.387, "color": "\033[91m", "emoji": "🔴", "desc": "Top 10% highest-risk — immediate review"},
    {"label": "HIGH",      "min_prob": 0.324, "color": "\033[31m", "emoji": "🟠", "desc": "Top 20% highest-risk — priority intervention"},
    {"label": "ELEVATED",  "min_prob": 0.282, "color": "\033[33m", "emoji": "🟡", "desc": "Top 30% highest-risk — heightened monitoring"},
    {"label": "MODERATE",  "min_prob": 0.120, "color": "\033[93m", "emoji": "🟢", "desc": "Above decision threshold — standard caution"},
    {"label": "LOW",       "min_prob": 0.000, "color": "\033[32m", "emoji": "⚪", "desc": "Below threshold — typical road conditions"},
]

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
WHITE = "\033[97m"
UNDERLINE = "\033[4m"


def get_risk_band(prob: float) -> dict:
    """Returns the risk band for a given calibrated probability."""
    for band in RISK_BANDS:
        if prob >= band["min_prob"]:
            return band
    return RISK_BANDS[-1]


# ==============================================================================
# MODEL LOADING
# ==============================================================================
def load_models():
    """Load all three task models from the models/ directory."""
    models = {}

    path_a = os.path.join(MODELS_DIR, "best_model_task_a.joblib")
    path_b = os.path.join(MODELS_DIR, "best_model_task_b.joblib")
    path_c = os.path.join(MODELS_DIR, "best_model_task_c.joblib")

    if os.path.exists(path_a):
        models["A"] = joblib.load(path_a)
    if os.path.exists(path_b):
        models["B"] = joblib.load(path_b)
    if os.path.exists(path_c):
        models["C"] = joblib.load(path_c)

    return models


# ==============================================================================
# PREDICTION ENGINE
# ==============================================================================
def predict_single(models: dict, scenario: dict) -> dict:
    """
    Runs all three task models on a single collision scenario.
    Returns calibrated probabilities, predictions, risk bands, and explanations.
    """
    # Build a single-row DataFrame with ALL_MODEL_FEATURES
    row = {}

    # Direct mappings
    row["speed_limit"] = scenario.get("speed_limit", 30)
    row["weather_conditions"] = scenario.get("weather_conditions", 1)
    row["road_type"] = scenario.get("road_type", 6)
    row["light_conditions"] = scenario.get("light_conditions", 1)
    row["road_surface_conditions"] = scenario.get("road_surface_conditions", 1)
    row["junction_detail"] = scenario.get("junction_detail", 0)
    row["urban_or_rural_area"] = scenario.get("urban_or_rural_area", 1)
    row["first_road_class"] = scenario.get("first_road_class", 3)
    row["junction_control"] = scenario.get("junction_control", -1)
    row["trunk_road_flag"] = scenario.get("trunk_road_flag", 2)
    row["carriageway_hazards"] = scenario.get("carriageway_hazards", 0)
    row["hour"] = scenario.get("hour", 12)
    row["day_of_week"] = scenario.get("day_of_week", 3)
    row["month"] = scenario.get("month", 6)

    # Vehicle features from user-facing mapping
    vehicle_info = map_user_vehicle_input(
        vehicle_type_name=scenario.get("vehicle_type", "Car"),
        age_of_vehicle=scenario.get("age_of_vehicle", None),
        driver_age_band=scenario.get("driver_age_band", None),
        number_of_vehicles=scenario.get("number_of_vehicles", 2),
    )
    row.update(vehicle_info)

    X = pd.DataFrame([row])[ALL_MODEL_FEATURES]
    results = {}

    # Task A: Binary
    if "A" in models:
        m = models["A"]
        raw_prob = m["pipeline"].predict_proba(X)[:, 1][0]
        cal_prob = float(m["calibrator"].predict_proba(np.array([raw_prob]))[0])
        threshold = m["optimal_threshold"]
        pred = int(cal_prob >= threshold)
        band = get_risk_band(cal_prob)

        results["task_a"] = {
            "prediction": "DANGEROUS" if pred == 1 else "MINOR",
            "probability": round(cal_prob, 4),
            "threshold": threshold,
            "risk_band": band,
            "model": f"{m['model_name']} ({m['imbalance_strategy']})",
        }

    # Task B: 3-Class
    if "B" in models:
        m = models["B"]
        raw_probs = m["pipeline"].predict_proba(X)[0]
        
        # Apply hierarchical threshold: Fatal first, then Serious
        tau_fatal = m["optimal_fatal_threshold"]
        tau_serious = m.get("optimal_serious_threshold", 0.45)

        # Calibrate fatal probability
        cal_fatal = float(m["fatal_calibrator"].predict_proba(np.array([raw_probs[0]]))[0])

        if cal_fatal >= tau_fatal:
            pred_class = 0
            pred_label = "FATAL"
        elif raw_probs[1] >= tau_serious:
            pred_class = 1
            pred_label = "SERIOUS"
        else:
            pred_class = 2
            pred_label = "SLIGHT"

        results["task_b"] = {
            "prediction": pred_label,
            "probabilities": {
                "Fatal": round(float(raw_probs[0]), 4),
                "Serious": round(float(raw_probs[1]), 4),
                "Slight": round(float(raw_probs[2]), 4),
            },
            "calibrated_fatal": round(cal_fatal, 4),
            "model": f"{m['model_name']} ({m['imbalance_strategy']})",
        }

    # Task C: Regression
    if "C" in models:
        m = models["C"]
        pred_count = float(m["pipeline"].predict(X)[0])
        pred_count = max(1.0, pred_count)  # Floor at 1 casualty

        results["task_c"] = {
            "predicted_casualties": round(pred_count, 2),
            "model": m["model_name"],
        }

    return results


# ==============================================================================
# INTERACTIVE TERMINAL UI
# ==============================================================================
def print_header():
    """Print the application header."""
    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}   🚗  RoadSense AI — Collision Risk Assessment Tool  🚗{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{DIM}   Predicts collision severity using UK DfT STATS19 models (2021-2023){RESET}")
    print(f"{DIM}   Models: LightGBM (Class-Weighted) | Platt-calibrated | F2-thresholded{RESET}")
    print()


def prompt_choice(prompt: str, options: dict, default=None) -> int:
    """Display a numbered menu and return the selected key."""
    print(f"\n  {BOLD}{prompt}{RESET}")
    for key, val in options.items():
        marker = " ←" if key == default else ""
        print(f"    [{key}] {val}{marker}")

    while True:
        raw = input(f"  {CYAN}Enter choice [{default}]: {RESET}").strip()
        if raw == "" and default is not None:
            return default
        try:
            choice = int(raw)
            if choice in options:
                return choice
        except ValueError:
            pass
        print(f"    {DIM}Invalid choice. Try again.{RESET}")


def prompt_number(prompt: str, default=None, min_val=None, max_val=None):
    """Prompt for a numeric input with validation."""
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"  {CYAN}{prompt}{suffix}: {RESET}").strip()
        if raw == "" and default is not None:
            return default
        if raw.lower() in ("skip", "none", ""):
            return None
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"    {DIM}Must be >= {min_val}{RESET}")
                continue
            if max_val is not None and val > max_val:
                print(f"    {DIM}Must be <= {max_val}{RESET}")
                continue
            return val
        except ValueError:
            print(f"    {DIM}Enter a number or 'skip'.{RESET}")


def prompt_vehicle_type() -> str:
    """Prompt user to select a vehicle type."""
    print(f"\n  {BOLD}Vehicle Type:{RESET}")
    for i, vt in enumerate(USER_VEHICLE_TYPES, 1):
        marker = " ←" if i == 1 else ""
        print(f"    [{i}] {vt}{marker}")

    while True:
        raw = input(f"  {CYAN}Enter choice [1]: {RESET}").strip()
        if raw == "":
            return USER_VEHICLE_TYPES[0]
        try:
            idx = int(raw)
            if 1 <= idx <= len(USER_VEHICLE_TYPES):
                return USER_VEHICLE_TYPES[idx - 1]
        except ValueError:
            pass
        print(f"    {DIM}Invalid choice. Try again.{RESET}")


def collect_scenario_interactive() -> dict:
    """Interactively collect a collision scenario from the user."""
    print(f"\n  {UNDERLINE}Describe the collision scenario:{RESET}")

    scenario = {}

    # Speed limit
    print(f"\n  {BOLD}Speed Limit (mph):{RESET}")
    for i, sl in enumerate(USER_SPEED_LIMITS, 1):
        marker = " ←" if sl == 30 else ""
        print(f"    [{i}] {sl} mph{marker}")
    while True:
        raw = input(f"  {CYAN}Enter choice [2]: {RESET}").strip()
        if raw == "":
            scenario["speed_limit"] = 30
            break
        try:
            idx = int(raw)
            if 1 <= idx <= len(USER_SPEED_LIMITS):
                scenario["speed_limit"] = USER_SPEED_LIMITS[idx - 1]
                break
        except ValueError:
            pass
        print(f"    {DIM}Invalid choice.{RESET}")

    # Weather
    scenario["weather_conditions"] = prompt_choice(
        "Weather Conditions:", USER_WEATHER_OPTIONS, default=1
    )

    # Road type
    scenario["road_type"] = prompt_choice(
        "Road Type:", USER_ROAD_TYPE_OPTIONS, default=6
    )

    # Light conditions
    light_opts = {1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit", 6: "Darkness - no lighting"}
    scenario["light_conditions"] = prompt_choice("Light Conditions:", light_opts, default=1)

    # Road surface
    surface_opts = {1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice", 5: "Flood over 3cm"}
    scenario["road_surface_conditions"] = prompt_choice("Road Surface:", surface_opts, default=1)

    # Urban/Rural
    area_opts = {1: "Urban", 2: "Rural"}
    scenario["urban_or_rural_area"] = prompt_choice("Urban or Rural:", area_opts, default=1)

    # Junction
    junction_opts = {0: "Not at junction", 1: "Roundabout", 2: "Mini-roundabout",
                     3: "T/staggered junction", 5: "Slip road", 6: "Crossroads",
                     7: "More than 4 arms", 8: "Private drive", 9: "Other junction"}
    scenario["junction_detail"] = prompt_choice("Junction Type:", junction_opts, default=0)

    # Vehicle type
    scenario["vehicle_type"] = prompt_vehicle_type()

    # Number of vehicles
    scenario["number_of_vehicles"] = int(prompt_number(
        "Number of vehicles involved", default=2, min_val=1, max_val=20
    ))

    # Time
    scenario["hour"] = int(prompt_number("Hour of day (0-23)", default=12, min_val=0, max_val=23))
    scenario["day_of_week"] = int(prompt_number("Day of week (1=Sun, 7=Sat)", default=3, min_val=1, max_val=7))
    scenario["month"] = int(prompt_number("Month (1-12)", default=6, min_val=1, max_val=12))

    # Optional: vehicle age and driver age band
    print(f"\n  {DIM}Optional (press Enter to skip):{RESET}")
    scenario["age_of_vehicle"] = prompt_number("Age of vehicle (years)", default=None, min_val=0, max_val=50)
    scenario["driver_age_band"] = prompt_number("Driver age band (1-11, DfT coding)", default=None, min_val=1, max_val=11)

    # Defaults for fields not collected
    scenario.setdefault("first_road_class", 3)  # A road
    scenario.setdefault("junction_control", -1 if scenario["junction_detail"] == 0 else 4)
    scenario.setdefault("trunk_road_flag", 2)  # Non-trunk
    scenario.setdefault("carriageway_hazards", 0)  # None

    return scenario


# ==============================================================================
# RESULTS DISPLAY
# ==============================================================================
def display_results(results: dict):
    """Display prediction results with rich formatting."""
    print(f"\n{'═' * 70}")
    print(f"{BOLD}{CYAN}   📊  PREDICTION RESULTS{RESET}")
    print(f"{'═' * 70}")

    # Task A
    if "task_a" in results:
        r = results["task_a"]
        band = r["risk_band"]
        pred_color = "\033[91m" if r["prediction"] == "DANGEROUS" else "\033[32m"

        print(f"\n  {BOLD}┌─ TASK A: Binary Severity Assessment{RESET}")
        print(f"  │")
        print(f"  │  Prediction:  {pred_color}{BOLD}{r['prediction']}{RESET}")
        print(f"  │  Probability: {BOLD}{r['probability']:.1%}{RESET} (threshold: {r['threshold']:.0%})")
        print(f"  │")
        print(f"  │  Risk Band:   {band['color']}{BOLD}{band['emoji']} {band['label']}{RESET}")
        print(f"  │               {DIM}{band['desc']}{RESET}")
        print(f"  │  Model:       {DIM}{r['model']}{RESET}")
        print(f"  └{'─' * 50}")

    # Task B
    if "task_b" in results:
        r = results["task_b"]
        severity_colors = {"FATAL": "\033[91m", "SERIOUS": "\033[33m", "SLIGHT": "\033[32m"}
        pred_color = severity_colors.get(r["prediction"], "")

        print(f"\n  {BOLD}┌─ TASK B: 3-Class Severity Breakdown{RESET}")
        print(f"  │")
        print(f"  │  Prediction:       {pred_color}{BOLD}{r['prediction']}{RESET}")
        print(f"  │")
        print(f"  │  Class Probabilities:")
        for cls, prob in r["probabilities"].items():
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  │    {cls:>8s}: {bar} {prob:.1%}")
        print(f"  │")
        print(f"  │  Calibrated Fatal:  {r['calibrated_fatal']:.2%}")
        print(f"  │  Model:             {DIM}{r['model']}{RESET}")
        print(f"  └{'─' * 50}")

    # Task C
    if "task_c" in results:
        r = results["task_c"]
        count = r["predicted_casualties"]
        count_display = f"{count:.1f}" if count != int(count) else str(int(count))

        print(f"\n  {BOLD}┌─ TASK C: Expected Casualty Count{RESET}")
        print(f"  │")
        print(f"  │  Predicted Casualties: {BOLD}{count_display}{RESET}")
        print(f"  │  Model:                {DIM}{r['model']}{RESET}")
        print(f"  └{'─' * 50}")

    print()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="RoadSense AI — Collision Risk Prediction")
    parser.add_argument("--batch", type=str, help="Path to CSV file for batch predictions")
    args = parser.parse_args()

    print_header()

    # Load models
    print(f"  {DIM}Loading models...{RESET}", end="", flush=True)
    models = load_models()
    loaded = ", ".join([f"Task {k}" for k in models.keys()])
    print(f"\r  ✓ Models loaded: {loaded}      ")

    if not models:
        print(f"\n  ❌ No models found in {MODELS_DIR}/. Run training first.")
        sys.exit(1)

    if args.batch:
        # Batch mode
        print(f"\n  Processing batch file: {args.batch}")
        batch_df = pd.read_csv(args.batch)
        all_results = []

        for i, row in batch_df.iterrows():
            scenario = row.to_dict()
            result = predict_single(models, scenario)
            flat = {"index": i}
            if "task_a" in result:
                flat["severity_binary"] = result["task_a"]["prediction"]
                flat["dangerous_prob"] = result["task_a"]["probability"]
                flat["risk_band"] = result["task_a"]["risk_band"]["label"]
            if "task_b" in result:
                flat["severity_3class"] = result["task_b"]["prediction"]
                flat["fatal_prob"] = result["task_b"]["probabilities"]["Fatal"]
            if "task_c" in result:
                flat["predicted_casualties"] = result["task_c"]["predicted_casualties"]
            all_results.append(flat)

        out_df = pd.DataFrame(all_results)
        out_path = args.batch.replace(".csv", "_predictions.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  ✓ Saved {len(out_df)} predictions to: {out_path}")
        return

    # Interactive mode
    while True:
        try:
            scenario = collect_scenario_interactive()
            results = predict_single(models, scenario)
            display_results(results)

            again = input(f"  {CYAN}Run another scenario? (y/n) [y]: {RESET}").strip().lower()
            if again == "n":
                break
        except KeyboardInterrupt:
            print(f"\n\n  {DIM}Exiting RoadSense AI. Stay safe on the roads! 🚗{RESET}\n")
            break

    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{DIM}  Thank you for using RoadSense AI.{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}\n")


if __name__ == "__main__":
    main()

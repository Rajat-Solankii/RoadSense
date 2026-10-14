"""
RoadSense AI: Indian Road Safety & Disaster Prevention Interactive Tool
========================================================================
Assesses collision severity and casualty impact tailored for Indian roads,
incorporating MoRTH empirical distributions, Indian climatic hazards,
and behavioral mindset factors (wrong-side driving, overloading, high-beam glare).

Features:
  - Task A: Binary Severity (Fatal/Grievous vs Minor)
  - Task B: 3-Class Granular Severity Breakdown (Fatal / Grievous / Minor)
  - Task C: Expected Casualty Count Regression
  - Disaster Risk Index (DRI) calculation (0 - 100%)
  - Preventive Action Directives & "Golden Hour" Emergency Protocol (112 / 1033)
  - Quick-load real-world Indian scenario presets (Yamuna Fog, Monsoon Ghat, etc.)

Usage:
  python predict_indian.py              # Interactive mode with presets
  python predict_indian.py --batch FILE # Batch CSV mode
"""

import os
import sys
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
    ROAD_CATEGORIES,
    ROAD_TYPES,
    WEATHER_CONDITIONS,
    ROAD_SURFACES,
    LIGHT_CONDITIONS,
    JUNCTION_TYPES,
    VEHICLE_TYPES,
    COLLISION_TYPES,
    SPEED_LIMITS_KMH,
    derive_most_vulnerable,
)
from src.disaster_prevention import DisasterPreventionEngine

MODELS_DIR = "models"

# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
WHITE = "\033[97m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
ORANGE = "\033[33m"
RED = "\033[91m"
MAGENTA = "\033[95m"
UNDERLINE = "\033[4m"


def load_models() -> dict:
    """Loads all three trained models."""
    models = {}
    path_a = os.path.join(MODELS_DIR, "model_task_a.joblib")
    if not os.path.exists(path_a):
        path_a = os.path.join(MODELS_DIR, "indian_model_task_a.joblib")

    path_b = os.path.join(MODELS_DIR, "model_task_b.joblib")
    if not os.path.exists(path_b):
        path_b = os.path.join(MODELS_DIR, "indian_model_task_b.joblib")

    path_c = os.path.join(MODELS_DIR, "model_task_c.joblib")
    if not os.path.exists(path_c):
        path_c = os.path.join(MODELS_DIR, "indian_model_task_c.joblib")

    if os.path.exists(path_a):
        models["A"] = joblib.load(path_a)
    if os.path.exists(path_b):
        models["B"] = joblib.load(path_b)
    if os.path.exists(path_c):
        models["C"] = joblib.load(path_c)

    return models


# Alias for backward compatibility
load_indian_models = load_models


def predict_scenario(models: dict, scenario: dict) -> dict:
    """Executes multi-task inference and disaster assessment on a scenario."""
    # Ensure most vulnerable participant is derived
    if "most_vulnerable_participant" not in scenario:
        scenario["most_vulnerable_participant"] = derive_most_vulnerable(
            scenario.get("primary_vehicle_type", "Car / Hatchback / Sedan"),
            scenario.get("secondary_vehicle_type", "None"),
        )

    # Defaults for categorical features
    scenario.setdefault("road_category", "National Highway (NH)")
    scenario.setdefault("road_type", "Divided Multi-Lane (4/6-Lane)")
    scenario.setdefault("weather_condition", "Clear / Sunny")
    scenario.setdefault("road_surface", "Dry Asphalt / Concrete")
    scenario.setdefault("light_condition", "Daylight")
    scenario.setdefault("junction_type", "Straight Road (Not at Junction)")
    scenario.setdefault("primary_vehicle_type", "Car / Hatchback / Sedan")
    scenario.setdefault("secondary_vehicle_type", "None")
    scenario.setdefault("collision_type", "Rear-End / Chain Pileup")
    scenario.setdefault("speed_limit_kmh", 80)

    # Defaults for optional numeric features
    scenario.setdefault("speed_delta_kmh", 0)
    scenario.setdefault("hour", 14)
    scenario.setdefault("day_of_week", 2)
    scenario.setdefault("month", 7)
    scenario.setdefault("number_of_vehicles", 2)
    scenario.setdefault("occupants_count", 3)

    # Behavioral flags
    for k in ["is_wrong_side", "is_overloaded", "safety_gear_violation",
              "high_beam_hazard", "stray_animal_hazard", "drowsy_window",
              "pothole_or_speedbreaker"]:
        scenario.setdefault(k, 0)

    # DataFrame formatting
    X_df = pd.DataFrame([scenario])[ALL_MODEL_FEATURES]
    preprocessor = models["A"]["preprocessor"]
    X_trans = preprocessor.transform(X_df)

    results = {}

    # Task A: Binary
    if "A" in models:
        m = models["A"]
        prob_dangerous = float(m["classifier"].predict_proba(X_trans)[:, 1][0])
        thresh = m["optimal_threshold"]
        pred_label = "DANGEROUS (FATAL/GRIEVOUS)" if prob_dangerous >= thresh else "MINOR"

        if prob_dangerous >= 0.70:
            band = {"label": "CRITICAL", "color": RED, "emoji": "🔴", "desc": "High probability of severe trauma"}
        elif prob_dangerous >= 0.45:
            band = {"label": "HIGH", "color": ORANGE, "emoji": "🟠", "desc": "Substantial injury probability"}
        elif prob_dangerous >= thresh:
            band = {"label": "ELEVATED", "color": YELLOW, "emoji": "🟡", "desc": "Above threshold — heightened risk"}
        else:
            band = {"label": "LOW", "color": GREEN, "emoji": "🟢", "desc": "Low probability of severe injury"}

        results["task_a"] = {
            "prediction": pred_label,
            "probability": round(prob_dangerous, 4),
            "threshold": round(thresh, 3),
            "risk_band": band,
            "model": m["model_name"],
        }

    # Task B: 3-Class
    if "B" in models:
        m = models["B"]
        probs_b = m["classifier"].predict_proba(X_trans)[0]
        p_fatal = float(probs_b[0])
        p_grievous = float(probs_b[1])
        p_minor = float(probs_b[2])

        tau_fatal = m["optimal_fatal_threshold"]
        tau_grievous = m.get("optimal_grievous_threshold", 0.40)

        if p_fatal >= tau_fatal:
            pred_b = "FATAL"
        elif p_grievous >= tau_grievous:
            pred_b = "GRIEVOUS INJURY"
        else:
            pred_b = "MINOR / PROPERTY DAMAGE"

        results["task_b"] = {
            "prediction": pred_b,
            "probabilities": {
                "Fatal": round(p_fatal, 4),
                "Grievous": round(p_grievous, 4),
                "Minor": round(p_minor, 4),
            },
            "fatal_threshold": round(tau_fatal, 3),
            "model": m["model_name"],
        }

    # Task C: Casualty Count
    if "C" in models:
        m = models["C"]
        z_pred = float(m["regressor"].predict(X_trans)[0])
        pred_count = max(1.0, z_pred + 1.0)
        results["task_c"] = {
            "predicted_casualties": round(pred_count, 1),
            "model": m["model_name"],
        }

    # Disaster Prevention Assessment
    b_probs = results.get("task_b", {}).get("probabilities", {"Fatal": 0.1, "Grievous": 0.3})
    disaster_assessment = DisasterPreventionEngine.assess_disaster_risk(scenario, b_probs)
    results["disaster"] = disaster_assessment

    return results


# Alias for backward compatibility
predict_indian_scenario = predict_scenario


def print_banner():
    print(f"\n{BOLD}{CYAN}{'═' * 76}{RESET}")
    print(f"{BOLD}{CYAN}   🇮🇳  RoadSense AI — Indian Road Safety & Disaster Prevention Tool  🇮🇳{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 76}{RESET}")
    print(f"{DIM}   Trained on Indian MoRTH & iRAD Standards | Weather & Behavioral Risk Engine{RESET}")
    print(f"{DIM}   Incorporating: Expressways, Ghats, Fog, Monsoon, Wrong-Side Driving & Overloading{RESET}")
    print()


def display_results(scenario: dict, results: dict):
    print(f"\n{'═' * 76}")
    print(f"{BOLD}{CYAN}   📊  MULTI-TASK COLLISION & DISASTER ASSESSMENT RESULTS{RESET}")
    print(f"{'═' * 76}")

    # Summary of input scenario
    actual_spd = scenario.get("speed_limit_kmh", 80) + scenario.get("speed_delta_kmh", 0)
    print(f"  {DIM}Scenario: {scenario.get('road_category')} | {scenario.get('weather_condition')}")
    print(f"  Vehicles: {scenario.get('primary_vehicle_type')} vs {scenario.get('secondary_vehicle_type')} | Speed: {actual_spd} km/h{RESET}")

    # ==========================================================================
    # 🎯 DIRECT PROBABLE ANSWER HIGHLIGHT
    # ==========================================================================
    probable_severity = results.get("task_b", {}).get("prediction", "UNKNOWN")
    prob_dangerous = results.get("task_a", {}).get("probability", 0.0)
    fatal_prob = results.get("task_b", {}).get("probabilities", {}).get("Fatal", 0.0)
    grievous_prob = results.get("task_b", {}).get("probabilities", {}).get("Grievous", 0.0)
    victims = results.get("task_c", {}).get("predicted_casualties", 1.0)
    dri = results.get("disaster", {}).get("disaster_risk_index", 50.0)
    dri_band = results.get("disaster", {}).get("risk_band", "MODERATE")

    color_choice = RED if "FATAL" in probable_severity or dri >= 70 else (ORANGE if "GRIEVOUS" in probable_severity or dri >= 50 else GREEN)

    print(f"\n  {BOLD}{WHITE}╔{'═' * 72}╗{RESET}")
    print(f"  {BOLD}{WHITE}║{color_choice}  🎯 PROBABLE PREDICTION SUMMARY                                        {WHITE}║{RESET}")
    print(f"  {BOLD}{WHITE}╠{'═' * 72}╣{RESET}")
    print(f"  {BOLD}{WHITE}║{RESET}  • {BOLD}Probable Collision Outcome:{RESET} {color_choice}{BOLD}{probable_severity}{RESET}")
    print(f"  {BOLD}{WHITE}║{RESET}    (Fatal: {fatal_prob:.1%} | Grievous Injury: {grievous_prob:.1%} | Severe Injury Risk: {prob_dangerous:.1%})")
    print(f"  {BOLD}{WHITE}║{RESET}  • {BOLD}Forecasted Casualties:{RESET}      {BOLD}{victims:.1f} victims{RESET} expected per incident")
    print(f"  {BOLD}{WHITE}║{RESET}  • {BOLD}Disaster Risk Escalation:{RESET}   {color_choice}{BOLD}{dri:.1f}% ({dri_band}){RESET}")
    if results.get("disaster", {}).get("mitigation_directives"):
        top_directive = results["disaster"]["mitigation_directives"][0]
        # Clean leading emoji for clean box formatting
        print(f"  {BOLD}{WHITE}║{RESET}  • {BOLD}Top Action to Prevent Disaster:{RESET}")
        print(f"  {BOLD}{WHITE}║{RESET}    👉 {top_directive}")
    print(f"  {BOLD}{WHITE}║{RESET}  • {BOLD}Post-Crash Life-Saving Rule:{RESET}")
    print(f"  {BOLD}{WHITE}║{RESET}    👉 DO NOT stand on carriageway! Evacuate IMMEDIATELY behind crash barrier.")
    print(f"  {BOLD}{WHITE}╚{'═' * 72}╝{RESET}")

    # Task A
    if "task_a" in results:
        r = results["task_a"]
        band = r["risk_band"]
        pred_color = RED if "DANGEROUS" in r["prediction"] else GREEN
        print(f"\n  {BOLD}┌─ TASK A: Binary Severity Assessment{RESET}")
        print(f"  │  Prediction:  {pred_color}{BOLD}{r['prediction']}{RESET}")
        print(f"  │  Probability: {BOLD}{r['probability']:.1%}{RESET} (Decision threshold: {r['threshold']:.1%})")
        print(f"  │  Risk Band:   {band['color']}{BOLD}{band['emoji']} {band['label']}{RESET} — {DIM}{band['desc']}{RESET}")
        print(f"  └{'─' * 60}")

    # Task B
    if "task_b" in results:
        r = results["task_b"]
        p_dict = r["probabilities"]
        color_map = {"FATAL": RED, "GRIEVOUS INJURY": ORANGE, "MINOR / PROPERTY DAMAGE": GREEN}
        print(f"\n  {BOLD}┌─ TASK B: 3-Class Severity Breakdown (MoRTH Classification){RESET}")
        print(f"  │  Primary Prediction: {color_map.get(r['prediction'], WHITE)}{BOLD}{r['prediction']}{RESET}")
        print(f"  │  Probability Distribution:")
        for cls, prob in p_dict.items():
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            c = RED if cls == "Fatal" else (ORANGE if cls == "Grievous" else GREEN)
            print(f"  │    {cls:>10s}: {c}{bar}{RESET} {prob:.1%}")
        print(f"  └{'─' * 60}")

    # Task C
    if "task_c" in results:
        r = results["task_c"]
        cnt = r["predicted_casualties"]
        cnt_str = f"{cnt:.1f}" if cnt != int(cnt) else str(int(cnt))
        print(f"\n  {BOLD}┌─ TASK C: Expected Casualty Count{RESET}")
        print(f"  │  Forecasted Victims: {BOLD}{MAGENTA}{cnt_str} people{RESET} per incident")
        print(f"  └{'─' * 60}")

    # Disaster Risk Assessment
    if "disaster" in results:
        d = results["disaster"]
        dri = d["disaster_risk_index"]
        dri_color = RED if dri >= 70 else (ORANGE if dri >= 50 else (YELLOW if dri >= 30 else GREEN))

        print(f"\n  {BOLD}{MAGENTA}╔{'═' * 74}╗{RESET}")
        print(f"  {BOLD}{MAGENTA}║  ⚠️  SYSTEMIC DISASTER ASSESSMENT & ACTIONABLE MITIGATION ENGINE         ║{RESET}")
        print(f"  {BOLD}{MAGENTA}╚{'═' * 74}╝{RESET}")
        print(f"  │")
        print(f"  │  Disaster Risk Index (DRI): {dri_color}{BOLD}{d['risk_emoji']} {dri:.1f}% — {d['risk_band']}{RESET}")
        print(f"  │  {DIM}{d['urgency_assessment']}{RESET}")
        print(f"  │")

        if d["risk_multipliers"]:
            print(f"  │  {BOLD}Active Indian Risk Catalysts:{RESET}")
            for m in d["risk_multipliers"]:
                print(f"  │    • {BOLD}{m['factor']}{RESET} ({m['impact']})")
                print(f"  │      {DIM}{m['reason']}{RESET}")
            print(f"  │")

        if d["escalation_threats"]:
            print(f"  │  {BOLD}{RED}Potential Catastrophic Escalations:{RESET}")
            for threat in d["escalation_threats"]:
                print(f"  │    🚨 [{threat['severity']}] {threat['title']}")
                print(f"  │       {threat['description']}")
            print(f"  │")

        if d["mitigation_directives"]:
            print(f"  │  {BOLD}{CYAN}Directives to Prevent a Big Disaster (Pre-Trip / In-Transit):{RESET}")
            for directive in d["mitigation_directives"]:
                print(f"  │    {directive}")
            print(f"  │")

        print(f"  │  {BOLD}{YELLOW}Post-Crash Emergency Containment (Preventing Secondary Pileups):{RESET}")
        for protocol in d["secondary_prevention_protocols"]:
            print(f"  │    {protocol}")
        print(f"  └{'─' * 74}")

    print()


def get_preset_scenarios() -> list:
    """Returns curated realistic Indian road collision scenarios."""
    return [
        {
            "name": "Yamuna Expressway: Dense Winter Smog Pileup (110 km/h, tailgating)",
            "scenario": {
                "road_category": "Expressway",
                "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather_condition": "Dense Winter Fog / Smog",
                "road_surface": "Dry Asphalt / Concrete",
                "light_condition": "Dusk / Dawn (Twilight)",
                "junction_type": "Straight Road (Not at Junction)",
                "primary_vehicle_type": "Car / Hatchback / Sedan",
                "secondary_vehicle_type": "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)",
                "collision_type": "Rear-End / Chain Pileup",
                "speed_limit_kmh": 120,
                "speed_delta_kmh": -10,  # Driving at 110 km/h in fog!
                "hour": 6,
                "day_of_week": 1,
                "month": 1,
                "number_of_vehicles": 5,
                "occupants_count": 6,
                "is_wrong_side": 0,
                "is_overloaded": 0,
                "safety_gear_violation": 1,
                "high_beam_hazard": 0,
                "stray_animal_hazard": 0,
                "drowsy_window": 0,
                "pothole_or_speedbreaker": 0,
            }
        },
        {
            "name": "Western Ghats: Heavy Monsoon Hairpin Descent (Two-Wheeler vs Bus)",
            "scenario": {
                "road_category": "Ghat / Hill Terrain Road",
                "road_type": "Undivided 2-Lane",
                "weather_condition": "Heavy Monsoon / Waterlogged",
                "road_surface": "Wet / Slippery",
                "light_condition": "Daylight",
                "junction_type": "Blind Curve / Ghat Hairpin Bend",
                "primary_vehicle_type": "Two-Wheeler (Motorcycle / Scooter)",
                "secondary_vehicle_type": "Bus (State Transport / Private / School)",
                "collision_type": "Head-On Collision",
                "speed_limit_kmh": 40,
                "speed_delta_kmh": 10,
                "hour": 15,
                "day_of_week": 5,
                "month": 8,
                "number_of_vehicles": 2,
                "occupants_count": 32,
                "is_wrong_side": 1, # Overtaking on blind turn
                "is_overloaded": 1, # Triple riding
                "safety_gear_violation": 1, # No helmets
                "high_beam_hazard": 0,
                "stray_animal_hazard": 0,
                "drowsy_window": 0,
                "pothole_or_speedbreaker": 1,
            }
        },
        {
            "name": "Delhi-Jaipur NH-48: Midnight Wrong-Side Tractor Trolley vs SUV",
            "scenario": {
                "road_category": "National Highway (NH)",
                "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather_condition": "Clear / Sunny",
                "road_surface": "Dry Asphalt / Concrete",
                "light_condition": "Night - Pitch Dark (No Streetlights)",
                "junction_type": "T-Junction / Unsignalized Median Cut",
                "primary_vehicle_type": "SUV / MUV / Compact SUV",
                "secondary_vehicle_type": "Tractor with Trolley (Sugarcane/Construction)",
                "collision_type": "Head-On Collision",
                "speed_limit_kmh": 100,
                "speed_delta_kmh": 15, # SUV speeding at 115 km/h
                "hour": 3,
                "day_of_week": 6,
                "month": 11,
                "number_of_vehicles": 2,
                "occupants_count": 8,
                "is_wrong_side": 1, # Tractor moving wrong side
                "is_overloaded": 1, # Sugarcane unlit trolley
                "safety_gear_violation": 0,
                "high_beam_hazard": 1,
                "stray_animal_hazard": 0,
                "drowsy_window": 1, # 03:00 AM
                "pothole_or_speedbreaker": 0,
            }
        },
        {
            "name": "Bangalore Outer Ring Road: Overcrowded Auto-Rickshaw vs LCV",
            "scenario": {
                "road_category": "Urban Arterial / Ring Road",
                "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather_condition": "Clear / Sunny",
                "road_surface": "Dry Asphalt / Concrete",
                "light_condition": "Daylight",
                "junction_type": "Flyover Ramp / Blind Slip Road",
                "primary_vehicle_type": "Three-Wheeler (Auto-Rickshaw / E-Rickshaw)",
                "secondary_vehicle_type": "Light Commercial Vehicle (LCV / Tempo / Tata Ace)",
                "collision_type": "Right Angle / T-Bone at Median Cut",
                "speed_limit_kmh": 50,
                "speed_delta_kmh": 10,
                "hour": 9,
                "day_of_week": 2,
                "month": 4,
                "number_of_vehicles": 2,
                "occupants_count": 7, # 6 passengers in auto
                "is_wrong_side": 0,
                "is_overloaded": 1,
                "safety_gear_violation": 1,
                "high_beam_hazard": 0,
                "stray_animal_hazard": 0,
                "drowsy_window": 0,
                "pothole_or_speedbreaker": 1,
            }
        },
    ]


def prompt_choice(title: str, items: list, default_idx: int = 0) -> str:
    print(f"\n  {BOLD}{title}{RESET}")
    for i, item in enumerate(items, 1):
        marker = " ←" if (i - 1) == default_idx else ""
        print(f"    [{i}] {item}{marker}")
    while True:
        raw = input(f"  {CYAN}Enter choice [{default_idx + 1}]: {RESET}").strip()
        if raw == "":
            return items[default_idx]
        try:
            choice = int(raw)
            if 1 <= choice <= len(items):
                return items[choice - 1]
        except ValueError:
            pass
        print(f"    {DIM}Invalid choice. Try again.{RESET}")


def prompt_yes_no(prompt: str, default: bool = False) -> int:
    def_str = "y" if default else "n"
    raw = input(f"  {CYAN}{prompt} (y/n) [{def_str}]: {RESET}").strip().lower()
    if raw == "":
        return 1 if default else 0
    return 1 if raw.startswith("y") else 0


def collect_interactive_scenario() -> dict:
    """Interactively prompts user for scenario features."""
    print(f"\n  {UNDERLINE}Describe the Indian Road Scenario:{RESET}")
    scenario = {}

    scenario["road_category"] = prompt_choice("Road Category / Hierarchy:", ROAD_CATEGORIES, default_idx=1)
    scenario["road_type"] = prompt_choice("Road Type / Carriageway:", ROAD_TYPES, default_idx=0)
    scenario["weather_condition"] = prompt_choice("Weather / Climatic Condition:", WEATHER_CONDITIONS, default_idx=0)
    scenario["road_surface"] = prompt_choice("Road Surface Condition:", ROAD_SURFACES, default_idx=0)
    scenario["light_condition"] = prompt_choice("Lighting Conditions:", LIGHT_CONDITIONS, default_idx=0)
    scenario["junction_type"] = prompt_choice("Junction / Geometry:", JUNCTION_TYPES, default_idx=0)

    # Speed limit
    print(f"\n  {BOLD}Speed Limit (km/h):{RESET}")
    for i, spd in enumerate(SPEED_LIMITS_KMH, 1):
        marker = " ←" if spd == 80 else ""
        print(f"    [{i}] {spd} km/h{marker}")
    while True:
        raw = input(f"  {CYAN}Enter choice [6]: {RESET}").strip()
        if raw == "":
            scenario["speed_limit_kmh"] = 80
            break
        try:
            idx = int(raw)
            if 1 <= idx <= len(SPEED_LIMITS_KMH):
                scenario["speed_limit_kmh"] = SPEED_LIMITS_KMH[idx - 1]
                break
        except ValueError:
            pass

    actual_raw = input(f"  {CYAN}Estimated Travel Speed in km/h (optional, default = speed limit): {RESET}").strip()
    if actual_raw.isdigit():
        actual_val = int(actual_raw)
        scenario["speed_delta_kmh"] = max(-30, actual_val - scenario["speed_limit_kmh"])
    else:
        scenario["speed_delta_kmh"] = 0

    # Vehicles
    scenario["primary_vehicle_type"] = prompt_choice("Primary Vehicle Type:", VEHICLE_TYPES, default_idx=0)
    sec_choices = VEHICLE_TYPES + ["None"]
    scenario["secondary_vehicle_type"] = prompt_choice("Secondary Vehicle Type (or None for solo crash):", sec_choices, default_idx=2)
    scenario["collision_type"] = prompt_choice("Collision Type / Geometry:", COLLISION_TYPES, default_idx=0)

    # Behavioral questions
    print(f"\n  {BOLD}Behavioral & Indian Road Mindset Risk Factors:{RESET}")
    scenario["is_wrong_side"] = prompt_yes_no("Wrong-side driving / Contraflow lane violation?", default=False)
    scenario["is_overloaded"] = prompt_yes_no("Overloading (triple riding on 2W / overloaded auto or truck)?", default=False)
    scenario["safety_gear_violation"] = prompt_yes_no("Safety gear non-compliance (no helmet or no seatbelt)?", default=True)
    scenario["high_beam_hazard"] = prompt_yes_no("Blinding high-beam glare on undivided road?", default=False)
    scenario["stray_animal_hazard"] = prompt_yes_no("Stray cattle or animal hazard on carriageway?", default=False)
    scenario["pothole_or_speedbreaker"] = prompt_yes_no("Unmarked speed bump, ditch or pothole crater present?", default=False)

    hour_raw = input(f"  {CYAN}Hour of the day (0-23) [14]: {RESET}").strip()
    scenario["hour"] = int(hour_raw) if hour_raw.isdigit() and 0 <= int(hour_raw) <= 23 else 14
    scenario["drowsy_window"] = 1 if 1 <= scenario["hour"] <= 5 else 0

    occ_raw = input(f"  {CYAN}Total occupants / people involved [2]: {RESET}").strip()
    scenario["occupants_count"] = int(occ_raw) if occ_raw.isdigit() and int(occ_raw) >= 1 else 2
    scenario["number_of_vehicles"] = 1 if scenario["secondary_vehicle_type"] == "None" else 2

    return scenario


def main():
    parser = argparse.ArgumentParser(description="RoadSense AI — Indian Road Safety & Disaster Mitigation")
    parser.add_argument("--batch", type=str, help="Path to batch CSV file")
    parser.add_argument("--quick", action="store_true", help="Run immediate prediction without menu")
    parser.add_argument("--preset", type=int, choices=[1, 2, 3, 4], help="Run a specific preset directly (1-4)")
    parser.add_argument("--road", type=str, default="National Highway (NH)", help="Road type/hierarchy")
    parser.add_argument("--weather", type=str, default="Clear / Sunny", help="Weather condition")
    parser.add_argument("--speed", type=int, default=80, help="Speed limit in km/h")
    parser.add_argument("--actual-speed", type=int, default=None, help="Actual travel speed in km/h")
    parser.add_argument("--vehicle", type=str, default="Two-Wheeler (Motorcycle / Scooter)", help="Primary vehicle")
    parser.add_argument("--other-vehicle", type=str, default="Car / Hatchback / Sedan", help="Secondary vehicle or None")
    parser.add_argument("--wrong-side", action="store_true", help="Wrong-side/contraflow driving")
    parser.add_argument("--overloaded", action="store_true", help="Overloaded vehicle / triple riding")
    parser.add_argument("--no-helmet", action="store_true", help="No helmet or unstrapped helmet")
    parser.add_argument("--no-seatbelt", action="store_true", help="No seatbelt worn")
    parser.add_argument("--high-beam", action="store_true", help="High-beam glare on undivided road")
    args = parser.parse_args()

    print_banner()

    print(f"  {DIM}Loading Indian Road Safety models...{RESET}", end="", flush=True)
    models = load_indian_models()
    loaded_tasks = ", ".join([f"Task {k}" for k in models.keys()])
    print(f"\r  ✓ Indian Models Loaded: {loaded_tasks}          \n")

    if not models:
        print(f"  ❌ No Indian models found in {MODELS_DIR}/. Run 'python -m src.indian_train' first.")
        sys.exit(1)

    # Direct preset or quick CLI flags
    presets = get_preset_scenarios()

    if args.preset:
        selected_scenario = presets[args.preset - 1]["scenario"]
        res = predict_indian_scenario(models, selected_scenario)
        display_results(selected_scenario, res)
        return

    # Check if user specified any custom flag or requested quick prediction
    custom_flag_set = any([
        args.quick,
        args.actual_speed is not None,
        args.wrong_side,
        args.overloaded,
        args.no_helmet,
        args.no_seatbelt,
        args.high_beam,
        args.road != "National Highway (NH)",
        args.weather != "Clear / Sunny",
        args.speed != 80,
        args.vehicle != "Two-Wheeler (Motorcycle / Scooter)",
        args.other-vehicle != "Car / Hatchback / Sedan" if hasattr(args, "other-vehicle") else False,
    ])

    if custom_flag_set:
        actual_spd = args.actual_speed if args.actual_speed is not None else args.speed
        speed_delta = actual_spd - args.speed

        # Build custom scenario from flags
        scenario = {
            "road_category": args.road,
            "road_type": "Divided Multi-Lane (4/6-Lane)" if "Expressway" in args.road or "Highway" in args.road else "Undivided 2-Lane",
            "weather_condition": args.weather,
            "road_surface": "Waterlogged / Potholed" if "Monsoon" in args.weather or "Rain" in args.weather else "Dry Asphalt / Concrete",
            "light_condition": "Daylight",
            "junction_type": "Straight Road (Not at Junction)",
            "primary_vehicle_type": args.vehicle,
            "secondary_vehicle_type": args.other_vehicle,
            "collision_type": "Head-On Collision" if args.wrong_side else "Rear-End / Chain Pileup",
            "speed_limit_kmh": args.speed,
            "speed_delta_kmh": speed_delta,
            "hour": 14,
            "day_of_week": 3,
            "month": 7,
            "number_of_vehicles": 1 if args.other_vehicle == "None" else 2,
            "occupants_count": 3 if args.overloaded else 2,
            "is_wrong_side": 1 if args.wrong_side else 0,
            "is_overloaded": 1 if args.overloaded else 0,
            "safety_gear_violation": 1 if (args.no_helmet or args.no_seatbelt) else 0,
            "high_beam_hazard": 1 if args.high_beam else 0,
            "stray_animal_hazard": 0,
            "drowsy_window": 0,
            "pothole_or_speedbreaker": 0,
        }
        res = predict_indian_scenario(models, scenario)
        display_results(scenario, res)
        return

    if args.batch:
        print(f"  Processing batch file: {args.batch}")
        b_df = pd.read_csv(args.batch)
        results_list = []
        for i, row in b_df.iterrows():
            sc = row.to_dict()
            res = predict_indian_scenario(models, sc)
            flat = {"index": i}
            if "task_a" in res:
                flat["severity_binary"] = res["task_a"]["prediction"]
                flat["dangerous_prob"] = res["task_a"]["probability"]
                flat["risk_band"] = res["task_a"]["risk_band"]["label"]
            if "task_b" in res:
                flat["severity_3class"] = res["task_b"]["prediction"]
                flat["fatal_prob"] = res["task_b"]["probabilities"]["Fatal"]
                flat["grievous_prob"] = res["task_b"]["probabilities"]["Grievous"]
            if "task_c" in res:
                flat["predicted_casualties"] = res["task_c"]["predicted_casualties"]
            if "disaster" in res:
                flat["disaster_risk_index"] = res["disaster"]["disaster_risk_index"]
                flat["disaster_risk_band"] = res["disaster"]["risk_band"]
            results_list.append(flat)

        out_path = args.batch.replace(".csv", "_indian_predictions.csv")
        pd.DataFrame(results_list).to_csv(out_path, index=False)
        print(f"  ✓ Saved batch predictions to: {out_path}")
        return

    # Interactive menu
    presets = get_preset_scenarios()

    while True:
        try:
            print(f"  {BOLD}Select an Option:{RESET}")
            for i, p in enumerate(presets, 1):
                print(f"    [{i}] Preset: {p['name']}")
            print(f"    [{len(presets) + 1}] Custom Interactive Scenario (Input your own parameters)")
            print(f"    [{len(presets) + 2}] Exit")

            choice_raw = input(f"\n  {CYAN}Enter choice [1]: {RESET}").strip()
            if choice_raw == "":
                choice = 1
            else:
                try:
                    choice = int(choice_raw)
                except ValueError:
                    choice = 1

            if choice == len(presets) + 2:
                break
            elif 1 <= choice <= len(presets):
                selected_scenario = presets[choice - 1]["scenario"]
            else:
                selected_scenario = collect_interactive_scenario()

            res = predict_indian_scenario(models, selected_scenario)
            display_results(selected_scenario, res)

            again = input(f"  {CYAN}Evaluate another scenario? (y/n) [y]: {RESET}").strip().lower()
            if again == "n":
                break
        except KeyboardInterrupt:
            break

    print(f"\n{BOLD}{CYAN}{'═' * 76}{RESET}")
    print(f"  {BOLD}Stay Alert, Stay Safe — Drive Defensively on Indian Roads! 🚗 🏍️ 🇮🇳{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 76}{RESET}\n")


if __name__ == "__main__":
    main()

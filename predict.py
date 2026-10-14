"""
RoadSense: Machine Learning Collision Severity & Casualty Predictor
===================================================================
A clean, production-grade machine learning prediction tool for road collision
severity and expected casualties.

Predicts:
  - 3-Class Severity: Fatal, Grievous Injury, Minor / Property Damage (Task B)
  - Binary Severity: Dangerous (Fatal/Grievous) vs Minor (Task A)
  - Expected Casualty Count: Predicted victims per incident (Task C)

Usage:
  python predict.py --quick              # Instant prediction on default scenario
  python predict.py --preset 1           # Run preset benchmark scenario
  python predict.py --road "Expressway"  # Custom road & vehicle parameters
  python predict.py --batch file.csv     # Bulk CSV prediction
  python predict.py                      # Interactive console prompt
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

MODELS_DIR = "models"

# ANSI Colors for terminal output
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


def load_models(models_dir: str = MODELS_DIR) -> dict:
    """Loads trained LightGBM models for Tasks A, B, and C."""
    models = {}
    path_a = os.path.join(models_dir, "model_task_a.joblib")
    path_b = os.path.join(models_dir, "model_task_b.joblib")
    path_c = os.path.join(models_dir, "model_task_c.joblib")

    if os.path.exists(path_a):
        models["A"] = joblib.load(path_a)
    if os.path.exists(path_b):
        models["B"] = joblib.load(path_b)
    if os.path.exists(path_c):
        models["C"] = joblib.load(path_c)

    return models


def predict_scenario(models: dict, scenario: dict) -> dict:
    """
    Executes machine learning inference across Tasks A, B, and C for an incident scenario.
    """
    sc = dict(scenario)

    # Derive most vulnerable participant if missing
    if "most_vulnerable_participant" not in sc:
        sc["most_vulnerable_participant"] = derive_most_vulnerable(
            sc.get("primary_vehicle_type", "Car / Hatchback / Sedan"),
            sc.get("secondary_vehicle_type", "None"),
        )

    # Defaults for categorical features
    sc.setdefault("road_category", "National Highway (NH)")
    sc.setdefault("road_type", "Divided Multi-Lane (4/6-Lane)")
    sc.setdefault("weather_condition", "Clear / Sunny")
    sc.setdefault("road_surface", "Dry Asphalt / Concrete")
    sc.setdefault("light_condition", "Daylight")
    sc.setdefault("junction_type", "Straight Road (Not at Junction)")
    sc.setdefault("primary_vehicle_type", "Car / Hatchback / Sedan")
    sc.setdefault("secondary_vehicle_type", "None")
    sc.setdefault("collision_type", "Rear-End / Chain Pileup")
    sc.setdefault("speed_limit_kmh", 80)

    # Defaults for numeric features
    sc.setdefault("speed_delta_kmh", 0)
    sc.setdefault("hour", 14)
    sc.setdefault("day_of_week", 2)
    sc.setdefault("month", 7)
    sc.setdefault("number_of_vehicles", 2)
    sc.setdefault("occupants_count", 3)

    # Defaults for behavioral indicators
    for k in ["is_wrong_side", "is_overloaded", "safety_gear_violation",
              "high_beam_hazard", "stray_animal_hazard", "drowsy_window",
              "pothole_or_speedbreaker"]:
        sc.setdefault(k, 0)

    # Build feature row and apply preprocessor
    X_df = pd.DataFrame([sc])[ALL_MODEL_FEATURES]
    preprocessor = models["A"]["preprocessor"]
    X_trans = preprocessor.transform(X_df)

    results = {}

    # Task A: Binary Severity (Dangerous vs Minor)
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
            "model": m.get("model_name", "LightGBM Binary"),
        }

    # Task B: 3-Class Severity Breakdown (Fatal / Grievous / Minor)
    if "B" in models:
        m = models["B"]
        probs_b = m["classifier"].predict_proba(X_trans)[0]
        p_fatal = float(probs_b[0])
        p_grievous = float(probs_b[1])
        p_minor = float(probs_b[2])

        tau_fatal = m.get("optimal_fatal_threshold", 0.35)

        if p_fatal >= 0.42 or (p_fatal >= tau_fatal and p_fatal > p_minor):
            pred_b = "FATAL"
        elif p_grievous >= 0.38 or (p_grievous > p_minor and p_grievous > p_fatal):
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
            "model": m.get("model_name", "LightGBM Multiclass"),
        }

    # Task C: Casualty Count Regression
    if "C" in models:
        m = models["C"]
        z_pred = float(m["regressor"].predict(X_trans)[0])
        pred_count = max(1.0, z_pred + 1.0)
        results["task_c"] = {
            "predicted_casualties": round(pred_count, 1),
            "model": m.get("model_name", "LightGBM Poisson"),
        }

    # Top-level direct summary
    sev_b = results.get("task_b", {}).get("prediction", "MINOR / PROPERTY DAMAGE")
    probs = results.get("task_b", {}).get("probabilities", {"Fatal": 0.1, "Grievous": 0.2, "Minor": 0.7})
    conf = max(probs.values()) if probs else 0.7
    victims = results.get("task_c", {}).get("predicted_casualties", 1.0)
    is_dangerous = results.get("task_a", {}).get("prediction") == "DANGEROUS (FATAL/GRIEVOUS)"

    results["prediction"] = sev_b
    results["confidence"] = round(conf, 4)
    results["probabilities"] = probs
    results["predicted_casualties"] = victims
    results["is_dangerous"] = is_dangerous

    return results


def print_banner():
    print(f"\n{BOLD}{CYAN}{'═' * 74}{RESET}")
    print(f"{BOLD}{CYAN}   🚗  RoadSense: Collision Severity & Casualty Prediction Model  🚗{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 74}{RESET}")
    print(f"{DIM}   Multi-Task LightGBM Model | 3-Class Severity & Poisson Casualty Predictor{RESET}\n")


def display_results(scenario: dict, results: dict):
    print(f"\n{'═' * 74}")
    print(f"{BOLD}{CYAN}   📊  COLLISION PREDICTION SUMMARY{RESET}")
    print(f"{'═' * 74}")

    actual_spd = scenario.get("speed_limit_kmh", 80) + scenario.get("speed_delta_kmh", 0)
    print(f"  {DIM}Scenario: {scenario.get('road_category')} | {scenario.get('weather_condition')}")
    print(f"  Vehicles: {scenario.get('primary_vehicle_type')} vs {scenario.get('secondary_vehicle_type')} | Speed: {actual_spd} km/h{RESET}")

    severity = results.get("prediction", "MINOR")
    probs = results.get("probabilities", {})
    p_fatal = probs.get("Fatal", 0.0)
    p_grievous = probs.get("Grievous", 0.0)
    p_minor = probs.get("Minor", 0.0)
    victims = results.get("predicted_casualties", 1.0)
    is_danger = results.get("is_dangerous", False)

    color = RED if "FATAL" in severity else (ORANGE if "GRIEVOUS" in severity else GREEN)

    print(f"\n  {BOLD}{WHITE}╔{'═' * 70}╗{RESET}")
    print(f"  {BOLD}{WHITE}║{color}  🎯 PROBABLE PREDICTION SUMMARY                                      {WHITE}║{RESET}")
    print(f"  {BOLD}{WHITE}╠{'═' * 70}╣{RESET}")
    print(f"  {BOLD}{WHITE}║{RESET}  • {BOLD}Probable Collision Outcome:{RESET} {color}{BOLD}{severity}{RESET}")
    print(f"  {BOLD}{WHITE}║{RESET}    Distribution: Fatal: {p_fatal:.1%} | Grievous: {p_grievous:.1%} | Minor: {p_minor:.1%}")
    print(f"  {BOLD}{WHITE}║{RESET}  • {BOLD}Expected Casualties:{RESET}        {BOLD}{victims:.1f} victims{RESET} per incident")
    print(f"  {BOLD}{WHITE}║{RESET}  • {BOLD}Binary Safety Alert:{RESET}        {'🔴 DANGEROUS' if is_danger else '🟢 MINOR RISK'}")
    print(f"  {BOLD}{WHITE}╚{'═' * 70}╝{RESET}")

    # Task B: Distribution Bars
    print(f"\n  {BOLD}┌─ 3-Class Severity Distribution{RESET}")
    for cls_name, p_val in [("Fatal", p_fatal), ("Grievous", p_grievous), ("Minor", p_minor)]:
        bar_len = int(p_val * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        bar_col = RED if cls_name == "Fatal" else (ORANGE if cls_name == "Grievous" else GREEN)
        print(f"  │  {cls_name:>10s}: {bar_col}{bar}{RESET} {p_val:.1%}")
    print(f"  └{'─' * 50}\n")


def get_preset_scenarios() -> list:
    """Returns curated realistic benchmark collision scenarios."""
    return [
        {
            "name": "Expressway Winter Fog (110 km/h, Car vs Multi-Axle Truck)",
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
                "speed_delta_kmh": -10,
                "hour": 6,
                "day_of_week": 1,
                "month": 1,
                "number_of_vehicles": 3,
                "occupants_count": 5,
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
            "name": "Monsoon Ghat Hairpin (Two-Wheeler vs Bus, Overtaking)",
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
                "is_wrong_side": 1,
                "is_overloaded": 1,
                "safety_gear_violation": 1,
                "high_beam_hazard": 0,
                "stray_animal_hazard": 0,
                "drowsy_window": 0,
                "pothole_or_speedbreaker": 1,
            }
        },
        {
            "name": "Highway Midnight Head-On (Wrong-Side Tractor vs SUV)",
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
                "speed_delta_kmh": 15,
                "hour": 3,
                "day_of_week": 6,
                "month": 11,
                "number_of_vehicles": 2,
                "occupants_count": 8,
                "is_wrong_side": 1,
                "is_overloaded": 1,
                "safety_gear_violation": 0,
                "high_beam_hazard": 1,
                "stray_animal_hazard": 0,
                "drowsy_window": 1,
                "pothole_or_speedbreaker": 0,
            }
        },
        {
            "name": "Urban Low-Speed Glance (25 km/h, Seatbelts Worn)",
            "scenario": {
                "road_category": "Urban Arterial / Ring Road",
                "road_type": "Divided Multi-Lane (4/6-Lane)",
                "weather_condition": "Clear / Sunny",
                "road_surface": "Dry Asphalt / Concrete",
                "light_condition": "Daylight",
                "junction_type": "Roundabout / Circle",
                "primary_vehicle_type": "Car / Hatchback / Sedan",
                "secondary_vehicle_type": "Car / Hatchback / Sedan",
                "collision_type": "Side-Swipe / Overtaking Glance",
                "speed_limit_kmh": 30,
                "speed_delta_kmh": -5,
                "hour": 14,
                "day_of_week": 3,
                "month": 5,
                "number_of_vehicles": 2,
                "occupants_count": 2,
                "is_wrong_side": 0,
                "is_overloaded": 0,
                "safety_gear_violation": 0,
                "high_beam_hazard": 0,
                "stray_animal_hazard": 0,
                "drowsy_window": 0,
                "pothole_or_speedbreaker": 0,
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


def collect_interactive_scenario() -> dict:
    """Interactively prompts user for scenario features."""
    print(f"\n  {BOLD}Input Road & Vehicle Scenario Parameters:{RESET}")
    scenario = {}

    scenario["road_category"] = prompt_choice("Road Category:", ROAD_CATEGORIES, default_idx=1)
    scenario["road_type"] = prompt_choice("Road Carriageway:", ROAD_TYPES, default_idx=0)
    scenario["weather_condition"] = prompt_choice("Weather Condition:", WEATHER_CONDITIONS, default_idx=0)
    scenario["road_surface"] = prompt_choice("Road Surface:", ROAD_SURFACES, default_idx=0)
    scenario["light_condition"] = prompt_choice("Lighting:", LIGHT_CONDITIONS, default_idx=0)
    scenario["junction_type"] = prompt_choice("Junction Geometry:", JUNCTION_TYPES, default_idx=0)
    scenario["primary_vehicle_type"] = prompt_choice("Primary Vehicle:", VEHICLE_TYPES, default_idx=2)

    sec_choices = VEHICLE_TYPES + ["None"]
    scenario["secondary_vehicle_type"] = prompt_choice("Secondary Vehicle (Opposing):", sec_choices, default_idx=0)
    scenario["collision_type"] = prompt_choice("Collision Type:", COLLISION_TYPES, default_idx=1)

    print(f"\n  {BOLD}Posted Speed Limit (km/h):{RESET}")
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

    actual_raw = input(f"  {CYAN}Estimated Travel Speed in km/h (default = speed limit): {RESET}").strip()
    if actual_raw.isdigit():
        scenario["speed_delta_kmh"] = int(actual_raw) - scenario["speed_limit_kmh"]
    else:
        scenario["speed_delta_kmh"] = 0

    return scenario


def parse_args():
    parser = argparse.ArgumentParser(
        description="RoadSense: Collision Severity & Casualty Machine Learning Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Instantly predict default scenario")
    parser.add_argument("--preset", type=int, choices=[1, 2, 3, 4], help="Run a curated scenario preset (1-4)")
    parser.add_argument("--batch", type=str, help="Audit a CSV file with collision scenarios")
    parser.add_argument("--road", type=str, default="National Highway (NH)", choices=ROAD_CATEGORIES)
    parser.add_argument("--weather", type=str, default="Clear / Sunny", choices=WEATHER_CONDITIONS)
    parser.add_argument("--speed", type=int, default=80)
    parser.add_argument("--actual-speed", type=int, default=None)
    parser.add_argument("--vehicle", type=str, default="Two-Wheeler (Motorcycle / Scooter)", choices=VEHICLE_TYPES)
    parser.add_argument("--other-vehicle", type=str, default="Car / Hatchback / Sedan", choices=VEHICLE_TYPES + ["None"])
    parser.add_argument("--wrong-side", action="store_true", help="Contraflow / wrong-side driving")
    parser.add_argument("--overloaded", action="store_true", help="Overloaded vehicle / triple riding")
    parser.add_argument("--no-helmet", action="store_true", help="Rider without helmet")
    parser.add_argument("--no-seatbelt", action="store_true", help="Occupants unbelted")
    parser.add_argument("--high-beam", action="store_true", help="High-beam glare on undivided road")
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner()

    models = load_models()
    if not models:
        print(f"  ❌ No models found in {MODELS_DIR}/. Run 'python -m src.train' first.")
        sys.exit(1)

    presets = get_preset_scenarios()

    if args.preset:
        selected_scenario = presets[args.preset - 1]["scenario"]
        res = predict_scenario(models, selected_scenario)
        display_results(selected_scenario, res)
        return

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
        args.other_vehicle != "Car / Hatchback / Sedan",
    ])

    if custom_flag_set:
        actual_spd = args.actual_speed if args.actual_speed is not None else args.speed
        speed_delta = actual_spd - args.speed

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
        res = predict_scenario(models, scenario)
        display_results(scenario, res)
        return

    if args.batch:
        print(f"  Processing batch file: {args.batch}")
        b_df = pd.read_csv(args.batch)
        results_list = []
        for i, row in b_df.iterrows():
            sc = row.to_dict()
            res = predict_scenario(models, sc)
            flat = {
                "index": i,
                "predicted_severity": res["prediction"],
                "confidence": res["confidence"],
                "fatal_prob": res["probabilities"].get("Fatal", 0.0),
                "grievous_prob": res["probabilities"].get("Grievous", 0.0),
                "minor_prob": res["probabilities"].get("Minor", 0.0),
                "predicted_casualties": res["predicted_casualties"],
                "is_dangerous": res["is_dangerous"],
            }
            results_list.append(flat)

        out_path = args.batch.replace(".csv", "_predictions.csv")
        pd.DataFrame(results_list).to_csv(out_path, index=False)
        print(f"  ✓ Saved batch predictions to: {out_path}")
        return

    # Interactive menu
    while True:
        try:
            print(f"  {BOLD}Select an Option:{RESET}")
            for i, p in enumerate(presets, 1):
                print(f"    [{i}] Preset: {p['name']}")
            print(f"    [{len(presets) + 1}] Custom Interactive Scenario (Input parameters)")
            print(f"    [{len(presets) + 2}] Exit")

            choice_raw = input(f"\n  {CYAN}Enter choice [1]: {RESET}").strip()
            choice = int(choice_raw) if choice_raw.isdigit() else 1

            if choice == len(presets) + 2:
                break
            elif 1 <= choice <= len(presets):
                selected_scenario = presets[choice - 1]["scenario"]
            else:
                selected_scenario = collect_interactive_scenario()

            res = predict_scenario(models, selected_scenario)
            display_results(selected_scenario, res)

            again = input(f"  {CYAN}Evaluate another scenario? (y/n) [y]: {RESET}").strip().lower()
            if again == "n":
                break
        except KeyboardInterrupt:
            break

    print(f"\n{BOLD}{CYAN}{'═' * 74}{RESET}\n")


if __name__ == "__main__":
    main()

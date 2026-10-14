"""
RoadSense: Machine Learning Prediction Web Server
=================================================
Flask REST API and web application for road collision severity assessment
and casualty prediction.

Endpoints:
  - GET  /                  : Web prediction dashboard
  - GET  /api/presets       : Curated benchmark collision scenarios
  - GET  /api/options       : Form category and parameter options
  - POST /api/predict       : Single incident ML severity prediction
  - POST /api/batch         : Batch CSV collision audit
  - GET  /api/sample_csv    : Downloadable sample CSV for testing
  - GET  /api/morth_stats   : MoRTH official accident statistics
"""

import os
import sys
import io
import csv
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

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
)
from predict import load_models, get_preset_scenarios, predict_scenario

app = Flask(__name__)

# Preload models once at server startup
print("[RoadSense] Loading machine learning models...")
MODELS = load_models()
print(f"[RoadSense] Loaded models: {list(MODELS.keys())}")


@app.route("/")
def index():
    """Serves the clean prediction dashboard."""
    return render_template("index.html")


@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Returns curated realistic road collision presets."""
    presets = get_preset_scenarios()
    metadata = [
        {"id": 0, "badge": "Expressway Fog", "icon": "🌫️", "severity_hint": "FATAL COLLISION", "color": "danger"},
        {"id": 1, "badge": "Monsoon Ghat Hairpin", "icon": "🌧️", "severity_hint": "GRIEVOUS INJURY", "color": "warning"},
        {"id": 2, "badge": "NH Midnight Head-On", "icon": "🚜", "severity_hint": "FATAL COLLISION", "color": "danger"},
        {"id": 3, "badge": "Urban Low-Speed Glance", "icon": "🚗", "severity_hint": "MINOR DAMAGE", "color": "success"},
    ]

    enriched = []
    for i, p in enumerate(presets):
        meta = metadata[i] if i < len(metadata) else {"id": i, "badge": "Preset", "icon": "🚗", "severity_hint": "RISK", "color": "info"}
        enriched.append({
            "id": i,
            "name": p["name"],
            "badge": meta["badge"],
            "icon": meta["icon"],
            "severity_hint": meta["severity_hint"],
            "color": meta["color"],
            "scenario": p["scenario"],
        })

    return jsonify(enriched)


@app.route("/api/options", methods=["GET"])
def api_options():
    """Returns lists of options for dropdown inputs."""
    return jsonify({
        "road_categories": ROAD_CATEGORIES,
        "road_types": ROAD_TYPES,
        "weather_conditions": WEATHER_CONDITIONS,
        "road_surfaces": ROAD_SURFACES,
        "light_conditions": LIGHT_CONDITIONS,
        "junction_types": JUNCTION_TYPES,
        "vehicle_types": VEHICLE_TYPES,
        "collision_types": COLLISION_TYPES,
        "speed_limits": SPEED_LIMITS_KMH,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Executes single scenario severity and casualty prediction."""
    try:
        scenario = request.get_json(force=True)
        if not scenario:
            return jsonify({"error": "No scenario data provided"}), 400

        results = predict_scenario(MODELS, scenario)
        return jsonify({
            "status": "success",
            "results": results,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Audits multiple scenarios from uploaded CSV or JSON records."""
    try:
        rows = []
        if "file" in request.files:
            file = request.files["file"]
            if file.filename.endswith(".csv"):
                stream = io.StringIO(file.stream.read().decode("utf-8", errors="replace"), newline=None)
                reader = csv.DictReader(stream)
                rows = list(reader)
            else:
                return jsonify({"error": "Only CSV files are supported"}), 400
        else:
            payload = request.get_json(force=True)
            rows = payload.get("records", [])

        if not rows:
            return jsonify({"error": "No records found to assess"}), 400

        assessed_rows = []
        fatal_count = 0
        grievous_count = 0
        minor_count = 0

        for idx, row in enumerate(rows):
            sc = dict(row)
            for num_col in ["speed_limit_kmh", "speed_delta_kmh", "number_of_vehicles", "occupants_count", "hour", "day_of_week", "month"]:
                if num_col in sc:
                    try:
                        sc[num_col] = int(sc[num_col])
                    except (ValueError, TypeError):
                        pass

            for b in ["is_wrong_side", "is_overloaded", "safety_gear_violation",
                      "high_beam_hazard", "stray_animal_hazard", "drowsy_window",
                      "pothole_or_speedbreaker"]:
                if b in sc:
                    try:
                        sc[b] = int(sc[b])
                    except (ValueError, TypeError):
                        pass

            res = predict_scenario(MODELS, sc)
            sev = res.get("prediction", "MINOR / PROPERTY DAMAGE")
            cas = res.get("predicted_casualties", 1.0)
            is_danger = res.get("is_dangerous", False)

            if "FATAL" in sev:
                fatal_count += 1
            elif "GRIEVOUS" in sev:
                grievous_count += 1
            else:
                minor_count += 1

            assessed_rows.append({
                "record_id": idx + 1,
                "label": sc.get("label", f"Record #{idx+1}"),
                "road_category": sc.get("road_category", "National Highway (NH)"),
                "weather": sc.get("weather_condition", "Clear / Sunny"),
                "primary_vehicle": sc.get("primary_vehicle_type", "Car / Hatchback / Sedan"),
                "speed_delta_kmh": sc.get("speed_delta_kmh", 0),
                "severity": sev,
                "confidence": res.get("confidence", 0.7),
                "probabilities": res.get("probabilities", {}),
                "casualties": round(cas, 1),
                "is_dangerous": is_danger,
            })

        return jsonify({
            "status": "success",
            "summary": {
                "total_records": len(rows),
                "fatal_count": fatal_count,
                "grievous_count": grievous_count,
                "minor_count": minor_count,
                "dangerous_count": sum(1 for r in assessed_rows if r["is_dangerous"]),
            },
            "records": assessed_rows,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sample_csv", methods=["GET"])
def api_sample_csv():
    """Generates a downloadable sample CSV for batch testing."""
    sample_data = [
        {
            "label": "Yamuna Exp Km-42 (Winter Morning)",
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
            "number_of_vehicles": 3,
            "occupants_count": 5,
            "is_wrong_side": 0,
            "is_overloaded": 0,
            "safety_gear_violation": 1,
            "high_beam_hazard": 0,
            "stray_animal_hazard": 0,
            "drowsy_window": 0,
            "pothole_or_speedbreaker": 0,
        },
        {
            "label": "Pune-Solapur NH-65 (Midnight Crossing)",
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
            "speed_delta_kmh": 20,
            "number_of_vehicles": 2,
            "occupants_count": 6,
            "is_wrong_side": 1,
            "is_overloaded": 1,
            "safety_gear_violation": 0,
            "high_beam_hazard": 1,
            "stray_animal_hazard": 0,
            "drowsy_window": 1,
            "pothole_or_speedbreaker": 0,
        },
        {
            "label": "Western Ghats Lonavala (Monsoon Curve)",
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
            "number_of_vehicles": 2,
            "occupants_count": 30,
            "is_wrong_side": 1,
            "is_overloaded": 1,
            "safety_gear_violation": 1,
            "high_beam_hazard": 0,
            "stray_animal_hazard": 0,
            "drowsy_window": 0,
            "pothole_or_speedbreaker": 1,
        },
    ]

    df = pd.DataFrame(sample_data)
    csv_str = df.to_csv(index=False)
    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=roadsense_batch_sample.csv"},
    )


@app.route("/api/morth_stats", methods=["GET"])
def api_morth_stats():
    """Serves official MoRTH statistics categorized for benchmarking."""
    stats_path = os.path.join("data", "morth_official_statistics.csv")
    if not os.path.exists(stats_path):
        return jsonify({"error": "Benchmark data file not found"}), 404

    df = pd.read_csv(stats_path, comment="#")
    records = df.to_dict(orient="records")
    return jsonify({
        "status": "success",
        "records": records,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 RoadSense Prediction Web App running at: http://localhost:5000")
    app.run(host="0.0.0.0", port=port, debug=False)

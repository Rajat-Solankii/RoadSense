"""
RoadSense AI: Web Frontend Server
=================================
Flask web server powering the interactive road safety assessment dashboard.
Exposes REST endpoints for presets, dropdown options, and multi-task AI predictions.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# Add project root to path
sys.path.insert(0, os.path.abspath("."))
from src.features import (
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

# Preload models once at startup
print("[RoadSense] Loading multi-task LightGBM models...")
MODELS = load_models()
print(f"[RoadSense] Loaded {len(MODELS)} models: {list(MODELS.keys())}")


@app.route("/")
def index():
    """Serves the main interactive dashboard."""
    return render_template("index.html")


@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Returns curated realistic Indian road collision presets."""
    presets = get_preset_scenarios()
    
    # Enrich presets with icons and badges
    metadata = [
        {
            "id": 0,
            "badge": "Expressway Fog",
            "icon": "🌫️",
            "severity_hint": "FATAL PILEUP",
            "color": "danger"
        },
        {
            "id": 1,
            "badge": "Monsoon Ghat Hairpin",
            "icon": "🌧️",
            "severity_hint": "GRIEVOUS INJURY",
            "color": "warning"
        },
        {
            "id": 2,
            "badge": "NH Midnight Wrong-Side",
            "icon": "🚜",
            "severity_hint": "FATAL COLLISION",
            "color": "danger"
        },
        {
            "id": 3,
            "badge": "Urban Auto T-Bone",
            "icon": "🛺",
            "severity_hint": "GRIEVOUS INJURY",
            "color": "warning"
        }
    ]
    
    enriched = []
    for i, p in enumerate(presets):
        meta = metadata[i] if i < len(metadata) else {"id": i, "badge": "Custom", "icon": "🚗", "severity_hint": "RISK", "color": "info"}
        enriched.append({
            "id": i,
            "name": p["name"],
            "badge": meta["badge"],
            "icon": meta["icon"],
            "severity_hint": meta["severity_hint"],
            "color": meta["color"],
            "scenario": p["scenario"]
        })
        
    return jsonify(enriched)


@app.route("/api/options", methods=["GET"])
def api_options():
    """Returns lists of options for form dropdowns."""
    return jsonify({
        "road_categories": ROAD_CATEGORIES,
        "road_types": ROAD_TYPES,
        "weather_conditions": WEATHER_CONDITIONS,
        "road_surfaces": ROAD_SURFACES,
        "light_conditions": LIGHT_CONDITIONS,
        "junction_types": JUNCTION_TYPES,
        "vehicle_types": VEHICLE_TYPES,
        "collision_types": COLLISION_TYPES,
        "speed_limits": SPEED_LIMITS_KMH
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Executes prediction on the provided road safety scenario payload."""
    try:
        scenario = request.get_json(force=True)
        if not scenario:
            return jsonify({"error": "No scenario data provided"}), 400

        # Run multi-task prediction & disaster prevention engine
        results = predict_scenario(MODELS, scenario)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 RoadSense AI Web App running at: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

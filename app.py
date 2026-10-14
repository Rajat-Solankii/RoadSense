"""
RoadSense: Professional Decision-Support Web Server
===================================================
Flask server providing REST APIs for:
  - Single incident severity & disaster prediction
  - 'What-If' counterfactual intervention comparison
  - Bulk / Batch CSV incident assessment & road safety audits
  - MoRTH & iRAD official road accident benchmarks
"""

import os
import sys
import io
import csv
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

# Add project root to path
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
from src.irad_database import IRADDatabase

app = Flask(__name__)

# Preload models once at startup
print("[RoadSense] Loading multi-task LightGBM models...")
MODELS = load_models()
print(f"[RoadSense] Loaded {len(MODELS)} models: {list(MODELS.keys())}")

# Initialize iRAD database
irad_db = IRADDatabase()


@app.route("/")
def index():
    """Serves the professional multi-tab dashboard."""
    return render_template("index.html")


@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Returns curated realistic Indian road collision presets."""
    presets = get_preset_scenarios()
    
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
    """Executes single scenario risk assessment."""
    try:
        scenario = request.get_json(force=True)
        if not scenario:
            return jsonify({"error": "No scenario data provided"}), 400

        results = predict_scenario(MODELS, scenario)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """Executes side-by-side comparison between baseline and mitigated conditions."""
    try:
        data = request.get_json(force=True)
        baseline = data.get("baseline", {})
        mitigated = data.get("mitigated", {})

        if not baseline or not mitigated:
            return jsonify({"error": "Both baseline and mitigated scenarios are required"}), 400

        res_base = predict_scenario(MODELS, baseline)
        res_mit = predict_scenario(MODELS, mitigated)

        # Calculate deltas
        base_fatal_prob = res_base.get("task_b", {}).get("probabilities", {}).get("Fatal", 0.0)
        mit_fatal_prob = res_mit.get("task_b", {}).get("probabilities", {}).get("Fatal", 0.0)
        fatal_reduction = max(0.0, (base_fatal_prob - mit_fatal_prob) * 100.0)

        base_cas = res_base.get("task_c", {}).get("predicted_casualties", 1.0)
        mit_cas = res_mit.get("task_c", {}).get("predicted_casualties", 1.0)
        casualties_saved = max(0.0, base_cas - mit_cas)

        base_dri = res_base.get("disaster", {}).get("disaster_risk_index", 0.0)
        mit_dri = res_mit.get("disaster", {}).get("disaster_risk_index", 0.0)
        dri_reduction = base_dri - mit_dri

        return jsonify({
            "status": "success",
            "baseline": res_base,
            "mitigated": res_mit,
            "impact": {
                "fatal_prob_reduction_pct": round(fatal_reduction, 1),
                "casualties_saved": round(casualties_saved, 1),
                "dri_reduction_pct": round(dri_reduction, 1),
                "baseline_severity": res_base.get("task_b", {}).get("prediction", "FATAL"),
                "mitigated_severity": res_mit.get("task_b", {}).get("prediction", "MINOR"),
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Audits multiple scenarios from uploaded CSV or JSON array."""
    try:
        rows = []
        if "file" in request.files:
            file = request.files["file"]
            if file.filename.endswith(".csv"):
                stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
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
        total_dri = 0.0

        for idx, row in enumerate(rows):
            # Parse & cast fields safely
            sc = dict(row)
            if "speed_limit_kmh" in sc:
                sc["speed_limit_kmh"] = int(sc["speed_limit_kmh"])
            if "speed_delta_kmh" in sc:
                sc["speed_delta_kmh"] = int(sc["speed_delta_kmh"])
            if "number_of_vehicles" in sc:
                sc["number_of_vehicles"] = int(sc["number_of_vehicles"])
            if "occupants_count" in sc:
                sc["occupants_count"] = int(sc["occupants_count"])

            for b in ["is_wrong_side", "is_overloaded", "safety_gear_violation",
                      "high_beam_hazard", "stray_animal_hazard", "drowsy_window",
                      "pothole_or_speedbreaker"]:
                if b in sc:
                    sc[b] = int(sc[b])

            res = predict_scenario(MODELS, sc)
            sev = res.get("task_b", {}).get("prediction", "MINOR")
            cas = res.get("task_c", {}).get("predicted_casualties", 1.0)
            dri = res.get("disaster", {}).get("disaster_risk_index", 0.0)

            if sev == "FATAL":
                fatal_count += 1
            elif sev == "GRIEVOUS INJURY":
                grievous_count += 1
            else:
                minor_count += 1

            total_dri += dri

            assessed_rows.append({
                "record_id": idx + 1,
                "label": sc.get("label", f"Record #{idx+1}"),
                "road_category": sc.get("road_category", "National Highway (NH)"),
                "weather": sc.get("weather_condition", "Clear / Sunny"),
                "primary_vehicle": sc.get("primary_vehicle_type", "Car / Hatchback / Sedan"),
                "speed_delta_kmh": sc.get("speed_delta_kmh", 0),
                "severity": sev,
                "casualties": round(cas, 1),
                "dri": round(dri, 1),
                "risk_band": res.get("disaster", {}).get("risk_band", "MODERATE"),
                "primary_directive": (res.get("disaster", {}).get("mitigation_directives") or ["Defensive driving"])[0]
            })

        avg_dri = round(total_dri / len(rows), 1) if rows else 0.0

        return jsonify({
            "status": "success",
            "summary": {
                "total_records": len(rows),
                "fatal_count": fatal_count,
                "grievous_count": grievous_count,
                "minor_count": minor_count,
                "average_dri": avg_dri,
                "critical_flag_count": sum(1 for r in assessed_rows if r["dri"] >= 75)
            },
            "records": assessed_rows
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sample_csv", methods=["GET"])
def api_sample_csv():
    """Generates a downloadable sample CSV for batch audit testing."""
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
            "number_of_vehicles": 4,
            "occupants_count": 8,
            "is_wrong_side": 0,
            "is_overloaded": 0,
            "safety_gear_violation": 1,
            "high_beam_hazard": 0,
            "stray_animal_hazard": 0,
            "drowsy_window": 0,
            "pothole_or_speedbreaker": 0
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
            "pothole_or_speedbreaker": 0
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
            "pothole_or_speedbreaker": 1
        },
        {
            "label": "Bangalore Outer Ring Road (Peak Hour)",
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
            "speed_delta_kmh": 5,
            "number_of_vehicles": 2,
            "occupants_count": 6,
            "is_wrong_side": 0,
            "is_overloaded": 1,
            "safety_gear_violation": 1,
            "high_beam_hazard": 0,
            "stray_animal_hazard": 0,
            "drowsy_window": 0,
            "pothole_or_speedbreaker": 1
        }
    ]

    df = pd.DataFrame(sample_data)
    csv_str = df.to_csv(index=False)
    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=roadsense_batch_sample.csv"}
    )


@app.route("/api/morth_stats", methods=["GET"])
def api_morth_stats():
    """Serves official MoRTH statistics categorized for dashboard benchmarking."""
    stats_path = os.path.join("data", "morth_official_statistics.csv")
    if not os.path.exists(stats_path):
        return jsonify({"error": "Benchmark data file not found"}), 404

    df = pd.read_csv(stats_path, comment="#")
    records = df.to_dict(orient="records")
    return jsonify({
        "status": "success",
        "records": records
    })


@app.route("/api/irad/incidents", methods=["GET"])
def api_irad_incidents():
    """Fetches verified iRAD / eDAR accident records enriched with multi-task AI assessment."""
    state = request.args.get("state", "All")
    limit = int(request.args.get("limit", 50))
    raw_incidents = irad_db.fetch_all(limit=limit, state=state)

    enriched = []
    total_fatal = 0
    total_grievous = 0
    total_dri = 0.0

    for inc in raw_incidents:
        # Check if AI assessment exists, otherwise calculate and cache it
        ai_sev = inc.get("ai_severity_predicted")
        ai_cas = inc.get("ai_predicted_casualties")
        ai_dri = inc.get("ai_disaster_risk_index")
        ai_band = inc.get("ai_risk_band")

        if not ai_sev or ai_dri is None:
            # Run model inference
            scenario_dict = {
                "road_category": inc["road_category"],
                "road_type": inc["road_type"],
                "weather_condition": inc["weather_condition"],
                "road_surface": inc["road_surface"],
                "light_condition": inc["light_condition"],
                "junction_type": inc["junction_type"],
                "primary_vehicle_type": inc["primary_vehicle_type"],
                "secondary_vehicle_type": inc["secondary_vehicle_type"],
                "collision_type": inc["collision_type"],
                "speed_limit_kmh": inc["speed_limit_kmh"],
                "speed_delta_kmh": inc["speed_delta_kmh"],
                "number_of_vehicles": inc["number_of_vehicles"],
                "occupants_count": inc["occupants_count"],
                "is_wrong_side": inc["is_wrong_side"],
                "is_overloaded": inc["is_overloaded"],
                "safety_gear_violation": inc["safety_gear_violation"],
                "high_beam_hazard": inc["high_beam_hazard"],
                "stray_animal_hazard": inc["stray_animal_hazard"],
                "drowsy_window": inc["drowsy_window"],
                "pothole_or_speedbreaker": inc["pothole_or_speedbreaker"],
            }
            res = predict_scenario(MODELS, scenario_dict)
            ai_sev = res.get("task_b", {}).get("prediction", "MINOR")
            ai_cas = res.get("task_c", {}).get("predicted_casualties", 1.0)
            ai_dri = res.get("disaster", {}).get("disaster_risk_index", 0.0)
            ai_band = res.get("disaster", {}).get("risk_band", "MODERATE")

            irad_db.update_ai_assessment(inc["accident_id"], ai_sev, ai_cas, ai_dri, ai_band)

        total_fatal += inc.get("actual_fatalities", 0)
        total_grievous += inc.get("actual_grievous", 0)
        total_dri += float(ai_dri)

        item = dict(inc)
        item["ai_severity_predicted"] = ai_sev
        item["ai_predicted_casualties"] = round(float(ai_cas), 1)
        item["ai_disaster_risk_index"] = round(float(ai_dri), 1)
        item["ai_risk_band"] = ai_band
        enriched.append(item)

    avg_dri = round(total_dri / len(enriched), 1) if enriched else 0.0

    return jsonify({
        "status": "success",
        "summary": {
            "total_incidents": len(enriched),
            "actual_fatalities": total_fatal,
            "actual_grievous": total_grievous,
            "average_dri": avg_dri,
            "active_states": list(set(r["state"] for r in enriched))
        },
        "incidents": enriched
    })


@app.route("/api/irad/simulate_live", methods=["POST"])
def api_irad_simulate_live():
    """Simulates a real-time incoming accident report from state highway police dispatch."""
    new_inc = irad_db.simulate_live_incoming_incident()

    # Immediately run model inference
    scenario_dict = {
        "road_category": new_inc["road_category"],
        "road_type": new_inc["road_type"],
        "weather_condition": new_inc["weather_condition"],
        "road_surface": new_inc["road_surface"],
        "light_condition": new_inc["light_condition"],
        "junction_type": new_inc["junction_type"],
        "primary_vehicle_type": new_inc["primary_vehicle_type"],
        "secondary_vehicle_type": new_inc["secondary_vehicle_type"],
        "collision_type": new_inc["collision_type"],
        "speed_limit_kmh": new_inc["speed_limit_kmh"],
        "speed_delta_kmh": new_inc["speed_delta_kmh"],
        "number_of_vehicles": new_inc["number_of_vehicles"],
        "occupants_count": new_inc["occupants_count"],
        "is_wrong_side": new_inc["is_wrong_side"],
        "is_overloaded": new_inc["is_overloaded"],
        "safety_gear_violation": new_inc["safety_gear_violation"],
        "high_beam_hazard": new_inc["high_beam_hazard"],
        "stray_animal_hazard": new_inc["stray_animal_hazard"],
        "drowsy_window": new_inc["drowsy_window"],
        "pothole_or_speedbreaker": new_inc["pothole_or_speedbreaker"],
    }
    res = predict_scenario(MODELS, scenario_dict)
    ai_sev = res.get("task_b", {}).get("prediction", "MINOR")
    ai_cas = res.get("task_c", {}).get("predicted_casualties", 1.0)
    ai_dri = res.get("disaster", {}).get("disaster_risk_index", 0.0)
    ai_band = res.get("disaster", {}).get("risk_band", "MODERATE")

    irad_db.update_ai_assessment(new_inc["accident_id"], ai_sev, ai_cas, ai_dri, ai_band)

    new_inc["ai_severity_predicted"] = ai_sev
    new_inc["ai_predicted_casualties"] = round(float(ai_cas), 1)
    new_inc["ai_disaster_risk_index"] = round(float(ai_dri), 1)
    new_inc["ai_risk_band"] = ai_band

    return jsonify({
        "status": "success",
        "incident": new_inc
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 RoadSense Professional Web App running at: http://localhost:5000")
    app.run(host="0.0.0.0", port=port, debug=False)

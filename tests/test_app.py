"""
Unit & Integration Tests for RoadSense Prediction Web Server
"""

import pytest
import json
import io
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Test that the homepage loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"RoadSense" in response.data
    assert b"Incident Prediction" in response.data
    assert b"Batch Audit (CSV)" in response.data


def test_api_presets_route(client):
    """Test that the presets endpoint returns a list of presets."""
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) >= 4
    first = data[0]
    assert "name" in first
    assert "scenario" in first
    assert "badge" in first
    assert "icon" in first


def test_api_options_route(client):
    """Test that the options endpoint returns valid category lists."""
    response = client.get("/api/options")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "road_categories" in data
    assert "weather_conditions" in data
    assert "vehicle_types" in data
    assert "speed_limits" in data
    assert len(data["road_categories"]) > 0


def test_api_predict_route(client):
    """Test prediction endpoint with a valid scenario payload."""
    payload = {
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
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert "results" in data
    res = data["results"]
    assert "prediction" in res
    assert "probabilities" in res
    assert "predicted_casualties" in res
    assert "is_dangerous" in res


def test_api_batch_route(client):
    """Test batch CSV audit endpoint."""
    csv_data = """label,road_category,weather_condition,primary_vehicle_type,secondary_vehicle_type,speed_limit_kmh,speed_delta_kmh,is_wrong_side,safety_gear_violation
Segment A,Expressway,Clear / Sunny,Car / Hatchback / Sedan,None,100,0,0,0
Segment B,National Highway (NH),Heavy Monsoon / Waterlogged,Two-Wheeler (Motorcycle / Scooter),Bus (State Transport / Private / School),60,15,1,1
"""
    data = {
        "file": (io.BytesIO(csv_data.encode("utf-8")), "test_batch.csv")
    }
    response = client.post("/api/batch", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = json.loads(response.data)
    assert res["status"] == "success"
    assert "summary" in res
    assert res["summary"]["total_records"] == 2
    assert len(res["records"]) == 2
    first = res["records"][0]
    assert "severity" in first
    assert "casualties" in first
    assert "is_dangerous" in first


def test_api_sample_csv_route(client):
    """Test downloading the sample CSV template."""
    response = client.get("/api/sample_csv")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert b"road_category" in response.data


def test_api_morth_stats_route(client):
    """Test fetching MoRTH benchmark statistics."""
    response = client.get("/api/morth_stats")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert isinstance(data["records"], list)
    assert len(data["records"]) > 10

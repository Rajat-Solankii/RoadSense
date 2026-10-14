"""
Unit & Integration Tests for RoadSense Web Application
"""

import pytest
import json
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
    assert b"Scenario Customizer" in response.data


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
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    res_json = json.loads(response.data)
    assert res_json["status"] == "success"
    results = res_json["results"]
    assert "task_a" in results
    assert "task_b" in results
    assert "task_c" in results
    assert "disaster" in results
    assert "prediction" in results["task_b"]
    assert "disaster_risk_index" in results["disaster"]

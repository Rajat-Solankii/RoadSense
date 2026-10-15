"""
RoadSense AI: Tests for Model Loading, Inference & Prediction Engine
"""

import pytest
import os
from predict import load_models, predict_scenario, get_preset_scenarios


@pytest.fixture(scope="module")
def models():
    """Load models fixture once for the module."""
    loaded = load_models()
    assert "A" in loaded, "Task A model must be loaded"
    assert "B" in loaded, "Task B model must be loaded"
    assert "C" in loaded, "Task C model must be loaded"
    return loaded


def test_predict_scenario_structure(models):
    """Verify predict_scenario outputs all expected tasks and valid values."""
    scenario = {
        "road_category": "National Highway (NH)",
        "road_type": "Divided Multi-Lane (4/6-Lane)",
        "weather_condition": "Clear / Sunny",
        "road_surface": "Dry Asphalt / Concrete",
        "light_condition": "Daylight",
        "junction_type": "Straight Road (Not at Junction)",
        "primary_vehicle_type": "Two-Wheeler (Motorcycle / Scooter)",
        "secondary_vehicle_type": "Car / Hatchback / Sedan",
        "speed_limit_kmh": 80,
        "speed_delta_kmh": 0,
        "is_wrong_side": 0,
        "is_overloaded": 0,
        "safety_gear_violation": 1,
    }

    result = predict_scenario(models, scenario)

    # Check Task A
    assert "task_a" in result
    assert result["task_a"]["prediction"] in ["DANGEROUS (FATAL/GRIEVOUS)", "MINOR"]
    assert 0.0 <= result["task_a"]["probability"] <= 1.0
    assert result["task_a"]["risk_band"]["label"] in ["LOW", "ELEVATED", "HIGH", "CRITICAL"]

    # Check Task B
    assert "task_b" in result
    assert result["task_b"]["prediction"] in ["FATAL", "GRIEVOUS INJURY", "MINOR / PROPERTY DAMAGE"]
    probs = result["task_b"]["probabilities"]
    assert pytest.approx(sum(probs.values()), 0.02) == 1.0

    # Check Task C
    assert "task_c" in result
    assert result["task_c"]["predicted_casualties"] >= 1.0

    # Check Top-level summary keys
    assert "prediction" in result
    assert "probabilities" in result
    assert "predicted_casualties" in result
    assert "is_dangerous" in result


def test_all_presets_valid(models):
    """Verify all curated benchmark presets execute and return valid predictions."""
    presets = get_preset_scenarios()
    assert len(presets) >= 4

    for preset in presets:
        sc = preset["scenario"]
        res = predict_scenario(models, sc)
        assert res["task_a"]["prediction"] is not None
        assert res["task_b"]["prediction"] is not None
        assert res["task_c"]["predicted_casualties"] >= 1.0
        assert res["prediction"] in ["FATAL", "GRIEVOUS INJURY", "MINOR / PROPERTY DAMAGE"]

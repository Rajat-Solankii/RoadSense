"""
RoadSense AI: Tests for Disaster Prevention & Emergency Mitigation Engine
"""

import pytest
from src.disaster_prevention import DisasterPreventionEngine


def test_assess_disaster_risk_fog_expressway():
    """Verify expressway dense fog triggers critical pileup escalation threats and directives."""
    scenario = {
        "road_category": "Expressway",
        "road_type": "Divided Multi-Lane (4/6-Lane)",
        "weather_condition": "Dense Winter Fog / Smog",
        "road_surface": "Dry Asphalt / Concrete",
        "light_condition": "Dusk / Dawn (Twilight)",
        "speed_limit_kmh": 120,
        "speed_delta_kmh": -10,
        "primary_vehicle_type": "Car / Hatchback / Sedan",
        "secondary_vehicle_type": "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)",
    }
    severity_probs = {"Fatal": 0.35, "Grievous": 0.55, "Minor": 0.10}

    result = DisasterPreventionEngine.assess_disaster_risk(scenario, severity_probs)

    assert "disaster_risk_index" in result
    assert result["disaster_risk_index"] >= 60.0
    assert result["risk_band"] in ["SEVERE HAZARD ALERT", "CATASTROPHIC DISASTER RISK"]

    # Threat verification
    threat_titles = [t["title"] for t in result["escalation_threats"]]
    assert "Chain-Reaction Expressway Pileup Hazard" in threat_titles

    # Mitigation directives verification
    directives_text = " ".join(result["mitigation_directives"])
    assert "DYNAMIC SPEED RESTRICTION" in directives_text
    assert "40 km/h" in directives_text

    # Secondary prevention protocols present
    assert len(result["secondary_prevention_protocols"]) >= 4
    protocols_text = " ".join(result["secondary_prevention_protocols"]).lower()
    assert "crash barrier" in protocols_text
    assert "1033" in protocols_text


def test_assess_disaster_risk_contraflow():
    """Verify wrong-side driving triggers head-on collision threat."""
    scenario = {
        "road_category": "National Highway (NH)",
        "road_type": "Divided Multi-Lane (4/6-Lane)",
        "weather_condition": "Clear / Sunny",
        "is_wrong_side": 1,
        "speed_limit_kmh": 80,
    }
    severity_probs = {"Fatal": 0.40, "Grievous": 0.45, "Minor": 0.15}

    result = DisasterPreventionEngine.assess_disaster_risk(scenario, severity_probs)
    threat_titles = [t["title"] for t in result["escalation_threats"]]
    assert "High-Energy Head-On Collision" in threat_titles


def test_assess_disaster_risk_monsoon():
    """Verify monsoon waterlogged roads trigger aquaplaning advisory."""
    scenario = {
        "road_category": "State Highway (SH)",
        "road_type": "Undivided 2-Lane",
        "weather_condition": "Heavy Monsoon / Waterlogged",
        "road_surface": "Wet / Slippery",
        "speed_limit_kmh": 60,
    }
    severity_probs = {"Fatal": 0.10, "Grievous": 0.30, "Minor": 0.60}

    result = DisasterPreventionEngine.assess_disaster_risk(scenario, severity_probs)
    threat_titles = [t["title"] for t in result["escalation_threats"]]
    assert "High-Speed Hydroplaning & Secondary Run-Off" in threat_titles

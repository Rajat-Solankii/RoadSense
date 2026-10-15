"""
RoadSense AI: Tests for Feature Engineering & Preprocessing Module
"""

import pytest
import numpy as np
import pandas as pd
from src.features import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    BEHAVIORAL_INDICATOR_FEATURES,
    build_preprocessor,
    derive_most_vulnerable,
    VULNERABILITY_RANK,
)


def test_feature_list_definitions():
    """Ensure feature lists are well-defined and non-overlapping."""
    assert len(CATEGORICAL_FEATURES) > 0
    assert len(NUMERICAL_FEATURES) > 0
    assert len(BEHAVIORAL_INDICATOR_FEATURES) > 0

    # Ensure no duplicates in ALL_MODEL_FEATURES
    assert len(ALL_MODEL_FEATURES) == len(set(ALL_MODEL_FEATURES))
    assert set(ALL_MODEL_FEATURES) == set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BEHAVIORAL_INDICATOR_FEATURES)


def test_derive_most_vulnerable():
    """Test hierarchical vulnerability ranking."""
    # Pedestrian > Car
    assert derive_most_vulnerable("Car / Hatchback / Sedan", "Bicycle / Pedestrian") == "Bicycle / Pedestrian"

    # Two-Wheeler > Heavy Truck
    assert derive_most_vulnerable("Two-Wheeler (Motorcycle / Scooter)", "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)") == "Two-Wheeler (Motorcycle / Scooter)"

    # Auto-Rickshaw > Bus
    assert derive_most_vulnerable("Three-Wheeler (Auto-Rickshaw / E-Rickshaw)", "Bus (State Transport / Private / School)") == "Three-Wheeler (Auto-Rickshaw / E-Rickshaw)"

    # Solo vehicle (v2 is None)
    assert derive_most_vulnerable("Car / Hatchback / Sedan", None) == "Car / Hatchback / Sedan"


def test_build_preprocessor():
    """Test ColumnTransformer construction and transformation."""
    preprocessor = build_preprocessor()
    assert preprocessor is not None

    # Create a dummy row matching ALL_MODEL_FEATURES
    dummy_data = {
        "road_category": ["National Highway (NH)"],
        "road_type": ["Divided Multi-Lane (4/6-Lane)"],
        "weather_condition": ["Clear / Sunny"],
        "road_surface": ["Dry Asphalt / Concrete"],
        "light_condition": ["Daylight"],
        "junction_type": ["Straight Road (Not at Junction)"],
        "primary_vehicle_type": ["Two-Wheeler (Motorcycle / Scooter)"],
        "secondary_vehicle_type": ["Car / Hatchback / Sedan"],
        "most_vulnerable_participant": ["Two-Wheeler (Motorcycle / Scooter)"],
        "collision_type": ["Rear-End / Chain Pileup"],
        "speed_limit_kmh": [80],
        "speed_delta_kmh": [0],
        "hour": [14],
        "day_of_week": [2],
        "month": [6],
        "number_of_vehicles": [2],
        "occupants_count": [3],
        "is_wrong_side": [0],
        "is_overloaded": [0],
        "safety_gear_violation": [1],
        "high_beam_hazard": [0],
        "stray_animal_hazard": [0],
        "drowsy_window": [0],
        "pothole_or_speedbreaker": [0],
    }
    df = pd.DataFrame(dummy_data)
    X_trans = preprocessor.fit_transform(df)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 1
    assert X_trans.shape[1] >= len(ALL_MODEL_FEATURES)
    assert not np.isnan(X_trans).any()

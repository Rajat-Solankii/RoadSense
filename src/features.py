"""
RoadSense AI: Feature Engineering & Domain Specification
=========================================================
Calibrated against official MoRTH (Ministry of Road Transport and Highways)
and iRAD (Integrated Road Accident Database) standards.

Features:
  - Road hierarchy (Expressways, NH, SH, MDR, Rural PMGSY, Ghats)
  - Speed limits in km/h (20 to 120 km/h) & overspeeding delta
  - Climatic conditions (Monsoon waterlogging, Dense winter fog/smog, Heatwaves, Dust storms)
  - Vehicle distribution (Two-wheelers, Autos/E-rickshaws, Tractors with trolleys, LCV, Trucks, Buses)
  - Behavioral risk variables (Wrong-side driving, overloading/triple riding, lack of helmets/seatbelts,
    high-beam glare, drowsy driving windows, stray cattle intrusion, unscientific speed-breakers/potholes)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ==============================================================================
# 1. DOMAIN LOOKUPS & METADATA
# ==============================================================================

ROAD_CATEGORIES = [
    "Expressway",
    "National Highway (NH)",
    "State Highway (SH)",
    "Major District Road (MDR)",
    "Rural / Village Road (PMGSY)",
    "Urban Arterial / Ring Road",
    "Ghat / Hill Terrain Road",
]

ROAD_TYPES = [
    "Divided Multi-Lane (4/6-Lane)",
    "Undivided 2-Lane",
    "Single Lane Road",
    "Service Road",
    "Flyover / Elevated Corridor",
    "Roundabout / Circle",
]

WEATHER_CONDITIONS = [
    "Clear / Sunny",
    "Heavy Monsoon / Waterlogged",
    "Dense Winter Fog / Smog",
    "Dust Storm / Andhi",
    "Extreme Summer Heatwave (>42°C)",
    "Hail / Cloudburst",
    "Light Rain / Drizzle",
]

ROAD_SURFACES = [
    "Dry Asphalt / Concrete",
    "Waterlogged / Potholed",
    "Wet / Slippery",
    "Unpaved / Muddy Gravel",
    "Under Construction / Loose Gravel",
]

LIGHT_CONDITIONS = [
    "Daylight",
    "Night - Streetlights Functional",
    "Night - Pitch Dark (No Streetlights)",
    "Dusk / Dawn (Twilight)",
]

JUNCTION_TYPES = [
    "Straight Road (Not at Junction)",
    "T-Junction / Unsignalized Median Cut",
    "Crossroads / 4-Way Intersection",
    "Roundabout",
    "Flyover Ramp / Blind Slip Road",
    "Blind Curve / Ghat Hairpin Bend",
]

VEHICLE_TYPES = [
    "Two-Wheeler (Motorcycle / Scooter)",
    "Three-Wheeler (Auto-Rickshaw / E-Rickshaw)",
    "Car / Hatchback / Sedan",
    "SUV / MUV / Compact SUV",
    "Bus (State Transport / Private / School)",
    "Light Commercial Vehicle (LCV / Tempo / Tata Ace)",
    "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)",
    "Tractor with Trolley (Sugarcane/Construction)",
    "Bicycle / Pedestrian",
]

COLLISION_TYPES = [
    "Head-On Collision",
    "Rear-End / Chain Pileup",
    "Right Angle / T-Bone at Median Cut",
    "Side-Swipe / Overtaking Glance",
    "Run-Off Road / Hit Tree, Pole or Ditch",
    "Hit Pedestrian / Stray Cattle",
    "Vehicle Rollover / Overturn",
]

SPEED_LIMITS_KMH = [20, 30, 40, 50, 60, 80, 100, 120]

# Vulnerability hierarchy (higher number = more vulnerable participant)
VULNERABILITY_RANK = {
    "Bicycle / Pedestrian": 5,
    "Two-Wheeler (Motorcycle / Scooter)": 4,
    "Three-Wheeler (Auto-Rickshaw / E-Rickshaw)": 3,
    "Car / Hatchback / Sedan": 2,
    "SUV / MUV / Compact SUV": 2,
    "Light Commercial Vehicle (LCV / Tempo / Tata Ace)": 1,
    "Tractor with Trolley (Sugarcane/Construction)": 1,
    "Bus (State Transport / Private / School)": 0,
    "Heavy Goods Vehicle (Multi-Axle Truck / Dumper / Trailer)": 0,
    "None": 0,
}

# ==============================================================================
# 2. FEATURE DEFINITIONS FOR MODEL PIPELINE
# ==============================================================================

CATEGORICAL_FEATURES = [
    "road_category",
    "road_type",
    "weather_condition",
    "road_surface",
    "light_condition",
    "junction_type",
    "primary_vehicle_type",
    "secondary_vehicle_type",
    "most_vulnerable_participant",
    "collision_type",
]

NUMERICAL_FEATURES = [
    "speed_limit_kmh",
    "speed_delta_kmh",           # Actual speed minus speed limit (>0 means overspeeding)
    "hour",
    "day_of_week",
    "month",
    "number_of_vehicles",
    "occupants_count",
]

BEHAVIORAL_INDICATOR_FEATURES = [
    "is_wrong_side",             # Contraflow / wrong-side driving
    "is_overloaded",             # Triple riding, overloaded passenger auto, freight overhang
    "safety_gear_violation",     # Helmet unstrapped/missing, no seatbelt
    "high_beam_hazard",          # Glare on undivided highway
    "stray_animal_hazard",       # Stray cattle/dog sudden intrusion
    "drowsy_window",             # 01:00 AM - 05:30 AM circadian low window
    "pothole_or_speedbreaker",   # Unmarked speed bump or broken road crater
]

ALL_MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BEHAVIORAL_INDICATOR_FEATURES

TARGET_BINARY = "severity_binary"       # 1 = Fatal/Grievous, 0 = Minor/Property Damage
TARGET_3CLASS = "severity_3class"       # 0 = Fatal, 1 = Grievous, 2 = Minor
TARGET_COUNT = "casualty_count"         # y >= 1


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a robust, production-grade ColumnTransformer.
    - Categorical: OneHotEncoder with handle_unknown='ignore'
    - Numerical: Median imputation + StandardScaler
    - Behavioral: Passthrough (binary 0/1)
    """
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("beh", "passthrough", BEHAVIORAL_INDICATOR_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def derive_most_vulnerable(v1: str, v2: Optional[str] = None) -> str:
    """Calculates the most vulnerable participant between colliding entities."""
    r1 = VULNERABILITY_RANK.get(v1, 2)
    r2 = VULNERABILITY_RANK.get(v2, -1) if v2 else -1
    return v1 if r1 >= r2 else v2

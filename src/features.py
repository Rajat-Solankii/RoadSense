"""
RoadSense AI: Feature Engineering & Preprocessing Pipeline
Responsible for:
  - Defining standard feature definitions across all models
  - User-facing input mapping for predict.py demo (mapping vehicle types to interaction flags)
  - Unified ColumnTransformer (OneHotEncoder with explicit 'Unknown' category + Median Scaler for numerics)
  - Time-based and Police-Force regional train/val/test splits
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

# ==============================================================================
# 1. FEATURE LISTS BY ROLE
# ==============================================================================
CORE_FEATURES = [
    "speed_limit",               # Proxy for operating speed
    "weather_conditions",
    "road_type",
    "most_vulnerable_vehicle_type"
]

EXTRA_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "light_conditions",
    "junction_detail",
    "road_surface_conditions",
    "urban_or_rural_area",
    "first_road_class",
    "junction_control",
    "trunk_road_flag",
    "carriageway_hazards",
    "number_of_vehicles",
    "any_motorcycle",
    "any_pedal_cycle",
    "any_hgv_or_bus",
    "age_of_vehicle_max",
    "age_of_vehicle_max_missing",
    "driver_age_band_youngest",
    "driver_age_band_youngest_missing",
]

CATEGORICAL_FEATURES = [
    "weather_conditions",
    "road_type",
    "light_conditions",
    "road_surface_conditions",
    "junction_detail",
    "urban_or_rural_area",
    "first_road_class",
    "junction_control",
    "trunk_road_flag",
    "carriageway_hazards",
    "most_vulnerable_vehicle_type",
]

NUMERICAL_FEATURES = [
    "speed_limit",
    "hour",
    "day_of_week",
    "month",
    "number_of_vehicles",
    "age_of_vehicle_max",
    "driver_age_band_youngest",
]

BINARY_INDICATOR_FEATURES = [
    "any_motorcycle",
    "any_pedal_cycle",
    "any_hgv_or_bus",
    "age_of_vehicle_max_missing",
    "driver_age_band_youngest_missing",
]

ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_INDICATOR_FEATURES

# ==============================================================================
# 2. USER-FACING OPTIONS & MAPPING FOR TERMINAL DEMO (predict.py)
# ==============================================================================
USER_VEHICLE_TYPES = [
    "Car",
    "Motorcycle",
    "Pedal cycle",
    "Van / Light Goods",
    "Bus or HGV",
    "Other / Unknown"
]

USER_SPEED_LIMITS = [20, 30, 40, 50, 60, 70]

USER_WEATHER_OPTIONS = {
    1: "Fine, no high winds",
    2: "Raining, no high winds",
    3: "Snowing, no high winds",
    4: "Fine + high winds",
    5: "Raining + high winds",
    6: "Snowing + high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown"
}

USER_ROAD_TYPE_OPTIONS = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown"
}

def map_user_vehicle_input(
    vehicle_type_name: str,
    age_of_vehicle: Optional[float] = None,
    driver_age_band: Optional[float] = None,
    number_of_vehicles: int = 2
) -> Dict[str, Any]:
    """
    Maps a user-friendly vehicle type string (e.g. from predict.py terminal prompt)
    into the exact collision-level vehicle interaction features:
      - any_motorcycle
      - any_pedal_cycle
      - any_hgv_or_bus
      - most_vulnerable_vehicle_type
      - age_of_vehicle_max & age_of_vehicle_max_missing
      - driver_age_band_youngest & driver_age_band_youngest_missing
    """
    v_clean = vehicle_type_name.strip().lower()
    
    any_moto = 0
    any_pedal = 0
    any_hgv = 0
    vuln_type = "car_taxi"
    
    if "pedal" in v_clean or "cycle" in v_clean or "bicycle" in v_clean:
        any_pedal = 1
        vuln_type = "pedal_cycle"
    elif "moto" in v_clean or "bike" in v_clean:
        any_moto = 1
        vuln_type = "motorcycle"
    elif "bus" in v_clean or "hgv" in v_clean or "truck" in v_clean or "coach" in v_clean:
        any_hgv = 1
        vuln_type = "hgv_or_bus"
    elif "van" in v_clean or "goods" in v_clean:
        vuln_type = "van"
    elif "car" in v_clean or "taxi" in v_clean:
        vuln_type = "car_taxi"
    else:
        vuln_type = "other_unknown"
        
    # Pedal cycles structurally have no vehicle registration age
    if any_pedal == 1 and age_of_vehicle is None:
        veh_age = np.nan
        veh_age_missing = 1
    elif age_of_vehicle is not None:
        veh_age = float(age_of_vehicle)
        veh_age_missing = 0
    else:
        veh_age = np.nan
        veh_age_missing = 1
        
    if driver_age_band is not None:
        youngest_driver = float(driver_age_band)
        driver_missing = 0
    else:
        youngest_driver = np.nan
        driver_missing = 1
        
    return {
        "any_motorcycle": any_moto,
        "any_pedal_cycle": any_pedal,
        "any_hgv_or_bus": any_hgv,
        "most_vulnerable_vehicle_type": vuln_type,
        "age_of_vehicle_max": veh_age,
        "age_of_vehicle_max_missing": veh_age_missing,
        "driver_age_band_youngest": youngest_driver,
        "driver_age_band_youngest_missing": driver_missing,
        "number_of_vehicles": number_of_vehicles
    }


# ==============================================================================
# 3. UNIFIED PREPROCESSING PIPELINE (ColumnTransformer)
# ==============================================================================
def _clean_and_stringify_categoricals(X):
    """Fills NaNs with 'Unknown' and converts all values to strings uniformly."""
    df = pd.DataFrame(X).copy()
    return df.fillna("Unknown").astype(str)

def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a reusable, leakage-free sklearn ColumnTransformer:
      - Categoricals: explicit 'Unknown' category -> OneHotEncoder(handle_unknown='ignore')
      - Numerics: SimpleImputer(median) -> StandardScaler()
      - Binary indicators: Passthrough
    """
    # 1. Categorical pipeline: explicit 'Unknown' category for any missing codes
    cat_pipeline = Pipeline(steps=[
        ("clean_str", FunctionTransformer(_clean_and_stringify_categoricals)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # 2. Numerical pipeline: median imputation + standard scaling
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_INDICATOR_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor


# ==============================================================================
# 4. TIME-BASED & REGION GENERALIZATION SPLITTING
# ==============================================================================
def get_time_based_split(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the master dataset into:
      - Train: 2021 and 2022
      - Val: 2022 (used for temporal cross-validation / early stopping)
      - Test: 2023 (unseen holdout year testing COVID distribution shift)
    """
    train_df = df[df["collision_year"].isin([2021, 2022])].copy()
    val_df = df[df["collision_year"] == 2022].copy()
    test_df = df[df["collision_year"] == 2023].copy()
    
    return train_df, val_df, test_df


def get_region_generalization_split(
    df: pd.DataFrame, 
    test_police_forces: List[int] = [1, 99]  # London Met (1) and Unallocated (99) as holdout regions
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset by police_force area to test regional spatial generalization:
      - Train: collisions in standard police forces
      - Test: collisions in completely unseen police force jurisdictions
    """
    train_reg = df[~df["police_force"].isin(test_police_forces)].copy()
    test_reg = df[df["police_force"].isin(test_police_forces)].copy()
    
    return train_reg, test_reg

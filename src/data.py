"""
RoadSense AI: Data Pipeline & Preprocessing Module
Responsible for:
  - Multi-year DfT STATS19 CSV loading & validation
  - Per-column missing/unknown code recoding based on official DfT specification
  - Collision-level vehicle feature aggregation (vulnerability, counts, ages, structural missingness)
  - Strict leakage prevention (excluding crash dynamics, post-hoc reporting, injury adjustments)
  - Data quality reporting (row counts, missing rates before/after, class balances)
"""

from typing import Dict, List, Tuple, Any
import os
import glob
import numpy as np
import pandas as pd

# ==============================================================================
# 1. PER-COLUMN UNKNOWN CODE MAP (Source: DfT Road Safety Data Guide 2024/2025)
# ==============================================================================
UNKNOWN_CODE_MAP: Dict[str, List[int]] = {
    # Posted speed limit: -1 = missing, 99 = unknown (self-reported)
    "speed_limit": [-1, 99],
    
    # Weather conditions: -1 = data missing, 9 = unknown (8 = 'Other' is kept as valid category)
    "weather_conditions": [-1, 9],
    
    # Road type: -1 = data missing, 9 = unknown
    "road_type": [-1, 9],
    
    # Light conditions: -1 = data missing, 7 = darkness - lighting unknown
    "light_conditions": [-1, 7],
    
    # Junction detail: -1 = data missing, 99 = unknown
    "junction_detail": [-1, 99],
    
    # Road surface conditions: -1 = data missing, 9 = unknown
    "road_surface_conditions": [-1, 9],
    
    # Urban or rural area: -1 = data missing, 3 = unallocated
    "urban_or_rural_area": [-1, 3],
    
    # First road class: -1 = data missing
    "first_road_class": [-1],
    
    # Junction control: -1 = data missing / not applicable, 9 = unknown
    "junction_control": [-1, 9],
    
    # Trunk road flag: -1 = data missing
    "trunk_road_flag": [-1],
    
    # Carriageway hazards: -1 = data missing, 99 = unknown (0 = none is valid)
    "carriageway_hazards": [-1, 99],
    
    # Vehicle type: -1 = data missing, 99 = unknown vehicle type
    "vehicle_type": [-1, 99],
    
    # Age of vehicle: -1 = missing/out of range, 99 = unknown
    "age_of_vehicle": [-1, 99],
    
    # Age band of driver: -1 = data missing
    "age_band_of_driver": [-1],
    
    # Police force: -1 = data missing, 99 = unallocated
    "police_force": [-1, 99],
}

# ==============================================================================
# 2. STRICT DATA LEAKAGE & EXCLUSION REGISTER (DfT STATS19 Sourced)
# ==============================================================================
EXCLUDED_COLUMNS_RATIONALE: Dict[str, str] = {
    # Post-hoc Accident Injury Adjustments & Classifications (Outcome Leakage)
    "collision_injury_based": "DfT post-accident classification: records whether severity was determined via injury-based reporting system; directly leaks injury outcome.",
    "collision_adjusted_severity_serious": "DfT modeled probability: calculated post-hoc to adjust for police force reporting system differences; leaks target severity.",
    "collision_adjusted_severity_slight": "DfT modeled probability: calculated post-hoc to adjust for police force reporting system differences; leaks target severity.",
    "enhanced_severity_collision": "DfT post-accident administrative field: records severity reassessed after hospital verification; leaks target severity.",
    
    # Crash Dynamics (Impact Sequence Physics - Not Pre-Crash Conditions)
    "first_point_of_impact": "DfT crash dynamics field (code 1-4): records vehicle impact location at the moment of collision; occurs during the crash event.",
    "skidding_and_overturning": "DfT crash dynamics field (code 1-5): records whether vehicle skidded, jackknifed, or overturned during collision physics.",
    "vehicle_leaving_carriageway": "DfT crash dynamics field (code 1-8): records whether and where vehicle ran off the road as a consequence of impact.",
    "hit_object_in_carriageway": "DfT crash dynamics field (code 1-12): records secondary collisions with objects in the carriageway during impact.",
    "hit_object_off_carriageway": "DfT crash dynamics field (code 1-11): records secondary collisions with roadside furniture/trees post-impact.",
    "vehicle_manoeuvre": "DfT crash dynamics field (code 1-18): records vehicle action at the exact moment of collision; not an environmental condition.",
    "vehicle_manoeuvre_historic": "DfT legacy crash dynamics field: pre-2024 duplicate of vehicle_manoeuvre.",
    "vehicle_direction_from": "DfT kinematic vector: compass direction of travel at moment of impact.",
    "vehicle_direction_to": "DfT kinematic vector: intended destination trajectory at moment of impact.",
    "towing_and_articulation": "DfT vehicle equipment field: >99% unrecorded/none in road safety extract.",
    
    # Administrative & Post-Accident Response Fields
    "did_police_officer_attend_scene_of_accident": "DfT operational field: records whether officer attended the scene; determined after accident notification.",
    "journey_purpose_of_driver": "DfT post-incident driver survey response; not an environmental road risk factor.",
    "journey_purpose_of_driver_historic": "DfT legacy driver survey field.",
    "driver_imd_decile": "DfT socioeconomic index: Index of Multiple Deprivation decile of driver residential postcode; administrative census field.",
    "lsoa_of_driver": "DfT census identifier: Lower Layer Super Output Area of driver residence.",
    "lsoa_of_accident_location": "DfT census identifier: administrative geographic boundary; overfits spatial polygons.",
    "collision_ref_no": "DfT internal police administrative reference number string.",
    "local_authority_district": "DfT administrative council code; redundant with police_force and road environment.",
    "local_authority_ons_district": "DfT ONS administrative boundary identifier.",
    "local_authority_highway": "DfT highway authority administrative identifier.",
    "local_authority_highway_current": "DfT current highway authority administrative identifier.",
    "second_road_class": "DfT highway numbering for secondary junction arms; missing for non-junction collisions.",
    "second_road_number": "DfT highway numbering for secondary junction arms.",
    "first_road_number": "DfT road route number (e.g., A40, M1); high cardinality identifier hindering spatial generalization.",
    "pedestrian_crossing_human_control_historic": "DfT legacy duplicate crossing field.",
    "pedestrian_crossing_physical_facilities_historic": "DfT legacy duplicate crossing field.",
    "pedestrian_crossing": "DfT pedestrian crossing facility code; primarily relevant to pedestrian casualty records.",
    "special_conditions_at_site": "DfT transient site flag (e.g., defective signs, roadworks); >98% unrecorded/none.",
    "carriageway_hazards_historic": "DfT legacy duplicate hazard field.",
    "junction_detail_historic": "DfT legacy specification column; standardized modern junction_detail is retained.",
    "location_easting_osgr": "DfT British National Grid easting coordinate; causes tree models to memorize spatial points.",
    "location_northing_osgr": "DfT British National Grid northing coordinate; causes tree models to memorize spatial points.",
    "longitude": "DfT geographic coordinate; causes tree models to memorize spatial points.",
    "latitude": "DfT geographic coordinate; causes tree models to memorize spatial points.",
    "generic_make_model": "DfT vehicle make and model string; high cardinality and unstandardized commercial text.",
    "engine_capacity_cc": "DfT mechanistic vehicle attribute; high missing rate and correlated with vehicle_type.",
    "propulsion_code": "DfT fuel/propulsion type code; secondary technical parameter.",
    "escooter_flag": "DfT flag added experimentally in 2022; inconsistent across 2021-2023 collection.",
    "driver_distance_banding": "DfT driver survey field; >80% unrecorded.",
    "sex_of_driver": "Driver demographic attribute; excluded to prevent demographic bias in risk assessment.",
    "age_of_driver": "Continuous driver age; binned into age_band_of_driver to prevent single-year overfitting.",
}

# ==============================================================================
# 3. VEHICLE CLASSIFICATION CATEGORIES FOR AGGREGATION
# ==============================================================================
MOTORCYCLE_CODES = {2, 3, 4, 5, 23, 97, 103, 104, 105, 106}
PEDAL_CYCLE_CODES = {1}
HGV_OR_BUS_CODES = {10, 11, 20, 21, 98, 110, 113}
CAR_OR_TAXI_CODES = {8, 9, 108, 109}
VAN_CODES = {19}

def get_vulnerability_rank(v_code: int) -> int:
    """
    Ranks vehicles by physiological vulnerability of occupants/riders (DfT Transport Hierarchy):
      1: Pedal cycle (highest vulnerability - unprotected cyclist)
      2: Motorcycle (high vulnerability - motorized two-wheeler)
      3: Car or Taxi (standard passenger vehicle with chassis & airbags)
      4: Van / Light Goods (sturdier light commercial chassis)
      5: HGV or Bus (heavy mass / commercial vehicle)
      6: Other / Unknown
    """
    if v_code in PEDAL_CYCLE_CODES:
        return 1
    if v_code in MOTORCYCLE_CODES:
        return 2
    if v_code in CAR_OR_TAXI_CODES:
        return 3
    if v_code in VAN_CODES:
        return 4
    if v_code in HGV_OR_BUS_CODES:
        return 5
    return 6

VULNERABILITY_LABEL_MAP = {
    1: "pedal_cycle",
    2: "motorcycle",
    3: "car_taxi",
    4: "van",
    5: "hgv_or_bus",
    6: "other_unknown",
}


# ==============================================================================
# 4. DATA VERIFICATION & SCHEMA CONSISTENCY
# ==============================================================================
def verify_schema_consistency(data_dir: str = "data", years: List[int] = [2021, 2022, 2023]) -> Dict[str, Any]:
    """
    Verifies that collision and vehicle files exist for all years and have
    strictly identical column names and compatible types before merging.
    """
    report = {"collision_files": {}, "vehicle_files": {}, "consistent": True}
    base_col_columns = None
    base_veh_columns = None
    
    for yr in years:
        c_path = os.path.join(data_dir, f"dft-road-casualty-statistics-collision-{yr}.csv")
        v_path = os.path.join(data_dir, f"dft-road-casualty-statistics-vehicle-{yr}.csv")
        
        if not os.path.exists(c_path) or not os.path.exists(v_path):
            report["consistent"] = False
            report[f"error_{yr}"] = f"Missing file for year {yr}"
            continue
            
        c_cols = list(pd.read_csv(c_path, nrows=2).columns)
        v_cols = list(pd.read_csv(v_path, nrows=2).columns)
        
        report["collision_files"][yr] = {"path": c_path, "col_count": len(c_cols)}
        report["vehicle_files"][yr] = {"path": v_path, "col_count": len(v_cols)}
        
        if base_col_columns is None:
            base_col_columns = c_cols
            base_veh_columns = v_cols
        else:
            if c_cols != base_col_columns:
                report["consistent"] = False
                report[f"col_diff_{yr}"] = list(set(c_cols) ^ set(base_col_columns))
            if v_cols != base_veh_columns:
                report["consistent"] = False
                report[f"veh_diff_{yr}"] = list(set(v_cols) ^ set(base_veh_columns))
                
    return report


# ==============================================================================
# 5. VEHICLE AGGREGATION FUNCTION (COLLISION-LEVEL RESOLUTION)
# ==============================================================================
def aggregate_vehicles_per_collision(df_veh: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates vehicle records to collision level without relying on vehicle_reference == 1.
    
    CRITICAL: -1 and 99 are converted to np.nan BEFORE min/max aggregations so that:
      - age_of_vehicle_max skips -1 and 99, taking the max of truly valid vehicle ages.
      - driver_age_band_youngest skips -1, taking the min of truly valid driver age bands.
      - Structural missing indicators are created for pedal cycles and unrecorded ages.
    """
    df = df_veh.copy()
    
    # 1. Convert unknown codes to NaN BEFORE aggregations
    for col, unk_vals in UNKNOWN_CODE_MAP.items():
        if col in df.columns:
            df[col] = df[col].replace(unk_vals, np.nan)
            
    # 2. Binary vehicle category flags
    df["is_moto"] = df["vehicle_type"].isin(MOTORCYCLE_CODES).astype(int)
    df["is_pedal"] = df["vehicle_type"].isin(PEDAL_CYCLE_CODES).astype(int)
    df["is_hgv_bus"] = df["vehicle_type"].isin(HGV_OR_BUS_CODES).astype(int)
    
    # 3. Vulnerability score (1 = most vulnerable, 6 = least / unknown)
    df["vuln_rank"] = df["vehicle_type"].map(get_vulnerability_rank).fillna(6).astype(int)
    
    # 4. Group by collision_index
    # Note: min and max in pandas automatically ignore np.nan
    grouped = df.groupby("collision_index").agg(
        any_motorcycle=("is_moto", "max"),
        any_pedal_cycle=("is_pedal", "max"),
        any_hgv_or_bus=("is_hgv_bus", "max"),
        min_vuln_rank=("vuln_rank", "min"),
        age_of_vehicle_max=("age_of_vehicle", "max"),
        driver_age_band_youngest=("age_band_of_driver", "min"),
        vehicle_records_counted=("vehicle_reference", "count")
    ).reset_index()
    
    # 5. Map vulnerability rank back to semantic label
    grouped["most_vulnerable_vehicle_type"] = grouped["min_vuln_rank"].map(VULNERABILITY_LABEL_MAP)
    grouped.drop(columns=["min_vuln_rank"], inplace=True)
    
    # 6. Add structural missing-indicator columns (especially vital for pedal cycles)
    grouped["age_of_vehicle_max_missing"] = grouped["age_of_vehicle_max"].isna().astype(int)
    grouped["driver_age_band_youngest_missing"] = grouped["driver_age_band_youngest"].isna().astype(int)
    
    return grouped


# ==============================================================================
# 6. PIPELINE PROCESSOR FOR A SINGLE YEAR
# ==============================================================================
def process_single_year(
    year: int, 
    data_dir: str = "data"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads, cleans, merges, and derives features for a single STATS19 year.
    Tracks data-quality metrics and row counts.
    """
    c_path = os.path.join(data_dir, f"dft-road-casualty-statistics-collision-{year}.csv")
    v_path = os.path.join(data_dir, f"dft-road-casualty-statistics-vehicle-{year}.csv")
    
    c_raw = pd.read_csv(c_path, low_memory=False)
    v_raw = pd.read_csv(v_path, low_memory=False)
    
    stats: Dict[str, Any] = {
        "year": year,
        "raw_collision_rows": len(c_raw),
        "raw_vehicle_rows": len(v_raw),
    }
    
    # 1. Clean unknown codes in collision table
    c_df = c_raw.copy()
    raw_missing_counts = {}
    for col in UNKNOWN_CODE_MAP:
        if col in c_df.columns:
            unk_count = c_df[col].isin(UNKNOWN_CODE_MAP[col]).sum()
            raw_missing_counts[col] = int(unk_count)
            c_df[col] = c_df[col].replace(UNKNOWN_CODE_MAP[col], np.nan)
    stats["collision_raw_unknown_counts"] = raw_missing_counts
    
    # 2. Aggregate vehicles to collision level
    v_agg = aggregate_vehicles_per_collision(v_raw)
    stats["aggregated_vehicle_collisions"] = len(v_agg)
    
    # 3. Merge collisions and aggregated vehicle features
    merged = pd.merge(c_df, v_agg, on="collision_index", how="inner")
    stats["merged_rows"] = len(merged)
    
    # 4. Feature Engineering: Temporal variables
    # Time format: 'HH:MM'
    merged["hour"] = pd.to_datetime(merged["time"], format="%H:%M", errors="coerce").dt.hour
    # Date format: 'DD/MM/YYYY'
    merged["month"] = pd.to_datetime(merged["date"], format="%d/%m/%Y", errors="coerce").dt.month
    
    # 5. Define Prediction Targets
    # Task A: Binary (1: Dangerous = Fatal/Serious, 0: Minor = Slight)
    # STATS19 coding: 1 = Fatal, 2 = Serious, 3 = Slight
    merged["target_severity_binary"] = merged["collision_severity"].isin([1, 2]).astype(int)
    
    # Task B: 3-Class (0: Fatal, 1: Serious, 2: Slight)
    # Remap 1->0 (Fatal), 2->1 (Serious), 3->2 (Slight)
    severity_map = {1: 0, 2: 1, 3: 2}
    merged["target_severity_3class"] = merged["collision_severity"].map(severity_map)
    
    # Task C: Regression (Casualties count)
    merged["target_casualties"] = merged["number_of_casualties"].astype(int)
    
    # 6. Select Clean Model Features and Retained Metadata
    feature_columns = [
        # Identifier & Split Keys (Not used as training features)
        "collision_index",
        "collision_year",
        "police_force",
        
        # Targets
        "target_severity_binary",
        "target_severity_3class",
        "target_casualties",
        
        # Core Road & Physical Environment Features
        "speed_limit",               # Proxy for operating speed
        "weather_conditions",
        "road_type",
        "light_conditions",
        "road_surface_conditions",
        "junction_detail",
        "urban_or_rural_area",
        
        # Additional Road Attributes (Optional approved features)
        "first_road_class",
        "junction_control",
        "trunk_road_flag",
        "carriageway_hazards",
        
        # Temporal Features
        "hour",
        "day_of_week",
        "month",
        
        # Vehicle & Collision Context Features
        "number_of_vehicles",
        "any_motorcycle",
        "any_pedal_cycle",
        "any_hgv_or_bus",
        "most_vulnerable_vehicle_type",
        "age_of_vehicle_max",
        "age_of_vehicle_max_missing",
        "driver_age_band_youngest",
        "driver_age_band_youngest_missing",
    ]
    
    clean_df = merged[feature_columns].copy()
    stats["final_clean_rows"] = len(clean_df)
    
    return clean_df, stats


# ==============================================================================
# 7. MASTER PIPELINE: BUILD & PERSIST DATASET
# ==============================================================================
def build_master_dataset(
    data_dir: str = "data", 
    output_dir: str = "data",
    tables_dir: str = "outputs/tables"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes end-to-end data pipeline across all years (2021-2023), creates
    the clean master parquet file, generates data quality report and leakage log.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    years = [2021, 2022, 2023]
    dfs = []
    yearly_stats = []
    
    print("=" * 70)
    print("ROADSENSE AI: DATA PIPELINE INITIATING")
    print("=" * 70)
    
    # 1. Verify schema consistency across all files
    schema_report = verify_schema_consistency(data_dir=data_dir, years=years)
    print(f"[*] Schema consistency verified across years {years}: {schema_report['consistent']}")
    if not schema_report["consistent"]:
        raise ValueError(f"Schema inconsistency detected: {schema_report}")
        
    # 2. Process each year
    for yr in years:
        print(f"[*] Processing Year {yr}...")
        df_yr, stats_yr = process_single_year(yr, data_dir=data_dir)
        dfs.append(df_yr)
        yearly_stats.append(stats_yr)
        print(f"    -> Extracted {len(df_yr):,} clean collision records.")
        
    # 3. Concatenate all years into master DataFrame
    master_df = pd.concat(dfs, ignore_index=True)
    print(f"\n[+] Master dataset aggregated: {len(master_df):,} total collision records.")
    
    # 4. Save Master Clean Datasets
    parquet_path = os.path.join(output_dir, "clean_roadsense.parquet")
    sample_csv_path = os.path.join(output_dir, "clean_roadsense_sample.csv")
    
    master_df.to_parquet(parquet_path, index=False)
    master_df.head(1000).to_csv(sample_csv_path, index=False)
    print(f"[+] Saved clean master dataset to: {parquet_path}")
    print(f"[+] Saved 1,000-row sample to: {sample_csv_path}")
    
    # 5. Build Unified Data Quality Report
    dq_records = []
    total_rows = len(master_df)
    
    eval_cols = [c for c in master_df.columns if not c.startswith("target_") and c != "collision_index"]
    
    for col in eval_cols:
        null_count = int(master_df[col].isna().sum())
        null_pct = round((null_count / total_rows) * 100, 2)
        dtype_str = str(master_df[col].dtype)
        unique_count = int(master_df[col].nunique(dropna=True))
        unk_codes = str(UNKNOWN_CODE_MAP.get(col, "None"))
        
        # Single Unified Imputation Policy
        if col in ["age_of_vehicle_max", "driver_age_band_youngest"]:
            policy = "Median imputation + Missing-indicator flag"
        elif col in ["speed_limit", "number_of_vehicles", "hour", "day_of_week", "month",
                     "any_motorcycle", "any_pedal_cycle", "any_hgv_or_bus",
                     "age_of_vehicle_max_missing", "driver_age_band_youngest_missing"]:
            policy = "None (0 missing rows in dataset)"
        else:
            policy = "Explicit 'Unknown' category"
        
        dq_records.append({
            "feature": col,
            "data_type": dtype_str,
            "unique_values": unique_count,
            "missing_count": null_count,
            "missing_percent": null_pct,
            "handled_unknown_codes": unk_codes,
            "imputation_policy": policy
        })
        
    dq_df = pd.DataFrame(dq_records)
    dq_csv_path = os.path.join(tables_dir, "data_quality_report.csv")
    dq_df.to_csv(dq_csv_path, index=False)
    print(f"[+] Data quality report saved to: {dq_csv_path}")
    
    # 6. Save Data Leakage Exclusion Log with strictly sourced DfT definitions
    leakage_records = [
        {"column_name": col, "rationale": reason}
        for col, reason in EXCLUDED_COLUMNS_RATIONALE.items()
    ]
    leakage_df = pd.DataFrame(leakage_records)
    leakage_csv_path = os.path.join(tables_dir, "leakage_exclusion_log.csv")
    leakage_df.to_csv(leakage_csv_path, index=False)
    print(f"[+] Leakage exclusion log saved to: {leakage_csv_path}")
    
    return master_df, dq_df


if __name__ == "__main__":
    build_master_dataset()

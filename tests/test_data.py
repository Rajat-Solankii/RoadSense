"""
RoadSense AI: Tests for Data Ingestion, Synthesis & MoRTH Database Module
"""

import os
import pytest
import pandas as pd
from src.features import ALL_MODEL_FEATURES, TARGET_BINARY, TARGET_3CLASS, TARGET_COUNT
from src.data import generate_synthetic_dataset, load_or_create_data, PARQUET_PATH, CSV_PATH


def test_generate_synthetic_dataset():
    """Verify synthetic generator generates valid bounded records with expected columns."""
    df = generate_synthetic_dataset(n_samples=100, random_state=42)

    assert len(df) == 100
    for col in ALL_MODEL_FEATURES:
        assert col in df.columns, f"Missing feature: {col}"

    assert TARGET_BINARY in df.columns
    assert TARGET_3CLASS in df.columns
    assert TARGET_COUNT in df.columns

    # Check target constraints
    assert set(df[TARGET_BINARY].unique()).issubset({0, 1})
    assert set(df[TARGET_3CLASS].unique()).issubset({0, 1, 2})
    assert (df[TARGET_COUNT] >= 1).all()

    # Check speed limits are valid
    assert (df["speed_limit_kmh"] >= 20).all()
    assert (df["speed_limit_kmh"] <= 120).all()


def test_load_or_create_data():
    """Verify loading from parquet works as expected."""
    df = load_or_create_data(force_recreate=False)
    assert df is not None
    assert len(df) > 0
    assert os.path.exists(PARQUET_PATH)


def test_morth_official_database_file():
    """Verify the official MoRTH statistics file exists and contains valid metrics."""
    morth_path = os.path.join("data", "morth_official_statistics.csv")
    assert os.path.exists(morth_path), "MoRTH official statistics CSV must exist"

    df = pd.read_csv(morth_path, comment="#")
    assert len(df) > 10
    assert "metric_name" in df.columns
    assert "year_2023" in df.columns
    assert "Two-Wheelers (Motorcycles / Scooters)" in df["metric_name"].values

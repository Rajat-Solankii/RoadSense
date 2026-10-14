"""
RoadSense AI: Tests for CLI Execution & Direct Prediction Arguments
"""

import subprocess
import sys
import pytest


def test_cli_quick():
    """Verify python predict.py --quick runs successfully and prints prediction summary."""
    cmd = [sys.executable, "predict.py", "--quick"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)

    assert result.returncode == 0
    assert "PROBABLE PREDICTION SUMMARY" in result.stdout
    assert "Probable Collision Outcome" in result.stdout
    assert "Disaster Risk Escalation" in result.stdout


def test_cli_preset():
    """Verify python predict.py --preset 1 runs successfully."""
    cmd = [sys.executable, "predict.py", "--preset", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)

    assert result.returncode == 0
    assert "Yamuna Expressway" in result.stdout or "Expressway" in result.stdout
    assert "PROBABLE PREDICTION SUMMARY" in result.stdout


def test_cli_custom_flags():
    """Verify python predict.py with custom flags runs successfully."""
    cmd = [
        sys.executable, "predict.py",
        "--road", "Expressway",
        "--weather", "Dense Winter Fog / Smog",
        "--speed", "120",
        "--actual-speed", "110",
        "--vehicle", "Car / Hatchback / Sedan"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)

    assert result.returncode == 0
    assert "PROBABLE PREDICTION SUMMARY" in result.stdout
    assert "Dense Winter Fog" in result.stdout

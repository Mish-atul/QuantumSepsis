"""
Quick test to verify Red Team denormalization fix
Run this locally before executing on GPU server
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.red_team import RedTeamAgent
from src.config import get_default_config

print("="*60)
print("RED TEAM DENORMALIZATION FIX TEST")
print("="*60)

# Create synthetic normalization stats
norm_stats = {
    "train_mean": {
        "heart_rate": 85.0,
        "sbp": 120.0,
        "dbp": 70.0,
        "map": 85.0,
        "temperature": 37.0,
        "resp_rate": 18.0,
        "spo2": 97.0,
        "gcs_total": 14.5,
        "lactate": 1.5,
        "wbc": 10.0,
        "creatinine": 1.0,
        "platelets": 200.0
    },
    "train_std": {
        "heart_rate": 15.0,
        "sbp": 20.0,
        "dbp": 12.0,
        "map": 15.0,
        "temperature": 0.8,
        "resp_rate": 5.0,
        "spo2": 3.0,
        "gcs_total": 2.0,
        "lactate": 1.0,
        "wbc": 5.0,
        "creatinine": 0.5,
        "platelets": 80.0
    }
}

# Test 1: Normal patient (raw values)
print("\nTest 1: Normal patient (raw values)")
print("-" * 60)
normal_raw = np.array([
    [80, 120, 70, 85, 37.0, 16, 98, 15, 1.0, 8.0, 0.9, 250],
    [82, 118, 72, 86, 37.0, 15, 97, 15, 1.1, 8.5, 0.9, 245],
    [78, 122, 68, 84, 36.8, 16, 98, 15, 1.0, 7.5, 1.0, 260],
    [81, 119, 71, 85, 36.9, 17, 97, 15, 1.0, 8.0, 0.9, 255],
    [79, 121, 69, 84, 37.1, 16, 98, 15, 0.9, 8.2, 0.9, 248],
    [80, 120, 70, 85, 37.0, 15, 98, 15, 1.0, 8.0, 0.9, 250],
], dtype=np.float64)

agent_raw = RedTeamAgent(use_normalized=False, norm_stats=None)
result_raw = agent_raw.evaluate(normal_raw)
print(f"Override: {result_raw.override_level}")
print(f"Active tripwires: {result_raw.n_active}")
print(f"Expected: WATCH (0 tripwires)")
assert result_raw.override_level == "WATCH", "FAILED: Should be WATCH"
print("✓ PASSED")

# Test 2: Normal patient (z-normalized, then denormalized)
print("\nTest 2: Normal patient (z-normalized)")
print("-" * 60)
# Z-normalize
means = np.array([norm_stats['train_mean'][f] for f in [
    'heart_rate', 'sbp', 'dbp', 'map', 'temperature', 'resp_rate',
    'spo2', 'gcs_total', 'lactate', 'wbc', 'creatinine', 'platelets'
]])
stds = np.array([norm_stats['train_std'][f] for f in [
    'heart_rate', 'sbp', 'dbp', 'map', 'temperature', 'resp_rate',
    'spo2', 'gcs_total', 'lactate', 'wbc', 'creatinine', 'platelets'
]])
normal_norm = (normal_raw - means) / stds

agent_norm = RedTeamAgent(use_normalized=True, norm_stats=norm_stats)
result_norm = agent_norm.evaluate(normal_norm)
print(f"Override: {result_norm.override_level}")
print(f"Active tripwires: {result_norm.n_active}")
print(f"Expected: WATCH (0 tripwires)")
assert result_norm.override_level == "WATCH", "FAILED: Should be WATCH"
print("✓ PASSED")

# Test 3: Septic patient (raw values)
print("\nTest 3: Septic patient (raw values)")
print("-" * 60)
septic_raw = np.array([
    [85, 115, 65, 80, 37.5, 18, 96, 15, 1.5, 10.0, 1.2, 200],
    [92, 110, 60, 75, 38.0, 20, 95, 15, 2.0, 12.0, 1.5, 180],
    [98, 105, 55, 72, 38.2, 22, 94, 14, 2.5, 14.0, 1.8, 160],
    [105, 100, 52, 68, 38.5, 24, 93, 14, 3.0, 16.0, 2.0, 140],
    [110, 95, 48, 65, 38.8, 26, 92, 13, 3.5, 18.0, 2.2, 120],
    [115, 90, 45, 62, 39.0, 28, 91, 13, 4.0, 20.0, 2.5, 100],
], dtype=np.float64)

result_septic_raw = agent_raw.evaluate(septic_raw)
print(f"Override: {result_septic_raw.override_level}")
print(f"Active tripwires: {result_septic_raw.n_active}")
print(f"Expected: CRITICAL (≥2 tripwires)")
assert result_septic_raw.override_level == "CRITICAL", "FAILED: Should be CRITICAL"
assert result_septic_raw.n_active >= 2, "FAILED: Should have ≥2 tripwires"
print("✓ PASSED")

# Test 4: Septic patient (z-normalized, then denormalized)
print("\nTest 4: Septic patient (z-normalized)")
print("-" * 60)
septic_norm = (septic_raw - means) / stds
result_septic_norm = agent_norm.evaluate(septic_norm)
print(f"Override: {result_septic_norm.override_level}")
print(f"Active tripwires: {result_septic_norm.n_active}")
print(f"Expected: CRITICAL (≥2 tripwires)")
assert result_septic_norm.override_level == "CRITICAL", "FAILED: Should be CRITICAL"
assert result_septic_norm.n_active >= 2, "FAILED: Should have ≥2 tripwires"
print("✓ PASSED")

# Test 5: Distribution test (100 random windows)
print("\nTest 5: Distribution test (100 random windows)")
print("-" * 60)
np.random.seed(42)
counts = {"WATCH": 0, "AMBER": 0, "CRITICAL": 0}

for i in range(100):
    # Generate random normalized window
    window = np.random.randn(6, 12) * 0.5  # Small variations around mean
    result = agent_norm.evaluate(window)
    counts[result.override_level] += 1

print(f"WATCH:    {counts['WATCH']}/100 ({counts['WATCH']}%)")
print(f"AMBER:    {counts['AMBER']}/100 ({counts['AMBER']}%)")
print(f"CRITICAL: {counts['CRITICAL']}/100 ({counts['CRITICAL']}%)")
print(f"Expected: Majority WATCH, some AMBER, few CRITICAL")

# Should NOT be 100% CRITICAL
assert counts['CRITICAL'] < 50, "FAILED: Too many CRITICAL alerts"
assert counts['WATCH'] > 30, "FAILED: Too few WATCH alerts"
print("✓ PASSED")

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
print("\nRed Team denormalization is working correctly!")
print("Safe to run on GPU server.")

"""
Module 4: Prediction Engine
Recursive 24-step forecasting using a trained model.
India-specific: uses IST-aware features.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from preprocessing import FEATURE_COLS


def build_feature_row(
    recent_demand: list,
    target_dt: datetime,
    rolling_7d_mean: float,
    rolling_7d_max: float,
    rolling_7d_std: float,
) -> np.ndarray:
    """Build one feature row matching FEATURE_COLS order."""
    hour        = target_dt.hour
    dow         = target_dt.weekday()
    month       = target_dt.month

    row = {
        "lag_1h":          recent_demand[-1],
        "lag_2h":          recent_demand[-2],
        "lag_3h":          recent_demand[-3],
        "lag_24h":         recent_demand[-24],
        "lag_48h":         recent_demand[-48],
        "lag_72h":         recent_demand[-72],
        "lag_96h":         recent_demand[-96],
        "rolling_7d_mean": rolling_7d_mean,
        "rolling_7d_max":  rolling_7d_max,
        "rolling_7d_std":  rolling_7d_std,
        "hour_sin":        np.sin(2 * np.pi * hour / 24),
        "hour_cos":        np.cos(2 * np.pi * hour / 24),
        "month_sin":       np.sin(2 * np.pi * month / 12),
        "month_cos":       np.cos(2 * np.pi * month / 12),
        "is_weekend":      int(dow >= 5),
        "day_of_week":     dow,
        "is_morning_peak": int(7 <= hour <= 10),
        "is_evening_peak": int(18 <= hour <= 22),
        "is_summer":       int(month in [4, 5, 6]),
        "is_winter":       int(month in [11, 12, 1]),
        "is_monsoon":      int(month in [7, 8, 9]),
    }

    return np.array([[row[f] for f in FEATURE_COLS]])


def predict_24h(model, recent_demand: list, start_datetime: datetime) -> List[dict]:
    """
    Recursively forecast next 24 hours.
    recent_demand must have at least 168 values (7 days).
    """
    if len(recent_demand) < 168:
        raise ValueError(f"Need ≥168 hours of history, got {len(recent_demand)}")

    buffer = list(recent_demand)
    rolling_7d_mean = float(np.mean(buffer[-168:]))
    rolling_7d_max  = float(np.max(buffer[-168:]))
    rolling_7d_std  = float(np.std(buffer[-168:]))

    results = []
    for step in range(24):
        target_dt = start_datetime + timedelta(hours=step)
        features  = build_feature_row(buffer, target_dt, rolling_7d_mean, rolling_7d_max, rolling_7d_std)
        predicted = float(model.predict(features)[0])
        predicted = max(0.0, round(predicted, 1))

        buffer.append(predicted)
        results.append({
            "hour":                step,
            "timestamp":          target_dt.isoformat(),
            "label":              f"{step:02d}:00",
            "predicted_demand_mw": predicted,
        })

    return results


def get_recent_from_csv(filepath: str, region_col: str = "demand_mw", hours: int = 168) -> list:
    """Read last N hours of a region column from demand.csv."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if region_col not in df.columns:
        raise ValueError(f"Column '{region_col}' not in {filepath}")
    return df[region_col].dropna().tail(hours).tolist()
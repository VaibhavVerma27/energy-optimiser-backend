"""
Module 4: Prediction Engine
----------------------------
Recursive 24-hour demand forecasting.
Each predicted hour feeds back as input for the next hour (recursive strategy).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List


def build_single_feature_row(
    recent_demand: list,      # Most recent demand values (at least 168 entries)
    target_datetime: datetime,
    rolling_7d_mean: float,
    rolling_7d_max: float,
) -> np.ndarray:
    """
    Construct one feature row for prediction.
    Mirrors the feature engineering from preprocessing.py.
    """
    hour = target_datetime.hour
    month = target_datetime.month
    day_of_week = target_datetime.weekday()
    is_weekend = int(day_of_week >= 5)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Lag features — index from end of recent_demand list
    lag_1h  = recent_demand[-1]
    lag_2h  = recent_demand[-2]
    lag_3h  = recent_demand[-3]
    lag_24h = recent_demand[-24]
    lag_48h = recent_demand[-48]
    lag_72h = recent_demand[-72]
    lag_96h = recent_demand[-96]

    return np.array([[
        lag_1h, lag_2h, lag_3h,
        lag_24h, lag_48h, lag_72h, lag_96h,
        rolling_7d_mean, rolling_7d_max,
        hour_sin, hour_cos,
        month_sin, month_cos,
        is_weekend, day_of_week,
    ]])


def predict_24h(
    model,
    recent_demand: list,   # At least 168 historical hourly values (7 days)
    start_datetime: datetime,
) -> List[dict]:
    """
    Recursively forecast the next 24 hours.

    Args:
        model: Trained scikit-learn model
        recent_demand: List of recent hourly demand values (at least 168)
        start_datetime: The first hour to forecast (e.g., datetime(2024, 7, 1, 0, 0))

    Returns:
        List of dicts: [{hour, timestamp, predicted_demand_mw}, ...]
    """
    if len(recent_demand) < 168:
        raise ValueError("Need at least 168 hours (7 days) of historical demand to forecast.")

    demand_buffer = list(recent_demand)  # Copy so we don't mutate the original
    results = []

    rolling_7d_mean = float(np.mean(demand_buffer[-168:]))
    rolling_7d_max  = float(np.max(demand_buffer[-168:]))

    for step in range(24):
        target_dt = start_datetime + timedelta(hours=step)
        features = build_single_feature_row(
            demand_buffer, target_dt, rolling_7d_mean, rolling_7d_max
        )
        predicted = float(model.predict(features)[0])
        predicted = max(0, predicted)  # Clamp negative predictions

        demand_buffer.append(predicted)  # Feed prediction back as lag for next step

        results.append({
            "hour": step,
            "timestamp": target_dt.isoformat(),
            "label": target_dt.strftime("%H:00"),
            "predicted_demand_mw": round(predicted, 1),
        })

    return results
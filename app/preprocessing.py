"""
Module 1 & 2: Data Preprocessing + Feature Engineering
-------------------------------------------------------
- Cleans raw hourly demand data
- Extracts time-based features
- Engineers lag features and rolling statistics
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load CSV with columns: [timestamp, demand_mw]
    Cleans missing values and enforces hourly resolution.
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Fill missing demand values with linear interpolation
    df["demand_mw"] = df["demand_mw"].interpolate(method="linear")

    # Drop duplicates, keep first
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)

    # Resample to hourly if needed
    df = df.set_index("timestamp").resample("h").mean().interpolate().reset_index()

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Module 2a: Time-based features.
    Adds hour, day_of_week, weekend flag, and cyclic sin/cos encodings.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclic encoding — prevents the model from treating hour 23 and 0 as far apart
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Module 2b: Lag and rolling features.
    Short-term lags: t-1, t-2, t-3
    Daily lags:      t-24, t-48, t-72, t-96
    Weekly stats:    rolling 7-day mean and max
    """
    df = df.copy()

    # Short-term lags
    for lag in [1, 2, 3]:
        df[f"lag_{lag}h"] = df["demand_mw"].shift(lag)

    # Same-hour on previous days
    for lag in [24, 48, 72, 96]:
        df[f"lag_{lag}h"] = df["demand_mw"].shift(lag)

    # Rolling 7-day window (168 hours)
    df["rolling_7d_mean"] = df["demand_mw"].shift(1).rolling(window=168).mean()
    df["rolling_7d_max"] = df["demand_mw"].shift(1).rolling(window=168).max()

    # Drop rows with NaN from lag creation
    df = df.dropna().reset_index(drop=True)

    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Returns X (features) and y (target) ready for model training.
    """
    feature_cols = [
        "lag_1h", "lag_2h", "lag_3h",
        "lag_24h", "lag_48h", "lag_72h", "lag_96h",
        "rolling_7d_mean", "rolling_7d_max",
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "is_weekend", "day_of_week",
    ]
    X = df[feature_cols].values
    y = df["demand_mw"].values
    return X, y, feature_cols


def preprocess_pipeline(filepath: str):
    """Full preprocessing pipeline — load → clean → features → matrix."""
    df = load_and_clean(filepath)
    df = add_time_features(df)
    df = add_lag_features(df)
    X, y, cols = build_feature_matrix(df)
    return df, X, y, cols
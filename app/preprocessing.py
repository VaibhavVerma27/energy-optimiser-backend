"""
Module 1 & 2: Data Preprocessing + Feature Engineering
India-only. Reads demand.csv produced by prepare_dataset.py.

Columns expected in demand.csv:
  timestamp, demand_mw,
  Northern_Region_mw, Western_Region_mw, Eastern_Region_mw,
  Southern_Region_mw, NorthEastern_Region_mw
"""

import pandas as pd
import numpy as np

REGION_COLS = [
    "Northern_Region_mw",
    "Western_Region_mw",
    "Eastern_Region_mw",
    "Southern_Region_mw",
    "NorthEastern_Region_mw",
]


def load_and_clean(filepath: str, region_col: str = "demand_mw") -> pd.DataFrame:
    """
    Load demand.csv and return a clean two-column df: timestamp + demand_mw.
    region_col can be 'demand_mw' (national) or any '*_Region_mw' column.
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if region_col not in df.columns:
        available = [c for c in df.columns if "mw" in c.lower()]
        raise ValueError(
            f"Column '{region_col}' not found.\n"
            f"Available MW columns: {available}"
        )

    df["demand_mw"] = pd.to_numeric(df[region_col], errors="coerce")
    df["demand_mw"] = df["demand_mw"].interpolate(method="linear")
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    df = df.set_index("timestamp").resample("h").mean().interpolate().reset_index()

    print(f"[{region_col}] {len(df):,} rows | "
          f"min={df['demand_mw'].min():,.0f} MW | "
          f"max={df['demand_mw'].max():,.0f} MW | "
          f"mean={df['demand_mw'].mean():,.0f} MW")

    return df[["timestamp", "demand_mw"]]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Cyclic encoding for hour and month
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # India-specific peak window flags
    df["is_morning_peak"] = ((df["hour"] >= 7)  & (df["hour"] <= 10)).astype(int)
    df["is_evening_peak"] = ((df["hour"] >= 18) & (df["hour"] <= 22)).astype(int)

    # India seasonal flags
    df["is_summer"] = df["month"].isin([4, 5, 6]).astype(int)
    df["is_winter"] = df["month"].isin([11, 12, 1]).astype(int)
    df["is_monsoon"]= df["month"].isin([7, 8, 9]).astype(int)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Short-term lags (t-1, t-2, t-3)
    for lag in [1, 2, 3]:
        df[f"lag_{lag}h"] = df["demand_mw"].shift(lag)

    # Same hour on previous days (t-24, t-48, t-72, t-96)
    for lag in [24, 48, 72, 96]:
        df[f"lag_{lag}h"] = df["demand_mw"].shift(lag)

    # 7-day rolling statistics
    df["rolling_7d_mean"] = df["demand_mw"].shift(1).rolling(window=168).mean()
    df["rolling_7d_max"]  = df["demand_mw"].shift(1).rolling(window=168).max()
    df["rolling_7d_std"]  = df["demand_mw"].shift(1).rolling(window=168).std()

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "lag_1h", "lag_2h", "lag_3h",
    "lag_24h", "lag_48h", "lag_72h", "lag_96h",
    "rolling_7d_mean", "rolling_7d_max", "rolling_7d_std",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "is_weekend", "day_of_week",
    "is_morning_peak", "is_evening_peak",
    "is_summer", "is_winter", "is_monsoon",
]


def build_feature_matrix(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["demand_mw"].values
    return X, y, FEATURE_COLS


def preprocess_pipeline(filepath: str, region_col: str = "demand_mw"):
    """Full pipeline: load → time features → lag features → matrix."""
    df = load_and_clean(filepath, region_col=region_col)
    df = add_time_features(df)
    df = add_lag_features(df)
    X, y, cols = build_feature_matrix(df)
    return df, X, y, cols
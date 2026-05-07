"""
Module 4: Prediction Engine
Recursive 24-step forecasting with:
  - All 21 base features
  - Holiday/festival features (7)
  - Weather features (5) — used when available
  - Confidence intervals via Random Forest tree variance
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Optional

from preprocessing import (
    BASE_FEATURE_COLS, HOLIDAY_FEATURE_COLS, WEATHER_FEATURE_COLS,
    FESTIVAL_WINDOWS, _get_holidays,
)


# ── Holiday lookup helpers ────────────────────────────────────────────────────
def _build_date_sets(year: int):
    """Return (national_holidays, major_festivals, full_set) for a given year."""
    years = [year - 1, year, year + 1]
    full  = _get_holidays(years)

    national = set()
    try:
        import holidays as _hl
        for yr in years:
            for d, name in _hl.India(years=yr).items():
                if any(k in name for k in ["Republic","Independence","Gandhi","Buddha",
                                            "Ambedkar","Christmas","Eid","Diwali"]):
                    national.add(d)
    except ImportError:
        for yr in years:
            national.update([date(yr,1,26), date(yr,8,15), date(yr,10,2)])

    festivals = set()
    for fl in FESTIVAL_WINDOWS.values():
        for start, end in fl:
            d = start
            while d <= end:
                festivals.add(d)
                d += timedelta(days=1)

    pre_festival = set()
    for fl in [FESTIVAL_WINDOWS["diwali"]]:
        for start, _ in fl:
            for i in range(1, 4):
                pre_festival.add(start - timedelta(days=i))

    return national, festivals, pre_festival, full | national | festivals


def _holiday_features(target_dt: datetime, date_sets) -> dict:
    national, festivals, pre_fest, all_dates = date_sets
    d = target_dt.date()
    prev_d = d - timedelta(days=1)

    def days_to_next(d_):
        for i in range(8):
            if (d_ + timedelta(days=i)) in all_dates:
                return i
        return 7

    diwali_days = set()
    for start, end in FESTIVAL_WINDOWS["diwali"]:
        cur = start
        while cur <= end:
            diwali_days.add(cur)
            cur += timedelta(days=1)

    return {
        "is_national_holiday": int(d in national),
        "is_major_festival":   int(d in festivals),
        "is_regional_holiday": int(d in all_dates and d not in national),
        "days_to_next_holiday": days_to_next(d),
        "day_after_holiday":   int(prev_d in all_dates),
        "is_pre_festival":     int(d in pre_fest),
        "is_diwali_window":    int(d in diwali_days),
    }


# ── Feature row builder ───────────────────────────────────────────────────────
def build_feature_row(
    recent_demand: list,
    target_dt: datetime,
    rolling_7d_mean: float,
    rolling_7d_max: float,
    rolling_7d_std: float,
    feature_cols: List[str],
    date_sets=None,
    weather_row: Optional[dict] = None,
) -> np.ndarray:
    """Build one feature vector matching the trained model's feature_cols."""
    hour  = target_dt.hour
    dow   = target_dt.weekday()
    month = target_dt.month

    base = {
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

    # Holiday features
    if date_sets is not None:
        base.update(_holiday_features(target_dt, date_sets))

    # Weather features — use forecast weather if provided, else
    # fall back to climatological mean for the hour/month
    if weather_row is not None:
        base.update(weather_row)
    else:
        # Climatological fallbacks (India annual averages)
        base["weather_temp_c"]       = _clim_temp(hour, month)
        base["weather_humidity_pct"] = _clim_humidity(month)
        base["weather_solar_wm2"]    = _clim_solar(hour, month)
        base["weather_heat_index"]   = base["weather_temp_c"]
        base["weather_cdh"]          = max(0, base["weather_temp_c"] - 24)

    # Build vector strictly in the order the model was trained on
    row = []
    for f in feature_cols:
        if f not in base:
            raise KeyError(f"Feature '{f}' not available for inference. "
                           f"Re-train the model or check feature_cols.")
        row.append(base[f])

    return np.array([row])


# ── Climatological fallbacks (used when no weather data is supplied) ──────────
def _clim_temp(hour: int, month: int) -> float:
    """Monthly mean temperature at 12:00 IST for Delhi — rough climatology."""
    monthly_peak = {1:15, 2:18, 3:25, 4:33, 5:38, 6:36,
                    7:32, 8:31, 9:29, 10:26, 11:19, 12:14}
    diurnal_amp = {1:10, 2:11, 3:13, 4:14, 5:14, 6:12,
                   7:8,  8:8,  9:9,  10:11, 11:10, 12:9}
    peak_t = monthly_peak.get(month, 25)
    amp    = diurnal_amp.get(month, 10)
    # Temperature follows a shifted cosine (coolest ~05:00, warmest ~14:00)
    offset = np.cos(2 * np.pi * (hour - 14) / 24)
    return round(peak_t - amp * 0.5 * (1 - offset), 1)

def _clim_humidity(month: int) -> float:
    monthly = {1:60, 2:55, 3:45, 4:30, 5:25, 6:50,
               7:75, 8:78, 9:68, 10:52, 11:55, 12:62}
    return float(monthly.get(month, 55))

def _clim_solar(hour: int, month: int) -> float:
    if hour < 6 or hour > 18:
        return 0.0
    monthly_peak = {1:450, 2:520, 3:600, 4:680, 5:700, 6:580,
                    7:380, 8:370, 9:450, 10:520, 11:460, 12:420}
    bell = np.exp(-0.5 * ((hour - 12) / 3.0) ** 2)
    return round(float(monthly_peak.get(month, 500)) * bell, 1)


# ── Confidence interval extraction ───────────────────────────────────────────
def _tree_predictions(model, X: np.ndarray) -> np.ndarray:
    """Get predictions from every tree in the Random Forest."""
    return np.array([tree.predict(X)[0] for tree in model.estimators_])


def _confidence_interval(tree_preds: np.ndarray, level: float = 0.80):
    """Return (lower, upper) prediction interval at `level` confidence."""
    lo = (1 - level) / 2 * 100
    hi = (1 + level) / 2 * 100
    return float(np.percentile(tree_preds, lo)), float(np.percentile(tree_preds, hi))


# ── Main forecast function ────────────────────────────────────────────────────
def predict_24h(
    model,
    recent_demand: list,
    start_datetime: datetime,
    feature_cols: Optional[List[str]] = None,
    weather_forecast: Optional[List[dict]] = None,
    confidence: bool = True,
    ci_level: float = 0.80,
) -> List[dict]:
    """
    Recursively forecast the next 24 hours.

    Args:
        model:            Trained sklearn RandomForestRegressor.
        recent_demand:    List of >= 168 hourly MW values (oldest first).
        start_datetime:   First forecast hour (aware or naive UTC datetime).
        feature_cols:     Exact feature column list the model was trained on.
                          If None, falls back to BASE_FEATURE_COLS (21 features).
        weather_forecast: List of 24 dicts with weather for each forecast hour.
                          Keys: weather_temp_c, weather_humidity_pct,
                                weather_solar_wm2, weather_heat_index, weather_cdh
                          If None, uses climatological fallbacks.
        confidence:       If True, compute 80% prediction interval per hour.
        ci_level:         Confidence level for intervals (default 0.80 = 80%).

    Returns:
        List of 24 dicts, one per forecast hour.
    """
    if len(recent_demand) < 168:
        raise ValueError(f"Need ≥168 hours of history, got {len(recent_demand)}")

    if feature_cols is None:
        feature_cols = BASE_FEATURE_COLS

    buffer = list(recent_demand)
    rolling_7d_mean = float(np.mean(buffer[-168:]))
    rolling_7d_max  = float(np.max(buffer[-168:]))
    rolling_7d_std  = float(np.std(buffer[-168:]))

    # Build holiday date sets once for the forecast window
    has_holidays = any(f in feature_cols for f in ["is_national_holiday", "is_major_festival"])
    date_sets = None
    if has_holidays:
        year = start_datetime.year
        date_sets = _build_date_sets(year)

    has_weather = any(f in feature_cols for f in WEATHER_FEATURE_COLS)

    results = []
    for step in range(24):
        target_dt  = start_datetime + timedelta(hours=step)
        wx_row     = weather_forecast[step] if (weather_forecast and has_weather) else None
        # If weather is in feature_cols but no forecast provided, use climatology
        if has_weather and wx_row is None:
            wx_row = {
                "weather_temp_c":       _clim_temp(target_dt.hour, target_dt.month),
                "weather_humidity_pct": _clim_humidity(target_dt.month),
                "weather_solar_wm2":    _clim_solar(target_dt.hour, target_dt.month),
                "weather_heat_index":   _clim_temp(target_dt.hour, target_dt.month),
                "weather_cdh":          max(0, _clim_temp(target_dt.hour, target_dt.month) - 24),
            }

        features = build_feature_row(
            buffer, target_dt,
            rolling_7d_mean, rolling_7d_max, rolling_7d_std,
            feature_cols, date_sets, wx_row,
        )

        # Point prediction
        predicted = float(model.predict(features)[0])
        predicted = max(0.0, round(predicted, 1))

        # Confidence interval
        ci_lower, ci_upper = None, None
        if confidence and hasattr(model, "estimators_"):
            tree_preds = _tree_predictions(model, features)
            ci_lower, ci_upper = _confidence_interval(tree_preds, ci_level)
            ci_lower = max(0.0, round(ci_lower, 1))
            ci_upper = max(0.0, round(ci_upper, 1))

        buffer.append(predicted)

        hour_result = {
            "hour":                 step,
            "timestamp":            target_dt.isoformat(),
            "label":                f"{step:02d}:00",
            "predicted_demand_mw":  predicted,
        }
        if ci_lower is not None:
            hour_result["ci_lower_mw"] = ci_lower
            hour_result["ci_upper_mw"] = ci_upper
            hour_result["ci_level"]    = ci_level
        if wx_row:
            hour_result["weather_temp_c"]      = wx_row.get("weather_temp_c")
            hour_result["weather_humidity_pct"] = wx_row.get("weather_humidity_pct")

        results.append(hour_result)

    return results


def get_recent_from_csv(filepath: str, region_col: str = "demand_mw", hours: int = 168) -> list:
    """Read last N hours of a region column from demand.csv."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if region_col not in df.columns:
        raise ValueError(f"Column '{region_col}' not in {filepath}")
    return df[region_col].dropna().tail(hours).tolist()
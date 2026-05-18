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
    original_history: Optional[list] = None,
) -> np.ndarray:
    """Build one feature vector matching the trained model's feature_cols.

    original_history: The unmodified demand history BEFORE any recursive predictions
    were appended. When provided, long-range lags (24h, 48h, 72h, 96h) use it
    instead of the accumulated buffer. This prevents recursive error from flattening
    multi-step forecasts — the 24h-ago value stays anchored to real history.
    """
    hour  = target_dt.hour
    dow   = target_dt.weekday()
    month = target_dt.month

    # Use original_history for long-range lags if available
    hist_long = original_history if original_history is not None else recent_demand

    # Safe lag accessor — never crashes on short context
    def safe_lag(src, n):
        if len(src) >= n:
            return src[-n]
        elif len(src) >= 1:
            return src[-1]  # repeat last known value
        return 0.0

    base = {
        "lag_1h":          safe_lag(recent_demand, 1),
        "lag_2h":          safe_lag(recent_demand, 2),
        "lag_3h":          safe_lag(recent_demand, 3),
        "lag_24h":         safe_lag(hist_long, 24),
        "lag_48h":         safe_lag(hist_long, 48),
        "lag_72h":         safe_lag(hist_long, 72),
        "lag_96h":         safe_lag(hist_long, 96),
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
# ── Direct multi-step forecast (LightGBM + 24 horizon models) ────────────────
def predict_24h_direct(
    horizon_models: dict,
    recent_demand: list,
    start_datetime: datetime,
    feature_cols: List[str],
    weather_forecast: Optional[List[dict]] = None,
    confidence: bool = True,
    ci_level: float = 0.80,
) -> List[dict]:
    """
    Direct multi-step forecasting — calls 24 separate models, one per horizon.

    Eliminates recursive error accumulation: each horizon model was trained
    to predict that specific number of hours ahead from the same input features.

    Args:
        horizon_models:   dict {1: model_h1, 2: model_h2, ..., 24: model_h24}
        recent_demand:    list of >=168 hourly MW values (oldest first)
        start_datetime:   first forecast hour (IST)
        feature_cols:     feature list — must match training
        weather_forecast: list of 24 dicts with weather for each forecast hour
        confidence:       compute CI using prediction std across model uncertainty
        ci_level:         confidence level (default 80%)

    Returns:
        List of 24 dicts — same format as predict_24h.
    """
    if len(recent_demand) < 168:
        raise ValueError(f"Need ≥168 hours of history, got {len(recent_demand)}")
    if not horizon_models:
        raise ValueError("No horizon models provided")

    # Use the EXACT same features for all horizon predictions.
    # The lags (lag_1h, lag_24h, etc) all come from the actual recent history —
    # NO recursive prediction-feeding. Each horizon model handles its own offset.
    rolling_7d_mean = float(np.mean(recent_demand[-168:]))
    rolling_7d_max  = float(np.max(recent_demand[-168:]))
    rolling_7d_std  = float(np.std(recent_demand[-168:]))

    # Holiday date sets
    has_holidays = any(f in feature_cols for f in ["is_national_holiday", "is_major_festival"])
    date_sets = _build_date_sets(start_datetime.year) if has_holidays else None
    has_weather = any(f in feature_cols for f in WEATHER_FEATURE_COLS)

    results = []
    for step in range(24):
        target_dt = start_datetime + timedelta(hours=step)
        horizon   = step + 1   # h=1 predicts 1h ahead, etc.

        # Get the model for this horizon (fall back to nearest)
        model = horizon_models.get(horizon)
        if model is None:
            available = sorted(horizon_models.keys())
            nearest = min(available, key=lambda h: abs(h - horizon))
            model = horizon_models[nearest]

        # Weather for this hour
        wx_row = weather_forecast[step] if (weather_forecast and has_weather) else None
        if has_weather and wx_row is None:
            wx_row = {
                "weather_temp_c":       _clim_temp(target_dt.hour, target_dt.month),
                "weather_humidity_pct": _clim_humidity(target_dt.month),
                "weather_solar_wm2":    _clim_solar(target_dt.hour, target_dt.month),
                "weather_heat_index":   _clim_temp(target_dt.hour, target_dt.month),
                "weather_cdh":          max(0, _clim_temp(target_dt.hour, target_dt.month) - 24),
            }

        # Build feature row using ACTUAL recent_demand for ALL lags
        # (no buffer with predictions appended — that's the whole point of direct)
        features = build_feature_row(
            recent_demand, target_dt,
            rolling_7d_mean, rolling_7d_max, rolling_7d_std,
            feature_cols, date_sets, wx_row,
            original_history=recent_demand,   # same as buffer for direct mode
        )

        # Wrap as DataFrame so feature names match training (no sklearn warning)
        features_df = pd.DataFrame(features, columns=feature_cols)

        # Point prediction
        predicted = float(model.predict(features_df)[0])
        predicted = max(0.0, round(predicted, 1))

        # Confidence interval — for LightGBM use bootstrap-like estimate from
        # model's training MAE if available, else use std across nearby horizons
        ci_lower, ci_upper = None, None
        if confidence:
            # Heuristic: CI widens with horizon (longer = more uncertainty)
            # Base sigma: ~3% of rolling mean for h=1, scaling to ~6% at h=24
            base_sigma = rolling_7d_mean * 0.03
            sigma      = base_sigma * (1.0 + (horizon - 1) / 23.0)
            from scipy.stats import norm
            try:
                z = norm.ppf((1 + ci_level) / 2)
            except Exception:
                z = 1.282 if ci_level <= 0.80 else 1.96
            ci_lower = max(0.0, round(predicted - z * sigma, 1))
            ci_upper = max(0.0, round(predicted + z * sigma, 1))

        hour_result = {
            "hour":                 step,
            "horizon":              horizon,
            "timestamp":            target_dt.isoformat(),
            "label":                f"{step:02d}:00",
            "predicted_demand_mw":  predicted,
            "model_type":           "direct_lgbm",
        }
        if ci_lower is not None:
            hour_result["ci_lower_mw"] = ci_lower
            hour_result["ci_upper_mw"] = ci_upper
            hour_result["ci_level"]    = ci_level
        if wx_row:
            hour_result["weather_temp_c"]       = wx_row.get("weather_temp_c")
            hour_result["weather_humidity_pct"] = wx_row.get("weather_humidity_pct")

        results.append(hour_result)

    return results


# ── Recursive forecast (legacy fallback) ─────────────────────────────────────
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
    original_history = list(recent_demand)

    rolling_7d_mean = float(np.mean(buffer[-168:]))
    rolling_7d_max  = float(np.max(buffer[-168:]))
    rolling_7d_std  = float(np.std(buffer[-168:]))

    # Compute the diurnal pattern from the last 7 days of history
    # This gives us the average hourly shape (0.0-1.0) relative to daily peak
    # We use this to anchor long-step predictions to the correct hour-of-day level
    # rather than letting recursive lag_1h dominate and flatten the forecast
    # ── Diurnal shape computation ─────────────────────────────────────────────
    # Build the expected hourly shape from the history.
    # CRITICAL: only use hours with real measured variation, not interpolated.
    # We detect interpolation by checking if adjacent hours have identical or
    # perfectly linear values — real demand always has some noise.
    # If coverage is poor, fall back to the hardcoded India demand shape derived
    # from MERIT India data (May 2026).
    INDIA_HOURLY_SHAPE = {
        # Derived from May 2026 MERIT India actual data (multiple days)
        # trough/mean = 0.879 at 04:00, peak/mean = 1.092 at 15:00
        # swing = 21.3% of mean — reflects current India demand pattern
        0:0.9741, 1:0.9486, 2:0.9264, 3:0.9088, 4:0.8792, 5:0.8951,
        6:0.9051, 7:0.9022, 8:0.9341, 9:0.9893, 10:1.0152, 11:1.0289,
        12:1.0321, 13:1.0174, 14:1.0648, 15:1.0918, 16:1.0694, 17:1.0376,
        18:1.0182, 19:1.0638, 20:1.0601, 21:1.0543, 22:1.0683, 23:1.0584,
    }

    daily_shapes = {}
    for day in range(7):
        day_start = len(buffer) - 168 + day * 24
        day_vals  = buffer[day_start: day_start + 24]
        day_mean  = np.mean(day_vals)
        if day_mean <= 0:
            continue
        day_std = np.std(day_vals)
        # Require 2% variation — 0.5% was too low and let interpolated days through
        if day_std / day_mean < 0.02:
            continue
        for h, v in enumerate(day_vals):
            if h not in daily_shapes:
                daily_shapes[h] = []
            daily_shapes[h].append(v / day_mean)

    if len(daily_shapes) >= 18:
        hourly_shape = {h: float(np.mean(v)) for h, v in daily_shapes.items()}
    else:
        hourly_shape = INDIA_HOURLY_SHAPE

    if hourly_shape:
        swing = max(hourly_shape.values()) - min(hourly_shape.values())
        if swing < 0.10:  # less than 10% swing — still too flat, use default
            hourly_shape = INDIA_HOURLY_SHAPE

    # ── Build holiday date sets (needed for bias check and forecast loop) ────
    has_holidays = any(f in feature_cols for f in ["is_national_holiday", "is_major_festival"])
    has_weather  = any(f in feature_cols for f in WEATHER_FEATURE_COLS)
    date_sets = _build_date_sets(start_datetime.year) if has_holidays else None

    # ── Per-hour bias correction ──────────────────────────────────────────────
    # Computes per-hour correction factors from recent history by comparing
    # what the model predicts vs what actually happened at each hour of day.
    # CRITICAL: must skip interpolated buffer slots (gaps filled by merit_parser
    # with linear interpolation). Interpolated values have actual≈predicted so
    # ratio≈1.0, but real peak hours need ratio≈1.05-1.08. Averaging real and
    # interpolated samples dilutes the correction by ~50%.
    #
    # Detection: a buffer slot is interpolated if the 24h window around it
    # has std/mean < 0.5%. Real demand always has >0.5% hourly variation.
    hour_bias = {h: 1.0 for h in range(24)}
    try:
        hour_actuals = {h: [] for h in range(24)}
        hour_preds   = {h: [] for h in range(24)}

        # Determine which buffer indices contain REAL measured data vs interpolated.
        # We know real dates from start_datetime and the merit_history timestamps.
        # The merit_parser fills gaps between uploaded files with linear interpolation.
        # Interpolated slots have ratio≈1.0 (actual≈predicted) which dilutes the
        # bias factors. We skip them by checking the target datetime's date.
        #
        # Real dates = dates for which MERIT CSV files were uploaded.
        # We detect them: if a buffer slot has a large local std it's real,
        # but the most reliable method is to check per-slot variance vs neighbours.
        # We use a conservative check: slot is real if abs(val - linear_interp) > 500 MW
        # (linear interpolation between adjacent real anchors deviates < 500 MW from
        # itself, but real demand deviates thousands of MW from any linear fit).
        buf_arr = np.array(buffer)
        real_idx = set()
        for i in range(2, len(buffer) - 2):
            # Linear interpolation between i-2 and i+2
            linear_interp = (buf_arr[i-2] + buf_arr[i+2]) / 2
            # If actual value deviates significantly from linear interpolation, it's real
            if abs(buf_arr[i] - linear_interp) > 300:
                real_idx.add(i)
        # Always include the last 24 known values (definitely real — most recent upload)
        for i in range(max(0, len(buffer) - 24), len(buffer)):
            real_idx.add(i)

        for back in range(1, min(97, len(buffer))):
            idx    = len(buffer) - back
            ctx    = buffer[:idx]
            if len(ctx) < 24:
                break
            # Skip interpolated slots
            if idx not in real_idx:
                continue
            actual    = buffer[idx]
            target_dt = start_datetime - timedelta(hours=back)
            hour      = target_dt.hour

            ctx_arr = np.array(ctx[-168:] if len(ctx) >= 168 else ctx)
            c_mean  = float(ctx_arr.mean())
            c_max   = float(ctx_arr.max())
            c_std   = float(ctx_arr.std())

            feats = build_feature_row(
                list(ctx), target_dt,
                c_mean, c_max, c_std,
                feature_cols, date_sets, None,
                original_history=list(ctx),
            )
            pred = float(model.predict(feats)[0])
            if pred > 0 and actual > 0:
                hour_actuals[hour].append(actual)
                hour_preds[hour].append(pred)

        # Compute per-hour ratio from real samples only
        computed = {}
        for h in range(24):
            if len(hour_preds[h]) >= 1:
                ratio = float(np.mean(hour_actuals[h])) / float(np.mean(hour_preds[h]))
                computed[h] = float(np.clip(ratio, 0.85, 1.25))

        if computed:
            mean_r = float(np.mean(list(computed.values())))
            for h in range(24):
                hour_bias[h] = computed.get(h, float(np.clip(mean_r, 0.85, 1.25)))
            # DEBUG — print so we can verify in server logs
            print(f"[BIAS] real_idx={len(real_idx)}, computed={len(computed)} hours, "
                  f"mean_r={mean_r:.4f}")
            for h in [4, 9, 15, 19]:
                print(f"[BIAS] h={h:02d}: n={len(hour_preds.get(h,[]))}, "
                      f"actual_avg={np.mean(hour_actuals[h]) if hour_actuals[h] else 0:,.0f}, "
                      f"pred_avg={np.mean(hour_preds[h]) if hour_preds[h] else 0:,.0f}, "
                      f"factor={hour_bias[h]:.4f}")
        else:
            print("[BIAS] WARNING: no computed factors, using 1.0 for all hours")

    except Exception as e:
        print(f"[BIAS] EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        pass  # keep hour_bias = 1.0 defaults

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
            original_history=original_history,
        )

        # Point prediction
        predicted_raw = float(model.predict(features)[0])
        predicted_raw = max(0.0, predicted_raw)

        # Diurnal blend — forces the prediction to follow the real daily shape.
        # The model's raw output varies by only ~5,000 MW across 24h (it relies
        # heavily on lag_1h which stays near the mean). The actual India demand
        # swings 40,000+ MW. Without blending, predictions are nearly flat.
        # Blend weight: starts at step 1 (not 5) and ramps to 60% by step 24.
        # At 60% the shape drives the amplitude, the model drives the level.
        if hourly_shape:
            expected_for_hour = rolling_7d_mean * hourly_shape.get(target_dt.hour, 1.0)
            blend_weight = min(0.60, (step + 1) / 24 * 0.60)
            predicted_raw = predicted_raw * (1 - blend_weight) + expected_for_hour * blend_weight

        # Per-hour bias correction — peak hours get larger correction than troughs
        corrected_raw = predicted_raw * hour_bias.get(target_dt.hour, 1.0)
        if step in [0, 5, 10, 15, 20]:  # print key steps
            print(f"[PRED] step={step:02d} h={target_dt.hour:02d} "
                  f"raw={predicted_raw:,.0f} "
                  f"bias=×{hour_bias.get(target_dt.hour,1.0):.4f} "
                  f"corrected={corrected_raw:,.0f}")
        predicted_raw = corrected_raw

        predicted = max(0.0, round(predicted_raw, 1))

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
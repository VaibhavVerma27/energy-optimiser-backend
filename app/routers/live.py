"""
routers/live.py
===============
Endpoints for live forecasting and prediction storage/comparison.

GET  /api/live/weather                  — current weather for all 5 regions (Open-Meteo)
POST /api/live/forecast                 — run forecast with live weather, save prediction
GET  /api/live/predictions              — list saved prediction runs
GET  /api/live/predictions/{run_id}     — get full saved prediction
GET  /api/live/compare/{run_id}         — compare prediction vs NPP actuals
POST /api/live/fetch-actuals            — fetch NPP actual data for a date and save
GET  /api/live/rolling-performance      — rolling MAE over saved predictions
"""

import os
import sys
import uuid
from datetime import datetime, date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from live_data_fetcher import (
    fetch_all_regions_live_weather,
    fetch_all_regions_weather_for_date,
    fetch_actual_demand_for_date,
    get_recent_demand_from_csv_or_live,
    get_recent_demand_scale,
)
from prediction_store import (
    init_db, save_forecast, save_actuals,
    list_predictions, get_prediction, delete_prediction,
    compute_comparison, get_rolling_performance,
)
from capacity_engine import compute_dynamic_capacity
from predictor import predict_24h
from model import load_region_model_with_meta
from routers.upload import load_merit_history

def _clim_temp_for_month(month: int) -> float:
    """Monthly mean peak temperature for Delhi — used for demand scaling."""
    return {1:15,2:18,3:25,4:33,5:38,6:36,7:32,8:31,9:29,10:26,11:19,12:14}.get(month,28)

router = APIRouter()

DATA_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demand.csv")
MODELS_DIR  = os.path.dirname(os.path.dirname(__file__))
ALL_REGIONS = [
    "Northern_Region_mw", "Western_Region_mw", "Southern_Region_mw",
    "Eastern_Region_mw",  "NorthEastern_Region_mw",
]
REGION_ID_MAP = {
    "Northern_Region_mw":     "Northern_Region",
    "Western_Region_mw":      "Western_Region",
    "Southern_Region_mw":     "Southern_Region",
    "Eastern_Region_mw":      "Eastern_Region",
    "NorthEastern_Region_mw": "NorthEastern_Region",
}


# ── GET /api/live/weather ─────────────────────────────────────────────────────
@router.get("/weather")
async def get_live_weather(hours_ahead: int = Query(default=24, ge=1, le=72)):
    """
    Fetch real-time weather for all 5 region cities from Open-Meteo.
    Returns per-region hourly weather for the next hours_ahead hours.
    """
    try:
        weather = fetch_all_regions_live_weather(hours_ahead=hours_ahead)
        # Summarise for quick display
        summary = {}
        for region_id, hourly in weather.items():
            if hourly:
                now = hourly[0]
                summary[region_id] = {
                    "city":            _city_name(region_id),
                    "current_temp_c":  now.get("temp_c"),
                    "current_humidity":now.get("humidity_pct"),
                    "current_solar_wm2":now.get("solar_wm2"),
                    "current_wind_ms": now.get("wind_speed_ms"),
                    "hourly":          hourly,
                    "source":          "Open-Meteo forecast API",
                }
            else:
                summary[region_id] = {"error": "fetch_failed", "hourly": []}
        return {"status": "ok", "fetched_at": datetime.utcnow().isoformat(), "regions": summary}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Weather fetch failed: {e}")


# ── POST /api/live/forecast ───────────────────────────────────────────────────
class LiveForecastRequest(BaseModel):
    save: bool = True
    notes: str = ""
    hours_ahead: int = 24
    manual_demand: Optional[dict] = None
    # For specific-date mode: pass ISO date string e.g. "2026-05-10T00:00:00"
    start_datetime: Optional[str] = None

@router.post("/forecast")
async def run_live_forecast(req: LiveForecastRequest):
    """
    Run a 24h forecast using:
    - Last 168h of actual demand from demand.csv as model history
    - Live weather from Open-Meteo for capacity engine
    - Current IST time as forecast start
    - Saves result to SQLite predictions DB if req.save=True

    This is the "best possible forecast for right now" endpoint.
    """
    # 1. Fetch weather for the correct date
    # Live mode  → Open-Meteo forecast (next 24h from now)
    # Past date  → Open-Meteo archive (actual measurements for that day)
    # Future     → Open-Meteo forecast (NWP prediction for that day, up to 16 days)
    # Beyond 16d → climatological averages with a clear flag
    weather_source_label = "unknown"
    try:
        if req.start_datetime:
            # Specific-date mode — fetch weather for the target date
            target_date = datetime.fromisoformat(
                req.start_datetime.replace("Z","")
            ).date()
            live_weather, weather_source_label = fetch_all_regions_weather_for_date(
                target_date, hours=req.hours_ahead
            )
        else:
            # Live mode — fetch current forecast
            live_weather = fetch_all_regions_live_weather(hours_ahead=req.hours_ahead)
            weather_source_label = "open-meteo-live"
    except Exception as e:
        live_weather = {}
        weather_source_label = "fetch_failed"

    # 2. Forecast start — IST for live mode, or user-specified date
    if req.start_datetime:
        # Specific-date mode: use the provided date at midnight IST
        start_dt = datetime.fromisoformat(req.start_datetime.replace("Z",""))
    else:
        # Live mode: next full hour in IST
        now_utc  = datetime.utcnow()
        now_ist  = now_utc + timedelta(hours=5, minutes=30)
        start_dt = now_ist.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    forecast_date = start_dt.date().isoformat()

    region_results = {}
    all_india_pred     = [0.0] * req.hours_ahead
    all_india_cap      = [0.0] * req.hours_ahead
    all_india_ci_lower = [0.0] * req.hours_ahead
    all_india_ci_upper = [0.0] * req.hours_ahead
    all_india_solar    = [0.0] * req.hours_ahead
    all_india_wind     = [0.0] * req.hours_ahead
    all_india_hydro    = [0.0] * req.hours_ahead
    all_india_thermal  = [0.0] * req.hours_ahead
    regions_ok         = 0

    # Load MERIT history once (uploaded CSV files = real 2026 All-India data)
    merit_history = load_merit_history()
    using_merit   = bool(merit_history.get("demand_mw"))

    doy = start_dt.timetuple().tm_yday

    # ── Step A: Run ONE All-India model with real All-India history ───────────
    # WHY: The All-India model was trained on 95K–237K MW. Current = 181–226K.
    # It is well-calibrated at current demand levels.
    # Regional models were trained at 2022 regional levels (e.g. Northern mean
    # = 48K MW). We only have All-India MERIT data, so regional history is
    # estimated by splitting (Northern = AI × 29.88%). This proxy sits in the
    # upper tail of each regional model's training distribution, producing
    # unreliable lag features. The All-India model avoids this entirely.
    try:
        orig = os.getcwd()
        os.chdir(MODELS_DIR)
        ai_model, ai_feature_cols = load_region_model_with_meta("demand_mw")
        os.chdir(orig)
    except FileNotFoundError:
        os.chdir(orig)
        raise HTTPException(500, "All-India model not found. Run train.py first.")

    # All-India history — use real MERIT data if available, else scaled CSV
    if using_merit:
        ai_history = merit_history["demand_mw"][-168:]
        ai_scale_info = {
            "source":         "merit_india_csv",
            "last_timestamp": merit_history.get("last_timestamp"),
            "mean_mw":        merit_history.get("mean_mw"),
            "coverage_pct":   merit_history.get("data_quality", {}).get("coverage_pct"),
            "total_scale":    1.0,
        }
    else:
        ai_history_raw, _, _ = get_recent_demand_from_csv_or_live(DATA_PATH, "demand_mw", 168)
        npp_scale  = get_recent_demand_scale(DATA_PATH, "demand_mw")
        base_scale = npp_scale["scale_factor"]
        ai_history = [round(v * base_scale, 1) for v in ai_history_raw]
        ai_scale_info = {
            "source":      npp_scale.get("source", "scaled_csv"),
            "total_scale": round(base_scale, 4),
        }

    # Weather for All-India — use Northern Region (Delhi) as representative
    # since it is the largest region and most weather-sensitive
    ai_wx_hourly = live_weather.get("Northern_Region", [])
    ai_weather_forecast = []
    for wx in ai_wx_hourly[:req.hours_ahead]:
        t     = wx.get("temp_c") or 25.0
        h_pct = wx.get("humidity_pct") or 60.0
        if t >= 27:
            heat_idx = t + 0.33 * (h_pct / 100 * 6.105 * 2.718 ** (17.27 * t / (237.7 + t))) - 4.0
            heat_idx = max(heat_idx, t)
        else:
            heat_idx = t
        ai_weather_forecast.append({
            "weather_temp_c":       t,
            "weather_humidity_pct": h_pct,
            "weather_solar_wm2":    wx.get("solar_wm2") or 0.0,
            "weather_heat_index":   round(heat_idx, 1),
            "weather_cdh":          round(max(0.0, t - 24.0), 1),
        })

    ai_forecast_raw = predict_24h(
        ai_model, ai_history, start_dt,
        feature_cols=ai_feature_cols,
        weather_forecast=ai_weather_forecast if ai_weather_forecast else None,
        confidence=True,
        ci_level=0.80,
    )

    # ── Step B: Split All-India prediction into regional estimates ────────────
    # Use dynamic hour×month shares (not fixed) for better regional accuracy
    for col in ALL_REGIONS:
        region_id = REGION_ID_MAP[col]
        try:
            wx_hourly = live_weather.get(region_id, [])
            enhanced  = []

            for f in ai_forecast_raw:
                h        = f["hour"]
                target_h = (start_dt.hour + h) % 24
                target_m = start_dt.month

                # Dynamic regional share for this hour and month
                from merit_parser import get_region_shares
                shares  = get_region_shares(target_h, target_m)
                share   = shares.get(col, 0.2)

                # Scale All-India prediction to this region
                ai_pred   = f["predicted_demand_mw"]
                reg_pred  = round(ai_pred * share, 1)
                reg_lower = round(f.get("ci_lower_mw", ai_pred) * share, 1) if f.get("ci_lower_mw") else None
                reg_upper = round(f.get("ci_upper_mw", ai_pred) * share, 1) if f.get("ci_upper_mw") else None

                # Per-region weather for capacity engine
                wx = wx_hourly[h] if h < len(wx_hourly) else {}
                cap = compute_dynamic_capacity(
                    region_id, h, start_dt.month,
                    current_demand_mw=reg_pred,
                    day_of_year=doy,
                    solar_irradiance_wm2=wx.get("solar_wm2"),
                    ambient_temp_c=wx.get("temp_c"),
                    wind_speed_ms=wx.get("wind_speed_ms"),
                )

                hour_result = {
                    **f,
                    "predicted_demand_mw":  reg_pred,
                    "ci_lower_mw":          reg_lower,
                    "ci_upper_mw":          reg_upper,
                    "capacity_mw":          round(cap.total_available_mw, 1),
                    "solar_available_mw":   cap.breakdown.get("solar",   0),
                    "wind_available_mw":    cap.breakdown.get("wind",    0),
                    "hydro_available_mw":   cap.breakdown.get("hydro",   0),
                    "thermal_available_mw": cap.breakdown.get("thermal", 0),
                    "weather_enhanced_cap": cap.weather_enhanced,
                    "solar_cf":             cap.capacity_factors.get("solar", 0),
                    "weather_temp_c":       wx.get("temp_c"),
                    "weather_solar_wm2":    wx.get("solar_wm2"),
                    "weather_wind_ms":      wx.get("wind_speed_ms"),
                    "region_share":         round(share, 4),
                }
                enhanced.append(hour_result)

                if h < req.hours_ahead:
                    all_india_pred[h]     += reg_pred
                    all_india_cap[h]      += cap.total_available_mw
                    all_india_ci_lower[h] += (reg_lower or reg_pred)
                    all_india_ci_upper[h] += (reg_upper or reg_pred)
                    all_india_solar[h]    += cap.breakdown.get("solar",   0)
                    all_india_wind[h]     += cap.breakdown.get("wind",    0)
                    all_india_hydro[h]    += cap.breakdown.get("hydro",   0)
                    all_india_thermal[h]  += cap.breakdown.get("thermal", 0)

            region_results[region_id] = {
                "forecast":   enhanced,
                "scale_info": ai_scale_info,
            }
            regions_ok += 1

        except Exception as e:
            region_results[region_id] = {"error": str(e)}

    # Determine weather source quality for run_id and response
    has_live_wx = any(bool(v) for v in live_weather.values())
    wx_source   = weather_source_label if has_live_wx else "no_weather"

    # Build run_id
    run_id = f"{forecast_date}T{start_dt.strftime('%H%M')}_{wx_source[:4]}"

    # All-India aggregated
    all_india_hours = [
        {
            "hour":                  h,
            "label":                 f"{h:02d}:00",
            "predicted_demand_mw":   round(all_india_pred[h],     1),
            "capacity_mw":           round(all_india_cap[h],      1),
            # CI bounds — sum of regional bounds (conservative: wider than true CI)
            "ci_lower_mw":           round(all_india_ci_lower[h], 1) if regions_ok > 0 else None,
            "ci_upper_mw":           round(all_india_ci_upper[h], 1) if regions_ok > 0 else None,
            # Generation source breakdown — sum of all 5 regions
            "solar_available_mw":    round(all_india_solar[h],    1),
            "wind_available_mw":     round(all_india_wind[h],     1),
            "hydro_available_mw":    round(all_india_hydro[h],    1),
            "thermal_available_mw":  round(all_india_thermal[h],  1),
            "historical_baseline_mw": round(all_india_pred[h] * 0.95, 1),
        }
        for h in range(req.hours_ahead)
    ]
    region_results["ALL_INDIA"] = {"forecast": all_india_hours, "scale_info": None}

    # Save to DB
    if req.save:
        try:
            init_db()
            flat_for_save = {
                k: v["forecast"]
                for k, v in region_results.items()
                if isinstance(v, dict) and "forecast" in v
                and isinstance(v["forecast"], list)
                and v["forecast"]
                and "predicted_demand_mw" in v["forecast"][0]
            }
            save_forecast(
                run_id=run_id,
                forecast_date=forecast_date,
                mode="live",
                weather_source=wx_source,
                model_features=33,
                all_india_peak_mw=max(all_india_pred) if all_india_pred else 0,
                region_forecasts=flat_for_save,
                notes=req.notes,
            )
        except Exception as e:
            pass   # don't fail the whole request if DB save fails

    return {
        "run_id":           run_id,
        "forecast_date":    forecast_date,
        "start_datetime":   start_dt.isoformat(),
        "weather_source":   wx_source,
        "model_features":   33,
        "saved":            req.save,
        "regions":          region_results,
    }


# ── GET /api/live/predictions ─────────────────────────────────────────────────
@router.get("/predictions")
async def list_saved_predictions(limit: int = Query(default=30, ge=1, le=200)):
    init_db()
    return {"predictions": list_predictions(limit=limit)}


# ── DELETE /api/live/predictions/{run_id} ─────────────────────────────────────
@router.delete("/predictions/{run_id}")
async def delete_saved_prediction(run_id: str):
    init_db()
    deleted = delete_prediction(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Prediction '{run_id}' not found")
    return {"deleted": True, "run_id": run_id}


# ── GET /api/live/predictions/{run_id} ────────────────────────────────────────
@router.get("/predictions/{run_id}")
async def get_saved_prediction(run_id: str):
    init_db()
    pred = get_prediction(run_id)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Prediction run '{run_id}' not found")
    return pred


# ── POST /api/live/fetch-actuals ──────────────────────────────────────────────
class FetchActualsRequest(BaseModel):
    date: str   # YYYY-MM-DD

@router.post("/fetch-actuals")
async def fetch_and_save_actuals(req: FetchActualsRequest):
    """
    Fetch actual generation data from NPP for a date and save to DB.
    NPP publishes data for the previous day by ~10am IST.
    Data available from 2013-03-31 onwards.
    """
    try:
        target = date.fromisoformat(req.date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    if target >= date.today():
        raise HTTPException(status_code=400, detail="Can only fetch actuals for past dates")

    try:
        actuals = fetch_actual_demand_for_date(target)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NPP fetch failed: {e}")

    if not actuals["available"]:
        return {
            "status": "not_available",
            "date": req.date,
            "message": "NPP data not available for this date. Try a weekday from 2013 onwards."
        }

    # Save to DB
    init_db()
    save_actuals(req.date, actuals)

    return {
        "status":   "saved",
        "date":     req.date,
        "dgr1":     actuals.get("dgr1"),
        "dgr8":     actuals.get("dgr8"),
    }


# ── GET /api/live/compare/{run_id} ────────────────────────────────────────────
@router.get("/compare/{run_id}")
async def compare_prediction_vs_actual(
    run_id: str,
    auto_fetch: bool = Query(default=True),
):
    """
    Compare a saved prediction against actual NPP data.
    If auto_fetch=True and no actuals exist, tries to fetch from NPP automatically.
    """
    init_db()
    pred = get_prediction(run_id)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Prediction '{run_id}' not found")

    # Try auto-fetching if needed
    if auto_fetch:
        try:
            forecast_date = date.fromisoformat(pred["forecast_date"])
            if forecast_date < date.today():
                actuals = fetch_actual_demand_for_date(forecast_date)
                if actuals["available"]:
                    save_actuals(pred["forecast_date"], actuals)
        except Exception:
            pass

    result = compute_comparison(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="Could not compute comparison")

    return result


# ── GET /api/live/rolling-performance ─────────────────────────────────────────
@router.get("/rolling-performance")
async def get_model_performance(days: int = Query(default=30, ge=7, le=365)):
    init_db()
    return get_rolling_performance(days=days)


# ── Helper ────────────────────────────────────────────────────────────────────
def _city_name(region_id: str) -> str:
    names = {
        "Northern_Region":     "New Delhi",
        "Western_Region":      "Mumbai",
        "Southern_Region":     "Chennai",
        "Eastern_Region":      "Kolkata",
        "NorthEastern_Region": "Guwahati",
    }
    return names.get(region_id, region_id)
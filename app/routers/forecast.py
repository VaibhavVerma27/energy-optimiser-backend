"""
API Router: /api/forecast
POST /api/forecast/all-regions   — Predict all 5 regions + national in one call
POST /api/forecast/region        — Predict a single region
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from predictor import predict_24h, get_recent_from_csv
from decision_engine import detect_overloads, demand_response, REGION_CAPACITIES_MW
from capacity_engine import compute_dynamic_capacity, REGION_INSTALLED
from model import load_region_model_with_meta

def _get_weather_for_forecast(start_dt, hours: int = 24) -> list:
    """Pull actual weather data from demand.csv for the forecast window if available."""
    try:
        import pandas as pd
        from datetime import timedelta
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        weather_hours = []
        for i in range(hours):
            ts = start_dt + timedelta(hours=i)
            row = df[df["timestamp"] == ts]
            if not row.empty:
                weather_hours.append({
                    "solar_irradiance_wm2": float(row["northern_solar_wm2"].iloc[0])
                        if "northern_solar_wm2" in row.columns else None,
                    "ambient_temp_c": float(row["northern_temp_c"].iloc[0])
                        if "northern_temp_c" in row.columns else None,
                    "wind_speed_ms": None,   # not in dataset yet
                })
            else:
                weather_hours.append({})
        return weather_hours if any(w for w in weather_hours) else []
    except Exception:
        return []

def _get_region_weather(start_dt, region_id: str, hours: int = 24) -> list:
    """Pull region-specific weather from demand.csv for the forecast window."""
    prefix_map = {
        "Northern_Region":     "northern",
        "Western_Region":      "western",
        "Southern_Region":     "southern",
        "Eastern_Region":      "eastern",
        "NorthEastern_Region": "ne",
        "ALL_INDIA":           "northern",
    }
    prefix = prefix_map.get(region_id, "northern")
    try:
        import pandas as pd
        from datetime import timedelta
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        weather_hours = []
        for i in range(hours):
            ts  = start_dt + timedelta(hours=i)
            row = df[df["timestamp"] == ts]
            if not row.empty:
                solar_col = f"{prefix}_solar_wm2"
                temp_col  = f"{prefix}_temp_c"
                weather_hours.append({
                    "solar_irradiance_wm2": float(row[solar_col].iloc[0])
                        if solar_col in row.columns else None,
                    "ambient_temp_c": float(row[temp_col].iloc[0])
                        if temp_col in row.columns else None,
                    "wind_speed_ms": None,
                })
            else:
                weather_hours.append({})
        return weather_hours if any(w.get("solar_irradiance_wm2") is not None for w in weather_hours) else []
    except Exception:
        return []

router = APIRouter()

DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demand.csv")
MODELS_DIR  = os.path.dirname(os.path.abspath(__file__)) + "/.."

ALL_REGION_COLS = [
    "Northern_Region_mw",
    "Western_Region_mw",
    "Eastern_Region_mw",
    "Southern_Region_mw",
    "NorthEastern_Region_mw",
]

REGION_LABEL = {
    "Northern_Region_mw":    "Northern Region",
    "Western_Region_mw":     "Western Region",
    "Eastern_Region_mw":     "Eastern Region",
    "Southern_Region_mw":    "Southern Region",
    "NorthEastern_Region_mw":"North-Eastern Region",
    "demand_mw":             "National",
}

REGION_ID_MAP = {
    "Northern_Region_mw":    "Northern_Region",
    "Western_Region_mw":     "Western_Region",
    "Eastern_Region_mw":     "Eastern Region",
    "Southern_Region_mw":    "Southern_Region",
    "NorthEastern_Region_mw":"NorthEastern_Region",
    "demand_mw":             "ALL_INDIA",
}


def _load_model(region_col: str):
    """Load model + feature_cols. Returns (model, feature_cols)."""
    orig_dir = os.getcwd()
    try:
        os.chdir(MODELS_DIR)
        model, feature_cols = load_region_model_with_meta(region_col)
        return model, feature_cols
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No model found for {region_col}.\n"
            f"Run: python train.py --data data/demand.csv"
        )
    finally:
        os.chdir(orig_dir)


def _forecast_one_region(region_col: str, recent_demand: Optional[List[float]], start_dt: datetime) -> dict:
    """Run forecast + dynamic capacity + decision engine for one region column."""
    model, feature_cols = _load_model(region_col)

    # Use provided data or load from CSV
    if recent_demand and len(recent_demand) >= 168:
        history = recent_demand
    else:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError("data/demand.csv not found. Run prepare_dataset.py first.")
        history = get_recent_from_csv(DATA_PATH, region_col=region_col, hours=168)

    forecast_raw = predict_24h(model, history, start_dt,
                                feature_cols=feature_cols,
                                confidence=True, ci_level=0.80)
    region_id    = REGION_ID_MAP.get(region_col, "ALL_INDIA")
    doy          = start_dt.timetuple().tm_yday

    # Compute dynamic capacity per hour — weather-enhanced when data available
    region_weather = _get_region_weather(start_dt, region_id, hours=24)
    enhanced_forecast = []
    capacity_timeline = []
    for f in forecast_raw:
        hour  = f["hour"]
        month = start_dt.month
        wx    = region_weather[hour] if hour < len(region_weather) else {}
        dyn   = compute_dynamic_capacity(
            region_id, hour, month,
            current_demand_mw=f["predicted_demand_mw"],
            day_of_year=doy,
            solar_irradiance_wm2=wx.get("solar_irradiance_wm2"),
            ambient_temp_c=wx.get("ambient_temp_c"),
            wind_speed_ms=wx.get("wind_speed_ms"),
        )
        dynamic_cap = dyn.total_available_mw

        enhanced_forecast.append({
            **f,
            "capacity_mw":            dynamic_cap,
            "historical_baseline_mw": round(f["predicted_demand_mw"] * 0.95),
            "solar_available_mw":     dyn.breakdown.get("solar", 0),
            "wind_available_mw":      dyn.breakdown.get("wind", 0),
            "hydro_available_mw":     dyn.breakdown.get("hydro", 0),
            "thermal_available_mw":   dyn.breakdown.get("thermal", 0),
            "renewable_available_mw": round(
                dyn.breakdown.get("solar", 0) +
                dyn.breakdown.get("wind", 0) +
                dyn.breakdown.get("hydro", 0), 1
            ),
            "capacity_alerts":        dyn.alerts,
            "weather_enhanced_cap":   dyn.weather_enhanced,
            "solar_cf":               dyn.capacity_factors.get("solar", 0),
            "wind_cf":                dyn.capacity_factors.get("wind", 0),
        })
        capacity_timeline.append({
            "hour":             hour,
            "label":            f["label"],
            "total_capacity_mw": dynamic_cap,
            "solar_cf":         dyn.capacity_factors.get("solar", 0),
            "wind_cf":          dyn.capacity_factors.get("wind", 0),
            "hydro_cf":         dyn.capacity_factors.get("hydro", 0),
            "thermal_cf":       dyn.capacity_factors.get("thermal", 0),
            "breakdown_mw":     dyn.breakdown,
            "alerts":           dyn.alerts,
        })

    # Detect overloads using per-hour dynamic capacity
    overload_results = []
    from decision_engine import OverloadResult
    for f in enhanced_forecast:
        pred = f["predicted_demand_mw"]
        cap  = f["capacity_mw"]
        overload_results.append(OverloadResult(
            hour=f["hour"], timestamp=f["timestamp"], label=f["label"],
            predicted_mw=pred, is_overload=pred > cap,
            excess_mw=round(max(0, pred - cap), 1), region=region_id,
        ))

    dr     = demand_response(overload_results, region=region_id)
    adj_map = {a["hour"]: a["adjusted_mw"] for a in dr["adjusted_curve"]}
    for f in enhanced_forecast:
        f["adjusted_demand_mw"] = adj_map.get(f["hour"], f["predicted_demand_mw"])

    peak    = max(enhanced_forecast, key=lambda x: x["predicted_demand_mw"])
    peak_cap = next((f["capacity_mw"] for f in enhanced_forecast if f["hour"] == peak["hour"]), 0)

    return {
        "region_col":        region_col,
        "region_label":      REGION_LABEL.get(region_col, region_col),
        "region_id":         region_id,
        "forecast":          enhanced_forecast,
        "capacity_timeline": capacity_timeline,
        "overload_summary": {
            "overload_detected":    dr["total_overload_hours"] > 0,
            "total_overload_hours": dr["total_overload_hours"],
            "peak_predicted_mw":    peak["predicted_demand_mw"],
            "peak_hour":            peak["label"],
            "capacity_mw":          peak_cap,
            "excess_mw":            round(max(0, peak["predicted_demand_mw"] - peak_cap), 1),
        },
        "demand_response": dr,
    }


# ── Request schemas ───────────────────────────────────────────────────────────

class AllRegionsForecastRequest(BaseModel):
    # Optional: supply 168+ recent values per region for more accurate predictions.
    # If omitted, uses last 168 rows from data/demand.csv automatically.
    Northern_Region_mw:    Optional[List[float]] = None
    Western_Region_mw:     Optional[List[float]] = None
    Eastern_Region_mw:     Optional[List[float]] = None
    Southern_Region_mw:    Optional[List[float]] = None
    NorthEastern_Region_mw:Optional[List[float]] = None
    demand_mw:             Optional[List[float]] = None
    start_datetime:        Optional[str] = None


class SingleRegionRequest(BaseModel):
    region_col:     str             # e.g. "Northern_Region_mw"
    recent_demand:  Optional[List[float]] = None
    start_datetime: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/all-regions")
def forecast_all_regions(req: AllRegionsForecastRequest):
    """
    Predict next 24h for all 5 regions + national demand simultaneously.
    Uses last 168 rows from data/demand.csv if no data provided.
    """
    start_dt = (
        datetime.fromisoformat(req.start_datetime)
        if req.start_datetime
        else datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    )

    supplied = {
        "demand_mw":              req.demand_mw,
        "Northern_Region_mw":     req.Northern_Region_mw,
        "Western_Region_mw":      req.Western_Region_mw,
        "Eastern_Region_mw":      req.Eastern_Region_mw,
        "Southern_Region_mw":     req.Southern_Region_mw,
        "NorthEastern_Region_mw": req.NorthEastern_Region_mw,
    }

    results = {}
    errors  = {}
    all_india_predicted  = [0.0] * 24
    all_india_adjusted   = [0.0] * 24

    cols_to_forecast = ["demand_mw"] + ALL_REGION_COLS

    for col in cols_to_forecast:
        try:
            data = _forecast_one_region(col, supplied.get(col), start_dt)
            results[col] = data
            if col != "demand_mw":
                for i, f in enumerate(data["forecast"]):
                    all_india_predicted[i] += f["predicted_demand_mw"]
                    all_india_adjusted[i]  += f["adjusted_demand_mw"]
        except FileNotFoundError as e:
            errors[col] = str(e)
        except Exception as e:
            errors[col] = str(e)

    if not results:
        raise HTTPException(status_code=503, detail=f"All forecasts failed: {errors}")

    # Build All-India aggregated series
    all_india_forecast = []
    for i in range(24):
        dt = start_dt + timedelta(hours=i)
        all_india_forecast.append({
            "hour": i,
            "timestamp": dt.isoformat(),
            "label": f"{i:02d}:00",
            "predicted_demand_mw":  round(all_india_predicted[i], 1),
            "adjusted_demand_mw":   round(all_india_adjusted[i], 1),
            "historical_baseline_mw": round(all_india_predicted[i] * 0.95),
            "capacity_mw": REGION_CAPACITIES_MW["ALL_INDIA"],
        })

    total_peak = max(all_india_predicted) if all_india_predicted else 0

    return {
        "regions":  results,
        "all_india": {
            "forecast":       all_india_forecast,
            "peak_mw":        round(total_peak, 1),
            "capacity_mw":    REGION_CAPACITIES_MW["ALL_INDIA"],
            "utilisation_pct": round((total_peak / REGION_CAPACITIES_MW["ALL_INDIA"]) * 100, 1),
        },
        "errors":       errors,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.post("/region")
def forecast_single_region(req: SingleRegionRequest):
    """Predict next 24h for a single region."""
    start_dt = (
        datetime.fromisoformat(req.start_datetime)
        if req.start_datetime
        else datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    )
    try:
        return _forecast_one_region(req.region_col, req.recent_demand, start_dt)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status")
def forecast_status():
    """Check which models are trained and data availability."""
    import glob as _glob

    models_found = {}
    cols = ["demand_mw"] + ALL_REGION_COLS
    for col in cols:
        path = os.path.join(MODELS_DIR, f"model_{col}.joblib")
        models_found[col] = os.path.exists(path)

    data_ok = os.path.exists(DATA_PATH)
    data_rows = 0
    data_date_range = None
    if data_ok:
        try:
            import pandas as pd
            df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
            data_rows = len(df)
            data_date_range = f"{df['timestamp'].min()} → {df['timestamp'].max()}"
        except Exception:
            pass

    return {
        "models": models_found,
        "all_models_ready": all(models_found.values()),
        "data_file": data_ok,
        "data_rows": data_rows,
        "data_date_range": data_date_range,
    }
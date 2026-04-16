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
    import joblib
    path = os.path.join(MODELS_DIR, f"model_{region_col}.joblib")
    if not os.path.exists(path):
        # Fallback to national model
        fallback = os.path.join(MODELS_DIR, "model_demand_mw.joblib")
        if os.path.exists(fallback):
            return joblib.load(fallback)
        raise FileNotFoundError(
            f"No model found for {region_col}.\n"
            f"Run: python train.py --data data/demand.csv"
        )
    return joblib.load(path)


def _forecast_one_region(region_col: str, recent_demand: Optional[List[float]], start_dt: datetime) -> dict:
    """Run forecast + decision engine for one region column."""
    model = _load_model(region_col)

    # Use provided data or load from CSV
    if recent_demand and len(recent_demand) >= 168:
        history = recent_demand
    else:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError("data/demand.csv not found. Run prepare_dataset.py first.")
        history = get_recent_from_csv(DATA_PATH, region_col=region_col, hours=168)

    forecast = predict_24h(model, history, start_dt)

    region_id = REGION_ID_MAP.get(region_col, "ALL_INDIA")
    capacity  = REGION_CAPACITIES_MW.get(region_id, REGION_CAPACITIES_MW["ALL_INDIA"])

    # Add capacity and baseline to each forecast hour
    for f in forecast:
        f["capacity_mw"] = capacity
        f["historical_baseline_mw"] = round(f["predicted_demand_mw"] * 0.95)

    overloads = detect_overloads(forecast, capacity_mw=capacity, region=region_id)
    dr = demand_response(overloads, region=region_id)

    # Add adjusted demand to forecast
    adj_map = {a["hour"]: a["adjusted_mw"] for a in dr["adjusted_curve"]}
    for f in forecast:
        f["adjusted_demand_mw"] = adj_map.get(f["hour"], f["predicted_demand_mw"])

    peak = max(forecast, key=lambda x: x["predicted_demand_mw"])

    return {
        "region_col":   region_col,
        "region_label": REGION_LABEL.get(region_col, region_col),
        "region_id":    region_id,
        "forecast":     forecast,
        "overload_summary": {
            "overload_detected":    dr["total_overload_hours"] > 0,
            "total_overload_hours": dr["total_overload_hours"],
            "peak_predicted_mw":    peak["predicted_demand_mw"],
            "peak_hour":            peak["label"],
            "capacity_mw":          capacity,
            "excess_mw":            round(max(0, peak["predicted_demand_mw"] - capacity), 1),
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
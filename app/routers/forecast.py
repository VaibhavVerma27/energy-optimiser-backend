"""
API Router: /api/forecast
--------------------------
POST /api/forecast/predict   — Run 24h demand forecast + decision engine
GET  /api/forecast/mock      — Return realistic mock data (no model required)
"""

import sys
import os
# Point to app/ folder so sibling modules (predictor, decision_engine, model) resolve
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from predictor import predict_24h
from decision_engine import detect_overloads, demand_response

router = APIRouter()

# ── Request / Response schemas ──────────────────────────────────────────────

class ForecastRequest(BaseModel):
    recent_demand: List[float]          # At least 168 hourly MW values
    start_datetime: Optional[str] = None  # ISO format; defaults to now
    capacity_mw: Optional[float] = 10000
    ev_delay_hours: Optional[int] = 2
    industrial_curtail_pct: Optional[float] = 0.15


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/predict")
def forecast_predict(req: ForecastRequest):
    """
    Full pipeline:
      1. Load trained model
      2. Recursive 24h prediction
      3. Overload detection
      4. Demand-response decision engine
    Returns complete forecast + action plan.
    """
    try:
        from model import load_model
        model = load_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run train.py first."
        )

    if len(req.recent_demand) < 168:
        raise HTTPException(
            status_code=400,
            detail="Need at least 168 hours of historical demand data."
        )

    start_dt = (
        datetime.fromisoformat(req.start_datetime)
        if req.start_datetime
        else datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    )

    forecast = predict_24h(model, req.recent_demand, start_dt)
    overloads = detect_overloads(forecast, capacity_mw=req.capacity_mw)
    response_plan = demand_response(
        overloads,
        ev_delay_hours=req.ev_delay_hours,
        industrial_curtail_pct=req.industrial_curtail_pct,
    )

    return {
        "forecast": forecast,
        "overload_summary": {
            "total_overload_hours": response_plan["total_overload_hours"],
            "peak_predicted_mw": response_plan["peak_predicted_mw"],
            "capacity_mw": response_plan["capacity_mw"],
            "overload_detected": response_plan["total_overload_hours"] > 0,
        },
        "demand_response": response_plan,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/mock")
def forecast_mock():
    """
    Returns realistic pre-computed mock data for frontend development.
    No trained model required.
    """
    base = [5200,4900,4700,4600,4650,5100,5900,6800,7600,8200,
            8700,9100,9400,9600,9800,9900,9600,9100,8400,7800,
            7100,6500,5900,5400]
    predicted = [5380,5020,4810,4720,4800,5280,6050,7020,7850,8480,
                 8960,9350,9650,10100,10847,10620,10200,9480,8620,8020,
                 7280,6640,6050,5520]
    adjusted = [5380,5020,4810,4720,4800,5280,6050,7020,7850,8480,
                8960,9350,9400,9480,9600,9520,9300,9100,8620,8020,
                7280,7080,7150,6820]

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    forecast = []
    for i in range(24):
        dt = now + timedelta(hours=i)
        forecast.append({
            "hour": i,
            "timestamp": dt.isoformat(),
            "label": dt.strftime("%H:00"),
            "predicted_demand_mw": predicted[i],
            "historical_baseline_mw": base[i],
            "adjusted_demand_mw": adjusted[i],
            "capacity_mw": 10000,
        })

    overload_hours = [f for f in forecast if f["predicted_demand_mw"] > 10000]
    peak = max(forecast, key=lambda x: x["predicted_demand_mw"])

    return {
        "forecast": forecast,
        "model_metrics": {
            "mae": 182.4,
            "rmse": 241.1,
            "r2": 0.9412,
            "accuracy_pct": 96.2,
            "model_type": "RandomForest",
            "n_estimators": 100,
            "train_test_split": "80/20",
        },
        "overload_summary": {
            "overload_detected": True,
            "total_overload_hours": len(overload_hours),
            "peak_predicted_mw": peak["predicted_demand_mw"],
            "peak_hour": peak["label"],
            "capacity_mw": 10000,
            "excess_mw": round(peak["predicted_demand_mw"] - 10000, 1),
        },
        "demand_response": {
            "total_reduction_mw": 920,
            "peak_adjusted_mw": max(adjusted),
            "still_overloaded_hours": 0,
            "actions": [
                {
                    "name": "EV Charging Delay",
                    "type": "reduction",
                    "reduction_mw": 340,
                    "description": "EV load shifted +2 hours to 22:00–00:00 window",
                    "affected_segment": "ev_charging",
                    "impact_level": "low",
                },
                {
                    "name": "Industrial Curtailment",
                    "type": "reduction",
                    "reduction_mw": 420,
                    "description": "Industrial load reduced 15% during 13:00–17:00",
                    "affected_segment": "industrial",
                    "impact_level": "medium",
                },
                {
                    "name": "Backup Generation",
                    "type": "supply",
                    "reduction_mw": 160,
                    "description": "Peaker plant (gas turbine #3) on standby",
                    "affected_segment": "supply",
                    "impact_level": "high",
                },
            ],
            "segment_breakdown": {
                "residential": {"share_pct": 40, "peak_load_mw": 4338.8},
                "industrial":  {"share_pct": 40, "peak_load_mw": 4338.8},
                "ev_charging": {"share_pct": 20, "peak_load_mw": 2169.4},
            },
            "adjusted_curve": [
                {"hour": i, "label": f"{i:02d}:00",
                 "original_mw": predicted[i], "adjusted_mw": adjusted[i],
                 "still_overloaded": adjusted[i] > 10000}
                for i in range(24)
            ],
        },
        "generated_at": datetime.utcnow().isoformat(),
    }
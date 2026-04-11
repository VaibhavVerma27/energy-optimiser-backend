"""
API Router: /api/demand
------------------------
GET /api/demand/recent        — Returns latest 168 hourly MW values from demand.csv
GET /api/demand/recent?hours=N — Returns latest N hourly values (min 168)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime

router = APIRouter()

# Path to the prepared dataset — relative to app/ folder
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demand.csv")


def load_recent_demand(hours: int = 168) -> list:
    """
    Reads demand.csv and returns the latest `hours` MW values as a list.
    Minimum 168 (7 days) required for the forecasting model.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"demand.csv not found at {DATA_PATH}. "
            "Run prepare_dataset.py first to generate it."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["demand_mw"])

    if len(df) < 168:
        raise ValueError(f"Dataset too small: {len(df)} rows. Need at least 168.")

    # Take the last `hours` rows
    recent = df.tail(hours)

    return {
        "recent_demand": recent["demand_mw"].round(1).tolist(),
        "hours": len(recent),
        "from_timestamp": recent["timestamp"].iloc[0].isoformat(),
        "to_timestamp": recent["timestamp"].iloc[-1].isoformat(),
        "mean_mw": round(recent["demand_mw"].mean(), 1),
        "max_mw": round(recent["demand_mw"].max(), 1),
        "min_mw": round(recent["demand_mw"].min(), 1),
    }


@router.get("/recent")
def get_recent_demand(
    hours: int = Query(default=168, ge=168, le=720,
                       description="Number of recent hourly values to return (min 168, max 720)")
):
    """
    Returns the latest hourly demand values from demand.csv.
    Use the returned `recent_demand` array directly in fetchLiveForecast().
    """
    try:
        result = load_recent_demand(hours)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load demand data: {str(e)}")
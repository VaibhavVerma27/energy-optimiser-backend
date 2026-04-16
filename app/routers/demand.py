"""
API Router: /api/demand
GET /api/demand/recent           — Last 168h for all regions
GET /api/demand/recent?region=X  — Last 168h for one region
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from decision_engine import REGION_CAPACITIES_MW

router = APIRouter()

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demand.csv")

ALL_MW_COLS = [
    "demand_mw",
    "Northern_Region_mw",
    "Western_Region_mw",
    "Eastern_Region_mw",
    "Southern_Region_mw",
    "NorthEastern_Region_mw",
]


@router.get("/recent")
def get_recent(
    region: str = Query(default=None, description="e.g. Northern_Region_mw or demand_mw"),
    hours:  int = Query(default=168, ge=168, le=720),
):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="data/demand.csv not found. Run prepare_dataset.py.")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if region:
        if region not in df.columns:
            avail = [c for c in df.columns if "mw" in c.lower()]
            raise HTTPException(status_code=422, detail=f"Column '{region}' not found. Available: {avail}")
        recent = df.tail(hours)
        cap = REGION_CAPACITIES_MW.get(region.replace("_mw",""), REGION_CAPACITIES_MW["ALL_INDIA"])
        return {
            "region":       region,
            "recent_demand":recent[region].round(1).tolist(),
            "hours":        len(recent),
            "from":         recent["timestamp"].iloc[0].isoformat(),
            "to":           recent["timestamp"].iloc[-1].isoformat(),
            "mean_mw":      round(recent[region].mean(), 1),
            "max_mw":       round(recent[region].max(), 1),
            "capacity_mw":  cap,
        }
    else:
        # Return all available columns
        recent = df.tail(hours)
        result = {
            "hours": len(recent),
            "from":  recent["timestamp"].iloc[0].isoformat(),
            "to":    recent["timestamp"].iloc[-1].isoformat(),
        }
        for col in ALL_MW_COLS:
            if col in recent.columns:
                result[col] = recent[col].round(1).tolist()
        return result
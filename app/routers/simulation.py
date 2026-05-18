"""
routers/simulation.py — Merit Order & Carbon/Cost Intensity API
"""
from fastapi import APIRouter, Query
from datetime import datetime, timedelta
from typing import List, Optional
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capacity_engine import compute_dynamic_capacity, REGION_INSTALLED
from merit_order import compute_merit_dispatch, compute_daily_insights
from prediction_store import list_predictions, get_prediction

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

ALL_REGIONS = [
    "Northern_Region", "Western_Region", "Southern_Region",
    "Eastern_Region", "NorthEastern_Region",
]


def _get_all_india_available(hour: int, month: int,
                              solar_wm2: Optional[float] = None,
                              temp_c: Optional[float] = None,
                              wind_ms: Optional[float] = None) -> dict:
    """Sum available capacity across all regions for a given hour."""
    totals = {"nuclear":0, "hydro":0, "solar":0, "wind":0, "other":0, "thermal":0}
    doy = datetime.now().timetuple().tm_yday
    for region in ALL_REGIONS:
        cap = compute_dynamic_capacity(
            region, hour, month,
            day_of_year=doy,
            solar_irradiance_wm2=solar_wm2,
            ambient_temp_c=temp_c,
            wind_speed_ms=wind_ms,
        )
        b = cap.breakdown
        totals["nuclear"]  += b.get("nuclear", 0)
        totals["hydro"]    += b.get("hydro", 0)
        totals["solar"]    += b.get("solar", 0)
        totals["wind"]     += b.get("wind", 0)
        totals["other"]    += b.get("other", 0)
        totals["thermal"]  += b.get("thermal", 0)
    return totals


@router.get("/merit-dispatch")
async def get_merit_dispatch(
    date: Optional[str] = Query(None, description="YYYY-MM-DD, defaults to today"),
):
    """
    Return 24-hour merit order dispatch, cost (₹/kWh), and carbon intensity (kg CO2/kWh)
    for All-India based on the saved demand forecast for the given date.
    """
    now = datetime.now()
    target_date = datetime.strptime(date, "%Y-%m-%d").date() if date else now.date()
    month = target_date.month

    # Get saved forecast for this date — use the demand_mw (All-India) region
    forecast_data = None
    try:
        all_preds = list_predictions(limit=30)
        match = next((p for p in all_preds if p.get("forecast_date") == str(target_date)), None)
        if match:
            full = get_prediction(match["run_id"])
            if full:
                # forecast_by_region has demand_mw as the All-India model
                ai_hours = full.get("forecast_by_region", {}).get("demand_mw", [])
                if not ai_hours:
                    # Fall back to ALL_INDIA region key
                    ai_hours = full.get("forecast_by_region", {}).get("ALL_INDIA", [])
                forecast_data = {"hours": [{"hour": h["hour"], "predicted_demand_mw": h["predicted_mw"]}
                                           for h in ai_hours]}
    except Exception:
        pass

    if forecast_data and forecast_data.get("hours"):
        demand_by_hour = {h["hour"]: h["predicted_demand_mw"]
                         for h in forecast_data["hours"]}
        forecast_source = "saved_forecast"
    else:
        # Fallback: use typical India demand profile scaled to current level
        TYPICAL_SHAPE = {
            0:0.9741,1:0.9486,2:0.9264,3:0.9088,4:0.8792,5:0.8951,
            6:0.9051,7:0.9022,8:0.9341,9:0.9893,10:1.0152,11:1.0289,
            12:1.0321,13:1.0174,14:1.0648,15:1.0918,16:1.0694,17:1.0376,
            18:1.0182,19:1.0638,20:1.0601,21:1.0543,22:1.0683,23:1.0584,
        }
        base_demand = 205000  # current All-India mean MW
        demand_by_hour = {h: base_demand * TYPICAL_SHAPE[h] for h in range(24)}
        forecast_source = "typical_profile"

    # Compute merit dispatch for each hour
    hours_result = []
    for h in range(24):
        demand = demand_by_hour.get(h, 205000)
        available = _get_all_india_available(h, month)
        merit_hour = compute_merit_dispatch(demand, available, h)
        hours_result.append({
            "hour":             merit_hour.hour,
            "label":            merit_hour.label,
            "demand_mw":        merit_hour.demand_mw,
            "gen_required_mw":  merit_hour.gen_required_mw,
            "dispatch":         merit_hour.dispatch,
            "renewable_mw":     merit_hour.renewable_mw,
            "thermal_mw":       merit_hour.thermal_mw,
            "renewable_pct":    merit_hour.renewable_pct,
            "avg_cost_rs_kwh":  merit_hour.avg_cost_rs_kwh,
            "marginal_cost":    merit_hour.marginal_cost,
            "co2_kg_kwh":       merit_hour.co2_kg_kwh,
            "co2_g_kwh":        round(merit_hour.co2_kg_kwh * 1000, 1),
            "co2_intensity_label": merit_hour.co2_intensity_label,
            "cost_label":       merit_hour.cost_label,
            "color":            merit_hour.color,
            "curtailed_mw":     merit_hour.curtailed_mw,
            "curtailed_pct":    merit_hour.curtailed_pct,
        })

    from merit_order import MeritHour
    merit_hours = [MeritHour(
        hour=h["hour"], label=h["label"],
        demand_mw=h["demand_mw"], gen_required_mw=h["gen_required_mw"],
        dispatch=h["dispatch"],
        renewable_mw=h["renewable_mw"], thermal_mw=h["thermal_mw"],
        renewable_pct=h["renewable_pct"],
        avg_cost_rs_kwh=h["avg_cost_rs_kwh"], marginal_cost=h["marginal_cost"],
        co2_kg_kwh=h["co2_kg_kwh"], co2_intensity_label=h["co2_intensity_label"],
        cost_label=h["cost_label"], color=h["color"],
        curtailed_mw=h["curtailed_mw"], curtailed_pct=h["curtailed_pct"],
    ) for h in hours_result]

    insights = compute_daily_insights(merit_hours)

    return {
        "date":            str(target_date),
        "month":           month,
        "forecast_source": forecast_source,
        "hours":           hours_result,
        "insights":        insights,
        "td_loss_pct":     19.2,
        "note": "Merit order dispatch: Nuclear→Hydro→Solar→Wind→Other RE→Coal(old)→Coal(new)→Gas",
    }
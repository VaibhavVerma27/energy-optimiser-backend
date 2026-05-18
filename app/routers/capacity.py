"""
API Router: /api/capacity
GET /api/capacity/now?region=X          — Current hour dynamic capacity
GET /api/capacity/24h?region=X&month=M  — 24h capacity profile
GET /api/capacity/all-regions           — All regions current capacity
GET /api/capacity/mix?region=X&month=M  — Generation mix summary
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Query
from capacity_engine import (
    compute_dynamic_capacity, get_generation_mix_summary, REGION_INSTALLED,
)

router = APIRouter()

IST = timezone(timedelta(hours=5, minutes=30))
ALL_REGIONS = list(REGION_INSTALLED.keys())


def _now_ist():
    return datetime.now(IST)


def _serialize_cap(cap, hour: int, label: str) -> dict:
    """Consistently serialize a DynamicCapacity object — used by all endpoints."""
    return {
        "hour":               hour,
        "label":              label,
        "total_available_mw": cap.total_available_mw,
        "installed_total_mw": cap.installed_total_mw,
        # breakdown_mw as nested object (what the frontend expects)
        "breakdown_mw": {
            "thermal": cap.breakdown.get("thermal", 0),
            "hydro":   cap.breakdown.get("hydro",   0),
            "solar":   cap.breakdown.get("solar",   0),
            "wind":    cap.breakdown.get("wind",    0),
            "nuclear": cap.breakdown.get("nuclear", 0),
            "other":   cap.breakdown.get("other",   0),
        },
        # also expose top-level for convenience
        "thermal_mw": cap.breakdown.get("thermal", 0),
        "hydro_mw":   cap.breakdown.get("hydro",   0),
        "solar_mw":   cap.breakdown.get("solar",   0),
        "wind_mw":    cap.breakdown.get("wind",    0),
        "nuclear_mw": cap.breakdown.get("nuclear", 0),
        "renewable_mw": round(
            cap.breakdown.get("solar", 0) +
            cap.breakdown.get("wind",  0) +
            cap.breakdown.get("hydro", 0), 1
        ),
        # capacity factors
        "capacity_factors": cap.capacity_factors,
        "solar_cf":   cap.capacity_factors.get("solar",   0),
        "wind_cf":    cap.capacity_factors.get("wind",    0),
        "hydro_cf":   cap.capacity_factors.get("hydro",   0),
        "thermal_cf": cap.capacity_factors.get("thermal", 0),
        "nuclear_cf": cap.capacity_factors.get("nuclear", 0),
        # alerts
        "alerts":               cap.alerts,
        "utilisation_headroom_mw": cap.utilisation_headroom_mw,
    }


@router.get("/now")
def capacity_now(region: str = Query(default="Northern_Region")):
    """Current dynamic capacity for one region."""
    now = _now_ist()
    cap = compute_dynamic_capacity(
        region, now.hour, now.month,
        day_of_year=now.timetuple().tm_yday,
    )
    result = _serialize_cap(cap, now.hour, now.strftime("%H:00"))
    result["region"] = region
    result["timestamp_ist"] = now.strftime("%Y-%m-%d %H:%M IST")
    result["month"] = now.month
    return result


@router.get("/24h")
def capacity_24h(
    region: str  = Query(default="Northern_Region"),
    month:  int  = Query(default=None, ge=1, le=12),
    # Optional per-hour weather arrays (comma-separated, 24 values each)
    solar_wm2:  str = Query(default=None, description="Comma-separated solar irradiance per hour (W/m²)"),
    temp_c:     str = Query(default=None, description="Comma-separated ambient temp per hour (°C)"),
    wind_ms:    str = Query(default=None, description="Comma-separated wind speed per hour (m/s)"),
):
    """
    24-hour dynamic capacity profile for one region.
    Pass per-hour weather arrays to get weather-adjusted capacity.
    Without weather, uses seasonal defaults.
    """
    now   = _now_ist()
    month = month or now.month
    doy   = now.timetuple().tm_yday

    def _parse_arr(s: str):
        if not s: return None
        try:
            vals = [float(x.strip()) for x in s.split(",")]
            return vals if len(vals) == 24 else None
        except Exception:
            return None

    solar_arr = _parse_arr(solar_wm2)
    temp_arr  = _parse_arr(temp_c)
    wind_arr  = _parse_arr(wind_ms)

    hours = []
    for h in range(24):
        cap = compute_dynamic_capacity(
            region, h, month, day_of_year=doy,
            solar_irradiance_wm2 = solar_arr[h] if solar_arr else None,
            ambient_temp_c       = temp_arr[h]  if temp_arr  else None,
            wind_speed_ms        = wind_arr[h]  if wind_arr  else None,
        )
        s = _serialize_cap(cap, h, f"{h:02d}:00")
        s["weather"] = {
            "solar_wm2":  round(solar_arr[h], 1) if solar_arr else None,
            "temp_c":     round(temp_arr[h],  1) if temp_arr  else None,
            "wind_ms":    round(wind_arr[h],  1) if wind_arr  else None,
        }
        hours.append(s)

    return {"region": region, "month": month, "hours": hours, "weather_used": bool(solar_arr or temp_arr or wind_arr)}


@router.get("/all-india-24h")
def capacity_all_india_24h(
    month:     int = Query(default=None, ge=1, le=12),
    solar_wm2: str = Query(default=None),
    temp_c:    str = Query(default=None),
    wind_ms:   str = Query(default=None),
):
    """
    24-hour All-India capacity profile: sums all 5 regions per hour.
    Accepts per-hour weather arrays (comma-separated, 24 values).
    This is the main endpoint for the capacity page chart.
    """
    now   = _now_ist()
    month = month or now.month
    doy   = now.timetuple().tm_yday

    def _parse_arr(s: str):
        if not s: return None
        try:
            vals = [float(x.strip()) for x in s.split(",")]
            return vals if len(vals) == 24 else None
        except Exception:
            return None

    solar_arr = _parse_arr(solar_wm2)
    temp_arr  = _parse_arr(temp_c)
    wind_arr  = _parse_arr(wind_ms)

    hours = []
    for h in range(24):
        totals = {k: 0.0 for k in ["thermal","hydro","solar","wind","nuclear","other","total","installed"]}
        for region in ALL_REGIONS:
            cap = compute_dynamic_capacity(
                region, h, month, day_of_year=doy,
                solar_irradiance_wm2 = solar_arr[h] if solar_arr else None,
                ambient_temp_c       = temp_arr[h]  if temp_arr  else None,
                wind_speed_ms        = wind_arr[h]  if wind_arr  else None,
            )
            for src in ["thermal","hydro","solar","wind","nuclear","other"]:
                totals[src] += cap.breakdown.get(src, 0)
            totals["total"]     += cap.total_available_mw
            totals["installed"] += cap.installed_total_mw

        renewable = totals["solar"] + totals["wind"] + totals["hydro"]
        hours.append({
            "hour":              h,
            "label":             f"{h:02d}:00",
            "total_available_mw": round(totals["total"], 1),
            "installed_total_mw": round(totals["installed"], 1),
            "breakdown_mw": {k: round(totals[k], 1) for k in ["thermal","hydro","solar","wind","nuclear","other"]},
            "renewable_mw":  round(renewable, 1),
            "thermal_mw":    round(totals["thermal"], 1),
            "renewable_pct": round(renewable / max(totals["total"], 1) * 100, 1),
            "weather": {
                "solar_wm2": round(solar_arr[h], 1) if solar_arr else None,
                "temp_c":    round(temp_arr[h],  1) if temp_arr  else None,
                "wind_ms":   round(wind_arr[h],  1) if wind_arr  else None,
            },
        })

    return {
        "month":        month,
        "weather_used": bool(solar_arr or temp_arr or wind_arr),
        "hours":        hours,
    }


@router.get("/all-regions")
def capacity_all_regions():
    """Current dynamic capacity for all 5 regions simultaneously."""
    now   = _now_ist()
    hour  = now.hour
    month = now.month
    doy   = now.timetuple().tm_yday

    result             = {}
    all_india_available = 0
    all_india_installed = 0
    all_india_renewable = 0
    all_india_thermal   = 0

    for region in ALL_REGIONS:
        cap = compute_dynamic_capacity(region, hour, month, day_of_year=doy)
        s   = _serialize_cap(cap, hour, f"{hour:02d}:00")
        result[region] = s
        all_india_available += cap.total_available_mw
        all_india_installed += cap.installed_total_mw
        all_india_renewable += s["renewable_mw"]
        all_india_thermal   += cap.breakdown.get("thermal", 0)

    return {
        "timestamp_ist": now.strftime("%Y-%m-%d %H:%M IST"),
        "hour":          hour,
        "month":         month,
        "regions":       result,
        "all_india": {
            "total_available_mw": round(all_india_available, 1),
            "installed_mw":       round(all_india_installed, 1),
            "renewable_mw":       round(all_india_renewable, 1),
            "thermal_mw":         round(all_india_thermal,   1),
            "renewable_pct":      round(all_india_renewable / max(all_india_available, 1) * 100, 1),
        },
    }


@router.get("/mix")
def capacity_mix(
    region: str = Query(default="Northern_Region"),
    month:  int = Query(default=None, ge=1, le=12),
):
    """Monthly generation mix summary."""
    now   = _now_ist()
    month = month or now.month
    return get_generation_mix_summary(region, month)
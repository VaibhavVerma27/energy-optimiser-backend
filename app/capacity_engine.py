"""
Dynamic Capacity Engine — India Grid (Weather-Enhanced)
=========================================================
Computes per-hour available generation per region from:

1. SOLAR  — weather-aware: uses actual solar irradiance (W/m²) from demand.csv
            when available, otherwise falls back to bell-curve seasonal model
2. WIND   — monsoon-driven seasonality + regional multipliers (no live wind speed yet)
3. HYDRO  — reservoir-level seasonality (post-monsoon full, pre-monsoon depleted)
4. THERMAL— PLF ~58% avg + maintenance schedule + deterministic forced outage
5. NUCLEAR— stable ~82% CUF with seasonal refuelling dips
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Installed capacity mix per region (MW) — CEA 2023-24 ─────────────────────
REGION_INSTALLED = {
    "Northern_Region": {
        "thermal": 62000, "hydro": 22000, "solar": 22000,
        "wind":     4000, "nuclear": 1680, "other":  3320,
    },
    "Western_Region": {
        "thermal": 70000, "hydro":  8000, "solar": 34000,
        "wind":    22000, "nuclear": 1840, "other":  4160,
    },
    "Southern_Region": {
        "thermal": 45000, "hydro": 14000, "solar": 20000,
        "wind":    20000, "nuclear": 2440, "other":  3560,
    },
    "Eastern_Region": {
        "thermal": 43000, "hydro":  4500, "solar":  4000,
        "wind":     1200, "nuclear":    0, "other":  2300,
    },
    "NorthEastern_Region": {
        "thermal":   900, "hydro":  1800, "solar":   500,
        "wind":       80, "nuclear":    0, "other":   220,
    },
}

# Peak irradiance (W/m²) at which a solar panel operates at nameplate capacity
# Panels are rated at 1000 W/m² (STC), but real-world effective peak ~950 W/m²
SOLAR_STC_IRRADIANCE = 950.0

# Regional temperature coefficient (%/°C above 25°C) — hotter = less efficient
SOLAR_TEMP_COEFF = 0.004   # 0.4% efficiency loss per °C above 25°C


# ── Solar CF ─────────────────────────────────────────────────────────────────

def solar_cf(
    hour: int, month: int, region: str,
    actual_irradiance_wm2: Optional[float] = None,
    ambient_temp_c: Optional[float] = None,
) -> float:
    """
    Solar capacity factor (0.0 – ~0.85).

    If actual_irradiance_wm2 is provided (from weather_fetcher.py data),
    uses it directly divided by STC irradiance — much more accurate.

    Otherwise falls back to bell-curve seasonal model.

    Temperature derating: panels lose ~0.4% efficiency per °C above 25°C.
    """
    if actual_irradiance_wm2 is not None and actual_irradiance_wm2 >= 0:
        # Weather-data path — direct physical calculation
        raw_cf = actual_irradiance_wm2 / SOLAR_STC_IRRADIANCE

        # Temperature derating (cell temp ≈ ambient + 25°C under irradiance)
        if ambient_temp_c is not None:
            cell_temp  = ambient_temp_c + (actual_irradiance_wm2 / 1000) * 25
            temp_derate = 1.0 - max(0.0, cell_temp - 25.0) * SOLAR_TEMP_COEFF
            raw_cf *= temp_derate

        # Regional efficiency multiplier (dust, soiling, panel age)
        region_eff = {
            "Northern_Region":     0.93,   # Rajasthan — dusty, regular cleaning
            "Western_Region":      0.94,   # Gujarat — well-maintained large parks
            "Southern_Region":     0.92,   # Karnataka / TN — humid, some soiling
            "Eastern_Region":      0.88,   # Odisha/Bihar — higher humidity loss
            "NorthEastern_Region": 0.82,   # Heavy monsoon soiling, hillside plants
        }
        return round(min(raw_cf * region_eff.get(region, 0.90), 1.0), 3)

    # ── Fallback: bell-curve seasonal model ──────────────────────────────────
    if hour < 5 or hour > 19:
        return 0.0

    bell = math.exp(-0.5 * ((hour - 12.0) / 3.5) ** 2)

    season_cf = {
        1: 0.72, 2: 0.76, 3: 0.82, 4: 0.88,
        5: 0.87, 6: 0.65, 7: 0.45, 8: 0.42,
        9: 0.55, 10: 0.72, 11: 0.70, 12: 0.68,
    }
    region_cf = {
        "Northern_Region":     0.82, "Western_Region":      0.85,
        "Southern_Region":     0.80, "Eastern_Region":      0.68,
        "NorthEastern_Region": 0.55,
    }
    return round(bell * season_cf.get(month, 0.70) * region_cf.get(region, 0.75), 3)


# ── Wind CF ───────────────────────────────────────────────────────────────────

def wind_cf(
    hour: int, month: int, region: str,
    wind_speed_ms: Optional[float] = None,
) -> float:
    """
    Wind capacity factor (0.0 – ~0.45).
    If wind_speed_ms is provided, uses power-curve model.
    Otherwise uses monsoon-driven seasonal model (70% May–Sep).
    """
    if wind_speed_ms is not None and wind_speed_ms >= 0:
        # Simplified wind power curve for India's hub heights (~90m)
        # Cut-in: 3 m/s, rated: 12 m/s, cut-out: 25 m/s
        if wind_speed_ms < 3.0:
            cf = 0.0
        elif wind_speed_ms >= 25.0:
            cf = 0.0   # storm cut-out
        elif wind_speed_ms >= 12.0:
            cf = 1.0   # at rated power
        else:
            # Cubic power law between cut-in and rated
            cf = ((wind_speed_ms - 3.0) / (12.0 - 3.0)) ** 3

        region_cf = {
            "Northern_Region":     0.45, "Western_Region":      1.00,
            "Southern_Region":     0.92, "Eastern_Region":      0.25,
            "NorthEastern_Region": 0.10,
        }
        return round(min(cf * region_cf.get(region, 0.50), 1.0), 3)

    # ── Fallback: seasonal model ──────────────────────────────────────────────
    season_cf = {
        1: 0.10, 2: 0.10, 3: 0.12, 4: 0.15,
        5: 0.28, 6: 0.38, 7: 0.45, 8: 0.42,
        9: 0.32, 10: 0.16, 11: 0.11, 12: 0.10,
    }
    region_cf = {
        "Northern_Region":     0.45, "Western_Region":      1.00,
        "Southern_Region":     0.92, "Eastern_Region":      0.25,
        "NorthEastern_Region": 0.10,
    }
    diurnal = {
        0:0.85,1:0.82,2:0.80,3:0.80,4:0.82,5:0.85,6:0.90,7:0.95,
        8:1.00,9:1.05,10:1.08,11:1.10,12:1.12,13:1.12,14:1.10,15:1.08,
        16:1.05,17:1.02,18:1.00,19:0.97,20:0.95,21:0.92,22:0.90,23:0.87,
    }
    return round(season_cf.get(month, 0.18) * region_cf.get(region, 0.50)
                 * diurnal.get(hour, 1.0), 3)


# ── Hydro CF ──────────────────────────────────────────────────────────────────

def hydro_cf(hour: int, month: int, region: str) -> float:
    """
    Hydro CF (0.33 – 0.75). Reservoir cycle or run-of-river (NE).
    """
    if region == "NorthEastern_Region":
        s = {1:0.35,2:0.32,3:0.30,4:0.32,5:0.45,6:0.62,
             7:0.72,8:0.70,9:0.60,10:0.48,11:0.40,12:0.37}
    else:
        s = {1:0.55,2:0.50,3:0.44,4:0.38,5:0.33,6:0.35,
             7:0.42,8:0.52,9:0.68,10:0.75,11:0.72,12:0.62}
    peak_boost = 1.1 if (7 <= hour <= 11 or 18 <= hour <= 22) else 1.0
    return round(s.get(month, 0.50) * peak_boost, 3)


# ── Thermal CF ────────────────────────────────────────────────────────────────

def thermal_cf(
    hour: int, month: int, region: str,
    outage_seed: int = 0,
    ambient_temp_c: Optional[float] = None,
) -> float:
    """
    Thermal PLF (~53–67%). Includes:
    - Seasonal maintenance schedule (lowest Mar–May)
    - Regional coal-supply factor
    - Deterministic forced outage (0–8%)
    - Heat-rate derating at very high ambient temps (>40°C):
      thermal efficiency drops ~0.1% per °C above 35°C
    """
    base_plf = {
        1:0.65,2:0.64,3:0.58,4:0.54,5:0.53,
        6:0.60,7:0.62,8:0.64,9:0.65,10:0.66,11:0.67,12:0.66,
    }
    plf = base_plf.get(month, 0.60)

    region_factor = {
        "Northern_Region":     0.98, "Western_Region":      1.00,
        "Southern_Region":     0.92, "Eastern_Region":      1.02,
        "NorthEastern_Region": 0.75,
    }

    # High ambient temp reduces thermal efficiency (condensers work harder)
    temp_factor = 1.0
    if ambient_temp_c is not None and ambient_temp_c > 35.0:
        temp_factor = 1.0 - (ambient_temp_c - 35.0) * 0.001  # 0.1% per °C

    pseudo_rand   = ((outage_seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff
    outage_factor = 1.0 - (pseudo_rand * 0.08)

    return round(plf * region_factor.get(region, 1.0) * temp_factor * outage_factor, 3)


def nuclear_cf(month: int) -> float:
    refuel = {3: 0.72, 4: 0.70, 5: 0.74}
    return refuel.get(month, 0.82)


# ── Main computation ──────────────────────────────────────────────────────────

@dataclass
class DynamicCapacity:
    region: str
    hour: int
    month: int
    total_available_mw: float
    installed_total_mw: float
    utilisation_headroom_mw: float
    breakdown: Dict[str, float]         = field(default_factory=dict)
    capacity_factors: Dict[str, float]  = field(default_factory=dict)
    alerts: List[str]                   = field(default_factory=list)
    weather_enhanced: bool              = False  # True when real weather data used


def compute_dynamic_capacity(
    region: str,
    hour: int,
    month: int,
    current_demand_mw: float = 0.0,
    day_of_year: int = 180,
    # Optional weather inputs (from weather_fetcher.py data)
    solar_irradiance_wm2: Optional[float] = None,
    ambient_temp_c: Optional[float]       = None,
    wind_speed_ms: Optional[float]        = None,
) -> DynamicCapacity:
    """
    Compute available generation for a region at one hour.

    When weather parameters are supplied, uses physical models:
      - solar_irradiance_wm2 → direct solar CF calculation with temp derating
      - ambient_temp_c       → thermal heat-rate derating above 35°C
      - wind_speed_ms        → cubic power-curve wind model (if available)

    Falls back to seasonal averages when weather is not supplied.
    """
    installed    = REGION_INSTALLED.get(region, REGION_INSTALLED["Northern_Region"])
    outage_seed  = hash(f"{region}_{day_of_year}") & 0x7fffffff
    has_weather  = solar_irradiance_wm2 is not None or ambient_temp_c is not None

    cf = {
        "solar":   solar_cf(hour, month, region, solar_irradiance_wm2, ambient_temp_c),
        "wind":    wind_cf(hour, month, region, wind_speed_ms),
        "hydro":   hydro_cf(hour, month, region),
        "thermal": thermal_cf(hour, month, region, outage_seed, ambient_temp_c),
        "nuclear": nuclear_cf(month),
        "other":   0.65,
    }

    available = {
        src: round(installed.get(src, 0) * cf.get(src, 0.5))
        for src in ["thermal", "hydro", "solar", "wind", "nuclear", "other"]
    }

    total_available = sum(available.values())
    headroom        = round(total_available - current_demand_mw, 1)

    alerts = []
    if cf["solar"] < 0.05:
        alerts.append("Solar offline (night)")
    elif has_weather and solar_irradiance_wm2 is not None and solar_irradiance_wm2 < 200:
        alerts.append(f"Low solar irradiance — {solar_irradiance_wm2:.0f} W/m² (cloud cover)")
    if month in [6,7,8,9] and cf["wind"] > 0.35:
        alerts.append("Monsoon wind boost active")
    if month in [4,5,6] and cf["hydro"] < 0.40:
        alerts.append("Low reservoir levels — hydro reduced")
    if month in [3,4,5] and cf["thermal"] < 0.57:
        alerts.append("Scheduled thermal maintenance")
    if ambient_temp_c is not None and ambient_temp_c > 40:
        alerts.append(f"High ambient temp {ambient_temp_c:.0f}°C — thermal derating active")
    if current_demand_mw > 0 and headroom < total_available * 0.05:
        alerts.append(f"⚠ Low headroom: only {headroom:,.0f} MW margin")
    if current_demand_mw > total_available:
        alerts.append(f"🔴 Overload: demand exceeds available by {abs(headroom):,.0f} MW")

    return DynamicCapacity(
        region=region, hour=hour, month=month,
        total_available_mw=round(total_available, 1),
        installed_total_mw=sum(installed.values()),
        utilisation_headroom_mw=headroom,
        breakdown=available,
        capacity_factors=cf,
        alerts=alerts,
        weather_enhanced=has_weather,
    )


def compute_24h_capacity(region: str, month: int, day_of_year: int = 180,
                          weather_data: Optional[List[dict]] = None) -> List[dict]:
    """
    24h capacity profile. weather_data is an optional list of 24 dicts with keys:
      solar_irradiance_wm2, ambient_temp_c, wind_speed_ms
    """
    results = []
    for h in range(24):
        wx = (weather_data[h] if weather_data and h < len(weather_data) else {})
        cap = compute_dynamic_capacity(
            region, h, month, day_of_year=day_of_year,
            solar_irradiance_wm2=wx.get("solar_irradiance_wm2"),
            ambient_temp_c=wx.get("ambient_temp_c"),
            wind_speed_ms=wx.get("wind_speed_ms"),
        )
        results.append({
            "hour": h, "label": f"{h:02d}:00",
            **{k: v for k, v in vars(cap).items()},
        })
    return results


def get_generation_mix_summary(region: str, month: int) -> dict:
    hourly    = [compute_dynamic_capacity(region, h, month) for h in range(24)]
    avg_total = sum(c.total_available_mw for c in hourly) / 24
    avg_src   = {}
    for src in ["thermal", "solar", "wind", "hydro", "nuclear", "other"]:
        avg_src[src] = round(sum(c.breakdown.get(src, 0) for c in hourly) / 24, 1)
    installed = REGION_INSTALLED.get(region, {})
    return {
        "region": region, "month": month,
        "avg_available_mw": round(avg_total, 1),
        "installed_mw": sum(installed.values()),
        "avg_by_source": avg_src,
        "renewable_pct": round(
            (avg_src.get("solar",0)+avg_src.get("wind",0)+avg_src.get("hydro",0))
            / max(avg_total,1) * 100, 1
        ),
    }
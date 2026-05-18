"""
Dynamic Capacity Engine — India Grid (Weather-Enhanced)
=========================================================
Source: CEA Monthly Installed Capacity Report March 2026
URL: npp.gov.in/public-reports/cea/monthly/installcap/2026/MAR/
As on: 31/03/2026

CEA verified regional totals:
  Northern:  143,446 MW  (was 127,353 MW as of July 2024)
  Western:   191,198 MW  (was 148,858 MW)
  Southern:  141,825 MW  (was 130,944 MW)
  Eastern:    50,016 MW  (was 35,570 MW)
  NE:          6,255 MW  (was 5,496 MW)
  All-India: 532,740 MW  (was 448,381 MW — +19% growth)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Installed capacity mix per region (MW) — CEA March 2026 ─────────────────
# Source: npp.gov.in location-wise state-wise installed capacity PDFs
# Thermal = Coal + Lignite + Gas + Diesel (CEA exact totals)
# Nuclear = CEA exact
# Hydro   = Large hydro CEA exact
# Solar + Wind derived from RES total (subtracted SHP/biomass/waste)
# Other   = SHP + Biomass + Waste-to-Energy (estimated as RES - solar - wind)

REGION_INSTALLED = {
    "Northern_Region": {
        # CEA totals: thermal=57,561 hydro=21,666 nuclear=2,220 RES=62,000
        "thermal": 57561,   # coal 50,269 + gas 1,580 + diesel 0 + lignite 0 (CEA exact)
        "nuclear":  2220,   # Narora 440 + Rawatbhata NR share 1,780
        "hydro":   21666,   # Tehri 2,400 + Nathpa Jhakri 1,530 + NJPC 1,500 + others
        "solar":   46000,   # Rajasthan dominates (~40GW) + UP/HP/Haryana (from RES 62,000)
        "wind":     9000,   # Rajasthan + UP wind, increased from 4,800
        "other":    7000,   # SHP + biomass (residual of RES 62,000)
        # Grand Total (CEA): 143,446 MW
    },
    "Western_Region": {
        # CEA totals: thermal=94,070 hydro=7,392 nuclear=3,240 RES=86,495
        "thermal": 94070,   # coal 83,271 + gas 1,400 + diesel 0 + lignite 0 (CEA exact)
        "nuclear":  3240,   # Kakrapar 2,160 + Rawatbhata WR share 1,080
        "hydro":    7392,   # Koyna 1,960 + Sardar Sarovar 1,450 + Indira Sagar + others
        "solar":   52000,   # Gujarat ~20GW + Rajasthan (WR portion) + MH + MP (from RES 86,495)
        "wind":    27000,   # Gujarat coastline ~15GW + MH + MP (India's highest wind)
        "other":    7495,   # SHP + biomass (residual)
        # Grand Total (CEA): 191,198 MW
    },
    "Southern_Region": {
        # CEA totals: thermal=54,052 hydro=13,596 nuclear=3,320 RES=70,857
        "thermal": 54052,   # coal 46,605 + gas 3,640 + diesel 460 + lignite 0 (CEA exact)
        "nuclear":  3320,   # Kudankulam 2,000 + Kaiga 880 + Madras NPP 440
        "hydro":   13596,   # Srisailam 1,670 + Nagarjunasagar 816 + Idukki 780 + others
        "solar":   46000,   # Karnataka (Pavagada 2GW+) + AP (Kurnool) + TN + Telangana
        "wind":    19000,   # TN ~10GW + AP ~6GW + Karnataka ~3GW (from RES 70,857)
        "other":    5857,   # SHP + biomass (residual)
        # Grand Total (CEA): 141,825 MW
    },
    "Eastern_Region": {
        # CEA totals: thermal=41,138 hydro=5,988 nuclear=0 RES=2,890
        "thermal": 41138,   # coal 41,045 + diesel 93 (CEA exact) — significant increase
        "nuclear":     0,   # Zero nuclear in Eastern Region
        "hydro":    5988,   # Maithon 60 + Panchet 80 + Hirakud 347.5 + Sikkim 2,282 + others
        "solar":    1800,   # Odisha + Bihar + West Bengal (from RES 2,890)
        "wind":      500,   # Odisha coast + WB (from RES 2,890)
        "other":     590,   # SHP + biomass
        # Grand Total (CEA): 50,016 MW
    },
    "NorthEastern_Region": {
        # CEA totals: thermal=2,451 hydro=2,773 nuclear=0 RES=1,031
        "thermal":  2451,   # coal 750 + gas 0 + diesel 36 + other thermal 1,665 (CEA exact)
        "nuclear":     0,   # Zero nuclear
        "hydro":    2773,   # Ranganadi 405 + Kopili 275 + Loktak 105 + Arunachal 1,865 + others
        "solar":     700,   # Assam + Meghalaya + others (from RES 1,031)
        "wind":       50,   # Negligible
        "other":     281,   # SHP (significant in NE hills)
        # Grand Total (CEA): 6,255 MW
    },
}

SOLAR_STC_IRRADIANCE = 950.0
SOLAR_TEMP_COEFF = 0.004


def solar_cf(
    hour: int, month: int, region: str,
    actual_irradiance_wm2: Optional[float] = None,
    ambient_temp_c: Optional[float] = None,
) -> float:
    if actual_irradiance_wm2 is not None and actual_irradiance_wm2 >= 0:
        raw_cf = actual_irradiance_wm2 / SOLAR_STC_IRRADIANCE
        if ambient_temp_c is not None:
            cell_temp  = ambient_temp_c + (actual_irradiance_wm2 / 1000) * 25
            temp_derate = 1.0 - max(0.0, cell_temp - 25.0) * SOLAR_TEMP_COEFF
            raw_cf *= temp_derate
        region_eff = {
            "Northern_Region":     0.93,
            "Western_Region":      0.94,
            "Southern_Region":     0.92,
            "Eastern_Region":      0.88,
            "NorthEastern_Region": 0.82,
        }
        return round(min(raw_cf * region_eff.get(region, 0.90), 1.0), 3)

    if hour < 5 or hour > 19:
        return 0.0
    bell = math.exp(-0.5 * ((hour - 12.0) / 3.5) ** 2)
    season_cf = {
        1:0.72,2:0.76,3:0.82,4:0.88,5:0.87,6:0.65,
        7:0.45,8:0.42,9:0.55,10:0.72,11:0.70,12:0.68,
    }
    region_cf = {
        "Northern_Region":0.82,"Western_Region":0.85,
        "Southern_Region":0.80,"Eastern_Region":0.68,
        "NorthEastern_Region":0.55,
    }
    return round(bell * season_cf.get(month, 0.70) * region_cf.get(region, 0.75), 3)


def wind_cf(
    hour: int, month: int, region: str,
    wind_speed_ms: Optional[float] = None,
) -> float:
    if wind_speed_ms is not None and wind_speed_ms >= 0:
        if wind_speed_ms < 3.0:
            cf = 0.0
        elif wind_speed_ms >= 25.0:
            cf = 0.0
        elif wind_speed_ms >= 12.0:
            cf = 1.0
        else:
            cf = ((wind_speed_ms - 3.0) / (12.0 - 3.0)) ** 3
        region_cf = {
            "Northern_Region":0.45,"Western_Region":1.00,
            "Southern_Region":0.92,"Eastern_Region":0.25,
            "NorthEastern_Region":0.10,
        }
        return round(min(cf * region_cf.get(region, 0.50), 1.0), 3)

    season_cf = {
        1:0.10,2:0.10,3:0.12,4:0.15,5:0.28,6:0.38,
        7:0.45,8:0.42,9:0.32,10:0.16,11:0.11,12:0.10,
    }
    region_cf = {
        "Northern_Region":0.45,"Western_Region":1.00,
        "Southern_Region":0.92,"Eastern_Region":0.25,
        "NorthEastern_Region":0.10,
    }
    diurnal = {
        0:0.85,1:0.82,2:0.80,3:0.80,4:0.82,5:0.85,
        6:0.90,7:0.95,8:1.00,9:1.05,10:1.08,11:1.10,
        12:1.12,13:1.12,14:1.10,15:1.08,16:1.05,17:1.02,
        18:1.00,19:0.97,20:0.95,21:0.92,22:0.90,23:0.87,
    }
    return round(season_cf.get(month, 0.18) * region_cf.get(region, 0.50)
                 * diurnal.get(hour, 1.0), 3)


def thermal_cf(
    hour: int, month: int, region: str,
    outage_seed: int = 0,
    ambient_temp_c: Optional[float] = None,
) -> float:
    """
    Thermal AVAILABILITY factor — fraction of installed thermal capacity
    that is available to dispatch at any given hour.

    KEY CORRECTION: Thermal plants are base-load. They do NOT vary by hour.
    A coal plant running at 03:00 is running at the same output as at 15:00.
    The old PLF-based hourly model was wrong — PLF is a utilisation metric,
    not an availability metric.

    Availability = 1 - planned_outage_rate
    Planned outages are SEASONAL (scheduled in summer/post-monsoon) not hourly.

    Typical India thermal availability by season (from CEA Annual Reports):
      Jan-Mar: 0.82 — high season, good cooling water, all units available
      Apr-May: 0.78 — pre-summer maintenance season, some units offline
      Jun-Sep: 0.80 — monsoon, some units on maintenance but demand is lower
      Oct-Dec: 0.84 — peak season, maximum units available

    Temperature derating: extreme heat (>40°C) reduces thermal output slightly
    because cooling water is warmer (condenser back-pressure effect).
    Effect: ~0.5% output reduction per °C above 40°C. Small but real.
    """
    # Monthly availability (1 - planned_outage_rate)
    monthly_avail = {
        1:0.82, 2:0.82, 3:0.80,
        4:0.78, 5:0.78, 6:0.80,
        7:0.80, 8:0.80, 9:0.80,
        10:0.84, 11:0.84, 12:0.84,
    }
    avail = monthly_avail.get(month, 0.80)

    # Regional adjustment — NE has older plants with lower availability
    region_factor = {
        "Northern_Region":      1.00,
        "Western_Region":       1.00,
        "Southern_Region":      0.97,  # slightly older fleet
        "Eastern_Region":       1.00,
        "NorthEastern_Region":  0.85,  # older gas turbines, higher outage rate
    }
    avail *= region_factor.get(region, 1.00)

    # Temperature derating above 40°C (condenser back-pressure effect)
    if ambient_temp_c is not None and ambient_temp_c > 40.0:
        avail *= (1.0 - (ambient_temp_c - 40.0) * 0.005)

    return round(max(0.50, min(avail, 0.95)), 3)


def hydro_cf(hour: int, month: int, region: str) -> float:
    """
    Hydro capacity factor — fraction of hydro capacity available + dispatchable.

    Hydro IS dispatchable unlike thermal. Reservoir-based hydro is held back
    during off-peak hours and released during morning/evening peaks.
    Run-of-river (dominant in NE and Northern Himalayan) is NOT dispatchable
    but follows river flow which peaks during snowmelt (Apr-Jun) and monsoon.

    May specific: Himalayan snowmelt makes May a HIGH hydro month for Northern/NE.
    Previous value of 0.33 for May was too low — corrected to 0.44.
    """
    if region == "NorthEastern_Region":
        # Mostly run-of-river — follows monsoon/snowmelt, no peak dispatch
        s = {1:0.35,2:0.32,3:0.35,4:0.42,5:0.52,6:0.65,
             7:0.72,8:0.70,9:0.60,10:0.48,11:0.40,12:0.37}
        # No dispatchable peak boost for run-of-river
        return round(s.get(month, 0.50), 3)
    else:
        # Mix of reservoir (dispatchable) and run-of-river
        # May updated from 0.33 → 0.44 (snowmelt peak + Himalayan run-of-river)
        s = {1:0.58,2:0.52,3:0.46,4:0.42,5:0.44,6:0.40,
             7:0.50,8:0.60,9:0.72,10:0.78,11:0.75,12:0.65}
        # Dispatchable reservoir hydro: 10% boost during peak demand hours
        peak_boost = 1.10 if (7 <= hour <= 11 or 18 <= hour <= 22) else 1.00
        return round(s.get(month, 0.55) * peak_boost, 3)


def nuclear_cf(month: int) -> float:
    refuel = {3:0.72,4:0.70,5:0.74}
    return refuel.get(month, 0.82)


@dataclass
class DynamicCapacity:
    region: str
    hour: int
    month: int
    total_available_mw: float
    installed_total_mw: float
    utilisation_headroom_mw: float
    breakdown: Dict[str, float]        = field(default_factory=dict)
    capacity_factors: Dict[str, float] = field(default_factory=dict)
    alerts: List[str]                  = field(default_factory=list)
    weather_enhanced: bool             = False


def compute_dynamic_capacity(
    region: str,
    hour: int,
    month: int,
    current_demand_mw: float = 0.0,
    day_of_year: int = 180,
    solar_irradiance_wm2: Optional[float] = None,
    ambient_temp_c: Optional[float]       = None,
    wind_speed_ms: Optional[float]        = None,
) -> DynamicCapacity:
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
        for src in ["thermal","hydro","solar","wind","nuclear","other"]
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
    results = []
    for h in range(24):
        wx  = (weather_data[h] if weather_data and h < len(weather_data) else {})
        cap = compute_dynamic_capacity(
            region, h, month, day_of_year=day_of_year,
            solar_irradiance_wm2=wx.get("solar_irradiance_wm2"),
            ambient_temp_c=wx.get("ambient_temp_c"),
            wind_speed_ms=wx.get("wind_speed_ms"),
        )
        results.append({"hour":h,"label":f"{h:02d}:00",**vars(cap)})
    return results


def get_generation_mix_summary(region: str, month: int) -> dict:
    hourly    = [compute_dynamic_capacity(region, h, month) for h in range(24)]
    avg_total = sum(c.total_available_mw for c in hourly) / 24
    avg_src   = {}
    for src in ["thermal","solar","wind","hydro","nuclear","other"]:
        avg_src[src] = round(sum(c.breakdown.get(src,0) for c in hourly) / 24, 1)
    installed = REGION_INSTALLED.get(region, {})
    return {
        "region":region,"month":month,
        "avg_available_mw":round(avg_total,1),
        "installed_mw":sum(installed.values()),
        "avg_by_source":avg_src,
        "renewable_pct":round(
            (avg_src.get("solar",0)+avg_src.get("wind",0)+avg_src.get("hydro",0))
            /max(avg_total,1)*100,1
        ),
    }
"""
Dynamic Capacity Engine — India Grid
======================================
Models real-time available capacity per region based on:

1. SOLAR: hourly sun angle + season + cloud cover (monsoon reduction)
   - Daytime only (sunrise ~06:00, sunset ~18:30 IST avg)
   - Peak at 12:00, near-zero before 06:00 and after 19:00
   - Summer (Apr-Jun): highest irradiance → highest CF
   - Monsoon (Jul-Sep): 40-60% reduction from cloud cover
   - Winter (Nov-Jan): moderate, shortest days

2. WIND: monsoon-driven seasonality + region variation
   - 70% of annual generation May-September (SW monsoon)
   - Western/Southern regions dominate (Gujarat, Tamil Nadu, Western Ghats)
   - Northern/Eastern regions have low wind capacity

3. HYDRO: reservoir-level seasonality
   - Post-monsoon (Sep-Nov): reservoirs full → highest hydro output
   - Pre-monsoon (Apr-Jun): low reservoirs → reduced hydro
   - NE Region: dominated by run-of-river hydro, highest in monsoon

4. THERMAL: planned maintenance + forced outage probability
   - Average PLF (Plant Load Factor) ~58% nationally (CERC data)
   - Maintenance scheduled in low-demand months (Mar-May)
   - Random forced outages modeled probabilistically
   - Coal supply constraints reduce effective capacity

Sources:
  - CEA 2023-24: Installed capacity by region and source
  - CERC: Average capacity factors by generation type
  - Research: India wind/solar seasonal patterns (Hunt et al. 2024)
  - Wikipedia: India wind power (70% generation May-Sep, 18% annual CUF)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List


# ── Installed capacity mix per region (MW) — CEA 2023-24 ─────────────────────
# Source: Central Electricity Authority, Executive Summary Sep 2023

REGION_INSTALLED = {
    "Northern_Region": {
        "thermal": 62000,   # Coal + gas dominant (NTPC Rihand, Singrauli, Dadri)
        "hydro":   22000,   # Himachal/Uttarakhand hydro (Tehri, Nathpa Jhakri)
        "solar":   22000,   # Rajasthan solar parks (Bhadla, REWA etc.)
        "wind":     4000,   # Limited wind (Rajasthan/Gujarat border)
        "nuclear":  1680,   # Narora NPP (UP)
        "other":    3320,   # Biomass, small hydro
    },
    "Western_Region": {
        "thermal": 70000,   # Largest thermal base (Mundra, Vindhyachal, Tiroda)
        "hydro":    8000,   # Koyna, Sardar Sarovar
        "solar":   34000,   # Gujarat + Maharashtra + Rajasthan (Bhadla in WR)
        "wind":    22000,   # Gujarat coastline (highest wind in India)
        "nuclear":  1840,   # Kakrapar + Rawatbhata
        "other":    4160,
    },
    "Southern_Region": {
        "thermal": 45000,   # Ramagundam, Talcher
        "hydro":   14000,   # AP, Karnataka, Kerala hydro
        "solar":   20000,   # Karnataka + AP + Tamil Nadu (Pavagada)
        "wind":    20000,   # Tamil Nadu + AP + Karnataka (highest wind historically)
        "nuclear":  2440,   # Kudankulam NPP
        "other":    3560,
    },
    "Eastern_Region": {
        "thermal": 43000,   # Dominant: Farakka, Kahalgaon, Talcher
        "hydro":    4500,   # Limited (Maithon, Panchet)
        "solar":    4000,   # Odisha, Bihar
        "wind":     1200,   # Very limited
        "nuclear":     0,
        "other":    2300,
    },
    "NorthEastern_Region": {
        "thermal":   900,   # Gas-based mostly (Assam)
        "hydro":    1800,   # Kopili, Loktak, Umiam (run-of-river dominant)
        "solar":     500,   # Limited
        "wind":       80,   # Negligible
        "nuclear":     0,
        "other":     220,   # Biomass
    },
}


# ── Capacity factor profiles ──────────────────────────────────────────────────

def solar_cf(hour: int, month: int, region: str) -> float:
    """
    Solar capacity factor: 0.0 to ~0.85
    Follows a bell curve centered on solar noon (~12:30 IST avg).
    Reduced during monsoon months due to cloud cover.
    Region affects peak CF (Western has highest irradiance).
    """
    # Solar only available daytime — simple bell curve
    if hour < 5 or hour > 19:
        return 0.0

    # Gaussian centered at hour 12 (solar noon), spread ~4h
    peak_hour   = 12.0
    sigma       = 3.5
    bell        = math.exp(-0.5 * ((hour - peak_hour) / sigma) ** 2)

    # Seasonal multiplier (India solar — highest pre-monsoon, lowest monsoon)
    season_cf = {
        1: 0.72, 2: 0.76, 3: 0.82, 4: 0.88,
        5: 0.87, 6: 0.65, 7: 0.45, 8: 0.42,
        9: 0.55, 10: 0.72, 11: 0.70, 12: 0.68,
    }
    seasonal = season_cf.get(month, 0.70)

    # Regional peak CF multiplier (irradiance varies)
    region_cf = {
        "Northern_Region":     0.82,   # Rajasthan excellent solar
        "Western_Region":      0.85,   # Gujarat + Rajasthan — best in India
        "Southern_Region":     0.80,   # Karnataka, AP, TN
        "Eastern_Region":      0.68,   # Moderate
        "NorthEastern_Region": 0.55,   # Lower irradiance, hills/clouds
    }
    r_cf = region_cf.get(region, 0.75)

    return round(bell * seasonal * r_cf, 3)


def wind_cf(hour: int, month: int, region: str) -> float:
    """
    Wind capacity factor: 0.0 to ~0.45
    70% of annual generation is May-September (SW monsoon).
    Western and Southern regions dominate India's wind output.
    Wind has slight diurnal variation (higher afternoon).
    Source: Wikipedia India wind power (18% annual avg CUF)
    """
    # Seasonal multiplier — 70% of output May-Sep
    # Annual mean = 0.18 CUF. Calibrate monthly to sum to that.
    season_cf = {
        1: 0.10, 2: 0.10, 3: 0.12, 4: 0.15,
        5: 0.28, 6: 0.38, 7: 0.45, 8: 0.42,
        9: 0.32, 10: 0.16, 11: 0.11, 12: 0.10,
    }
    seasonal = season_cf.get(month, 0.18)

    # Regional multiplier — Western and Southern dominate
    region_cf = {
        "Northern_Region":     0.45,   # Gujarat border/Rajasthan — decent
        "Western_Region":      1.00,   # Gujarat coastline — best in India
        "Southern_Region":     0.92,   # Tamil Nadu, AP, Western Ghats
        "Eastern_Region":      0.25,   # Limited wind resource
        "NorthEastern_Region": 0.10,   # Very limited
    }
    r_cf = region_cf.get(region, 0.50)

    # Diurnal variation: wind picks up midday-evening
    diurnal = {
        0: 0.85, 1: 0.82, 2: 0.80, 3: 0.80, 4: 0.82, 5: 0.85,
        6: 0.90, 7: 0.95, 8: 1.00, 9: 1.05, 10: 1.08, 11: 1.10,
        12: 1.12, 13: 1.12, 14: 1.10, 15: 1.08, 16: 1.05, 17: 1.02,
        18: 1.00, 19: 0.97, 20: 0.95, 21: 0.92, 22: 0.90, 23: 0.87,
    }

    return round(seasonal * r_cf * diurnal.get(hour, 1.0), 3)


def hydro_cf(hour: int, month: int, region: str) -> float:
    """
    Hydro capacity factor: 0.0 to ~0.75
    Post-monsoon (Sep-Nov): reservoirs full → peak output
    Pre-monsoon (Apr-Jun): depleted reservoirs → reduced
    NE region: run-of-river, peaks during monsoon itself
    """
    if region == "NorthEastern_Region":
        # Run-of-river: peaks during monsoon rainfall
        season_cf = {
            1: 0.35, 2: 0.32, 3: 0.30, 4: 0.32, 5: 0.45,
            6: 0.62, 7: 0.72, 8: 0.70, 9: 0.60, 10: 0.48, 11: 0.40, 12: 0.37,
        }
    else:
        # Reservoir-based: peaks post-monsoon when reservoirs are full
        season_cf = {
            1: 0.55, 2: 0.50, 3: 0.44, 4: 0.38, 5: 0.33,
            6: 0.35, 7: 0.42, 8: 0.52, 9: 0.68, 10: 0.75, 11: 0.72, 12: 0.62,
        }

    # Hydro runs on demand — dispatchable, so CF is relatively flat hourly
    # but slightly higher during peak demand hours (grid operators schedule it)
    peak_boost = 1.1 if (7 <= hour <= 11 or 18 <= hour <= 22) else 1.0

    return round(season_cf.get(month, 0.50) * peak_boost, 3)


def thermal_cf(hour: int, month: int, region: str, outage_seed: int = 0) -> float:
    """
    Thermal (coal) capacity factor: 0.45 to 0.72
    National average PLF ~58% (CERC 2023)
    Maintenance outages scheduled in low-demand months (Mar-May)
    Forced outages represented as small random reduction
    """
    # Base PLF by season — maintenance in spring
    base_plf = {
        1: 0.65, 2: 0.64, 3: 0.58, 4: 0.54, 5: 0.53,
        6: 0.60, 7: 0.62, 8: 0.64, 9: 0.65, 10: 0.66, 11: 0.67, 12: 0.66,
    }
    plf = base_plf.get(month, 0.60)

    # Regional variation — coal supply reliability
    region_factor = {
        "Northern_Region":     0.98,   # Good coal supply (UP pithead plants)
        "Western_Region":      1.00,   # Largest thermal base
        "Southern_Region":     0.92,   # Some coal supply constraints
        "Eastern_Region":      1.02,   # Near coalfields (Odisha, Jharkhand)
        "NorthEastern_Region": 0.75,   # Mostly gas, supply constraints
    }

    # Simulate forced outage: deterministic pseudo-random based on day/region
    # This gives consistent values for same inputs without true randomness
    pseudo_rand = ((outage_seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff
    outage_factor = 1.0 - (pseudo_rand * 0.08)  # 0-8% forced outage

    return round(plf * region_factor.get(region, 1.0) * outage_factor, 3)


def nuclear_cf(month: int) -> float:
    """Nuclear: highly stable baseload, ~80-85% CUF. Minor seasonal refueling."""
    refuel = {3: 0.72, 4: 0.70, 5: 0.74}  # Spring refueling outages
    return refuel.get(month, 0.82)


# ── Main capacity calculation ─────────────────────────────────────────────────

@dataclass
class DynamicCapacity:
    region: str
    hour: int
    month: int
    total_available_mw: float
    installed_total_mw: float
    utilisation_headroom_mw: float  # available - current demand
    breakdown: Dict[str, float] = field(default_factory=dict)  # source → available MW
    capacity_factors: Dict[str, float] = field(default_factory=dict)  # source → CF
    alerts: List[str] = field(default_factory=list)


def compute_dynamic_capacity(
    region: str,
    hour: int,
    month: int,
    current_demand_mw: float = 0.0,
    day_of_year: int = 180,  # used for thermal outage seed
) -> DynamicCapacity:
    """
    Compute available generation capacity for a region at a specific hour/month.
    Returns total available MW and per-source breakdown.
    """
    installed = REGION_INSTALLED.get(region, REGION_INSTALLED["Northern_Region"])
    outage_seed = hash(f"{region}_{day_of_year}") & 0x7fffffff

    # Capacity factors for each source
    cf = {
        "solar":   solar_cf(hour, month, region),
        "wind":    wind_cf(hour, month, region),
        "hydro":   hydro_cf(hour, month, region),
        "thermal": thermal_cf(hour, month, region, outage_seed),
        "nuclear": nuclear_cf(month),
        "other":   0.65,  # biomass/small hydro — stable baseload
    }

    # Available MW = installed × CF
    available = {
        source: round(installed.get(source, 0) * cf.get(source, 0.5))
        for source in ["thermal", "hydro", "solar", "wind", "nuclear", "other"]
    }

    total_available = sum(available.values())
    headroom = round(total_available - current_demand_mw, 1)

    alerts = []
    # Solar unavailable at night
    if cf["solar"] < 0.05:
        alerts.append("Solar offline (night)")
    # Monsoon wind boost
    if month in [6, 7, 8, 9] and cf["wind"] > 0.35:
        alerts.append("Monsoon wind boost active")
    # Low hydro pre-monsoon
    if month in [4, 5, 6] and cf["hydro"] < 0.40:
        alerts.append("Low reservoir levels — hydro reduced")
    # Thermal maintenance
    if month in [3, 4, 5] and cf["thermal"] < 0.57:
        alerts.append("Scheduled thermal maintenance")
    # Near capacity
    if current_demand_mw > 0 and headroom < total_available * 0.05:
        alerts.append(f"⚠ Low headroom: only {headroom:,.0f} MW margin")
    if current_demand_mw > total_available:
        alerts.append(f"🔴 Overload: demand exceeds available by {abs(headroom):,.0f} MW")

    return DynamicCapacity(
        region=region,
        hour=hour,
        month=month,
        total_available_mw=round(total_available, 1),
        installed_total_mw=sum(installed.values()),
        utilisation_headroom_mw=headroom,
        breakdown=available,
        capacity_factors=cf,
        alerts=alerts,
    )


def compute_24h_capacity(region: str, month: int, day_of_year: int = 180) -> List[dict]:
    """Return dynamic capacity for all 24 hours of a given month."""
    return [
        {
            "hour":   h,
            "label":  f"{h:02d}:00",
            **vars(compute_dynamic_capacity(region, h, month, day_of_year=day_of_year)),
        }
        for h in range(24)
    ]


def get_generation_mix_summary(region: str, month: int) -> dict:
    """Average generation mix for a region in a given month."""
    hourly = [compute_dynamic_capacity(region, h, month) for h in range(24)]
    avg_total = sum(c.total_available_mw for c in hourly) / 24
    avg_by_source = {}
    for source in ["thermal", "solar", "wind", "hydro", "nuclear", "other"]:
        avg_by_source[source] = round(
            sum(c.breakdown.get(source, 0) for c in hourly) / 24, 1
        )

    installed = REGION_INSTALLED.get(region, {})
    return {
        "region":           region,
        "month":            month,
        "avg_available_mw": round(avg_total, 1),
        "installed_mw":     sum(installed.values()),
        "avg_by_source":    avg_by_source,
        "renewable_pct":    round(
            (avg_by_source.get("solar", 0) + avg_by_source.get("wind", 0) + avg_by_source.get("hydro", 0))
            / max(avg_total, 1) * 100, 1
        ),
    }
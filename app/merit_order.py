"""
merit_order.py — Merit Order Dispatch & Carbon/Cost Intensity Engine
=====================================================================
Computes hour-by-hour generation mix, cost, and carbon intensity for India's
grid based on predicted demand and available capacity by source.

Merit order (cheapest dispatch first):
  1. Nuclear    ₹1.5/kWh  0.000 kg CO2/kWh  must-run baseload
  2. Hydro      ₹0.5/kWh  0.000 kg CO2/kWh  run-of-river must-run + reservoir dispatchable
  3. Solar      ₹2.8/kWh  0.000 kg CO2/kWh  must-run when available
  4. Wind       ₹3.2/kWh  0.000 kg CO2/kWh  must-run when available
  5. Other RE   ₹4.0/kWh  0.050 kg CO2/kWh  biomass + SHP
  6. Coal (old) ₹4.2/kWh  0.820 kg CO2/kWh  old PPAs, low variable cost
  7. Coal (new) ₹6.0/kWh  0.820 kg CO2/kWh  new PPAs, higher tariff
  8. Gas        ₹7.5/kWh  0.450 kg CO2/kWh  peaking plant, expensive

Cost data sources:
  - MERIT India (meritindia.in) variable cost data
  - CERC benchmark tariff orders 2024-25
  - IISD India energy cost research 2025 (₹6.64/kWh new coal PPAs)
  - India average emission factor: 0.477 kg CO2/kWh (CEA 2024)

T&D losses: 19.2% (CEA FY2023-24) — generation must be higher than demand
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


# ── Source parameters ────────────────────────────────────────────────────────
# variable_cost: ₹/kWh  (what the grid pays to dispatch one more kWh)
# co2_kg_kwh:   kg CO2 per kWh generated
# label:        display name

MERIT_ORDER = [
    {"id": "nuclear",   "label": "Nuclear",       "variable_cost": 1.50, "co2_kg_kwh": 0.000, "color": "#c084fc"},
    {"id": "hydro",     "label": "Hydro",          "variable_cost": 0.50, "co2_kg_kwh": 0.000, "color": "#00d4aa"},
    {"id": "solar",     "label": "Solar",          "variable_cost": 2.80, "co2_kg_kwh": 0.000, "color": "#ffd60a"},
    {"id": "wind",      "label": "Wind",           "variable_cost": 3.20, "co2_kg_kwh": 0.000, "color": "#4da6ff"},
    {"id": "other",     "label": "Other RE",       "variable_cost": 4.00, "co2_kg_kwh": 0.050, "color": "#888888"},
    {"id": "coal_old",  "label": "Coal (old PPA)", "variable_cost": 4.20, "co2_kg_kwh": 0.820, "color": "#ff6b35"},
    {"id": "coal_new",  "label": "Coal (new PPA)", "variable_cost": 6.00, "co2_kg_kwh": 0.820, "color": "#ff4d6a"},
    {"id": "gas",       "label": "Gas",            "variable_cost": 7.50, "co2_kg_kwh": 0.450, "color": "#ffb347"},
]

# T&D loss factor: generation must cover this overhead
TD_LOSS_FACTOR = 1.192  # 19.2% losses

# Old coal is ~60% of thermal fleet (pre-2015 PPAs), new coal ~35%, gas ~5%
COAL_OLD_FRACTION = 0.60
COAL_NEW_FRACTION = 0.35
GAS_FRACTION      = 0.05


@dataclass
class MeritHour:
    hour:           int
    label:          str    # "00:00", "01:00" etc
    demand_mw:      float  # predicted demand at bus bar
    gen_required_mw: float # demand / (1 - TD_loss) — what plants must generate

    # Dispatch mix (MW generated from each source)
    dispatch: Dict[str, float] = field(default_factory=dict)

    # Aggregate metrics
    renewable_mw:     float = 0.0
    thermal_mw:       float = 0.0
    renewable_pct:    float = 0.0

    # Cost & carbon
    avg_cost_rs_kwh:  float = 0.0  # weighted avg ₹/kWh
    marginal_cost:    float = 0.0  # cost of last (most expensive) source dispatched
    co2_kg_kwh:       float = 0.0  # kg CO2 per kWh consumed
    co2_intensity_label: str = ""  # "Low", "Moderate", "High"
    cost_label:       str = ""     # "Cheap", "Moderate", "Expensive"
    color:            str = "#00d4aa"  # green/amber/red

    # Curtailment (renewable available but not needed)
    curtailed_mw:     float = 0.0
    curtailed_pct:    float = 0.0


def compute_merit_dispatch(
    demand_mw: float,
    available_by_source: Dict[str, float],
    hour: int,
) -> MeritHour:
    """
    Given predicted demand and available capacity per source,
    compute the merit-order dispatch, cost, and carbon intensity.

    available_by_source: {
        "nuclear": MW, "hydro": MW, "solar": MW, "wind": MW,
        "other": MW, "thermal": MW  (thermal is split into old/new/gas)
    }
    """
    label = f"{hour:02d}:00"

    # Generation required = demand adjusted for T&D losses
    gen_required = demand_mw * TD_LOSS_FACTOR

    # Split thermal into old coal / new coal / gas using fleet fractions
    thermal_available = available_by_source.get("thermal", 0)
    avail = {
        "nuclear":  available_by_source.get("nuclear", 0),
        "hydro":    available_by_source.get("hydro", 0),
        "solar":    available_by_source.get("solar", 0),
        "wind":     available_by_source.get("wind", 0),
        "other":    available_by_source.get("other", 0),
        "coal_old": thermal_available * COAL_OLD_FRACTION,
        "coal_new": thermal_available * COAL_NEW_FRACTION,
        "gas":      thermal_available * GAS_FRACTION,
    }

    # Dispatch in merit order
    remaining = gen_required
    dispatch   = {}
    total_cost = 0.0
    total_co2  = 0.0   # kg per hour = kg/kWh × MW × 1000 (MW→kW)
    marginal_source = None

    total_renewable_available = (avail["solar"] + avail["wind"] +
                                  avail["hydro"] + avail["nuclear"] + avail["other"])

    for src in MERIT_ORDER:
        sid  = src["id"]
        cap  = avail.get(sid, 0)
        used = min(remaining, cap)
        dispatch[sid] = round(used, 1)
        if used > 0:
            total_cost += used * src["variable_cost"]  # MW × ₹/kWh = ₹/h × 1000
            total_co2  += used * src["co2_kg_kwh"]    # MW × kg/kWh = kg/h × 1000
            marginal_source = src
        remaining  = max(0, remaining - used)
        if remaining <= 0:
            break

    # Curtailment: renewable available beyond what was needed
    renewable_dispatched = sum(dispatch.get(s, 0)
                               for s in ["nuclear","hydro","solar","wind","other"])
    renewable_available  = total_renewable_available
    curtailed = max(0, renewable_available - renewable_dispatched)

    thermal_dispatched = (dispatch.get("coal_old", 0) +
                          dispatch.get("coal_new", 0) +
                          dispatch.get("gas", 0))

    renewable_pct = (renewable_dispatched / gen_required * 100) if gen_required > 0 else 0

    # Weighted average cost ₹/kWh
    avg_cost = total_cost / gen_required if gen_required > 0 else 0

    # CO2 intensity kg/kWh at point of consumption (includes T&D losses)
    co2_kwh = total_co2 / (demand_mw * 1000) if demand_mw > 0 else 0

    # Labels and colour
    if co2_kwh < 0.25:
        ci_label = "Very Low"; color = "#00d4aa"
    elif co2_kwh < 0.40:
        ci_label = "Low";       color = "#4ade80"
    elif co2_kwh < 0.55:
        ci_label = "Moderate";  color = "#ffd60a"
    elif co2_kwh < 0.70:
        ci_label = "High";      color = "#ffb347"
    else:
        ci_label = "Very High"; color = "#ff4d6a"

    if avg_cost < 3.0:
        cost_label = "Very Cheap"
    elif avg_cost < 4.5:
        cost_label = "Cheap"
    elif avg_cost < 5.5:
        cost_label = "Moderate"
    elif avg_cost < 6.5:
        cost_label = "Expensive"
    else:
        cost_label = "Very Expensive"

    return MeritHour(
        hour=hour,
        label=label,
        demand_mw=round(demand_mw, 1),
        gen_required_mw=round(gen_required, 1),
        dispatch=dispatch,
        renewable_mw=round(renewable_dispatched, 1),
        thermal_mw=round(thermal_dispatched, 1),
        renewable_pct=round(renewable_pct, 1),
        avg_cost_rs_kwh=round(avg_cost, 3),
        marginal_cost=round(marginal_source["variable_cost"], 2) if marginal_source else 0,
        co2_kg_kwh=round(co2_kwh, 4),
        co2_intensity_label=ci_label,
        cost_label=cost_label,
        color=color,
        curtailed_mw=round(curtailed, 1),
        curtailed_pct=round(curtailed / renewable_available * 100, 1) if renewable_available > 0 else 0.0,
    )


def compute_daily_insights(hours: List[MeritHour]) -> Dict:
    """
    Compute actionable insights from the 24-hour merit dispatch:
    - Best hours to run flexible loads (cheapest + cleanest)
    - Worst hours (most expensive + dirtiest)
    - Potential savings from shifting load
    - Total renewable curtailment
    """
    if not hours:
        return {}

    # Rank hours by cost + carbon (lower = better)
    scored = sorted(hours, key=lambda h: h.avg_cost_rs_kwh + h.co2_kg_kwh * 5)
    best_hours  = [h.hour for h in scored[:4]]
    worst_hours = [h.hour for h in scored[-4:]]

    peak_hour = max(hours, key=lambda h: h.demand_mw)
    cleanest  = min(hours, key=lambda h: h.co2_kg_kwh)
    dirtiest  = max(hours, key=lambda h: h.co2_kg_kwh)
    cheapest  = min(hours, key=lambda h: h.avg_cost_rs_kwh)
    expensive = max(hours, key=lambda h: h.avg_cost_rs_kwh)

    # Shifting 5 GW from worst to best hours
    shift_mw = 5000  # 5 GW flexible load shift
    cost_saving_per_kwh = expensive.avg_cost_rs_kwh - cheapest.avg_cost_rs_kwh
    co2_saving_per_kwh  = dirtiest.co2_kg_kwh - cleanest.co2_kg_kwh
    # Assuming 4 hours of shift
    shift_kwh = shift_mw * 4 * 1000  # kWh
    cost_saving_crore = cost_saving_per_kwh * shift_kwh / 1e7  # ₹ crore
    co2_saving_tonnes = co2_saving_per_kwh * shift_kwh / 1000  # tonnes

    total_curtailed = sum(h.curtailed_mw for h in hours)
    avg_renewable_pct = sum(h.renewable_pct for h in hours) / len(hours)
    avg_co2 = sum(h.co2_kg_kwh for h in hours) / len(hours)
    avg_cost = sum(h.avg_cost_rs_kwh for h in hours) / len(hours)

    return {
        "best_hours":          best_hours,
        "worst_hours":         worst_hours,
        "peak_demand_hour":    peak_hour.hour,
        "peak_demand_mw":      peak_hour.demand_mw,
        "cleanest_hour":       cleanest.hour,
        "cleanest_co2":        cleanest.co2_kg_kwh,
        "dirtiest_hour":       dirtiest.hour,
        "dirtiest_co2":        dirtiest.co2_kg_kwh,
        "cheapest_hour":       cheapest.hour,
        "cheapest_cost":       cheapest.avg_cost_rs_kwh,
        "expensive_hour":      expensive.hour,
        "expensive_cost":      expensive.avg_cost_rs_kwh,
        "cost_ratio":          round(expensive.avg_cost_rs_kwh / cheapest.avg_cost_rs_kwh, 2) if cheapest.avg_cost_rs_kwh > 0 else 1.0,
        "avg_renewable_pct":   round(avg_renewable_pct, 1),
        "avg_co2_kg_kwh":      round(avg_co2, 4),
        "avg_cost_rs_kwh":     round(avg_cost, 3),
        "total_curtailed_gwh": round(total_curtailed / 1000, 1),
        "shift_saving_crore":  round(cost_saving_crore, 1),
        "shift_co2_saving_t":  round(co2_saving_tonnes, 0),
        "merit_order":         [{"id": s["id"], "label": s["label"],
                                  "cost": s["variable_cost"], "co2": s["co2_kg_kwh"],
                                  "color": s["color"]} for s in MERIT_ORDER],
    }
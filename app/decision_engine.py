"""
Modules 5 & 6: Overload Detection + Decision Engine
-----------------------------------------------------
Calibrated for India's 5-region POSOCO grid (CEA 2023-24 data).

Region capacities (installed, CEA 2023-24):
  Northern  : 115,000 MW  (UP, Delhi, Rajasthan, Punjab, Haryana, HP, J&K)
  Western   : 130,000 MW  (Gujarat, Maharashtra, MP, Chhattisgarh, Goa)
  Southern  :  95,000 MW  (AP, Telangana, Karnataka, Tamil Nadu, Kerala)
  Eastern   :  55,000 MW  (West Bengal, Odisha, Bihar, Jharkhand)
  NE        :   4,500 MW  (Assam, Meghalaya, Manipur, Mizoram, Nagaland etc)
  All-India : ~399,500 MW

Costs in Indian Rupees (₹/MWh):
  Backup peaker (gas)   : ₹8,500/MWh
  Industrial curtailment: ₹2,200/MWh savings
  EV shift incentive    : ₹600/MWh
  CO₂ gas peaker        : 490 kg/MWh
  CO₂ grid average      : 820 kg/MWh (India coal-heavy mix)
"""

from typing import List, Dict
from dataclasses import dataclass, field

# ── Region capacities (MW) ────────────────────────────────────────────────────
REGION_CAPACITIES_MW = {
    "Northern_Region":      115000,
    "Western_Region":       130000,
    "Southern_Region":       95000,
    "Eastern_Region":        55000,
    "NorthEastern_Region":    4500,
    "ALL_INDIA":            399500,
}

# Default capacity for single-region mode
DEFAULT_CAPACITY_MW = 399500

# Demand segment shares
SEGMENT_SHARES = {
    "residential": 0.26,   # India: lower residential share vs US
    "industrial":  0.45,   # India: higher industrial share
    "agriculture": 0.18,   # India-specific: large agricultural load
    "commercial":  0.11,
}

# India-specific EV share (low, but growing)
EV_SHARE = 0.04

# Cost constants (₹/MWh)
COST_BACKUP_PER_MWH     = 8500
COST_INDUSTRY_SAVE_MWH  = 2200
COST_EV_INCENTIVE_MWH   = 600
COST_AGRI_INCENTIVE_MWH = 400

# Carbon (kg CO₂/MWh)
CO2_BACKUP_KG_MWH  = 490   # Gas peaker
CO2_GRID_KG_MWH    = 820   # India grid average (coal-heavy)


@dataclass
class OverloadResult:
    hour: int
    timestamp: str
    label: str
    predicted_mw: float
    is_overload: bool
    excess_mw: float
    region: str = "ALL_INDIA"


@dataclass
class ActionResult:
    name: str
    type: str
    reduction_mw: float
    description: str
    affected_segment: str
    impact_level: str
    cost_inr: float = 0.0
    co2_kg: float = 0.0


@dataclass
class HourAdjustment:
    hour: int
    label: str
    original_mw: float
    adjusted_mw: float
    actions_applied: List[str] = field(default_factory=list)


def detect_overloads(
    forecast: List[dict],
    capacity_mw: float = DEFAULT_CAPACITY_MW,
    region: str = "ALL_INDIA",
) -> List[OverloadResult]:
    """Module 5: Flag each hour where predicted demand exceeds capacity."""
    results = []
    cap = REGION_CAPACITIES_MW.get(region, capacity_mw)
    for entry in forecast:
        pred = entry["predicted_demand_mw"]
        excess = max(0.0, pred - cap)
        results.append(OverloadResult(
            hour=entry["hour"],
            timestamp=entry["timestamp"],
            label=entry["label"],
            predicted_mw=pred,
            is_overload=pred > cap,
            excess_mw=round(excess, 1),
            region=region,
        ))
    return results


def demand_response(
    overload_results: List[OverloadResult],
    ev_delay_hours: int = 2,
    industrial_curtail_pct: float = 0.15,
    agri_shift_pct: float = 0.10,
    backup_supply_mw: float = 2000.0,
    capacity_mw: float = None,
    region: str = "ALL_INDIA",
) -> Dict:
    """
    Module 6: India-specific demand response cascade.

    Strategy order:
    1. EV charging delay (low impact — small share but easy to shift)
    2. Agricultural pump load shift (India-specific — schedulable loads)
    3. Industrial curtailment (highest MW impact)
    4. Backup generation (gas peakers / diesel)
    """
    n = len(overload_results)
    cap = capacity_mw or REGION_CAPACITIES_MW.get(region, DEFAULT_CAPACITY_MW)
    demand = [r.predicted_mw for r in overload_results]
    actions_summary: List[ActionResult] = []

    # Strategy 1: EV Delay
    ev_reduction = {}
    for r in overload_results:
        if r.is_overload:
            ev_load = r.predicted_mw * EV_SHARE
            demand[r.hour] -= ev_load
            ev_reduction[r.hour] = ev_load
            target = min(r.hour + ev_delay_hours, n - 1)
            demand[target] += ev_load

    total_ev = sum(ev_reduction.values())
    if total_ev > 0:
        actions_summary.append(ActionResult(
            name="EV Charging Delay",
            type="reduction",
            reduction_mw=round(total_ev / max(len(ev_reduction), 1), 1),
            description=f"EV load shifted +{ev_delay_hours}h to off-peak window (4% of demand)",
            affected_segment="ev_charging",
            impact_level="low",
            cost_inr=round((total_ev / 1000) * COST_EV_INCENTIVE_MWH),
            co2_kg=0,
        ))

    # Strategy 2: Agricultural pump shift (India-specific)
    agri_reduction = {}
    for i, r in enumerate(overload_results):
        if demand[i] > cap:
            agri_load = r.predicted_mw * SEGMENT_SHARES["agriculture"]
            cut = agri_load * agri_shift_pct
            demand[i] -= cut
            agri_reduction[r.hour] = cut

    total_agri = sum(agri_reduction.values())
    if total_agri > 0:
        actions_summary.append(ActionResult(
            name="Agricultural Pump Deferral",
            type="reduction",
            reduction_mw=round(total_agri / max(len(agri_reduction), 1), 1),
            description=f"Irrigation pump loads deferred by 2-3h (18% agri segment, {int(agri_shift_pct*100)}% shifted)",
            affected_segment="agriculture",
            impact_level="low",
            cost_inr=round((total_agri / 1000) * COST_AGRI_INCENTIVE_MWH),
            co2_kg=0,
        ))

    # Strategy 3: Industrial curtailment
    industrial_reduction = {}
    for i, r in enumerate(overload_results):
        if demand[i] > cap:
            ind_load = r.predicted_mw * SEGMENT_SHARES["industrial"]
            cut = ind_load * industrial_curtail_pct
            demand[i] -= cut
            industrial_reduction[r.hour] = cut

    total_ind = sum(industrial_reduction.values())
    if total_ind > 0:
        actions_summary.append(ActionResult(
            name="Industrial Curtailment",
            type="reduction",
            reduction_mw=round(total_ind / max(len(industrial_reduction), 1), 1),
            description=f"Industrial segment ({int(SEGMENT_SHARES['industrial']*100)}%) reduced {int(industrial_curtail_pct*100)}% under DSM regulations",
            affected_segment="industrial",
            impact_level="medium",
            cost_inr=round((total_ind / 1000) * COST_INDUSTRY_SAVE_MWH * -1),  # negative = savings
            co2_kg=round((total_ind / 1000) * CO2_GRID_KG_MWH * -1),
        ))

    # Strategy 4: Backup generation
    backup_hours = []
    total_backup = 0
    for i, r in enumerate(overload_results):
        if demand[i] > cap:
            add = min(demand[i] - cap, backup_supply_mw)
            demand[i] -= add
            total_backup += add
            backup_hours.append(r.hour)

    if backup_hours:
        actions_summary.append(ActionResult(
            name="Backup Generation",
            type="supply",
            reduction_mw=round(backup_supply_mw, 1),
            description=f"Gas peaker / diesel DG sets activated during hours {backup_hours[:3]}{'...' if len(backup_hours)>3 else ''}",
            affected_segment="supply",
            impact_level="high",
            cost_inr=round((total_backup / 1000) * COST_BACKUP_PER_MWH),
            co2_kg=round((total_backup / 1000) * CO2_BACKUP_KG_MWH),
        ))

    # Build adjusted curve
    adjustments = []
    for i, r in enumerate(overload_results):
        applied = []
        if r.hour in ev_reduction:       applied.append("EV Delay")
        if r.hour in agri_reduction:     applied.append("Agri Shift")
        if r.hour in industrial_reduction: applied.append("Industrial Cut")
        if r.hour in backup_hours:       applied.append("Backup Gen")
        adjustments.append(HourAdjustment(
            hour=r.hour,
            label=r.label,
            original_mw=round(r.predicted_mw, 1),
            adjusted_mw=round(demand[i], 1),
            actions_applied=applied,
        ))

    overload_hours   = [r for r in overload_results if r.is_overload]
    peak_predicted   = max((r.predicted_mw for r in overload_results), default=0)
    peak_adjusted    = max(demand)
    total_reduction  = sum(r.predicted_mw - demand[r.hour] for r in overload_results)
    still_overloaded = sum(1 for i, r in enumerate(overload_results) if demand[i] > cap)

    # Cost totals
    total_cost_inr = sum(a.cost_inr for a in actions_summary)
    total_co2_kg   = sum(a.co2_kg for a in actions_summary)

    return {
        "region": region,
        "capacity_mw": cap,
        "peak_predicted_mw": round(peak_predicted, 1),
        "peak_adjusted_mw": round(peak_adjusted, 1),
        "total_overload_hours": len(overload_hours),
        "still_overloaded_hours": still_overloaded,
        "total_reduction_mw": round(total_reduction, 1),
        "total_cost_inr": round(total_cost_inr),
        "total_co2_kg": round(total_co2_kg),
        "actions": [
            {
                "name": a.name, "type": a.type,
                "reduction_mw": a.reduction_mw,
                "description": a.description,
                "affected_segment": a.affected_segment,
                "impact_level": a.impact_level,
                "cost_inr": a.cost_inr,
                "co2_kg": a.co2_kg,
            }
            for a in actions_summary
        ],
        "adjusted_curve": [
            {
                "hour": adj.hour, "label": adj.label,
                "original_mw": adj.original_mw,
                "adjusted_mw": adj.adjusted_mw,
                "actions_applied": adj.actions_applied,
                "still_overloaded": adj.adjusted_mw > cap,
            }
            for adj in adjustments
        ],
        "segment_breakdown": {
            seg: {
                "share_pct": int(share * 100),
                "peak_load_mw": round(peak_predicted * share, 1),
            }
            for seg, share in {**SEGMENT_SHARES, "ev_charging": EV_SHARE}.items()
        },
        "region_capacities": REGION_CAPACITIES_MW,
    }
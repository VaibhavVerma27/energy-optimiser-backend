"""
Module 5 & 6: Overload Detection + Decision Engine
----------------------------------------------------
- Compares predicted demand against grid capacity
- Detects overload windows
- Simulates demand-response strategies:
    1. EV charging delay (shift 20% load +2 hours)
    2. Industrial curtailment (reduce 40% segment by X%)
    3. Backup generation (add supply)
    4. Residential flex (optional voluntary reduction)
"""

from typing import List, Dict
from dataclasses import dataclass, field


GRID_CAPACITY_MW = 10_000

# Demand segment shares
SEGMENT_SHARES = {
    "residential": 0.40,
    "industrial":  0.40,
    "ev_charging":  0.20,
}


@dataclass
class OverloadResult:
    hour: int
    timestamp: str
    label: str
    predicted_mw: float
    is_overload: bool
    excess_mw: float


@dataclass
class ActionResult:
    name: str
    type: str                   # 'reduction' | 'supply'
    reduction_mw: float
    description: str
    affected_segment: str
    impact_level: str           # 'low' | 'medium' | 'high'


@dataclass
class HourAdjustment:
    hour: int
    label: str
    original_mw: float
    adjusted_mw: float
    actions_applied: List[str] = field(default_factory=list)


def detect_overloads(
    forecast: List[dict],
    capacity_mw: float = GRID_CAPACITY_MW,
) -> List[OverloadResult]:
    """
    Module 5: Flag each hour where predicted demand exceeds capacity.
    """
    results = []
    for entry in forecast:
        pred = entry["predicted_demand_mw"]
        excess = max(0.0, pred - capacity_mw)
        results.append(OverloadResult(
            hour=entry["hour"],
            timestamp=entry["timestamp"],
            label=entry["label"],
            predicted_mw=pred,
            is_overload=pred > capacity_mw,
            excess_mw=round(excess, 1),
        ))
    return results


def demand_response(
    overload_results: List[OverloadResult],
    ev_delay_hours: int = 2,
    industrial_curtail_pct: float = 0.15,
    backup_supply_mw: float = 500.0,
    residential_flex_pct: float = 0.05,
) -> Dict:
    """
    Module 6: Decision Engine

    Applies demand-response strategies to overloaded hours and returns
    an adjusted demand curve with per-action savings breakdown.

    Strategy cascade (applied in order until overload resolved):
    1. EV charging delay
    2. Industrial curtailment
    3. Backup generation
    4. Residential flex (optional)
    """
    n = len(overload_results)
    adjustments: List[HourAdjustment] = []
    actions_summary: List[ActionResult] = []
    total_ev_shifted = {}  # hour → MW shifted in from earlier hours

    # Build mutable demand array
    demand = [r.predicted_mw for r in overload_results]

    # --- Strategy 1: EV Charging Delay ---
    # Shift EV load from overload hours to hour + ev_delay_hours
    ev_reduction_per_hour = {}
    for r in overload_results:
        if r.is_overload:
            ev_load = r.predicted_mw * SEGMENT_SHARES["ev_charging"]
            ev_reduction_per_hour[r.hour] = ev_load
            demand[r.hour] -= ev_load
            # Add shifted load to a later hour (if within 24h window)
            target = r.hour + ev_delay_hours
            if target < n:
                demand[target] += ev_load
                total_ev_shifted[target] = total_ev_shifted.get(target, 0) + ev_load

    total_ev_shifted_mw = sum(ev_reduction_per_hour.values())
    if total_ev_shifted_mw > 0:
        actions_summary.append(ActionResult(
            name="EV Charging Delay",
            type="reduction",
            reduction_mw=round(total_ev_shifted_mw / max(len(ev_reduction_per_hour), 1), 1),
            description=f"EV load (20% of demand) shifted +{ev_delay_hours}h to off-peak window",
            affected_segment="ev_charging",
            impact_level="low",
        ))

    # --- Strategy 2: Industrial Curtailment ---
    industrial_savings = {}
    for i, r in enumerate(overload_results):
        if demand[i] > GRID_CAPACITY_MW:
            industrial_load = r.predicted_mw * SEGMENT_SHARES["industrial"]
            cut = industrial_load * industrial_curtail_pct
            demand[i] -= cut
            industrial_savings[r.hour] = cut

    total_industrial_cut = sum(industrial_savings.values())
    if total_industrial_cut > 0:
        actions_summary.append(ActionResult(
            name="Industrial Curtailment",
            type="reduction",
            reduction_mw=round(total_industrial_cut / max(len(industrial_savings), 1), 1),
            description=f"Industrial segment reduced by {int(industrial_curtail_pct*100)}% during overload hours",
            affected_segment="industrial",
            impact_level="medium",
        ))

    # --- Strategy 3: Backup Generation ---
    backup_hours = []
    for i, r in enumerate(overload_results):
        if demand[i] > GRID_CAPACITY_MW:
            supply_needed = min(demand[i] - GRID_CAPACITY_MW, backup_supply_mw)
            demand[i] -= supply_needed  # Effectively adds supply
            backup_hours.append(r.hour)

    if backup_hours:
        actions_summary.append(ActionResult(
            name="Backup Generation",
            type="supply",
            reduction_mw=round(backup_supply_mw, 1),
            description=f"Peaker plant activated during hours: {backup_hours}",
            affected_segment="supply",
            impact_level="high",
        ))

    # --- Strategy 4: Residential Flex (Optional) ---
    for i, r in enumerate(overload_results):
        if demand[i] > GRID_CAPACITY_MW:
            res_load = r.predicted_mw * SEGMENT_SHARES["residential"]
            cut = res_load * residential_flex_pct
            demand[i] -= cut

    # Build per-hour adjustment results
    for i, r in enumerate(overload_results):
        applied = []
        if r.hour in ev_reduction_per_hour:
            applied.append("EV Delay")
        if r.hour in industrial_savings:
            applied.append("Industrial Cut")
        if r.hour in backup_hours:
            applied.append("Backup Gen")
        adjustments.append(HourAdjustment(
            hour=r.hour,
            label=r.label,
            original_mw=round(r.predicted_mw, 1),
            adjusted_mw=round(demand[i], 1),
            actions_applied=applied,
        ))

    # Summary statistics
    overload_hours = [r for r in overload_results if r.is_overload]
    peak_predicted = max((r.predicted_mw for r in overload_results), default=0)
    peak_adjusted = max(demand)
    total_reduction = sum(r.predicted_mw - demand[r.hour] for r in overload_results)
    still_overloaded = sum(1 for i, r in enumerate(overload_results) if demand[i] > GRID_CAPACITY_MW)

    return {
        "capacity_mw": GRID_CAPACITY_MW,
        "peak_predicted_mw": round(peak_predicted, 1),
        "peak_adjusted_mw": round(peak_adjusted, 1),
        "total_overload_hours": len(overload_hours),
        "still_overloaded_hours": still_overloaded,
        "total_reduction_mw": round(total_reduction, 1),
        "actions": [
            {
                "name": a.name,
                "type": a.type,
                "reduction_mw": a.reduction_mw,
                "description": a.description,
                "affected_segment": a.affected_segment,
                "impact_level": a.impact_level,
            }
            for a in actions_summary
        ],
        "adjusted_curve": [
            {
                "hour": adj.hour,
                "label": adj.label,
                "original_mw": adj.original_mw,
                "adjusted_mw": adj.adjusted_mw,
                "actions_applied": adj.actions_applied,
                "still_overloaded": adj.adjusted_mw > GRID_CAPACITY_MW,
            }
            for adj in adjustments
        ],
        "segment_breakdown": {
            seg: {
                "share_pct": int(share * 100),
                "peak_load_mw": round(peak_predicted * share, 1),
            }
            for seg, share in SEGMENT_SHARES.items()
        },
    }
"""
Modules 5 & 6: Overload Detection + Decision Engine
India POSOCO Grid — CEA 2023-24

Key design: each OverloadResult carries its OWN per-hour dynamic capacity
so demand_response strategies work against the actual available MW at that
hour (not a single fixed number for all 24 hours).
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field

# ── Installed capacities (MW) — CEA 2023-24 ─────────────────────────────────
REGION_CAPACITIES_MW = {
    "Northern_Region":      115000,
    "Western_Region":       130000,
    "Southern_Region":       95000,
    "Eastern_Region":        55000,
    "NorthEastern_Region":    4500,
    "ALL_INDIA":            399500,
}
DEFAULT_CAPACITY_MW = 399500

# India demand segment shares (different from US/Europe)
SEGMENT_SHARES = {
    "residential": 0.26,
    "industrial":  0.45,
    "agriculture": 0.18,  # India-specific: schedulable irrigation pumps
    "commercial":  0.11,
}
EV_SHARE = 0.04  # Low but growing

# India cost constants (₹/MWh)
COST_BACKUP_PER_MWH    = 8500
COST_INDUSTRY_SAVE_MWH = 2200
COST_EV_INCENTIVE_MWH  = 600
COST_AGRI_INCENTIVE_MWH= 400

# Carbon (kg CO₂/MWh)
CO2_BACKUP_KG_MWH = 490   # Gas peaker
CO2_GRID_KG_MWH   = 820   # India coal-heavy grid average


@dataclass
class OverloadResult:
    hour: int
    timestamp: str
    label: str
    predicted_mw: float
    is_overload: bool
    excess_mw: float
    region: str = "ALL_INDIA"
    # Per-hour dynamic capacity — set by forecast router
    available_capacity_mw: float = 0.0


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


def detect_overloads(
    forecast: List[dict],
    capacity_mw: float = DEFAULT_CAPACITY_MW,
    region: str = "ALL_INDIA",
) -> List[OverloadResult]:
    """
    Flag each hour where predicted demand exceeds available capacity.
    Reads per-hour 'capacity_mw' from forecast if present (dynamic),
    otherwise uses the fixed capacity_mw fallback.
    """
    results = []
    fixed_cap = REGION_CAPACITIES_MW.get(region, capacity_mw)
    for entry in forecast:
        pred = entry["predicted_demand_mw"]
        # Use dynamic per-hour capacity if the forecast has it
        cap = float(entry.get("capacity_mw") or fixed_cap)
        excess = max(0.0, pred - cap)
        results.append(OverloadResult(
            hour=entry["hour"],
            timestamp=entry["timestamp"],
            label=entry["label"],
            predicted_mw=pred,
            is_overload=pred > cap,
            excess_mw=round(excess, 1),
            region=region,
            available_capacity_mw=cap,  # store per-hour capacity
        ))
    return results


def demand_response(
    overload_results: List[OverloadResult],
    ev_delay_hours: int = 2,
    industrial_curtail_pct: float = 0.15,
    agri_shift_pct: float = 0.10,
    backup_supply_mw: float = 2000.0,
    region: str = "ALL_INDIA",
) -> Dict:
    """
    India demand response cascade — uses per-hour dynamic capacity from
    each OverloadResult.available_capacity_mw.

    Cascade order:
    1. EV charging delay          (low impact, easy to shift)
    2. Agricultural pump deferral (India-specific, schedulable)
    3. Industrial curtailment     (highest MW impact under DSM)
    4. Backup generation          (gas peakers / diesel, last resort)
    """
    n = len(overload_results)
    if n == 0:
        return _empty_response(region)

    # Mutable demand array — we reduce this as strategies are applied
    demand = [r.predicted_mw for r in overload_results]
    # Per-hour capacity array (dynamic)
    caps = [
        r.available_capacity_mw if r.available_capacity_mw > 0
        else REGION_CAPACITIES_MW.get(region, DEFAULT_CAPACITY_MW)
        for r in overload_results
    ]
    actions_summary: List[ActionResult] = []

    # ── Strategy 1: EV Charging Delay ────────────────────────────────────────
    ev_reduction = {}
    for i, r in enumerate(overload_results):
        if r.is_overload:
            ev_load = r.predicted_mw * EV_SHARE
            demand[i] -= ev_load
            ev_reduction[r.hour] = ev_load
            target = min(i + ev_delay_hours, n - 1)
            demand[target] += ev_load

    total_ev = sum(ev_reduction.values())
    if total_ev > 0:
        actions_summary.append(ActionResult(
            name="EV Charging Delay",
            type="reduction",
            reduction_mw=round(total_ev / max(len(ev_reduction), 1), 1),
            description=f"EV loads (4% of demand) shifted +{ev_delay_hours}h to off-peak window",
            affected_segment="ev_charging",
            impact_level="low",
            cost_inr=round((total_ev / 1000) * COST_EV_INCENTIVE_MWH),
            co2_kg=0,
        ))

    # ── Strategy 2: Agricultural Pump Deferral ────────────────────────────────
    agri_reduction = {}
    for i, r in enumerate(overload_results):
        if demand[i] > caps[i]:
            cut = r.predicted_mw * SEGMENT_SHARES["agriculture"] * agri_shift_pct
            demand[i] -= cut
            agri_reduction[r.hour] = cut

    total_agri = sum(agri_reduction.values())
    if total_agri > 0:
        actions_summary.append(ActionResult(
            name="Agricultural Pump Deferral",
            type="reduction",
            reduction_mw=round(total_agri / max(len(agri_reduction), 1), 1),
            description=f"Irrigation pumps deferred 2-3h (18% agri segment, {int(agri_shift_pct*100)}% shifted)",
            affected_segment="agriculture",
            impact_level="low",
            cost_inr=round((total_agri / 1000) * COST_AGRI_INCENTIVE_MWH),
            co2_kg=0,
        ))

    # ── Strategy 3: Industrial Curtailment ────────────────────────────────────
    industrial_reduction = {}
    for i, r in enumerate(overload_results):
        if demand[i] > caps[i]:
            cut = r.predicted_mw * SEGMENT_SHARES["industrial"] * industrial_curtail_pct
            demand[i] -= cut
            industrial_reduction[r.hour] = cut

    total_ind = sum(industrial_reduction.values())
    if total_ind > 0:
        actions_summary.append(ActionResult(
            name="Industrial Curtailment",
            type="reduction",
            reduction_mw=round(total_ind / max(len(industrial_reduction), 1), 1),
            description=f"Industrial loads (45% segment) reduced {int(industrial_curtail_pct*100)}% under DSM",
            affected_segment="industrial",
            impact_level="medium",
            cost_inr=round((total_ind / 1000) * COST_INDUSTRY_SAVE_MWH * -1),
            co2_kg=round((total_ind / 1000) * CO2_GRID_KG_MWH * -1),
        ))

    # ── Strategy 4: Backup Generation ────────────────────────────────────────
    backup_hours = []
    total_backup = 0.0
    for i, r in enumerate(overload_results):
        if demand[i] > caps[i]:
            needed = demand[i] - caps[i]
            add = min(needed, backup_supply_mw)
            demand[i] -= add
            total_backup += add
            backup_hours.append(r.label)

    if total_backup > 0:
        actions_summary.append(ActionResult(
            name="Backup Generation",
            type="supply",
            reduction_mw=round(total_backup / max(len(backup_hours), 1), 1),
            description=f"Gas peakers / diesel DG sets activated at hours: {', '.join(backup_hours[:4])}{'...' if len(backup_hours)>4 else ''}",
            affected_segment="supply",
            impact_level="high",
            cost_inr=round((total_backup / 1000) * COST_BACKUP_PER_MWH),
            co2_kg=round((total_backup / 1000) * CO2_BACKUP_KG_MWH),
        ))

    # ── Build adjusted curve ──────────────────────────────────────────────────
    adjustments = []
    for i, r in enumerate(overload_results):
        applied = []
        if r.hour in ev_reduction:         applied.append("EV Delay")
        if r.hour in agri_reduction:       applied.append("Agri Shift")
        if r.hour in industrial_reduction: applied.append("Industrial Cut")
        if r.label in backup_hours:        applied.append("Backup Gen")
        adjustments.append({
            "hour":             r.hour,
            "label":            r.label,
            "original_mw":      round(r.predicted_mw, 1),
            "adjusted_mw":      round(demand[i], 1),
            "capacity_mw":      caps[i],
            "actions_applied":  applied,
            "still_overloaded": demand[i] > caps[i],
        })

    # ── Summary stats ─────────────────────────────────────────────────────────
    overload_hours    = [r for r in overload_results if r.is_overload]
    peak_predicted    = max((r.predicted_mw for r in overload_results), default=0)
    peak_adjusted     = max(demand)
    total_reduction   = sum(r.predicted_mw - demand[r.hour] for r in overload_results)
    still_overloaded  = sum(1 for i in range(n) if demand[i] > caps[i])
    total_cost_inr    = sum(a.cost_inr for a in actions_summary)
    total_co2_kg      = sum(a.co2_kg for a in actions_summary)

    return {
        "region":                region,
        "total_overload_hours":  len(overload_hours),
        "still_overloaded_hours":still_overloaded,
        "peak_predicted_mw":     round(peak_predicted, 1),
        "peak_adjusted_mw":      round(peak_adjusted, 1),
        "total_reduction_mw":    round(total_reduction, 1),
        "total_cost_inr":        round(total_cost_inr),
        "total_co2_kg":          round(total_co2_kg),
        "actions": [
            {
                "name":             a.name,
                "type":             a.type,
                "reduction_mw":     a.reduction_mw,
                "description":      a.description,
                "affected_segment": a.affected_segment,
                "impact_level":     a.impact_level,
                "cost_inr":         a.cost_inr,
                "co2_kg":           a.co2_kg,
            }
            for a in actions_summary
        ],
        "adjusted_curve": adjustments,
        "segment_breakdown": {
            seg: {
                "share_pct":     int(share * 100),
                "peak_load_mw":  round(peak_predicted * share, 1),
            }
            for seg, share in {**SEGMENT_SHARES, "ev_charging": EV_SHARE}.items()
        },
    }


def _empty_response(region: str) -> Dict:
    return {
        "region": region, "total_overload_hours": 0, "still_overloaded_hours": 0,
        "peak_predicted_mw": 0, "peak_adjusted_mw": 0, "total_reduction_mw": 0,
        "total_cost_inr": 0, "total_co2_kg": 0,
        "actions": [], "adjusted_curve": [],
        "segment_breakdown": {
            seg: {"share_pct": int(s*100), "peak_load_mw": 0}
            for seg, s in {**SEGMENT_SHARES, "ev_charging": EV_SHARE}.items()
        },
    }
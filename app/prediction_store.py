"""
prediction_store.py
===================
SQLite-based store for saving forecasts and comparing against actuals.

Tables:
  predictions       — one row per forecast run (metadata)
  forecast_hours    — one row per forecast hour (24 per run)
  actuals           — one row per actual data point fetched from NPP

This enables the comparison page to:
  1. Show any past prediction
  2. Fetch the NPP actual for that date
  3. Compute MAE, MAPE, peak error, and display side-by-side chart
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# Store the DB in the data directory next to demand.csv
DB_PATH = Path(__file__).parent / "data" / "predictions.db"


def init_db():
    """Create database and tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT UNIQUE NOT NULL,         -- e.g. "2024-05-15T18:00_live"
            created_at      TEXT NOT NULL,                -- ISO datetime
            forecast_date   TEXT NOT NULL,                -- date being forecast e.g. "2024-05-16"
            mode            TEXT NOT NULL DEFAULT 'auto', -- 'auto' | 'custom' | 'live'
            weather_source  TEXT,                         -- 'live' | 'csv' | 'seasonal'
            model_features  INTEGER,                      -- 33
            notes           TEXT,
            all_india_peak_mw REAL,
            has_actuals     INTEGER DEFAULT 0             -- 1 when NPP data fetched
        );

        CREATE TABLE IF NOT EXISTS forecast_hours (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            region          TEXT NOT NULL,
            hour            INTEGER NOT NULL,             -- 0-23
            timestamp       TEXT NOT NULL,               -- ISO
            predicted_mw    REAL NOT NULL,
            ci_lower_mw     REAL,
            ci_upper_mw     REAL,
            capacity_mw     REAL,
            solar_mw        REAL,
            wind_mw         REAL,
            hydro_mw        REAL,
            thermal_mw      REAL,
            is_overload     INTEGER DEFAULT 0,
            weather_temp_c  REAL,
            weather_solar_wm2 REAL,
            FOREIGN KEY (run_id) REFERENCES predictions(run_id)
        );

        CREATE TABLE IF NOT EXISTS actuals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            actual_date     TEXT NOT NULL,
            region          TEXT NOT NULL,
            actual_daily_mu REAL,                         -- from DGR1 (MWh/1000)
            ntpc_available_mw REAL,                       -- from DGR8
            ntpc_actual_mu  REAL,                         -- from DGR8
            source          TEXT,                         -- 'NPP_DGR1' | 'NPP_DGR8'
            fetched_at      TEXT NOT NULL,
            UNIQUE(actual_date, region, source)
        );

        CREATE TABLE IF NOT EXISTS model_performance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            region          TEXT NOT NULL,
            mae_mw          REAL,
            mape_pct        REAL,
            peak_error_mw   REAL,
            peak_error_pct  REAL,
            computed_at     TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(forecast_date);
        CREATE INDEX IF NOT EXISTS idx_hours_run  ON forecast_hours(run_id, region);
        CREATE INDEX IF NOT EXISTS idx_actuals_date ON actuals(actual_date, region);
        """)
    return str(DB_PATH)


@contextmanager
def _conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Save a forecast run ───────────────────────────────────────────────────────

def save_forecast(
    run_id: str,
    forecast_date: str,
    mode: str,
    weather_source: str,
    model_features: int,
    all_india_peak_mw: float,
    region_forecasts: dict,   # {region_id: [24 forecast hour dicts]}
    notes: str = "",
):
    """
    Save a complete forecast run to the database.
    region_forecasts: dict keyed by region_id, each value is the list of
    forecast hour dicts from predict_24h().
    """
    init_db()
    created_at = datetime.utcnow().isoformat()

    with _conn() as conn:
        # Remove any existing forecast for the same date so there is always
        # exactly one entry per forecast_date in the list.
        conn.execute("""
            DELETE FROM forecast_hours WHERE run_id IN (
                SELECT run_id FROM predictions WHERE forecast_date = ? AND run_id != ?
            )
        """, (forecast_date, run_id))
        conn.execute("""
            DELETE FROM predictions WHERE forecast_date = ? AND run_id != ?
        """, (forecast_date, run_id))

        # Upsert predictions row
        conn.execute("""
            INSERT OR REPLACE INTO predictions
            (run_id, created_at, forecast_date, mode, weather_source,
             model_features, notes, all_india_peak_mw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, created_at, forecast_date, mode, weather_source,
               model_features, notes, all_india_peak_mw))

        # Delete old forecast hours for this run_id (idempotent)
        conn.execute("DELETE FROM forecast_hours WHERE run_id = ?", (run_id,))

        # Insert forecast hours
        rows = []
        for region_id, hours in region_forecasts.items():
            for h in hours:
                rows.append((
                    run_id, region_id, h.get("hour", 0), h.get("timestamp", ""),
                    h.get("predicted_demand_mw", 0),
                    h.get("ci_lower_mw"), h.get("ci_upper_mw"),
                    h.get("capacity_mw"),
                    h.get("solar_available_mw"), h.get("wind_available_mw"),
                    h.get("hydro_available_mw"), h.get("thermal_available_mw"),
                    1 if h.get("predicted_demand_mw", 0) > h.get("capacity_mw", 9e9) else 0,
                    h.get("weather_temp_c"), h.get("weather_solar_wm2"),
                ))
        conn.executemany("""
            INSERT INTO forecast_hours
            (run_id, region, hour, timestamp, predicted_mw, ci_lower_mw, ci_upper_mw,
             capacity_mw, solar_mw, wind_mw, hydro_mw, thermal_mw, is_overload,
             weather_temp_c, weather_solar_wm2)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)

    return run_id


def save_actuals(actual_date: str, actuals_data: dict):
    """
    Save actual demand/generation from NPP for a date.
    actuals_data: return value of live_data_fetcher.fetch_actual_demand_for_date()
    """
    init_db()
    fetched_at = datetime.utcnow().isoformat()

    with _conn() as conn:
        # DGR1 daily generation totals
        if actuals_data.get("dgr1") and actuals_data["dgr1"].get("regions"):
            for region_id, mu_value in actuals_data["dgr1"]["regions"].items():
                conn.execute("""
                    INSERT OR REPLACE INTO actuals
                    (actual_date, region, actual_daily_mu, source, fetched_at)
                    VALUES (?, ?, ?, 'NPP_DGR1', ?)
                """, (actual_date, region_id, mu_value, fetched_at))

        # DGR8 NTPC capacity/actual
        if actuals_data.get("dgr8") and actuals_data["dgr8"].get("regions"):
            for region_id, vals in actuals_data["dgr8"]["regions"].items():
                conn.execute("""
                    INSERT OR REPLACE INTO actuals
                    (actual_date, region, ntpc_available_mw, ntpc_actual_mu,
                     source, fetched_at)
                    VALUES (?, ?, ?, ?, 'NPP_DGR8', ?)
                """, (actual_date, region_id,
                      vals.get("available_mw"), vals.get("actual_mu"),
                      fetched_at))

        # Mark prediction as having actuals
        conn.execute("""
            UPDATE predictions SET has_actuals = 1
            WHERE forecast_date = ?
        """, (actual_date,))


# ── Query functions ───────────────────────────────────────────────────────────

def delete_prediction(run_id: str) -> bool:
    """Delete a saved forecast run and all its hourly data. Returns True if found and deleted."""
    init_db()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT run_id FROM predictions WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not existing:
            return False
        conn.execute("DELETE FROM forecast_hours WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM predictions WHERE run_id = ?", (run_id,))
    return True


def list_predictions(limit: int = 30) -> list:
    """Return recent prediction runs, newest first."""
    init_db()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT run_id, created_at, forecast_date, mode, weather_source,
                   model_features, all_india_peak_mw, has_actuals, notes
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_prediction(run_id: str) -> Optional[dict]:
    """Get full prediction detail including all forecast hours."""
    init_db()
    with _conn() as conn:
        meta = conn.execute(
            "SELECT * FROM predictions WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not meta:
            return None

        hours = conn.execute("""
            SELECT region, hour, timestamp, predicted_mw, ci_lower_mw, ci_upper_mw,
                   capacity_mw, solar_mw, wind_mw, hydro_mw, thermal_mw,
                   is_overload, weather_temp_c
            FROM forecast_hours WHERE run_id = ?
            ORDER BY region, hour
        """, (run_id,)).fetchall()

    # Group hours by region
    by_region = {}
    for h in hours:
        r = h["region"]
        if r not in by_region:
            by_region[r] = []
        by_region[r].append(dict(h))

    return {**dict(meta), "forecast_by_region": by_region}


def get_actuals_for_date(actual_date: str) -> dict:
    """Get all saved actuals for a date."""
    init_db()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT region, actual_daily_mu, ntpc_available_mw, ntpc_actual_mu,
                   source, fetched_at
            FROM actuals WHERE actual_date = ?
        """, (actual_date,)).fetchall()
    return {r["region"]: dict(r) for r in rows}


def compute_comparison(run_id: str) -> Optional[dict]:
    """
    Compare a saved prediction against actual NPP data.
    Returns per-region metrics: MAE, MAPE, peak error.

    Comparison method:
    - Predicted: sum of hourly predicted_mw / 1000 = predicted daily MU
    - Actual: actual_daily_mu from NPP DGR1
    - Difference in MU and percentage
    """
    pred = get_prediction(run_id)
    if not pred:
        return None

    actuals = get_actuals_for_date(pred["forecast_date"])
    if not actuals:
        return {"run_id": run_id, "error": "No actuals available yet for this date"}

    comparison = {
        "run_id":         run_id,
        "forecast_date":  pred["forecast_date"],
        "mode":           pred["mode"],
        "weather_source": pred["weather_source"],
        "regions":        {},
    }

    total_pred_mu = 0
    total_actual_mu = 0

    # Check if we have All-India actuals from a MERIT CSV upload
    # This is more reliable than regional NPP data
    all_india_actual    = actuals.get("ALL_INDIA", {})
    all_india_mu        = all_india_actual.get("actual_daily_mu")
    all_india_hourly_rec = actuals.get("ALL_INDIA_HOURLY", {})
    hourly_actual_json  = all_india_hourly_rec.get("ntpc_actual_mu")
    hourly_actual: dict = {}
    if hourly_actual_json and isinstance(hourly_actual_json, str):
        import json as _json
        try:
            hourly_actual = _json.loads(hourly_actual_json)
        except Exception:
            pass

    for region_id, hours in pred["forecast_by_region"].items():
        if region_id == "ALL_INDIA":
            continue

        pred_daily_mu = sum(h["predicted_mw"] for h in hours) / 1000
        pred_peak_mw  = max(h["predicted_mw"] for h in hours)

        # Priority: region-specific actual → All-India × region share
        actual    = actuals.get(region_id, {})
        actual_mu = actual.get("actual_daily_mu") or actual.get("ntpc_actual_mu")

        # If no region actual but we have All-India from MERIT upload,
        # estimate regional actual using the SAME shares used at forecast time.
        # Using static BASE_REGION_SHARES here would cause a mismatch:
        # predictions used dynamic hour×month shares (e.g. Northern peak 32.38%)
        # while actuals would be split at 29.88% → artificial overprediction.
        # Fix: average the per-hour region_share stored in the forecast hours.
        if not actual_mu and all_india_mu:
            stored_shares = [h.get("region_share") for h in hours if h.get("region_share")]
            if stored_shares:
                reg_share = sum(stored_shares) / len(stored_shares)
            else:
                from merit_parser import BASE_REGION_SHARES
                col_key   = region_id + "_mw"
                reg_share = BASE_REGION_SHARES.get(col_key, 0.2)
            actual_mu = round(all_india_mu * reg_share, 2)

        if actual_mu:
            error_mu  = pred_daily_mu - actual_mu
            error_pct = (error_mu / actual_mu) * 100 if actual_mu else None
            mae       = abs(error_mu)
            mape      = abs(error_pct) if error_pct is not None else None
            total_pred_mu   += pred_daily_mu
            total_actual_mu += actual_mu
        else:
            error_mu = error_pct = mae = mape = None

        comparison["regions"][region_id] = {
            "predicted_daily_mu":  round(pred_daily_mu, 2),
            "actual_daily_mu":     round(actual_mu, 2) if actual_mu else None,
            "error_mu":            round(error_mu, 2) if error_mu is not None else None,
            "error_pct":           round(error_pct, 2) if error_pct is not None else None,
            "mae_mu":              round(mae, 2) if mae is not None else None,
            "mape_pct":            round(mape, 2) if mape is not None else None,
            "predicted_peak_mw":   round(pred_peak_mw),
            "ntpc_available_mw":   actual.get("ntpc_available_mw"),
            "hourly_forecast":     hours,
            "hourly_actual":       hourly_actual,  # from MERIT upload
        }

    # All-India summary — use uploaded MERIT total if available
    if all_india_mu:
        all_india_pred_mu = sum(
            sum(h["predicted_mw"] for h in hrs) / 1000
            for rid, hrs in pred["forecast_by_region"].items()
            if rid != "ALL_INDIA"
        )
        err = all_india_pred_mu - all_india_mu
        comparison["all_india_summary"] = {
            "predicted_mu": round(all_india_pred_mu, 2),
            "actual_mu":    round(all_india_mu, 2),
            "error_mu":     round(err, 2),
            "error_pct":    round(err / all_india_mu * 100, 2),
            "mae_mu":       round(abs(err), 2),
            "source":       "MERIT_CSV_UPLOAD",
        }
    elif total_pred_mu > 0 and total_actual_mu > 0:
        all_india_err = total_pred_mu - total_actual_mu
        comparison["all_india_summary"] = {
            "predicted_mu":  round(total_pred_mu, 2),
            "actual_mu":     round(total_actual_mu, 2),
            "error_mu":      round(all_india_err, 2),
            "error_pct":     round((all_india_err / total_actual_mu) * 100, 2),
            "mae_mu":        round(abs(all_india_err), 2),
            "source":        "regional_sum",
        }

    return comparison


def get_rolling_performance(days: int = 30) -> dict:
    """
    Compute rolling MAE over the last N days where we have both
    predictions and actuals saved.
    """
    init_db()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT p.run_id, p.forecast_date, f.region,
                   SUM(f.predicted_mw) / 1000.0 AS pred_mu,
                   a.actual_daily_mu
            FROM predictions p
            JOIN forecast_hours f ON f.run_id = p.run_id
            JOIN actuals a ON a.actual_date = p.forecast_date
                           AND a.region = f.region
                           AND a.source = 'NPP_DGR1'
            WHERE p.has_actuals = 1
              AND p.created_at >= datetime('now', '-' || ? || ' days')
            GROUP BY p.run_id, p.forecast_date, f.region
            ORDER BY p.forecast_date DESC
        """, (days,)).fetchall()

    if not rows:
        return {"days": days, "samples": 0, "mae_mu": None}

    errors = [abs(r["pred_mu"] - r["actual_daily_mu"])
              for r in rows if r["actual_daily_mu"]]
    return {
        "days":     days,
        "samples":  len(errors),
        "mae_mu":   round(sum(errors) / len(errors), 2) if errors else None,
        "min_error":round(min(errors), 2) if errors else None,
        "max_error":round(max(errors), 2) if errors else None,
    }
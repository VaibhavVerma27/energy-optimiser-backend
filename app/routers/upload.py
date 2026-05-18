"""
routers/upload.py
=================
Handles MERIT India CSV file uploads.

POST /api/upload/merit-demand
    - Accepts one or more CSV files from the live page
    - Parses them with merit_parser
    - Returns the 168h history ready for the live forecast
    - Optionally saves to data/merit_history.json for automatic reuse

GET /api/upload/merit-status
    - Returns info about the currently saved MERIT history
    - Shows coverage, last timestamp, mean demand
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from merit_parser import build_history_from_merit_files, get_latest_hourly_demand, parse_merit_csv, resample_to_hourly

router   = APIRouter()
DATA_DIR = Path(__file__).parent.parent / "data"
MERIT_CACHE = DATA_DIR / "merit_history.json"


@router.post("/merit-demand")
async def upload_merit_demand(files: List[UploadFile] = File(...)):
    """
    Upload one or more MERIT India Demand Met CSV files.

    How to get the files:
      1. Open npp.gov.in/dashBoard/gc-map-dashboard-meritchart
      2. Under "Real Time Demand Met Data" → click "Previous Data"
      3. Select each of the last 7 days → Download Excel (saves as CSV)
      4. Upload all files here together

    The endpoint will:
    - Parse all files
    - Resample 4-min readings to hourly
    - Split All-India demand into per-region estimates
    - Save the result to data/merit_history.json
    - Return the history ready for immediate use in live forecasting
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    raw_sources = []
    filenames   = []
    for f in files:
        content = await f.read()
        raw_sources.append(content)
        filenames.append(f.filename)

    try:
        history = build_history_from_merit_files(raw_sources, target_hours=168)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse files: {e}")

    # Save to cache for automatic reuse by live forecast
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_data = {
        **history,
        "saved_at":  datetime.utcnow().isoformat(),
        "filenames": filenames,
    }
    with open(MERIT_CACHE, "w") as fout:
        json.dump(cache_data, fout)

    # Warn about gaps so user knows which days to download
    gap_warning = None
    if history["data_quality"]["gaps_filled"] > 24:
        gap_warning = (
            f"{history['data_quality']['gaps_filled']} hours were interpolated "
            f"(coverage {history['data_quality']['coverage_pct']}%). "
            f"For best accuracy upload a file for every day in the 7-day window."
        )

    return {
        "status":            "ok",
        "files_processed":   len(files),
        "filenames":         filenames,
        "hours_available":   history["hours_available"],
        "hours_today":       history["hours_today"],
        "first_timestamp":   history["first_timestamp"],
        "last_timestamp":    history["last_timestamp"],
        "mean_demand_mw":    history["mean_mw"],
        "peak_demand_mw":    history["peak_mw"],
        "trough_demand_mw":  history["trough_mw"],
        "data_quality":      history["data_quality"],
        "gap_warning":       gap_warning,
        "message": (
            f"Successfully built {history['hours_available']}h history from {len(files)} file(s). "
            f"Coverage: {history['data_quality']['coverage_pct']}%."
            + (f" ⚠ {gap_warning}" if gap_warning else " Ready to forecast.")
        ),
    }


@router.get("/merit-status")
async def get_merit_status():
    """Check if MERIT history is available and how fresh it is."""
    if not MERIT_CACHE.exists():
        return {
            "available":   False,
            "message":     "No MERIT data uploaded yet. Upload CSV files to use real demand data.",
        }

    try:
        with open(MERIT_CACHE) as f:
            data = json.load(f)

        last_ts  = datetime.fromisoformat(data["last_timestamp"])
        saved_at = datetime.fromisoformat(data["saved_at"])
        age_hrs  = (datetime.utcnow() - saved_at).total_seconds() / 3600

        return {
            "available":         True,
            "hours_available":   data["hours_available"],
            "hours_today":       data.get("hours_today", 0),
            "first_timestamp":   data["first_timestamp"],
            "last_timestamp":    data["last_timestamp"],
            "saved_at":          data["saved_at"],
            "age_hours":         round(age_hrs, 1),
            "mean_demand_mw":    data.get("mean_mw", 0),
            "peak_demand_mw":    data.get("peak_mw", 0),
            "coverage_pct":      data.get("data_quality", {}).get("coverage_pct", 0),
            "filenames":         data.get("filenames", []),
            "stale":             age_hrs > 6,   # warn if >6h old
            "message": (
                f"MERIT data from {last_ts.strftime('%d %b %H:%M')} IST · "
                f"{data['hours_available']}h history · "
                f"Coverage {data.get('data_quality',{}).get('coverage_pct',0)}%"
                + (" · ⚠ Stale, please re-upload today's file" if age_hrs > 6 else "")
            ),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@router.post("/actual-demand")
async def upload_actual_demand(files: List[UploadFile] = File(...)):
    """
    Upload one or more MERIT India Demand Met CSV files for the compare page.

    Unlike /merit-demand (which builds 168h history for forecasting),
    this endpoint extracts the actual daily and hourly demand from each
    uploaded file and saves it to the SQLite actuals table for comparison.

    One file = one day's actuals. Upload multiple files for multiple days.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    from prediction_store import init_db, save_actuals
    import sqlite3

    init_db()
    results = []

    for f in files:
        content = await f.read()
        try:
            df = parse_merit_csv(content)
            df = df.sort_values("timestamp")

            # Get the date from the data
            actual_date = df["timestamp"].dt.date.iloc[0].isoformat()

            # Integration for daily total (trapezoidal — more accurate than avg×24)
            df["minutes"] = df["timestamp"].diff().dt.total_seconds().fillna(240) / 60
            df["mwh"]     = df["demand_mw"] * df["minutes"] / 60
            total_mu      = round(df["mwh"].sum() / 1000, 3)   # MU = MWh/1000

            # Hourly profile
            hourly = resample_to_hourly(df).set_index("hour_start")
            hourly_mw = {
                row.name.strftime("%H:00"): int(row["demand_mw"])
                for _, row in hourly.iterrows()
            }

            # Save to actuals table
            # Use ALL_INDIA key for the total, and store hourly as JSON
            import json
            from datetime import datetime as _dt
            from pathlib import Path

            DB_PATH = Path(__file__).parent.parent / "data" / "predictions.db"
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute("""
                INSERT OR REPLACE INTO actuals
                (actual_date, region, actual_daily_mu, source, fetched_at)
                VALUES (?, 'ALL_INDIA', ?, 'MERIT_CSV_UPLOAD', ?)
            """, (actual_date, total_mu, _dt.utcnow().isoformat()))

            # Also store hourly profile as a separate source
            conn.execute("""
                INSERT OR REPLACE INTO actuals
                (actual_date, region, actual_daily_mu, ntpc_actual_mu, source, fetched_at)
                VALUES (?, 'ALL_INDIA_HOURLY', ?, ?, 'MERIT_CSV_HOURLY', ?)
            """, (actual_date, total_mu,
                  json.dumps(hourly_mw),   # store as JSON in ntpc_actual_mu field
                  _dt.utcnow().isoformat()))

            # Mark any matching predictions as having actuals
            conn.execute("""
                UPDATE predictions SET has_actuals = 1
                WHERE forecast_date = ?
            """, (actual_date,))
            conn.commit()
            conn.close()

            results.append({
                "status":        "saved",
                "date":          actual_date,
                "filename":      f.filename,
                "daily_mu":      total_mu,
                "daily_avg_mw":  round(df["demand_mw"].mean()),
                "peak_mw":       int(df["demand_mw"].max()),
                "trough_mw":     int(df["demand_mw"].min()),
                "hours_covered": int(df["timestamp"].dt.hour.nunique()),
                "hourly_mw":     hourly_mw,
            })

        except Exception as e:
            results.append({
                "status":   "error",
                "filename": f.filename,
                "error":    str(e),
            })

    saved = [r for r in results if r["status"] == "saved"]
    return {
        "files_processed": len(files),
        "saved":           len(saved),
        "results":         results,
        "message": f"Saved actuals for {len(saved)} day(s): " +
                   ", ".join(r["date"] for r in saved if "date" in r),
    }


def load_merit_history() -> dict:
    """
    Load saved MERIT history for use in live forecast.
    Returns the full history dict, or empty dict if not available.
    """
    if not MERIT_CACHE.exists():
        return {}
    try:
        with open(MERIT_CACHE) as f:
            return json.load(f)
    except Exception:
        return {}
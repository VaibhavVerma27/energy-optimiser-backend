"""
merit_parser.py
===============
Parses MERIT India "Demand Met" CSV files downloaded from:
  npp.gov.in/dashBoard/gc-map-dashboard-meritchart
  → "Previous Data" → "Download Excel" (saves as CSV)

The file has three columns:
  Source  | Value         | Time
  DEMAND MET | 2,14,991  | 09/05/2026 00:01

Value is All-India demand met in MW using Indian number format (lakh notation):
  "2,14,991" = 214,991 MW (commas every 2 digits after first group)

This module:
  1. Parses one or more such CSV files
  2. Resamples 4-minute readings to hourly averages
  3. Splits All-India total into per-region estimates using historical
     regional load shares derived from the POSOCO training dataset
  4. Returns a dict ready to be used as the model's 168h history buffer
     — replacing the stale 2022 CSV data entirely

Regional load shares (from POSOCO 2017-2022 average):
  Northern  ~29.9%  Western  ~31.4%  Southern  ~25.9%
  Eastern   ~11.6%  NE        ~1.2%
These are remarkably stable across years (±1-2%) because they reflect
the long-run economic and population geography of India.
"""

import io
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# ── Regional load shares ──────────────────────────────────────────────────────
# Base shares from 2017-2022 POSOCO average
BASE_REGION_SHARES = {
    "Northern_Region_mw":      0.2988,
    "Western_Region_mw":       0.3139,
    "Southern_Region_mw":      0.2592,
    "Eastern_Region_mw":       0.1159,
    "NorthEastern_Region_mw":  0.0122,
}

# Dynamic share table: varies by hour AND month
# Northern share peaks in May-June afternoons (heavy AC in Delhi/Rajasthan)
# Southern share is relatively stable (milder climate)
# NE is nearly constant at ~1.2%
# Derived from POSOCO demand pattern analysis and CEA load research studies
def _build_dynamic_shares() -> dict:
    table = {}
    for h in range(24):
        table[h] = {}
        for m in range(1, 13):
            nr = 0.2988; wr = 0.3139; sr = 0.2592; er = 0.1159; ne = 0.0122
            # Summer afternoon AC surge in Northern Region (May-Jun, 12-18h)
            if m in [4, 5, 6] and 12 <= h <= 18:
                nr += 0.025; sr -= 0.012; er -= 0.008; wr -= 0.003; ne -= 0.002
            # Summer night in Northern: less AC, share drops
            elif m in [4, 5, 6] and 0 <= h <= 5:
                nr -= 0.015; sr += 0.007; wr += 0.005; er += 0.002; ne += 0.001
            # Winter morning peak Northern (Dec-Jan, 7-10h): heating + industrial
            elif m in [12, 1, 2] and 7 <= h <= 10:
                nr += 0.010; sr -= 0.005; er -= 0.003; wr -= 0.002
            # Monsoon: NE hydro peaks
            if m in [7, 8, 9]:
                ne += 0.002; nr -= 0.001; wr -= 0.001
            # Normalise to exactly 1.0
            tot = nr + wr + sr + er + ne
            table[h][m] = {
                "Northern_Region_mw":     round(nr / tot, 5),
                "Western_Region_mw":      round(wr / tot, 5),
                "Southern_Region_mw":     round(sr / tot, 5),
                "Eastern_Region_mw":      round(er / tot, 5),
                "NorthEastern_Region_mw": round(ne / tot, 5),
            }
    return table

DYNAMIC_SHARES = _build_dynamic_shares()

def get_region_shares(hour: int, month: int) -> dict:
    """Return region shares for a specific hour and month."""
    return DYNAMIC_SHARES.get(hour, {}).get(month, BASE_REGION_SHARES)

ALL_REGION_COLS = list(BASE_REGION_SHARES.keys())

# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_indian_number(s: str) -> int:
    """
    Convert Indian-format number string to int.
    "2,14,991" → 214991
    "2,09,178" → 209178
    """
    return int(str(s).replace(",", ""))


def parse_merit_csv(source: Union[str, bytes, io.IOBase]) -> pd.DataFrame:
    """
    Parse a single MERIT India demand CSV file.

    Accepts:
        - File path (str or Path)
        - Raw bytes (from file upload)
        - File-like object

    Returns DataFrame with columns:
        timestamp (datetime, IST), demand_mw (int, All-India MW)
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    elif isinstance(source, bytes):
        df = pd.read_csv(io.BytesIO(source))
    else:
        df = pd.read_csv(source)

    # Normalise column names (case-insensitive)
    df.columns = [c.strip().upper() for c in df.columns]

    if "VALUE" not in df.columns or "TIME" not in df.columns:
        raise ValueError(
            f"Expected columns 'Source', 'Value', 'Time'. Got: {list(df.columns)}"
        )

    # Parse MW values from Indian number format
    df["demand_mw"] = df["VALUE"].apply(_parse_indian_number)

    # Parse timestamps — format DD/MM/YYYY HH:MM
    df["timestamp"] = pd.to_datetime(df["TIME"].str.strip(), format="%d/%m/%Y %H:%M")

    # Keep only needed columns and sort
    df = df[["timestamp", "demand_mw"]].drop_duplicates().sort_values("timestamp")
    df = df.reset_index(drop=True)

    # Basic sanity checks
    if df["demand_mw"].max() < 50_000:
        raise ValueError(
            f"Values look too low (max={df['demand_mw'].max():,} MW). "
            "Expected All-India demand ~150,000-250,000 MW."
        )
    if df["demand_mw"].max() > 600_000:
        raise ValueError(
            f"Values look too high (max={df['demand_mw'].max():,} MW). "
            "Check the file format."
        )

    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 4-minute MERIT data to hourly averages.
    Each hour label represents the floor (e.g. 14:00 = average of 14:01-14:59 readings).
    """
    df = df.set_index("timestamp")
    hourly = df["demand_mw"].resample("h").mean().dropna()
    hourly = hourly.round(0).astype(int)
    return hourly.reset_index().rename(columns={"timestamp": "hour_start"})


# ── Multi-file processing ─────────────────────────────────────────────────────

def build_history_from_merit_files(
    file_sources: list,
    target_hours: int = 168,
) -> dict:
    """
    Parse multiple MERIT CSV files, combine into a continuous hourly history,
    and split into per-region estimates.

    Args:
        file_sources: List of file paths, bytes objects, or file-like objects.
                      Should cover the last 7 days (168h). Today's partial file
                      can be included — it extends coverage to the current hour.
        target_hours: Number of hours to return (default 168 = 7 days).

    Returns dict:
        {
            "demand_mw":              [168 floats],   # All-India hourly MW
            "Northern_Region_mw":     [168 floats],
            "Western_Region_mw":      [168 floats],
            "Southern_Region_mw":     [168 floats],
            "Eastern_Region_mw":      [168 floats],
            "NorthEastern_Region_mw": [168 floats],
            "last_timestamp":         "2026-05-09T18:00:00",
            "first_timestamp":        "2026-05-02T19:00:00",
            "hours_available":        168,
            "hours_today":            19,
            "mean_mw":                209234,
            "peak_mw":                226672,
            "trough_mw":              191098,
            "data_quality": {
                "gaps_filled":   3,     # hours interpolated
                "coverage_pct":  98.2,  # % of target window with data
            }
        }
    """
    # Parse and combine all files
    frames = []
    for src in file_sources:
        try:
            df = parse_merit_csv(src)
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: Could not parse file: {e}")
            continue

    if not frames:
        raise ValueError("No valid MERIT CSV files could be parsed.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")

    # Resample to hourly
    hourly = resample_to_hourly(combined)
    hourly = hourly.set_index("hour_start")

    # Determine how many complete hours from the most recent
    last_hour  = hourly.index.max()
    first_needed = last_hour - timedelta(hours=target_hours - 1)
    full_range = pd.date_range(first_needed, last_hour, freq="h")

    # Reindex to the full range and fill gaps
    hourly = hourly.reindex(full_range)
    gaps_filled = int(hourly["demand_mw"].isna().sum())
    hourly["demand_mw"] = hourly["demand_mw"].interpolate(method="linear").bfill().ffill()
    hourly["demand_mw"] = hourly["demand_mw"].round(0).astype(int)

    all_india = hourly["demand_mw"].tolist()
    timestamps = hourly.index.tolist()

    # Split into regional estimates using dynamic hour×month shares
    # instead of a single fixed percentage — this captures the fact that
    # Northern Region is ~32% of demand at 15:00 in May (extreme AC load)
    # but only ~28% at 02:00 in May (most AC switched off overnight)
    regional = {col: [] for col in BASE_REGION_SHARES}
    for ts, v in zip(timestamps, all_india):
        h = ts.hour
        m = ts.month
        shares = get_region_shares(h, m)
        for col in BASE_REGION_SHARES:
            regional[col].append(round(v * shares[col], 0))

    # Count today's hours (last partial day)
    today = last_hour.date()
    hours_today = int((hourly.index.date == today).sum())

    # Build result
    result = {
        "demand_mw":              all_india,
        **regional,
        "last_timestamp":         last_hour.isoformat(),
        "first_timestamp":        full_range[0].isoformat(),
        "hours_available":        len(all_india),
        "hours_today":            hours_today,
        "mean_mw":                int(np.mean(all_india)),
        "peak_mw":                int(np.max(all_india)),
        "trough_mw":              int(np.min(all_india)),
        "data_quality": {
            "gaps_filled":   gaps_filled,
            "coverage_pct":  round((len(full_range) - gaps_filled) / len(full_range) * 100, 1),
        },
    }

    return result


def get_latest_hourly_demand(
    file_sources: list,
) -> dict:
    """
    Get the single most recent hourly All-India demand from uploaded files.
    Used to update the history anchor without processing the full 168h.

    Returns:
        { "demand_mw": 214991, "timestamp": "2026-05-09T18:00:00", "source": "merit_csv" }
    """
    frames = []
    for src in file_sources:
        try:
            frames.append(parse_merit_csv(src))
        except Exception:
            continue

    if not frames:
        return {}

    combined = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
    latest_4min = combined.iloc[-1]
    latest_hour = resample_to_hourly(combined).iloc[-1]

    return {
        "demand_mw":    int(latest_hour["demand_mw"]),
        "timestamp":    latest_hour["hour_start"].isoformat(),
        "raw_latest":   int(latest_4min["demand_mw"]),
        "raw_time":     latest_4min["timestamp"].isoformat(),
        "source":       "merit_csv",
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    test_files = []
    if len(sys.argv) > 1:
        test_files = sys.argv[1:]
    else:
        # Look for test files in uploads
        for p in [
            "/mnt/user-data/uploads/Demand_Met_Data_2026-05-08.csv",
            "/mnt/user-data/uploads/Demand_Met_Data_2026-05-09.csv",
        ]:
            if os.path.exists(p):
                test_files.append(p)

    if not test_files:
        print("No test files found. Pass CSV paths as arguments.")
        sys.exit(0)

    print(f"Testing with {len(test_files)} file(s)...\n")
    result = build_history_from_merit_files(test_files, target_hours=168)

    print(f"First timestamp:    {result['first_timestamp']}")
    print(f"Last timestamp:     {result['last_timestamp']}")
    print(f"Hours available:    {result['hours_available']}")
    print(f"Hours today:        {result['hours_today']}")
    print(f"All-India mean:     {result['mean_mw']:,} MW")
    print(f"All-India peak:     {result['peak_mw']:,} MW")
    print(f"Coverage:           {result['data_quality']['coverage_pct']}%")
    print(f"Gaps filled:        {result['data_quality']['gaps_filled']}")

    print(f"\nPer-region last 3 hours:")
    for col in ALL_REGION_COLS:
        vals = result[col][-3:]
        print(f"  {col:<28}: {[f'{v:,.0f}' for v in vals]}")

    print(f"\nAll-India last 24 hours:")
    all_india = result["demand_mw"]
    for i, v in enumerate(all_india[-24:]):
        h = (result['last_timestamp'][:13])  # just the date+hour
        print(f"  H-{23-i:2d}  {v:>9,} MW")
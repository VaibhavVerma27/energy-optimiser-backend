"""
weather_fetcher.py
==================
Downloads hourly weather data for the 5 representative POSOCO region cities
from Open-Meteo (free, no API key required) and merges it into demand.csv.

Variables fetched per city:
  - temperature_2m          (°C)  — strongest demand driver
  - relativehumidity_2m     (%)   — affects AC efficiency
  - shortwave_radiation     (W/m²) — improves solar capacity model

Representative cities (chosen as regional load centres):
  Northern  → New Delhi      (28.6139°N, 77.2090°E)
  Western   → Mumbai         (19.0760°N, 72.8777°E)
  Southern  → Chennai        (13.0827°N, 80.2707°E)
  Eastern   → Kolkata        (22.5726°N, 88.3639°E)
  NE        → Guwahati       (26.1445°N, 91.7362°E)

Usage:
    # Pull weather for the entire date range in demand.csv (recommended)
    python weather_fetcher.py --data data/demand.csv

    # Pull a specific range
    python weather_fetcher.py --data data/demand.csv --start 2019-01-01 --end 2022-12-31

    # Dry run — show what would be fetched without writing anything
    python weather_fetcher.py --data data/demand.csv --dry-run

Output:
    Overwrites data/demand.csv with 15 new columns added:
    northern_temp_c, northern_humidity_pct, northern_solar_wm2,
    western_temp_c, western_humidity_pct, western_solar_wm2,
    southern_temp_c, southern_humidity_pct, southern_solar_wm2,
    eastern_temp_c, eastern_humidity_pct, eastern_solar_wm2,
    ne_temp_c, ne_humidity_pct, ne_solar_wm2

Note: Open-Meteo has a 1-year-per-request limit on the archive API.
      This script automatically chunks large date ranges into yearly batches.
"""

import sys
import os
import time
import argparse
import json
from datetime import datetime, timedelta
from urllib.request import urlopen
from urllib.error import URLError
from urllib.parse import urlencode

import pandas as pd
import numpy as np

# ── Region → representative city ─────────────────────────────────────────────
CITIES = {
    "northern": {"lat": 28.6139, "lon": 77.2090, "name": "New Delhi"},
    "western":  {"lat": 19.0760, "lon": 72.8777, "name": "Mumbai"},
    "southern": {"lat": 13.0827, "lon": 80.2707, "name": "Chennai"},
    "eastern":  {"lat": 22.5726, "lon": 88.3639, "name": "Kolkata"},
    "ne":       {"lat": 26.1445, "lon": 91.7362, "name": "Guwahati"},
}

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
VARIABLES = "temperature_2m,relativehumidity_2m,shortwave_radiation"
TIMEZONE = "Asia/Kolkata"


def fetch_year(lat: float, lon: float, year: int) -> pd.DataFrame:
    """Fetch one year of hourly weather for a lat/lon from Open-Meteo."""
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "hourly":    VARIABLES,
        "timezone":  TIMEZONE,
    }
    url = f"{OPEN_METEO_ARCHIVE}?{urlencode(params)}"

    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            break
        except URLError as e:
            if attempt == 2:
                raise RuntimeError(f"Failed after 3 attempts: {e}") from e
            print(f"  Retry {attempt + 1}/3 …")
            time.sleep(2)

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp":    pd.to_datetime(hourly["time"]),
        "temp_c":       hourly["temperature_2m"],
        "humidity_pct": hourly["relativehumidity_2m"],
        "solar_wm2":    hourly["shortwave_radiation"],
    })
    return df


def fetch_region(region: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch weather for a region across all years in range."""
    city = CITIES[region]
    print(f"  {city['name']} ({region}): {start_year}–{end_year}", end="", flush=True)

    frames = []
    for year in range(start_year, end_year + 1):
        print(f" {year}", end="", flush=True)
        df = fetch_year(city["lat"], city["lon"], year)
        frames.append(df)
        time.sleep(0.3)   # be polite to the free API

    print()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={
        "temp_c":       f"{region}_temp_c",
        "humidity_pct": f"{region}_humidity_pct",
        "solar_wm2":    f"{region}_solar_wm2",
    })
    return combined


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive extra features from raw weather values.
    These are domain-knowledge transformations that help the model
    more than the raw numbers alone.
    """
    df = df.copy()

    # Effective temperature = feels-like heat index (simplified Steadman)
    # High humidity + high temp = much higher AC load
    for region in CITIES:
        t = df.get(f"{region}_temp_c")
        h = df.get(f"{region}_humidity_pct")
        if t is not None and h is not None:
            # Simplified heat index — only meaningful above 27°C
            heat_idx = t + 0.33 * (h / 100 * 6.105 * np.exp(17.27 * t / (237.7 + t))) - 4.0
            df[f"{region}_heat_index"] = heat_idx.clip(lower=t)  # never below actual temp

    # Cooling degree hours (CDH): hours above 24°C, weighted by excess
    # Northern region AC load kicks in hard above 35°C
    for region in CITIES:
        t = df.get(f"{region}_temp_c")
        if t is not None:
            df[f"{region}_cdh"] = (t - 24).clip(lower=0)

    return df


def merge_weather(demand_path: str, dry_run: bool = False,
                  start_date: str = None, end_date: str = None):
    """Main entry point: load demand.csv, fetch weather, merge, save."""

    print(f"\nLoading {demand_path} …")
    df = pd.read_csv(demand_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} rows | {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Determine year range to fetch
    t_start = pd.to_datetime(start_date) if start_date else df["timestamp"].min()
    t_end   = pd.to_datetime(end_date)   if end_date   else df["timestamp"].max()
    start_year = t_start.year
    end_year   = t_end.year

    if dry_run:
        print(f"\nDRY RUN — would fetch {start_year}–{end_year} for:")
        for region, city in CITIES.items():
            print(f"  {region}: {city['name']} ({city['lat']}°N, {city['lon']}°E)")
        print("\nNo files written.")
        return

    # Check for existing weather columns — skip regions already fetched
    existing = [r for r in CITIES if f"{r}_temp_c" in df.columns]
    if existing:
        print(f"\nAlready have weather for: {existing}")
        print("Fetching remaining regions only. Delete columns to re-fetch.")

    print(f"\nFetching hourly weather {start_year}–{end_year} …")
    print("Source: Open-Meteo archive API (free, no key required)\n")

    for region in CITIES:
        if region in existing:
            print(f"  Skipping {region} (already in CSV)")
            continue
        try:
            weather_df = fetch_region(region, start_year, end_year)
            # Merge on timestamp — left join keeps all demand rows
            df = df.merge(weather_df, on="timestamp", how="left")
        except Exception as e:
            print(f"\n  WARNING: Failed to fetch {region}: {e}")
            print(f"  Filling {region} weather with NaN — will be interpolated")
            for col in [f"{region}_temp_c", f"{region}_humidity_pct", f"{region}_solar_wm2"]:
                df[col] = np.nan

    # Interpolate any gaps (missing hours, API gaps)
    weather_cols = [c for c in df.columns if any(
        c.startswith(r) for r in CITIES
    ) and c != "timestamp"]
    df[weather_cols] = df[weather_cols].interpolate(method="linear").ffill().bfill()

    # Add derived features
    df = add_weather_features(df)

    # Save
    df.to_csv(demand_path, index=False)
    new_cols = [c for c in df.columns if any(c.startswith(r) for r in CITIES)]
    print(f"\n✓ Saved {demand_path}")
    print(f"  Added {len(new_cols)} weather columns: {new_cols}")
    print(f"\nNext step: retrain models to use weather features:")
    print(f"  python train.py --data {demand_path} --use-weather")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch weather data for India Smart Grid")
    parser.add_argument("--data",    default="data/demand.csv", help="Path to demand.csv")
    parser.add_argument("--start",   default=None,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None,  help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched, don't write")
    args = parser.parse_args()

    merge_weather(args.data, dry_run=args.dry_run,
                  start_date=args.start, end_date=args.end)
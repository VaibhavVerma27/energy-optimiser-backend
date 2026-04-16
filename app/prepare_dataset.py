"""
prepare_dataset.py
------------------
Converts the India Hourly Load .xlsx file into demand.csv for training.

Expected Excel columns (exact names from the file):
  - datetime
  - National Hourly Demand
  - Northen Region Hourly Demand       (note: typo in source data)
  - Western Region Hourly Demand
  - Eastern Region Hourly Demand
  - Southern Region Hourly Demand
  - North-Eastern Region Hourly Demand

Usage:
    python prepare_dataset.py --file data/india_load.xlsx
    python prepare_dataset.py --file data/india_load.xlsx --start-year 2019

Output: data/demand.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import glob
import argparse

# Map from Excel column names → internal standard names
# Handles typos in source data (e.g. "Northen" instead of "Northern")
COLUMN_MAP = {
    "National Hourly Demand":              "demand_mw",
    "Northen Region Hourly Demand":        "Northern_Region_mw",
    "Northern Region Hourly Demand":       "Northern_Region_mw",
    "Western Region Hourly Demand":        "Western_Region_mw",
    "Eastern Region Hourly Demand":        "Eastern_Region_mw",
    "Southern Region Hourly Demand":       "Southern_Region_mw",
    "North-Eastern Region Hourly Demand":  "NorthEastern_Region_mw",
    "North Eastern Region Hourly Demand":  "NorthEastern_Region_mw",
    "NorthEastern Region Hourly Demand":   "NorthEastern_Region_mw",
}

DATETIME_ALIASES = ["datetime", "date", "timestamp", "time", "date_time", "DateTime", "Datetime"]

INDIA_CAPACITIES_MW = {
    "Northern_Region":     115000,
    "Western_Region":      130000,
    "Southern_Region":      95000,
    "Eastern_Region":       55000,
    "NorthEastern_Region":   4500,
}


def find_file(data_dir: str) -> str:
    """Find first xlsx or csv in data_dir."""
    for ext in ["*.xlsx", "*.xls", "*.csv"]:
        files = glob.glob(os.path.join(data_dir, ext))
        if files:
            return files[0]
    raise FileNotFoundError(f"No xlsx/xls/csv found in {data_dir}/")


def load_raw(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        print(f"Reading Excel: {os.path.basename(filepath)}")
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        print(f"Reading CSV: {os.path.basename(filepath)}")
        df = pd.read_csv(filepath)
    print(f"Raw shape: {df.shape}")
    print(f"Raw columns: {list(df.columns)}")
    return df


def detect_datetime_col(df: pd.DataFrame) -> str:
    for alias in DATETIME_ALIASES:
        for col in df.columns:
            if col.strip().lower() == alias.lower():
                return col
    # Fallback: first column
    return df.columns[0]


def convert(filepath: str, output_path: str, start_year: int = None):
    df = load_raw(filepath)

    # Detect datetime column
    dt_col = detect_datetime_col(df)
    print(f"Using datetime column: '{dt_col}'")
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    df = df.rename(columns={dt_col: "timestamp"})

    # Rename demand columns to standard names
    rename_map = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in COLUMN_MAP:
            rename_map[col] = COLUMN_MAP[col_stripped]
    df = df.rename(columns=rename_map)
    print(f"Renamed columns: {rename_map}")

    # Check we have the national demand column
    if "demand_mw" not in df.columns:
        # Try to derive from region columns if all present
        region_cols = [c for c in df.columns if c.endswith("_Region_mw")]
        if region_cols:
            print("National demand not found — computing from regions sum")
            df["demand_mw"] = df[region_cols].sum(axis=1)
        else:
            raise ValueError(
                "Could not find 'National Hourly Demand' column.\n"
                f"Available columns: {list(df.columns)}\n"
                "Check column names match the expected format."
            )

    df = df.sort_values("timestamp").reset_index(drop=True)

    if start_year:
        before = len(df)
        df = df[df["timestamp"].dt.year >= start_year].reset_index(drop=True)
        print(f"Filtered to {start_year}+: {before:,} → {len(df):,} rows")

    # Convert all MW columns to numeric
    mw_cols = [c for c in df.columns if c.endswith("_mw") or c == "demand_mw"]
    for col in mw_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicates, resample to strict hourly
    df = df.drop_duplicates("timestamp")
    keep_cols = ["timestamp", "demand_mw"] + [c for c in df.columns if c.endswith("_Region_mw")]
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.set_index("timestamp").resample("h").mean().interpolate(method="linear").reset_index()
    df = df.dropna(subset=["demand_mw"])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*55}")
    print(f"Saved {len(df):,} rows → {output_path}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nColumn summary:")
    for col in df.columns:
        if col == "timestamp":
            continue
        region = col.replace("_mw", "").replace("_Region", "")
        cap = INDIA_CAPACITIES_MW.get(col.replace("_mw", ""), None)
        peak = df[col].max()
        mean = df[col].mean()
        util = f" → {(peak/cap)*100:.1f}% utilisation" if cap else ""
        print(f"  {col:<32}: mean={mean:>8,.0f} MW  peak={peak:>8,.0f} MW{util}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare India load Excel for Smart Grid ML pipeline")
    parser.add_argument("--file",       default=None,             help="Path to xlsx/csv file (auto-detected if omitted)")
    parser.add_argument("--data-dir",   default="data",           help="Folder to search for data file")
    parser.add_argument("--output",     default="data/demand.csv",help="Output CSV path")
    parser.add_argument("--start-year", type=int, default=None,   help="Only keep data from this year onwards")
    args = parser.parse_args()

    filepath = args.file
    if not filepath:
        try:
            filepath = find_file(args.data_dir)
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print(f"\nPlace your India load xlsx file in: {os.path.abspath(args.data_dir)}/")
            print("Then run: python prepare_dataset.py")
            return

    convert(filepath, args.output, args.start_year)

    print(f"\nNext steps:")
    print(f"  python train.py --data {args.output}")
    print(f"  uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
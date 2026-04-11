"""
prepare_dataset.py
------------------
Downloads the PJM Hourly Energy Consumption dataset from Kaggle
and converts it into the format expected by this project.

STEP 1 — Get the dataset (pick one option):

  Option A: Kaggle CLI
    pip install kaggle
    # Put your kaggle.json API key in ~/.kaggle/kaggle.json
    kaggle datasets download -d robikscube/hourly-energy-consumption
    unzip hourly-energy-consumption.zip -d data/

  Option B: Manual download
    1. Go to https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
    2. Click Download
    3. Unzip and place PJME_hourly.csv inside:  app/data/PJME_hourly.csv

STEP 2 — Run this script (from inside app/ folder):
    python prepare_dataset.py

STEP 3 — Train the model:
    python train.py --data data/demand.csv --model rf
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import argparse

RAW_FILES = {
    "PJME":  ("PJME_hourly.csv",  "PJME_MW"),
    "AEP":   ("AEP_hourly.csv",   "AEP_MW"),
    "DOM":   ("DOM_hourly.csv",   "DOM_MW"),
    "COMED": ("COMED_hourly.csv", "COMED_MW"),
    "NI":    ("NI_hourly.csv",    "NI_MW"),
}


def convert(input_path: str, mw_col: str, output_path: str, start_year: int = 2012):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["Datetime"])
    df = df.rename(columns={"Datetime": "timestamp", mw_col: "demand_mw"})
    df = df[["timestamp", "demand_mw"]].dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter to recent years
    df = df[df["timestamp"].dt.year >= start_year]

    # Deduplicate and resample to strict hourly
    df = df.drop_duplicates("timestamp")
    df = df.set_index("timestamp").resample("h").mean().interpolate().reset_index()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} rows to {output_path}")
    print(f"Date range : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Demand range: {df['demand_mw'].min():.0f} MW – {df['demand_mw'].max():.0f} MW")
    print(f"Mean demand : {df['demand_mw'].mean():.0f} MW")
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare PJM dataset for Smart Grid system")
    parser.add_argument("--region",     default="PJME", choices=list(RAW_FILES.keys()))
    parser.add_argument("--data-dir",   default="data", help="Folder containing raw Kaggle CSVs")
    parser.add_argument("--output",     default="data/demand.csv")
    parser.add_argument("--start-year", type=int, default=2012)
    args = parser.parse_args()

    raw_file, mw_col = RAW_FILES[args.region]
    input_path = os.path.join(args.data_dir, raw_file)

    if not os.path.exists(input_path):
        print(f"\nERROR: File not found: {input_path}")
        print("\nDownload the dataset first:")
        print("  Option A — Kaggle CLI:")
        print("    pip install kaggle")
        print(f"    kaggle datasets download -d robikscube/hourly-energy-consumption")
        print(f"    unzip hourly-energy-consumption.zip -d {args.data_dir}/")
        print("\n  Option B — Manual:")
        print("    https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption")
        print(f"    Place PJME_hourly.csv in:  {os.path.abspath(args.data_dir)}/")
        return

    convert(input_path, mw_col, args.output, start_year=args.start_year)

    print("\nNext steps:")
    print(f"  python train.py --data {args.output} --model rf")
    print("  uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
"""
train.py — Train one model per region + one for national demand.

Usage:
    python train.py --data data/demand.csv
    python train.py --data data/demand.csv --model lr   (linear regression)

Trains and saves:
    model_demand_mw.joblib                — All-India / National
    model_Northern_Region_mw.joblib       — Northern Region
    model_Western_Region_mw.joblib        — Western Region
    model_Eastern_Region_mw.joblib        — Eastern Region
    model_Southern_Region_mw.joblib       — Southern Region
    model_NorthEastern_Region_mw.joblib   — North-Eastern Region
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pandas as pd
from preprocessing import preprocess_pipeline
from model import full_training_pipeline

ALL_COLUMNS = [
    "demand_mw",
    "Northern_Region_mw",
    "Western_Region_mw",
    "Eastern_Region_mw",
    "Southern_Region_mw",
    "NorthEastern_Region_mw",
]


def train_one(data_path: str, region_col: str, model_type: str) -> dict:
    model_path = f"model_{region_col}.joblib"
    print(f"\n{'='*55}")
    print(f"Training: {region_col}")
    print(f"{'='*55}")

    df, X, y, feature_names = preprocess_pipeline(data_path, region_col=region_col)
    print(f"Samples: {X.shape[0]:,} | Features: {X.shape[1]}")

    model, metrics, importances = full_training_pipeline(
        X, y, feature_names, model_type=model_type, model_path=model_path,
    )

    print(f"  MAE:      {metrics['mae']:>10,.1f} MW")
    print(f"  RMSE:     {metrics['rmse']:>10,.1f} MW")
    print(f"  R²:       {metrics['r2']:>10.4f}")
    print(f"  Accuracy: {metrics['accuracy_pct']:>9.1f}%")

    if importances:
        top5 = list(importances.items())[:5]
        print(f"  Top features: {', '.join(f'{k}={v:.3f}' for k,v in top5)}")

    print(f"  Saved → {model_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True, help="Path to demand.csv")
    parser.add_argument("--model", default="rf", choices=["rf", "lr"],
                        help="rf=RandomForest (default), lr=LinearRegression")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: {args.data} not found.")
        print("Run: python prepare_dataset.py --file data/your_file.xlsx")
        return

    # Detect which columns are actually in the CSV
    df_cols = pd.read_csv(args.data, nrows=0).columns.tolist()
    cols_to_train = [c for c in ALL_COLUMNS if c in df_cols]
    missing = [c for c in ALL_COLUMNS if c not in df_cols]

    if missing:
        print(f"Note: Skipping {missing} (not in CSV)")

    all_metrics = {}
    for col in cols_to_train:
        try:
            all_metrics[col] = train_one(args.data, col, args.model)
        except Exception as e:
            print(f"  FAILED for {col}: {e}")

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Column':<35} {'MAE':>10} {'R²':>8} {'Acc%':>8}")
    print(f"{'-'*65}")
    for col, m in all_metrics.items():
        print(f"{col:<35} {m['mae']:>10,.1f} {m['r2']:>8.4f} {m['accuracy_pct']:>7.1f}%")
    print(f"\nAll models saved. Start server: uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
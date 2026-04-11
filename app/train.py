"""
train.py — Run this once to train and save the model.

Usage (run from inside the app/ folder):
    cd app
    python train.py --data data/demand.csv --model rf

Expected CSV format:
    timestamp,demand_mw
    2022-01-01 00:00:00,5241.3
    2022-01-01 01:00:00,4987.1
"""

import sys
import os

# Point to app/ folder so sibling modules resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from preprocessing import preprocess_pipeline
from model import full_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train Smart Grid demand forecasting model")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "lr"], help="Model type")
    args = parser.parse_args()

    print(f"Loading and preprocessing: {args.data}")
    df, X, y, feature_names = preprocess_pipeline(args.data)
    print(f"Dataset shape: {X.shape[0]} samples x {X.shape[1]} features")

    print(f"\nTraining {args.model.upper()} model...")
    model, metrics, importances = full_training_pipeline(X, y, feature_names, model_type=args.model)

    print("\n-- Model Performance ------------------")
    print(f"  MAE:      {metrics['mae']:.2f} MW")
    print(f"  RMSE:     {metrics['rmse']:.2f} MW")
    print(f"  R2:       {metrics['r2']:.4f}")
    print(f"  Accuracy: {metrics['accuracy_pct']:.1f}%")

    if importances:
        print("\n-- Top 5 Feature Importances ----------")
        for feat, imp in list(importances.items())[:5]:
            print(f"  {feat:<22} {imp:.4f}")

    print("\nModel saved to model.joblib")


if __name__ == "__main__":
    main()
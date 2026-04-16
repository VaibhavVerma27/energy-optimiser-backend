"""
Module 3: ML Model — Training & Evaluation
Supports saving/loading separate models per region.
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_MODEL_PATH = "model.joblib"


def train_test_split_ts(X, y, split_ratio=0.8):
    """Chronological split — no shuffle."""
    idx = int(len(X) * split_ratio)
    return X[:idx], X[idx:], y[:idx], y[idx:]


def train_model(X_train, y_train, model_type="rf"):
    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=100, max_depth=12,
            min_samples_leaf=4, n_jobs=-1, random_state=42,
        )
    elif model_type == "lr":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model: {model_type}")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "accuracy_pct": round((1 - mae / np.mean(y_test)) * 100, 2),
        "n_test_samples": len(y_test),
    }


def save_model(model, path=DEFAULT_MODEL_PATH):
    joblib.dump(model, path)


def load_model(path=DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}. Run train.py first.")
    return joblib.load(path)


def load_region_model(region: str):
    """Load region-specific model, fall back to all-india model."""
    region_path = f"model_{region}.joblib"
    if os.path.exists(region_path):
        return joblib.load(region_path)
    if os.path.exists(DEFAULT_MODEL_PATH):
        return joblib.load(DEFAULT_MODEL_PATH)
    raise FileNotFoundError("No model found. Run: python train.py --data data/demand.csv --model rf")


def feature_importances(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return {}
    return dict(sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True))


def full_training_pipeline(X, y, feature_names, model_type="rf", model_path=DEFAULT_MODEL_PATH):
    X_train, X_test, y_train, y_test = train_test_split_ts(X, y)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    model = train_model(X_train, y_train, model_type)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, model_path)
    importances = feature_importances(model, feature_names)
    return model, metrics, importances
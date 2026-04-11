"""
Module 3: ML Model — Training & Evaluation
-------------------------------------------
- Linear Regression (baseline)
- Random Forest (recommended)
- Train/test split: 80/20 chronological
- Metrics: MAE, RMSE
- Model persistence with joblib
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_PATH = "model.joblib"


def train_test_split_ts(X: np.ndarray, y: np.ndarray, split_ratio: float = 0.8):
    """
    Chronological train/test split — NO shuffle.
    Always keep the most recent data as test set.
    """
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str = "rf"):
    """
    Train either a Random Forest or Linear Regression model.

    Args:
        model_type: 'rf' for Random Forest, 'lr' for Linear Regression
    """
    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "lr":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'rf' or 'lr'.")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model predictions.
    Returns MAE, RMSE, and R² score.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "accuracy_pct": round((1 - mae / np.mean(y_test)) * 100, 2),
        "n_test_samples": len(y_test),
    }


def save_model(model, path: str = MODEL_PATH):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}. Train first.")
    return joblib.load(path)


def feature_importances(model, feature_names: list) -> dict:
    """Return feature importances for Random Forest models."""
    if not hasattr(model, "feature_importances_"):
        return {}
    importances = model.feature_importances_
    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    ))


def full_training_pipeline(X: np.ndarray, y: np.ndarray, feature_names: list, model_type: str = "rf"):
    """
    End-to-end: split → train → evaluate → save.
    Returns model and metrics dict.
    """
    X_train, X_test, y_train, y_test = train_test_split_ts(X, y)
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

    model = train_model(X_train, y_train, model_type=model_type)
    metrics = evaluate_model(model, X_test, y_test)

    print(f"MAE:  {metrics['mae']:.2f} MW")
    print(f"RMSE: {metrics['rmse']:.2f} MW")
    print(f"R²:   {metrics['r2']:.4f}")

    save_model(model)
    importances = feature_importances(model, feature_names)

    return model, metrics, importances
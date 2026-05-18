"""
model.py — Training & loading for region+horizon specific models.

This is the LightGBM-based version with DIRECT multi-step forecasting.

Key changes from previous Random Forest version:
  • Default model = LightGBM (better at peaks, no averaging bias)
  • Direct multi-step: 24 horizon-specific models per region
    instead of 1 recursive model
  • Confidence intervals via LightGBM quantile regression (not tree variance)

File naming convention:
  model_<region>_h<H>.joblib       — direct horizon model
    e.g. model_Northern_Region_mw_h1.joblib   = 1h ahead
         model_Northern_Region_mw_h24.joblib  = 24h ahead

  model_<region>.joblib            — legacy recursive model (fallback)
"""

import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble  import RandomForestRegressor
from sklearn.metrics   import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

DEFAULT_MODEL_PATH = "model.joblib"


def train_test_split_ts(X, y, split_ratio=0.85):
    """Chronological split — no shuffle. 85/15 default for time series."""
    idx = int(len(X) * split_ratio)
    return X[:idx], X[idx:], y[:idx], y[idx:]


def train_model(X_train, y_train, model_type: str = "lgbm",
                X_val=None, y_val=None, feature_names=None):
    """
    Train a single model. Default LightGBM.

    LightGBM hyperparameters chosen for short-term load forecasting:
      n_estimators=500    — gradient boosting needs more rounds than RF
      learning_rate=0.05  — slow learning for stability on time series
      num_leaves=63       — leaf-wise growth, deeper than RF default
      min_child_samples=20 — prevent overfitting on small leaf nodes
      reg_alpha=0.1       — L1 regularisation for feature selection
      reg_lambda=0.1      — L2 regularisation for stability
      subsample=0.85      — row sampling for variance reduction
      colsample_bytree=0.85 — column sampling for diversity

    feature_names: when provided, X_train is wrapped in a DataFrame so the
    model stores feature names. This avoids the sklearn UserWarning at
    prediction time and ensures column-order safety.
    """
    # Wrap as DataFrame if feature names provided — eliminates warning
    if feature_names is not None and not hasattr(X_train, "columns"):
        import pandas as pd
        X_train = pd.DataFrame(X_train, columns=list(feature_names))
        if X_val is not None and not hasattr(X_val, "columns"):
            X_val = pd.DataFrame(X_val, columns=list(feature_names))

    if model_type == "lgbm":
        if not HAS_LGBM:
            print("  WARNING: lightgbm not installed, falling back to RandomForest")
            model_type = "rf"
        else:
            model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            # Use validation set with early stopping if provided
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
                )
            else:
                model.fit(X_train, y_train)
            return model

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
    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "mae":           round(mae, 2),
        "rmse":          round(rmse, 2),
        "r2":            round(r2, 4),
        "accuracy_pct":  round((1 - mae / np.mean(y_test)) * 100, 2),
        "n_test_samples": len(y_test),
    }


def save_model(model, path: str = DEFAULT_MODEL_PATH, feature_cols=None,
               horizon: int = None, model_type: str = None):
    """Save model + sidecar metadata JSON."""
    joblib.dump(model, path)
    meta = {"path": path}
    if feature_cols is not None:
        meta["feature_cols"] = list(feature_cols)
        meta["n_features"]   = len(feature_cols)
    if horizon is not None:
        meta["horizon"] = horizon
    if model_type is not None:
        meta["model_type"] = model_type
    if meta:
        meta_path = path.replace(".joblib", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def load_model(path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}. Run train.py first.")
    return joblib.load(path)


def load_feature_cols(path: str = DEFAULT_MODEL_PATH):
    """Load feature_cols from sidecar JSON. Falls back to BASE_FEATURE_COLS."""
    from preprocessing import BASE_FEATURE_COLS
    meta_path = path.replace(".joblib", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get("feature_cols", list(BASE_FEATURE_COLS))
    return list(BASE_FEATURE_COLS)


def load_region_model(region: str):
    """Legacy single-model loader — used for fallback to recursive prediction."""
    region_path = f"model_{region}.joblib"
    if os.path.exists(region_path):
        return joblib.load(region_path)
    if os.path.exists(DEFAULT_MODEL_PATH):
        return joblib.load(DEFAULT_MODEL_PATH)
    raise FileNotFoundError("No model found. Run: python train.py --data data/demand.csv")


def load_region_model_with_meta(region: str):
    """Legacy single-model loader + feature_cols. Used as fallback."""
    region_path = f"model_{region}.joblib"
    actual_path = region_path if os.path.exists(region_path) else DEFAULT_MODEL_PATH
    if not os.path.exists(actual_path):
        raise FileNotFoundError(
            "No model found. Run: python train.py --data data/demand.csv"
        )
    model = joblib.load(actual_path)
    feature_cols = load_feature_cols(actual_path)
    return model, feature_cols


# ── Direct multi-step model loading ──────────────────────────────────────────

def horizon_model_path(region: str, horizon: int) -> str:
    """Return file path for a horizon-specific model."""
    return f"model_{region}_h{horizon}.joblib"


def load_horizon_models(region: str, horizons: range = range(1, 25)):
    """
    Load all 24 horizon-specific models for a region.

    Returns:
        (models, feature_cols)
        models: dict {horizon: model}  for each horizon successfully loaded
        feature_cols: list of features (taken from the first loaded model)
    """
    models = {}
    feature_cols = None
    for h in horizons:
        path = horizon_model_path(region, h)
        if os.path.exists(path):
            models[h] = joblib.load(path)
            if feature_cols is None:
                feature_cols = load_feature_cols(path)

    if not models:
        return None, None
    return models, feature_cols


def has_direct_models(region: str) -> bool:
    """Check if at least 24 horizon models exist for a region."""
    return all(
        os.path.exists(horizon_model_path(region, h))
        for h in range(1, 25)
    )


# ── Feature importance ───────────────────────────────────────────────────────

def feature_importances(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return {}
    importances = model.feature_importances_
    if importances.sum() > 0:
        importances = importances / importances.sum()
    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1], reverse=True,
    ))


def full_training_pipeline(X, y, feature_names, model_type="lgbm",
                           model_path=DEFAULT_MODEL_PATH, horizon=None):
    """Standard train/eval/save pipeline."""
    X_train, X_test, y_train, y_test = train_test_split_ts(X, y)
    # Further split train into train+val for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split_ts(X_train, y_train, split_ratio=0.9)
    model   = train_model(X_tr, y_tr, model_type, X_val=X_val, y_val=y_val,
                          feature_names=feature_names)

    # Wrap test set with feature names too so predict() doesn't warn
    if feature_names is not None and not hasattr(X_test, "columns"):
        import pandas as pd
        X_test_df = pd.DataFrame(X_test, columns=list(feature_names))
    else:
        X_test_df = X_test
    metrics = evaluate_model(model, X_test_df, y_test)
    save_model(model, model_path,
               feature_cols=list(feature_names),
               horizon=horizon, model_type=model_type)
    importances = feature_importances(model, feature_names)
    return model, metrics, importances
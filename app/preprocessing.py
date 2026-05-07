"""
Module 1 & 2: Data Preprocessing + Feature Engineering
India-only. Reads demand.csv produced by prepare_dataset.py.

Feature count: 21 base + up to 25 weather/holiday features = up to 46 total
  Base (21):    lags, rolling stats, cyclic time, calendar, India peaks/seasons
  Weather (25): per-region temp, humidity, solar, heat index, CDH
  Holidays (7): is_national_holiday, is_major_festival, is_regional_holiday,
                days_to_holiday, day_after_holiday, is_pre_festival, holiday_type

Weather columns are optional — model gracefully falls back to 21 features
if weather data is not present in demand.csv (i.e. weather_fetcher.py
has not been run yet).
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Tuple, List

# ── Holiday calendar ──────────────────────────────────────────────────────────
try:
    import holidays as _holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False

# India national + state holidays (2015-2030)
def _build_india_holidays(years):
    """Build comprehensive India holiday set including major festivals."""
    h_set = set()

    if _HOLIDAYS_AVAILABLE:
        for yr in years:
            india_h = _holidays_lib.India(years=yr)
            h_set.update(india_h.keys())

    # Major festivals not always in the `holidays` package
    # Diwali: approximate dates (varies by year, lunar calendar)
    DIWALI = [
        date(2017, 10, 19), date(2018, 11,  7), date(2019, 10, 27),
        date(2020, 11, 14), date(2021, 11,  4), date(2022, 10, 24),
        date(2023, 11, 12), date(2024, 11,  1), date(2025, 10, 20),
    ]
    # Holi
    HOLI = [
        date(2017,  3, 13), date(2018,  3,  2), date(2019,  3, 21),
        date(2020,  3, 10), date(2021,  3, 29), date(2022,  3, 18),
        date(2023,  3,  8), date(2024,  3, 25), date(2025,  3, 14),
    ]
    # Eid ul-Fitr (approximate)
    EID = [
        date(2017,  6, 26), date(2018,  6, 16), date(2019,  6,  5),
        date(2020,  5, 24), date(2021,  5, 13), date(2022,  5,  3),
        date(2023,  4, 21), date(2024,  4, 10), date(2025,  3, 30),
    ]
    # Durga Puja / Navratri peak day
    DURGA = [
        date(2017,  9, 30), date(2018, 10, 18), date(2019, 10,  7),
        date(2020, 10, 25), date(2021, 10, 15), date(2022,  5,  3),
        date(2023, 10, 24), date(2024, 10, 12), date(2025, 10,  2),
    ]
    for festival_list in [DIWALI, HOLI, EID, DURGA]:
        h_set.update(festival_list)

    return h_set


# Cache the holiday set so we don't rebuild on every call
_HOLIDAY_CACHE = {}

def _get_holidays(years):
    key = tuple(sorted(set(years)))
    if key not in _HOLIDAY_CACHE:
        _HOLIDAY_CACHE[key] = _build_india_holidays(key)
    return _HOLIDAY_CACHE[key]


# High-impact festival windows (demand drops for multiple days)
FESTIVAL_WINDOWS = {
    "diwali": [
        (date(2017, 10, 17), date(2017, 10, 21)),
        (date(2018, 11,  5), date(2018, 11,  9)),
        (date(2019, 10, 25), date(2019, 10, 29)),
        (date(2020, 11, 12), date(2020, 11, 16)),
        (date(2021, 11,  2), date(2021, 11,  6)),
        (date(2022, 10, 22), date(2022, 10, 26)),
        (date(2023, 11, 10), date(2023, 11, 14)),
        (date(2024, 10, 30), date(2024, 11,  3)),
    ],
}


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7 holiday/festival features to the dataframe.

    Features added:
      is_national_holiday   — Republic Day, Independence Day, Gandhi Jayanti
      is_major_festival     — Diwali, Holi, Eid, Durga Puja
      is_regional_holiday   — state-level holidays (from holidays package)
      days_to_next_holiday  — 0 on holiday, 1 day before, 2 two days before, etc. (capped at 7)
      day_after_holiday     — 1 if the day immediately follows a holiday
      is_pre_festival       — 1 in the 3 days leading up to Diwali/Eid (demand spikes)
      is_diwali_window      — 1 during the 5-day Diwali festival (demand drops ~15%)
    """
    df = df.copy()
    dates = df["timestamp"].dt.date

    # Collect year range
    years = list(range(df["timestamp"].dt.year.min(), df["timestamp"].dt.year.max() + 2))
    holidays_set = _get_holidays(years)

    # Classify each holiday as national vs regional
    national_holidays = set()
    if _HOLIDAYS_AVAILABLE:
        for yr in years:
            for d, name in _holidays_lib.India(years=yr).items():
                if any(k in name for k in ["Republic", "Independence", "Gandhi", "Buddha",
                                            "Ambedkar", "Christmas", "Eid", "Diwali"]):
                    national_holidays.add(d)
    else:
        # Fallback hardcoded national holidays
        for yr in years:
            national_holidays.update([
                date(yr, 1, 26), date(yr, 8, 15), date(yr, 10, 2),
            ])

    major_festivals = set()
    for fl in FESTIVAL_WINDOWS.values():
        for start, end in fl:
            d = start
            while d <= end:
                major_festivals.add(d)
                d += timedelta(days=1)

    df["is_national_holiday"] = dates.apply(lambda d: int(d in national_holidays))
    df["is_major_festival"]   = dates.apply(lambda d: int(d in major_festivals))
    df["is_regional_holiday"] = dates.apply(
        lambda d: int(d in holidays_set and d not in national_holidays)
    )

    # days_to_next_holiday (look-ahead 7 days)
    date_set = holidays_set | national_holidays | major_festivals
    def days_to_next(d):
        for i in range(8):
            if (d + timedelta(days=i)) in date_set:
                return i
        return 7
    df["days_to_next_holiday"] = dates.apply(days_to_next)

    # day_after_holiday
    df["day_after_holiday"] = dates.apply(
        lambda d: int((d - timedelta(days=1)) in date_set)
    )

    # pre-festival: 3 days before Diwali or Eid (demand SPIKES for shopping/travel)
    pre_festival_days = set()
    for fl in [FESTIVAL_WINDOWS["diwali"]]:
        for start, _ in fl:
            for i in range(1, 4):
                pre_festival_days.add(start - timedelta(days=i))
    df["is_pre_festival"] = dates.apply(lambda d: int(d in pre_festival_days))

    # Diwali window: demand drops ~15%
    diwali_days = set()
    for start, end in FESTIVAL_WINDOWS["diwali"]:
        d = start
        while d <= end:
            diwali_days.add(d)
            d += timedelta(days=1)
    df["is_diwali_window"] = dates.apply(lambda d: int(d in diwali_days))

    return df


# ── Weather feature builder ───────────────────────────────────────────────────
WEATHER_REGIONS = ["northern", "western", "southern", "eastern", "ne"]
WEATHER_RAW_COLS = [f"{r}_{v}" for r in WEATHER_REGIONS
                    for v in ["temp_c", "humidity_pct", "solar_wm2"]]
WEATHER_DERIVED_COLS = [f"{r}_{v}" for r in WEATHER_REGIONS
                        for v in ["heat_index", "cdh"]]

# Use Northern (Delhi) as the "national" representative for weather
# since it's the largest region and most sensitive to AC-driven demand
NATIONAL_WEATHER_COLS = [
    "northern_temp_c", "northern_humidity_pct", "northern_solar_wm2",
    "northern_heat_index", "northern_cdh",
]


def _has_weather(df: pd.DataFrame) -> bool:
    return any(c in df.columns for c in WEATHER_RAW_COLS)


def add_weather_features_to_df(df: pd.DataFrame, region_col: str = "demand_mw") -> pd.DataFrame:
    """
    Select the right weather columns for the region being modelled.
    For national demand → use Northern (Delhi) weather.
    For a region → use that region's own weather columns.
    """
    df = df.copy()
    if not _has_weather(df):
        return df   # weather not available — training without it

    # Map region_col → weather region prefix
    region_map = {
        "demand_mw":             "northern",   # national uses Delhi
        "Northern_Region_mw":    "northern",
        "Western_Region_mw":     "western",
        "Southern_Region_mw":    "southern",
        "Eastern_Region_mw":     "eastern",
        "NorthEastern_Region_mw":"ne",
    }
    prefix = region_map.get(region_col, "northern")

    # Rename that region's weather to generic names so FEATURE_COLS stays clean
    weather_cols_for_region = {
        f"{prefix}_temp_c":       "weather_temp_c",
        f"{prefix}_humidity_pct": "weather_humidity_pct",
        f"{prefix}_solar_wm2":    "weather_solar_wm2",
        f"{prefix}_heat_index":   "weather_heat_index",
        f"{prefix}_cdh":          "weather_cdh",
    }
    existing = {k: v for k, v in weather_cols_for_region.items() if k in df.columns}
    df = df.rename(columns=existing)
    return df


# ── Region column load / clean ────────────────────────────────────────────────
REGION_COLS = [
    "Northern_Region_mw", "Western_Region_mw", "Eastern_Region_mw",
    "Southern_Region_mw", "NorthEastern_Region_mw",
]


def load_and_clean(filepath: str, region_col: str = "demand_mw") -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if region_col not in df.columns:
        available = [c for c in df.columns if "mw" in c.lower()]
        raise ValueError(f"Column '{region_col}' not found. Available: {available}")

    df["demand_mw"] = pd.to_numeric(df[region_col], errors="coerce")
    df["demand_mw"] = df["demand_mw"].interpolate(method="linear")
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    df = df.set_index("timestamp").resample("h").mean().interpolate().reset_index()

    print(f"[{region_col}] {len(df):,} rows | "
          f"min={df['demand_mw'].min():,.0f} MW | "
          f"max={df['demand_mw'].max():,.0f} MW | "
          f"mean={df['demand_mw'].mean():,.0f} MW")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["is_morning_peak"] = ((df["hour"] >= 7)  & (df["hour"] <= 10)).astype(int)
    df["is_evening_peak"] = ((df["hour"] >= 18) & (df["hour"] <= 22)).astype(int)
    df["is_summer"]   = df["month"].isin([4, 5, 6]).astype(int)
    df["is_winter"]   = df["month"].isin([11, 12, 1]).astype(int)
    df["is_monsoon"]  = df["month"].isin([7, 8, 9]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 2, 3, 24, 48, 72, 96]:
        df[f"lag_{lag}h"] = df["demand_mw"].shift(lag)
    df["rolling_7d_mean"] = df["demand_mw"].shift(1).rolling(window=168).mean()
    df["rolling_7d_max"]  = df["demand_mw"].shift(1).rolling(window=168).max()
    df["rolling_7d_std"]  = df["demand_mw"].shift(1).rolling(window=168).std()
    df = df.dropna().reset_index(drop=True)
    return df


# ── Feature column lists ──────────────────────────────────────────────────────
BASE_FEATURE_COLS = [
    "lag_1h", "lag_2h", "lag_3h",
    "lag_24h", "lag_48h", "lag_72h", "lag_96h",
    "rolling_7d_mean", "rolling_7d_max", "rolling_7d_std",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_weekend", "day_of_week",
    "is_morning_peak", "is_evening_peak",
    "is_summer", "is_winter", "is_monsoon",
]

HOLIDAY_FEATURE_COLS = [
    "is_national_holiday", "is_major_festival", "is_regional_holiday",
    "days_to_next_holiday", "day_after_holiday",
    "is_pre_festival", "is_diwali_window",
]

WEATHER_FEATURE_COLS = [
    "weather_temp_c", "weather_humidity_pct", "weather_solar_wm2",
    "weather_heat_index", "weather_cdh",
]

# FEATURE_COLS is computed dynamically based on what's available in the data
def get_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = list(BASE_FEATURE_COLS)
    for col in HOLIDAY_FEATURE_COLS:
        if col in df.columns:
            cols.append(col)
    for col in WEATHER_FEATURE_COLS:
        if col in df.columns:
            cols.append(col)
    return cols

# Keep backward-compatible alias
FEATURE_COLS = BASE_FEATURE_COLS


def build_feature_matrix(df: pd.DataFrame) -> Tuple:
    feature_cols = get_feature_cols(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[feature_cols].values
    y = df["demand_mw"].values
    return X, y, feature_cols


def preprocess_pipeline(filepath: str, region_col: str = "demand_mw",
                         use_weather: bool = True, use_holidays: bool = True):
    """
    Full pipeline:
      load → time features → lag features
      → (optional) holiday features
      → (optional) weather features
      → feature matrix

    use_weather=True will use weather columns if present in demand.csv.
    use_holidays=True will add holiday/festival flags.
    Pass use_weather=False or use_holidays=False to disable.
    """
    df = load_and_clean(filepath, region_col=region_col)
    df = add_time_features(df)
    df = add_lag_features(df)

    if use_holidays:
        df = add_holiday_features(df)
        hol_count = sum(1 for c in HOLIDAY_FEATURE_COLS if c in df.columns)
        print(f"  Holiday features: {hol_count} columns added")

    if use_weather and _has_weather(df):
        df = add_weather_features_to_df(df, region_col=region_col)
        wx_count = sum(1 for c in WEATHER_FEATURE_COLS if c in df.columns)
        print(f"  Weather features: {wx_count} columns added for {region_col}")
    elif use_weather:
        print(f"  Weather: not found in CSV — run weather_fetcher.py first")

    X, y, cols = build_feature_matrix(df)
    n_base = len(BASE_FEATURE_COLS)
    n_hol  = sum(1 for c in cols if c in HOLIDAY_FEATURE_COLS)
    n_wx   = sum(1 for c in cols if c in WEATHER_FEATURE_COLS)
    print(f"  Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features "
          f"(base={n_base}, holidays={n_hol}, weather={n_wx})")

    return df, X, y, cols
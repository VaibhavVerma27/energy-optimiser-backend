"""
live_data_fetcher.py
====================
Fetches real data from two public sources:

1. Open-Meteo FORECAST API (truly real-time, no API key)
   → Current + next 7 days hourly: temp, humidity, solar irradiance, wind speed
   → Used for live forecasting with real weather instead of climatological fallbacks

2. National Power Portal (NPP / CEA Daily Reports)
   → URL pattern: npp.gov.in/public-reports/cea/daily/dgr/DD-MM-YYYY/dgrN-YYYY-MM-DD.xls
   → DGR1:  All India + Region-wise daily generation totals (MU)
   → DGR8:  NTPC region-wise monitored/available capacity + actual generation
   → Used for the comparison page: predicted vs actual per day
   → Data available from 31 March 2013 onwards

Note: No live hourly demand API exists publicly for India.
      NPP provides only daily totals. We use these for:
        - Daily MAE comparison (predicted daily total vs actual daily total)
        - Rolling model performance tracking
"""

import json
import math
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, date
from typing import Optional
from urllib.parse import urlencode

import pandas as pd

# ── City coordinates for 5 POSOCO regions ────────────────────────────────────
REGION_CITIES = {
    "Northern_Region":     {"lat": 28.6139, "lon": 77.2090, "name": "New Delhi"},
    "Western_Region":      {"lat": 19.0760, "lon": 72.8777, "name": "Mumbai"},
    "Southern_Region":     {"lat": 13.0827, "lon": 80.2707, "name": "Chennai"},
    "Eastern_Region":      {"lat": 22.5726, "lon": 88.3639, "name": "Kolkata"},
    "NorthEastern_Region": {"lat": 26.1445, "lon": 91.7362, "name": "Guwahati"},
}

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE  = "https://archive-api.open-meteo.com/v1/archive"
NPP_BASE = "https://npp.gov.in/public-reports/cea/daily/dgr"


# ── 1. Open-Meteo live weather ────────────────────────────────────────────────

def fetch_live_weather(region: str, hours_ahead: int = 24) -> list:
    """
    Fetch real-time weather forecast for a region's representative city.
    Returns list of dicts with keys:
      timestamp, temp_c, humidity_pct, solar_wm2, wind_speed_ms
    for the next `hours_ahead` hours starting from the current hour (IST).

    Uses Open-Meteo forecast API — free, no key, updates every hour.
    """
    city = REGION_CITIES.get(region)
    if not city:
        raise ValueError(f"Unknown region: {region}")

    params = {
        "latitude":  city["lat"],
        "longitude": city["lon"],
        "hourly":    "temperature_2m,relativehumidity_2m,shortwave_radiation,windspeed_10m",
        "timezone":  "Asia/Kolkata",
        "forecast_days": max(2, math.ceil(hours_ahead / 24) + 1),
    }
    url = f"{OPEN_METEO_FORECAST}?{urlencode(params)}"

    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.URLError as e:
            if attempt == 2:
                raise RuntimeError(f"Open-Meteo forecast failed: {e}") from e
            time.sleep(1)

    hourly = data["hourly"]
    now_str = datetime.now().strftime("%Y-%m-%dT%H:00")

    # Find current hour index
    try:
        start_idx = hourly["time"].index(now_str)
    except ValueError:
        start_idx = 0

    result = []
    for i in range(start_idx, min(start_idx + hours_ahead, len(hourly["time"]))):
        result.append({
            "timestamp":    hourly["time"][i],
            "temp_c":       hourly["temperature_2m"][i],
            "humidity_pct": hourly["relativehumidity_2m"][i],
            "solar_wm2":    hourly["shortwave_radiation"][i],
            "wind_speed_ms":hourly["windspeed_10m"][i],
        })
    return result


def fetch_all_regions_live_weather(hours_ahead: int = 24) -> dict:
    """
    Fetch live weather for all 5 regions in parallel-ish sequence.
    Returns dict: {region_id: [list of hourly weather dicts]}
    Falls back gracefully if a single region fails.
    """
    result = {}
    for region in REGION_CITIES:
        try:
            result[region] = fetch_live_weather(region, hours_ahead)
            time.sleep(0.2)   # polite to free API
        except Exception as e:
            print(f"  WARNING: Weather fetch failed for {region}: {e}")
            result[region] = []
    return result


def fetch_weather_for_date(region: str, target_date: date, hours: int = 24) -> tuple:
    """
    Fetch hourly weather for a specific date for a region's city.
    Automatically selects the correct Open-Meteo API:

      Past dates          → Archive API  (historical measurements)
      Today + up to 16d   → Forecast API (NWP model output)
      Beyond 16 days      → Climatological fallback (monthly averages)

    Returns:
        (weather_list, source_label)
        weather_list: list of hourly dicts with temp_c, humidity_pct, solar_wm2, wind_speed_ms
        source_label: "archive", "forecast", or "climatology"
    """
    city  = REGION_CITIES.get(region)
    if not city:
        return [], "unknown_region"

    today     = date.today()
    days_diff = (target_date - today).days   # negative = past, 0 = today, positive = future

    # ── Past dates: use Archive API ──────────────────────────────────────────
    if days_diff < 0:
        start_str = target_date.strftime("%Y-%m-%d")
        end_str   = target_date.strftime("%Y-%m-%d")
        params = {
            "latitude":   city["lat"],
            "longitude":  city["lon"],
            "start_date": start_str,
            "end_date":   end_str,
            "hourly":     "temperature_2m,relativehumidity_2m,shortwave_radiation,windspeed_10m",
            "timezone":   "Asia/Kolkata",
        }
        url = f"{OPEN_METEO_ARCHIVE}?{urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            hourly = data["hourly"]
            result = []
            for i in range(min(hours, len(hourly["time"]))):
                result.append({
                    "timestamp":    hourly["time"][i],
                    "temp_c":       hourly["temperature_2m"][i],
                    "humidity_pct": hourly["relativehumidity_2m"][i],
                    "solar_wm2":    hourly["shortwave_radiation"][i],
                    "wind_speed_ms":hourly["windspeed_10m"][i],
                })
            return result[:hours], "archive"
        except Exception as e:
            print(f"  WARNING: Archive API failed for {region} {target_date}: {e}")
            return _climatological_weather(region, target_date, hours), "climatology_fallback"

    # ── Today + up to 16 days ahead: use Forecast API ────────────────────────
    if 0 <= days_diff <= 15:
        forecast_days = days_diff + 2   # +2 to ensure we cover the full target day
        params = {
            "latitude":      city["lat"],
            "longitude":     city["lon"],
            "hourly":        "temperature_2m,relativehumidity_2m,shortwave_radiation,windspeed_10m",
            "timezone":      "Asia/Kolkata",
            "forecast_days": min(forecast_days, 16),
        }
        url = f"{OPEN_METEO_FORECAST}?{urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            hourly  = data["hourly"]
            # Find the hours that belong to target_date
            target_str = target_date.strftime("%Y-%m-%d")
            result = []
            for i, ts in enumerate(hourly["time"]):
                if ts.startswith(target_str):
                    result.append({
                        "timestamp":    ts,
                        "temp_c":       hourly["temperature_2m"][i],
                        "humidity_pct": hourly["relativehumidity_2m"][i],
                        "solar_wm2":    hourly["shortwave_radiation"][i],
                        "wind_speed_ms":hourly["windspeed_10m"][i],
                    })
            if result:
                return result[:hours], "forecast"
        except Exception as e:
            print(f"  WARNING: Forecast API failed for {region} {target_date}: {e}")

        return _climatological_weather(region, target_date, hours), "climatology_fallback"

    # ── Beyond 16 days: climatological fallback ───────────────────────────────
    return _climatological_weather(region, target_date, hours), "climatology"


def _climatological_weather(region: str, target_date: date, hours: int = 24) -> list:
    """
    Generate climatological weather for a date when API data is unavailable.
    Based on monthly averages for each region's city.
    """
    m = target_date.month

    # Monthly mean temperature (°C) per region city
    CLIM_TEMP = {
        "Northern_Region":     [15,18,25,33,38,36,32,31,29,26,19,14],
        "Western_Region":      [24,25,28,30,32,31,29,29,28,30,27,24],
        "Southern_Region":     [26,28,31,33,35,33,31,31,30,28,26,25],
        "Eastern_Region":      [20,23,29,33,34,32,30,30,29,28,23,19],
        "NorthEastern_Region": [16,18,23,26,26,27,28,28,27,23,18,14],
    }
    CLIM_HUMID = {
        "Northern_Region":     [55,45,40,25,22,40,70,72,60,45,50,55],
        "Western_Region":      [65,65,70,75,80,85,88,87,85,80,72,65],
        "Southern_Region":     [75,75,70,70,68,70,78,80,78,75,74,75],
        "Eastern_Region":      [70,68,65,60,65,78,85,85,80,75,72,70],
        "NorthEastern_Region": [75,72,70,68,72,85,90,90,85,80,75,75],
    }

    temp   = CLIM_TEMP.get(region, CLIM_TEMP["Northern_Region"])[m-1]
    humid  = CLIM_HUMID.get(region, CLIM_HUMID["Northern_Region"])[m-1]

    result = []
    for h in range(hours):
        # Simple bell-curve solar (0 before 6am, peaks at noon)
        if 6 <= h <= 18:
            solar = 800 * math.exp(-0.5 * ((h - 12) / 3.0) ** 2)
            # Monsoon cloud reduction
            if m in [6,7,8,9]:
                solar *= 0.45
            elif m in [5,10]:
                solar *= 0.75
        else:
            solar = 0.0

        # Slight diurnal temperature variation (±3°C)
        t_adj = temp + 3 * math.sin(math.pi * (h - 6) / 12) if 6 <= h <= 18 else temp - 2

        result.append({
            "timestamp":    f"{target_date.strftime('%Y-%m-%d')}T{h:02d}:00",
            "temp_c":       round(t_adj, 1),
            "humidity_pct": humid,
            "solar_wm2":    round(solar, 1),
            "wind_speed_ms":3.5,   # moderate default
        })
    return result


def fetch_all_regions_weather_for_date(target_date: date, hours: int = 24) -> tuple:
    """
    Fetch weather for all 5 regions for a specific date.
    Returns (weather_dict, source_summary).
    """
    result  = {}
    sources = {}
    for region in REGION_CITIES:
        try:
            wx, src = fetch_weather_for_date(region, target_date, hours)
            result[region]  = wx
            sources[region] = src
            time.sleep(0.15)
        except Exception as e:
            print(f"  WARNING: Weather fetch failed for {region}: {e}")
            result[region]  = _climatological_weather(region, target_date, hours)
            sources[region] = "climatology_error_fallback"

    # Summarise sources
    unique_sources = set(sources.values())
    if len(unique_sources) == 1:
        source_summary = list(unique_sources)[0]
    else:
        source_summary = "+".join(sorted(unique_sources))

    return result, source_summary
    """
    Convert live weather list into the weather_override dict format
    expected by the forecast router.
    Uses average of first 12 daytime hours for temp/humidity,
    and flags that per-hour solar is available.
    """
    if not live_weather:
        return {}
    temps = [w["temp_c"] for w in live_weather if w.get("temp_c") is not None]
    hums  = [w["humidity_pct"] for w in live_weather if w.get("humidity_pct") is not None]
    # For capacity engine: pass the per-hour solar directly via weather_data list
    return {
        "temp_c":       round(sum(temps) / len(temps), 1) if temps else 30.0,
        "humidity":     round(sum(hums)  / len(hums),  1) if hums  else 60.0,
        "hourly_data":  live_weather,   # full per-hour weather for capacity engine
    }


# ── 2. NPP Daily Reports ──────────────────────────────────────────────────────

def _npp_url(report_date: date, report_num: int) -> tuple:
    """Build NPP report URL for a given date and report number.
    Returns (pdf_url, xls_url).
    """
    d = report_date.strftime("%d-%m-%Y")
    y = report_date.strftime("%Y-%m-%d")
    base = f"{NPP_BASE}/{d}"
    return (
        f"{base}/dgr{report_num}-{y}.pdf",
        f"{base}/dgr{report_num}-{y}.xls",
    )


def fetch_npp_daily_overview(report_date: date) -> Optional[dict]:
    """
    Fetch DGR1 (All India/Region-wise Power Generation Overview) for a date.
    Returns dict with keys: date, all_india_mu, regions: {name: mu}
    or None if not available (e.g. future date, weekend with no update).

    The XLS has this structure (rows 5-11 approx):
      Region | Today's Actual (MU) | ...
    """
    _, xls_url = _npp_url(report_date, 1)
    try:
        with urllib.request.urlopen(xls_url, timeout=15) as resp:
            raw = resp.read()
        # Parse XLS using xlrd via pandas
        import io
        df = pd.read_excel(io.BytesIO(raw), header=None, engine="xlrd")

        # Find region rows by scanning for known region names
        region_map = {
            "northern": "Northern_Region",
            "western":  "Western_Region",
            "southern": "Southern_Region",
            "eastern":  "Eastern_Region",
            "north eastern": "NorthEastern_Region",
            "north-eastern": "NorthEastern_Region",
            "northeastern": "NorthEastern_Region",
            "all india": "ALL_INDIA",
        }
        regions_found = {}
        for _, row in df.iterrows():
            for col_idx, val in enumerate(row):
                if isinstance(val, str):
                    key = val.strip().lower()
                    for pattern, region_id in region_map.items():
                        if pattern in key:
                            # Look for numeric values in the same row
                            nums = [v for v in row[col_idx+1:] if isinstance(v, (int, float))]
                            if nums:
                                regions_found[region_id] = round(float(nums[0]), 2)
                            break

        if not regions_found:
            return None

        return {
            "date":     report_date.isoformat(),
            "source":   "NPP/CEA DGR1",
            "url":      xls_url,
            "regions":  regions_found,
            "all_india_mu": regions_found.get("ALL_INDIA"),
        }

    except Exception as e:
        return None


def fetch_npp_ntpc_overview(report_date: date) -> Optional[dict]:
    """
    Fetch DGR8 (NTPC Region-wise Generation Overview) for a date.
    Returns dict with per-region: monitored_mw, available_mw, actual_mu
    This gives available capacity data for comparison.
    """
    pdf_url, _ = _npp_url(report_date, 8)
    try:
        # Try to get the PDF and extract text
        import urllib.request
        with urllib.request.urlopen(pdf_url, timeout=15) as resp:
            pdf_bytes = resp.read()

        # Extract text from PDF
        import io
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            # Fallback: basic text extraction
            text = pdf_bytes.decode("latin-1", errors="ignore")

        # Parse region data from text (matches the format seen in the PDF)
        # Format: "Northern 13294.06 9050.00 204.93 172.67 ..."
        #          Region   Monitored Available Program Actual
        import re
        region_patterns = {
            "Northern":     "Northern_Region",
            "Western":      "Western_Region",
            "Southern":     "Southern_Region",
            "Eastern":      "Eastern_Region",
            "North Eastern":"NorthEastern_Region",
        }
        result = {"date": report_date.isoformat(), "source": "NPP/CEA DGR8", "regions": {}}
        for label, region_id in region_patterns.items():
            pattern = rf"{label}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            m = re.search(pattern, text)
            if m:
                result["regions"][region_id] = {
                    "monitored_mw":  float(m.group(1)),
                    "available_mw":  float(m.group(2)),
                    "program_mu":    float(m.group(3)),
                    "actual_mu":     float(m.group(4)),
                }
        return result if result["regions"] else None

    except Exception as e:
        return None


def fetch_actual_demand_for_date(target_date: date) -> Optional[dict]:
    """
    Main function: fetch all available actual demand/generation data for a date.
    Combines DGR1 (generation overview) and DGR8 (NTPC overview).

    Returns:
    {
        "date": "2024-05-15",
        "dgr1": { ... generation totals ... },
        "dgr8": { ... NTPC capacity/actual ... },
        "available": True/False
    }
    """
    dgr1 = fetch_npp_daily_overview(target_date)
    dgr8 = fetch_npp_ntpc_overview(target_date)

    return {
        "date":      target_date.isoformat(),
        "dgr1":      dgr1,
        "dgr8":      dgr8,
        "available": dgr1 is not None or dgr8 is not None,
    }


# ── 3. Historical demand for model input ─────────────────────────────────────

def get_recent_demand_from_csv_or_live(
    csv_path: str,
    region_col: str,
    hours: int = 168,
) -> tuple:
    """
    Get recent demand values for model input.
    Returns (values: list, source: str, last_timestamp: str)

    Strategy:
    1. Read from demand.csv (always available, up to training cutoff)
    2. Return the last `hours` values with their timestamps
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        if region_col not in df.columns:
            raise ValueError(f"Column {region_col} not in {csv_path}")

        recent = df[[region_col, "timestamp"]].dropna().tail(hours)
        values = recent[region_col].tolist()
        last_ts = recent["timestamp"].iloc[-1].isoformat() if len(recent) > 0 else None

        return values, "demand.csv", last_ts
    except Exception as e:
        raise RuntimeError(f"Could not load demand data: {e}") from e



# ── 5. Recent demand scale factor from NPP ────────────────────────────────────

def get_recent_demand_scale(
    csv_path: str,
    region_col: str,
    npp_date: date = None,
) -> dict:
    """
    Compute a data-driven scale factor to adjust the 2022 history buffer
    to today's actual demand level.

    Method:
      1. Fetch yesterday's actual generation from NPP DGR1 (daily MU)
      2. Convert MU → average MW (MU×1000 / 24)
      3. Compute scale = actual_avg_mw / csv_history_avg_mw

    This replaces the 1.04^years approximation with actual observed data.

    Returns dict:
      {
        "scale_factor": float,         # multiply 2022 history by this
        "actual_avg_mw": float,        # yesterday's average MW from NPP
        "csv_avg_mw": float,           # 2022 history average MW
        "source": "npp_dgr1" | "growth_estimate",
        "date": str,
      }
    """
    # Region col → DGR1 region name mapping
    region_name_map = {
        "Northern_Region_mw":     "Northern_Region",
        "Western_Region_mw":      "Western_Region",
        "Southern_Region_mw":     "Southern_Region",
        "Eastern_Region_mw":      "Eastern_Region",
        "NorthEastern_Region_mw": "NorthEastern_Region",
    }
    region_id = region_name_map.get(region_col, "Northern_Region")

    # Get CSV average for this region (last 168h = 7 days)
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        csv_vals = df[region_col].dropna().tail(168).tolist() if region_col in df.columns else []
        csv_avg_mw = float(sum(csv_vals) / len(csv_vals)) if csv_vals else 0.0
    except Exception:
        csv_avg_mw = 0.0

    if csv_avg_mw <= 0:
        return {"scale_factor": 1.0, "source": "no_csv_data", "date": str(date.today())}

    # Try to get actual from NPP for yesterday (or up to 3 days back)
    if npp_date is None:
        npp_date = date.today() - timedelta(days=1)

    for days_back in range(4):
        attempt_date = npp_date - timedelta(days=days_back)
        try:
            dgr1 = fetch_npp_daily_overview(attempt_date)
            if dgr1 and dgr1.get("regions", {}).get(region_id):
                actual_mu = dgr1["regions"][region_id]
                # MU = million units = million kWh = 1000 MWh
                actual_avg_mw = (actual_mu * 1000) / 24.0
                if actual_avg_mw > csv_avg_mw * 0.5:  # sanity: at least 50% of csv value
                    scale = actual_avg_mw / csv_avg_mw
                    # Cap scale to reasonable range: 0.7x to 2.0x
                    scale = max(0.7, min(2.0, scale))
                    return {
                        "scale_factor":  round(scale, 4),
                        "actual_avg_mw": round(actual_avg_mw, 1),
                        "csv_avg_mw":    round(csv_avg_mw, 1),
                        "actual_mu":     actual_mu,
                        "source":        "npp_dgr1",
                        "date":          attempt_date.isoformat(),
                    }
        except Exception:
            continue

    # Fallback: growth estimate
    from datetime import datetime as _dt
    years = max(0, (_dt.now().year - 2022) + (_dt.now().month - 12) / 12)
    growth = round((1.04 ** years), 4)
    return {
        "scale_factor": growth,
        "actual_avg_mw": round(csv_avg_mw * growth, 1),
        "csv_avg_mw":    round(csv_avg_mw, 1),
        "source":        "growth_estimate_4pct_pa",
        "date":          str(date.today()),
    }

# ── 4. Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Open-Meteo live weather fetch...")
    try:
        wx = fetch_live_weather("Northern_Region", hours_ahead=6)
        print(f"  Northern Region next 6h: {len(wx)} hours")
        if wx:
            print(f"  Hour 0: {wx[0]['timestamp']} | "
                  f"{wx[0]['temp_c']}°C | "
                  f"{wx[0]['solar_wm2']} W/m² | "
                  f"{wx[0]['wind_speed_ms']} m/s")
    except Exception as e:
        print(f"  Weather fetch failed (expected in sandbox): {e}")

    print("\nTesting NPP data fetch for yesterday...")
    yesterday = date.today() - timedelta(days=1)
    try:
        data = fetch_actual_demand_for_date(yesterday)
        print(f"  Date: {data['date']}")
        print(f"  Available: {data['available']}")
        if data.get("dgr8") and data["dgr8"].get("regions"):
            for r, v in list(data["dgr8"]["regions"].items())[:2]:
                print(f"  {r}: available={v['available_mw']} MW, actual={v['actual_mu']} MU")
    except Exception as e:
        print(f"  NPP fetch failed (expected in sandbox): {e}")
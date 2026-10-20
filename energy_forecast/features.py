"""Feature engineering for day-ahead load forecasting.

Every load-derived feature is lagged by at least 24 hours, so the model only
uses information that would be available when a day-ahead forecast is issued.
Temperature is treated as a known input (in production it comes from a
weather forecast).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import is_holiday

LAGS = (24, 48, 168)
FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "temperature_f",
    "cdd",
    "hdd",
    "temp_rolling_24h",
    *[f"lag_{lag}" for lag in LAGS],
    "rolling_mean_24h_lag24",
    "rolling_max_24h_lag24",
    "rolling_mean_168h_lag24",
]


def build_features(segment_df: pd.DataFrame) -> pd.DataFrame:
    """Build features for a single segment. Rows without full lag history are dropped."""
    df = segment_df.sort_values("timestamp").set_index("timestamp").copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_holiday"] = is_holiday(idx).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    df["cdd"] = (df["temperature_f"] - 65).clip(lower=0)
    df["hdd"] = (65 - df["temperature_f"]).clip(lower=0)
    df["temp_rolling_24h"] = df["temperature_f"].rolling(24, min_periods=1).mean()

    load = df["load_mw"]
    for lag in LAGS:
        df[f"lag_{lag}"] = load.shift(lag)
    shifted = load.shift(24)
    df["rolling_mean_24h_lag24"] = shifted.rolling(24).mean()
    df["rolling_max_24h_lag24"] = shifted.rolling(24).max()
    df["rolling_mean_168h_lag24"] = shifted.rolling(168).mean()
    return df.dropna(subset=FEATURES).reset_index()

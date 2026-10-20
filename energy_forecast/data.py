"""Synthetic hourly load data for three customer segments.

The generator produces a weather-driven series that behaves like real utility
load data: temperature-sensitive residential demand with evening peaks,
business-hours commercial demand, and shift-based industrial demand that drops
on holidays. It exists so the pipeline can be run end to end without access to
proprietary meter data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = ("residential", "commercial", "industrial")

# Fixed-date US federal holidays are enough to give the models a holiday signal.
_HOLIDAYS = {(1, 1), (7, 4), (11, 11), (12, 25)}


def _temperature(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Hourly temperature in Fahrenheit with seasonal, diurnal and weather noise."""
    doy = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()
    seasonal = 55 + 25 * np.sin(2 * np.pi * (doy - 105) / 365.25)
    diurnal = 8 * np.sin(2 * np.pi * (hour - 9) / 24)
    # Multi-day weather fronts: an AR(1) process on top of the seasonal mean.
    fronts = np.zeros(len(index))
    shocks = rng.normal(0, 0.6, len(index))
    for i in range(1, len(index)):
        fronts[i] = 0.995 * fronts[i - 1] + shocks[i]
    return seasonal + diurnal + fronts


def is_holiday(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([(d.month, d.day) in _HOLIDAYS for d in index])


def generate_load(
    start: str = "2023-01-01",
    end: str = "2025-12-31 23:00",
    seed: int = 7,
) -> pd.DataFrame:
    """Return a long-format frame: timestamp, segment, temperature_f, load_mw."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="h")
    temp = _temperature(index, rng)
    hour = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy() < 5
    holiday = is_holiday(index)
    workday = weekday & ~holiday

    # Heating and cooling degree effects drive weather-sensitive demand.
    cdd = np.clip(temp - 65, 0, None)
    hdd = np.clip(65 - temp, 0, None)
    # Slow load growth (~2%/yr) anchored to a fixed epoch so separate drops line up.
    years = (index - pd.Timestamp("2023-01-01")).total_seconds().to_numpy() / (365.25 * 86400)
    trend = 0.02 * years

    evening = np.exp(-((hour - 19) ** 2) / 8) + 0.6 * np.exp(-((hour - 7) ** 2) / 4)
    residential = (
        420 + 180 * evening + 9.0 * cdd + 5.5 * hdd + 40 * (~workday)
    ) * (1 + trend)

    business = ((hour >= 8) & (hour <= 18)).astype(float)
    commercial = (300 + 380 * business * workday + 6.0 * cdd + 2.5 * hdd) * (1 + trend)

    shift = np.where((hour >= 6) & (hour < 22), 1.0, 0.7)
    industrial = (900 * shift * np.where(workday, 1.0, 0.55) + 1.2 * cdd) * (1 + trend)

    frames = []
    for name, base, noise in (
        ("residential", residential, 0.035),
        ("commercial", commercial, 0.03),
        ("industrial", industrial, 0.025),
    ):
        load = base * (1 + rng.normal(0, noise, len(index)))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": index,
                    "segment": name,
                    "temperature_f": temp.round(2),
                    "load_mw": load.round(2),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)

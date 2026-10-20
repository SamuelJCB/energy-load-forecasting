import pandas as pd
import pytest

from energy_forecast.data import SEGMENTS, generate_load
from energy_forecast.features import FEATURES, LAGS, build_features
from energy_forecast.ingest import ingest_directory, load_store, validate
from energy_forecast.model import train_segment


@pytest.fixture(scope="module")
def history():
    return generate_load(start="2023-01-01", end="2024-03-31 23:00", seed=1)


def test_generator_shape(history):
    assert set(history["segment"]) == set(SEGMENTS)
    assert (history["load_mw"] > 0).all()
    assert history.groupby("segment").size().nunique() == 1


def test_features_never_leak_same_day_load(history):
    seg = history[history["segment"] == "residential"]
    feats = build_features(seg)
    assert min(LAGS) >= 24
    assert "load_mw" not in FEATURES
    row = feats.iloc[500]
    lookup = seg.set_index("timestamp")["load_mw"]
    assert row["lag_24"] == pytest.approx(lookup[row["timestamp"] - pd.Timedelta(hours=24)])


def test_validate_rejects_bad_rows():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00", "not a date", "2024-01-01 01:00", "2024-01-01 02:00"],
            "segment": ["residential", "residential", "spaceship", "commercial"],
            "temperature_f": [30, 30, 30, 30],
            "load_mw": [500, 500, 500, -1],
        }
    )
    clean, rejected = validate(df)
    assert len(clean) == 1 and rejected == 3


def test_ingest_dedupes_and_archives(tmp_path, history):
    landing, store = tmp_path / "landing", tmp_path / "store.csv"
    landing.mkdir()
    sample = history.head(100)
    sample.to_csv(landing / "a.csv", index=False)
    sample.to_csv(landing / "b.csv", index=False)  # exact resend
    report = ingest_directory(landing, store)
    assert report.rows_added == 100 and report.duplicates_dropped == 100
    assert not list(landing.glob("*.csv"))
    assert len(load_store(store)) == 100


@pytest.mark.parametrize("segment", SEGMENTS)
def test_model_beats_seasonal_naive(history, segment):
    seg = history[history["segment"] == segment]
    _, result = train_segment(seg, test_days=21)
    assert result.xgb_mape < result.baseline_mape

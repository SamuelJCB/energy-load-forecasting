"""Per-segment XGBoost models, evaluation, and drift-triggered retraining."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import FEATURES, build_features

PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    early_stopping_rounds=50,
)


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


@dataclass
class SegmentResult:
    segment: str
    train_rows: int
    test_rows: int
    xgb_mape: float
    baseline_mape: float
    xgb_mae: float
    baseline_mae: float
    trained_at: str

    @property
    def improvement_pct(self) -> float:
        return (self.baseline_mape - self.xgb_mape) / self.baseline_mape * 100


class LevelNormalizedXGB:
    """XGBoost on load relative to its trailing weekly mean.

    Trees cannot extrapolate beyond the levels they saw in training, so a raw-MW
    model breaks when a segment's demand steps up (a new large customer, load
    growth). Predicting the ratio to the trailing 168h mean (lagged 24h) lets the
    model track level shifts within about a week.
    """

    BASE = "rolling_mean_168h_lag24"

    def __init__(self, **params):
        self.booster = xgb.XGBRegressor(**params)

    def fit(self, X, y, eval_set=None, **kw):
        es = [(Xv, yv / Xv[self.BASE]) for Xv, yv in eval_set] if eval_set else None
        self.booster.fit(X, y / X[self.BASE], eval_set=es, **kw)
        return self

    def predict(self, X):
        return self.booster.predict(X) * X[self.BASE].to_numpy()

    def save_model(self, path):
        self.booster.save_model(path)

    def load_model(self, path):
        self.booster.load_model(path)


def split_by_time(feats: pd.DataFrame, test_days: int):
    cutoff = feats["timestamp"].max() - pd.Timedelta(days=test_days)
    return feats[feats["timestamp"] <= cutoff], feats[feats["timestamp"] > cutoff]


def train_segment(segment_df: pd.DataFrame, test_days: int = 28, seed: int = 0):
    """Train on everything before the holdout window; evaluate on the holdout."""
    feats = build_features(segment_df)
    train, test = split_by_time(feats, test_days)
    # The last 14 training days act as the early-stopping validation set.
    fit, val = split_by_time(train, 14)

    model = LevelNormalizedXGB(**PARAMS, random_state=seed)
    model.fit(fit[FEATURES], fit["load_mw"], eval_set=[(val[FEATURES], val["load_mw"])], verbose=False)

    pred = model.predict(test[FEATURES])
    # Seasonal-naive baseline: same hour last week.
    baseline = test["lag_168"]
    result = SegmentResult(
        segment=str(segment_df["segment"].iloc[0]),
        train_rows=len(train),
        test_rows=len(test),
        xgb_mape=mape(test["load_mw"], pred),
        baseline_mape=mape(test["load_mw"], baseline),
        xgb_mae=mae(test["load_mw"], pred),
        baseline_mae=mae(test["load_mw"], baseline),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return model, result


def save(model: LevelNormalizedXGB, result: SegmentResult, model_dir: Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / f"{result.segment}.json")
    (model_dir / f"{result.segment}.meta.json").write_text(json.dumps(asdict(result), indent=2))


def load(segment: str, model_dir: Path) -> tuple[LevelNormalizedXGB, dict]:
    model = LevelNormalizedXGB()
    model.load_model(Path(model_dir) / f"{segment}.json")
    meta = json.loads((Path(model_dir) / f"{segment}.meta.json").read_text())
    return model, meta


def recent_error(model: LevelNormalizedXGB, segment_df: pd.DataFrame, days: int = 14) -> float:
    """MAPE of a saved model on the most recent `days` of data."""
    feats = build_features(segment_df)
    recent = feats[feats["timestamp"] > feats["timestamp"].max() - pd.Timedelta(days=days)]
    return mape(recent["load_mw"], model.predict(recent[FEATURES]))


def needs_retrain(model, meta: dict, segment_df: pd.DataFrame, tolerance: float = 1.25) -> tuple[bool, float]:
    """Retrain when recent error drifts past `tolerance` x the error measured at training time."""
    err = recent_error(model, segment_df)
    return err > meta["xgb_mape"] * tolerance, err

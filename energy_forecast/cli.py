"""Command-line entry point.

    python -m energy_forecast generate     # write synthetic history to data/landing/
    python -m energy_forecast ingest       # validate + merge landing files into data/store.csv
    python -m energy_forecast train        # train one model per segment, write report
    python -m energy_forecast retrain      # ingest new data, retrain only segments that drifted
    python -m energy_forecast plot         # actual vs forecast for the last two weeks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import model as m
from .data import SEGMENTS, generate_load
from .ingest import ingest_directory, load_store

DATA = Path("data")
LANDING, STORE, MODELS = DATA / "landing", DATA / "store.csv", Path("models")


def cmd_generate(args) -> None:
    df = generate_load(start=args.start, end=args.end, seed=args.seed)
    if args.shift_segment:
        # Simulate a structural change, e.g. a large new data center joining the commercial class.
        mask = df["segment"] == args.shift_segment
        df.loc[mask, "load_mw"] = (df.loc[mask, "load_mw"] * (1 + args.shift_pct / 100)).round(2)
    LANDING.mkdir(parents=True, exist_ok=True)
    # Split into monthly drops to mimic how meter data actually arrives.
    for period, chunk in df.groupby(df["timestamp"].dt.to_period("M")):
        chunk.to_csv(LANDING / f"load_{period}.csv", index=False)
    print(f"wrote {df['timestamp'].dt.to_period('M').nunique()} monthly files to {LANDING}/")


def cmd_ingest(args) -> None:
    print(ingest_directory(LANDING, STORE))


def _train(segments, store: pd.DataFrame) -> list[m.SegmentResult]:
    results = []
    for seg in segments:
        model, result = m.train_segment(store[store["segment"] == seg])
        m.save(model, result, MODELS)
        results.append(result)
    return results


def _print(results: list[m.SegmentResult]) -> None:
    print(f"{'segment':<12} {'XGB MAPE':>9} {'naive MAPE':>11} {'improvement':>12}")
    for r in results:
        print(f"{r.segment:<12} {r.xgb_mape:>8.2f}% {r.baseline_mape:>10.2f}% {r.improvement_pct:>11.1f}%")


def cmd_train(args) -> None:
    store = load_store(STORE)
    if store.empty:
        raise SystemExit("store is empty; run `generate` and `ingest` first")
    _print(_train(SEGMENTS, store))


def cmd_retrain(args) -> None:
    print("ingest:", ingest_directory(LANDING, STORE))
    store = load_store(STORE)
    stale = []
    for seg in SEGMENTS:
        seg_df = store[store["segment"] == seg]
        if not (MODELS / f"{seg}.json").exists():
            stale.append(seg)
            continue
        model, meta = m.load(seg, MODELS)
        drift, err = m.needs_retrain(model, meta, seg_df, tolerance=args.tolerance)
        print(f"{seg:<12} recent MAPE {err:.2f}% vs trained {meta['xgb_mape']:.2f}% -> {'retrain' if drift else 'ok'}")
        if drift or args.force:
            stale.append(seg)
    if stale:
        _print(_train(stale, store))
    else:
        print("all models within tolerance; nothing retrained")


def cmd_plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .features import FEATURES, build_features

    store = load_store(STORE)
    fig, axes = plt.subplots(len(SEGMENTS), 1, figsize=(11, 8), sharex=True)
    for ax, seg in zip(axes, SEGMENTS):
        model, meta = m.load(seg, MODELS)
        feats = build_features(store[store["segment"] == seg])
        window = feats[feats["timestamp"] > feats["timestamp"].max() - pd.Timedelta(days=args.days)]
        ax.plot(window["timestamp"], window["load_mw"], color="#1f2937", lw=1.4, label="actual")
        ax.plot(window["timestamp"], model.predict(window[FEATURES]), color="#2563eb", lw=1.2, label="XGBoost day-ahead")
        ax.plot(window["timestamp"], window["lag_168"], color="#9ca3af", lw=1, ls="--", label="same hour last week")
        ax.set_title(f"{seg}  (holdout MAPE {meta['xgb_mape']:.1f}% vs naive {meta['baseline_mape']:.1f}%)", loc="left", fontsize=10)
        ax.set_ylabel("MW")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="energy_forecast")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--start", default="2023-01-01")
    g.add_argument("--end", default="2025-12-31 23:00")
    g.add_argument("--shift-segment", choices=SEGMENTS, help="apply a step change to one segment")
    g.add_argument("--shift-pct", type=float, default=15.0)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_generate)
    sub.add_parser("ingest").set_defaults(func=cmd_ingest)
    sub.add_parser("train").set_defaults(func=cmd_train)
    r = sub.add_parser("retrain")
    r.add_argument("--tolerance", type=float, default=1.25)
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_retrain)
    pl = sub.add_parser("plot")
    pl.add_argument("--days", type=int, default=14)
    pl.add_argument("--out", default="docs/forecast.png")
    pl.set_defaults(func=cmd_plot)
    args = p.parse_args(argv)
    args.func(args)

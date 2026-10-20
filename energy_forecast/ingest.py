"""Automated ingestion: validate new meter drops and merge them into the store.

New files land in a directory (for example an SFTP drop or a cloud bucket
sync). `ingest_directory` validates each file, removes duplicates and bad
rows, appends clean data to the store, and moves processed files to an
archive so nothing is ingested twice.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import SEGMENTS

REQUIRED_COLUMNS = ("timestamp", "segment", "temperature_f", "load_mw")


@dataclass
class IngestReport:
    files: list[str] = field(default_factory=list)
    rows_added: int = 0
    rows_rejected: int = 0
    duplicates_dropped: int = 0

    def __str__(self) -> str:
        return (
            f"{len(self.files)} file(s) | +{self.rows_added} rows | "
            f"{self.rows_rejected} rejected | {self.duplicates_dropped} duplicates"
        )


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Coerce types and drop rows that would corrupt training. Returns (clean, n_rejected)."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["temperature_f"] = pd.to_numeric(df["temperature_f"], errors="coerce")
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    ok = (
        df["timestamp"].notna()
        & df["segment"].isin(SEGMENTS)
        & df["temperature_f"].between(-40, 130)
        & (df["load_mw"] > 0)
    )
    return df[ok], int((~ok).sum())


def load_store(store: Path) -> pd.DataFrame:
    if not store.exists():
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    return pd.read_csv(store, parse_dates=["timestamp"])


def ingest_directory(landing: Path, store: Path, archive: Path | None = None) -> IngestReport:
    landing, store = Path(landing), Path(store)
    archive = Path(archive) if archive else landing / "_processed"
    report = IngestReport()
    files = sorted(landing.glob("*.csv"))
    if not files:
        return report

    existing = load_store(store)
    batches = []
    for path in files:
        clean, rejected = validate(pd.read_csv(path))
        batches.append(clean)
        report.files.append(path.name)
        report.rows_rejected += rejected

    incoming = pd.concat(batches, ignore_index=True)
    combined = pd.concat([existing, incoming], ignore_index=True)
    before = len(combined)
    # Later drops win: a corrected reading replaces the original one.
    combined = combined.drop_duplicates(["timestamp", "segment"], keep="last")
    report.duplicates_dropped = before - len(combined)
    report.rows_added = len(combined) - len(existing)

    store.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["segment", "timestamp"]).to_csv(store, index=False)

    archive.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.move(str(path), archive / path.name)
    return report

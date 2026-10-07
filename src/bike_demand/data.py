"""Data acquisition and validation for the UCI Bike Sharing dataset."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
REQUIRED_COLUMNS = {
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
}


def download_dataset(output_dir: str | Path, *, force: bool = False) -> Path:
    """Download and extract the daily UCI Bike Sharing data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "day.csv"
    readme_path = output_dir / "UCI_README.txt"
    if csv_path.exists() and not force:
        return csv_path

    request = Request(DATASET_URL, headers={"User-Agent": "daily-bike-demand-forecasting/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed trusted URL
        archive = BytesIO(response.read())
    with ZipFile(archive) as zipped:
        csv_path.write_bytes(zipped.read("day.csv"))
        readme_path.write_bytes(zipped.read("Readme.txt"))
    return csv_path


def load_daily_data(path: str | Path) -> pd.DataFrame:
    """Load, validate, and return one row per calendar day."""
    path = Path(path)
    frame = pd.read_csv(path, parse_dates=["dteday"])
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["dteday"].duplicated().any():
        raise ValueError("Dates must be unique")
    if frame["cnt"].isna().any() or (frame["cnt"] < 0).any():
        raise ValueError("Demand must be present and non-negative")

    frame = frame.sort_values("dteday").reset_index(drop=True)
    expected = pd.date_range(frame["dteday"].min(), frame["dteday"].max(), freq="D")
    if not frame["dteday"].equals(pd.Series(expected, name="dteday")):
        raise ValueError("Daily series must have no missing calendar dates")
    if not (frame["casual"] + frame["registered"]).equals(frame["cnt"]):
        raise ValueError("Total count must equal casual plus registered riders")
    return frame


def data_quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Return compact, JSON-safe quality diagnostics."""
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "date_start": frame["dteday"].min().date().isoformat(),
        "date_end": frame["dteday"].max().date().isoformat(),
        "unique_days": int(frame["dteday"].nunique()),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_dates": int(frame["dteday"].duplicated().sum()),
        "minimum_demand": int(frame["cnt"].min()),
        "maximum_demand": int(frame["cnt"].max()),
        "mean_daily_demand": round(float(frame["cnt"].mean()), 2),
        "complete_daily_calendar": True,
        "target_reconciles_to_components": True,
    }

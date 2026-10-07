from pathlib import Path

import pandas as pd
import pytest

from bike_demand.data import load_daily_data


DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "day.csv"


def test_dataset_has_complete_daily_calendar():
    frame = load_daily_data(DATA_PATH)
    expected = pd.date_range(frame["dteday"].min(), frame["dteday"].max(), freq="D")
    assert len(frame) == 731
    assert frame["dteday"].tolist() == expected.tolist()


def test_total_demand_reconciles():
    frame = load_daily_data(DATA_PATH)
    assert (frame["casual"] + frame["registered"] == frame["cnt"]).all()


def test_duplicate_dates_are_rejected(tmp_path):
    frame = pd.read_csv(DATA_PATH)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "bad.csv"
    duplicated.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_daily_data(path)


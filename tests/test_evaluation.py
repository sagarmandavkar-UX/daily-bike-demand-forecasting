import numpy as np
import pytest

from bike_demand.evaluation import forecast_metrics


def test_perfect_forecast_has_zero_error():
    actual = np.array([10.0, 20.0, 30.0])
    metrics = forecast_metrics(actual, actual, np.arange(1, 30, dtype=float))
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["mase"] == 0


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        forecast_metrics(np.array([1, 2]), np.array([1]), np.arange(20))


def test_wape_uses_total_actual_demand():
    metrics = forecast_metrics(np.array([100.0, 200.0]), np.array([90.0, 220.0]), np.arange(1, 30, dtype=float))
    assert metrics["wape_pct"] == pytest.approx(10.0)


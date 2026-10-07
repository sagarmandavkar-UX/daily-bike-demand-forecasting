import numpy as np
import pytest

from bike_demand.forecasting import empirical_prediction_interval, forecast_model, rolling_origin_splits


def test_seasonal_naive_repeats_last_week():
    training = np.arange(1, 15, dtype=float)
    forecast = forecast_model("seasonal_naive", training, 10).values
    assert forecast.tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]


def test_rolling_splits_do_not_touch_holdout():
    splits = rolling_origin_splits(731, horizon=30, folds=3, final_holdout=30)
    assert len(splits) == 3
    assert max(splits[-1][1]) == 700
    assert min(splits[0][1]) == 611
    assert all(max(train) < min(validation) for train, validation in splits)


def test_prediction_interval_contains_point_for_centered_residuals():
    point = np.array([10.0, 20.0])
    residuals = np.linspace(-4, 4, 100)
    lower, upper = empirical_prediction_interval(point, residuals)
    assert np.all(lower < point)
    assert np.all(point < upper)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        forecast_model("lstm", np.arange(100), 30)


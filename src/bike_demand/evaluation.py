"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np


def forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    training: np.ndarray,
    *,
    seasonal_period: int = 7,
) -> dict[str, float]:
    """Calculate scale-dependent and scale-free forecast metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    training = np.asarray(training, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    if len(training) <= seasonal_period:
        raise ValueError("Training history is too short for seasonal MASE")

    error = predicted - actual
    absolute_error = np.abs(error)
    nonzero = actual != 0
    mape = np.mean(absolute_error[nonzero] / actual[nonzero]) * 100 if nonzero.any() else np.nan
    denominator = np.sum(np.abs(actual))
    mase_scale = np.mean(np.abs(training[seasonal_period:] - training[:-seasonal_period]))
    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(mape),
        "wape_pct": float(np.sum(absolute_error) / denominator * 100) if denominator else np.nan,
        "mase": float(np.mean(absolute_error) / mase_scale) if mase_scale else np.nan,
        "bias": float(np.mean(error)),
    }


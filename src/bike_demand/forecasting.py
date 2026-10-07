"""Forecast models and time-aware validation helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODEL_NAMES = ("seasonal_naive", "drift", "holt_winters", "sarima")


@dataclass(frozen=True)
class ForecastOutput:
    values: np.ndarray
    residuals: np.ndarray


def rolling_origin_splits(
    n_observations: int,
    *,
    horizon: int = 30,
    folds: int = 3,
    final_holdout: int = 30,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window folds that end before the final holdout."""
    first_validation_start = n_observations - final_holdout - folds * horizon
    if first_validation_start <= 2 * horizon:
        raise ValueError("Not enough history for the requested folds")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(folds):
        validation_start = first_validation_start + fold * horizon
        train = np.arange(validation_start)
        validation = np.arange(validation_start, validation_start + horizon)
        splits.append((train, validation))
    return splits


def _seasonal_naive(training: np.ndarray, horizon: int, seasonal_period: int) -> ForecastOutput:
    if len(training) < seasonal_period:
        raise ValueError("Seasonal naive requires at least one full season")
    pattern = training[-seasonal_period:]
    forecast = np.resize(pattern, horizon).astype(float)
    residuals = training[seasonal_period:] - training[:-seasonal_period]
    return ForecastOutput(forecast, residuals)


def _drift(training: np.ndarray, horizon: int) -> ForecastOutput:
    slope = (training[-1] - training[0]) / max(len(training) - 1, 1)
    steps = np.arange(1, horizon + 1)
    forecast = training[-1] + slope * steps
    fitted = training[0] + slope * np.arange(len(training))
    return ForecastOutput(forecast.astype(float), training - fitted)


def forecast_model(
    model_name: str,
    training: np.ndarray | pd.Series,
    horizon: int,
    *,
    seasonal_period: int = 7,
) -> ForecastOutput:
    """Fit one model and forecast the requested horizon."""
    values = np.asarray(training, dtype=float)
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    if model_name == "seasonal_naive":
        return _seasonal_naive(values, horizon, seasonal_period)
    if model_name == "drift":
        return _drift(values, horizon)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_name == "holt_winters":
            fitted = ExponentialSmoothing(
                values,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=seasonal_period,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
            forecast = fitted.forecast(horizon)
            residuals = values - np.asarray(fitted.fittedvalues)
        else:
            fitted = SARIMAX(
                values,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, seasonal_period),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=120)
            forecast = fitted.get_forecast(horizon).predicted_mean
            residuals = np.asarray(fitted.resid)
    residuals = residuals[np.isfinite(residuals)]
    warmup = min(max(seasonal_period * 2, 10), len(residuals) // 4)
    return ForecastOutput(np.maximum(np.asarray(forecast, dtype=float), 0), residuals[warmup:])


def empirical_prediction_interval(
    point_forecast: np.ndarray,
    residuals: np.ndarray,
    *,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an empirical residual interval around a point forecast."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 20:
        raise ValueError("At least 20 residuals are required for an interval")
    alpha = 1 - level
    lower_error, upper_error = np.quantile(residuals, [alpha / 2, 1 - alpha / 2])
    point_forecast = np.asarray(point_forecast, dtype=float)
    return (
        np.maximum(point_forecast + lower_error, 0),
        np.maximum(point_forecast + upper_error, 0),
    )


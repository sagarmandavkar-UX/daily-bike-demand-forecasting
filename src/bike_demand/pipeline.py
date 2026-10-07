"""End-to-end demand forecasting analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .data import data_quality_summary, load_daily_data
from .evaluation import forecast_metrics
from .forecasting import MODEL_NAMES, empirical_prediction_interval, forecast_model, rolling_origin_splits
from .visualization import create_figures

MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive (7-day)",
    "drift": "Drift",
    "holt_winters": "Holt-Winters",
    "sarima": "SARIMA",
}


def _rounded_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 3) for key, value in metrics.items()}


def run_pipeline(
    data_path: str | Path,
    report_dir: str | Path,
    *,
    horizon: int = 30,
    folds: int = 3,
) -> dict[str, object]:
    """Run validation, model selection, holdout testing, and future forecasting."""
    frame = load_daily_data(data_path)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = report_dir / "figures"
    values = frame["cnt"].to_numpy(dtype=float)
    dates = frame["dteday"]
    splits = rolling_origin_splits(len(frame), horizon=horizon, folds=folds, final_holdout=horizon)

    cv_rows: list[dict[str, object]] = []
    for fold_number, (train_idx, validation_idx) in enumerate(splits, start=1):
        training = values[train_idx]
        actual = values[validation_idx]
        for model_name in MODEL_NAMES:
            prediction = forecast_model(model_name, training, horizon).values
            metrics = forecast_metrics(actual, prediction, training)
            cv_rows.append(
                {
                    "fold": fold_number,
                    "train_end": dates.iloc[train_idx[-1]].date().isoformat(),
                    "validation_start": dates.iloc[validation_idx[0]].date().isoformat(),
                    "validation_end": dates.iloc[validation_idx[-1]].date().isoformat(),
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    **_rounded_metrics(metrics),
                }
            )
    cv = pd.DataFrame(cv_rows)
    comparison = (
        cv.groupby(["model", "model_label"], as_index=False)
        .agg(cv_mae=("mae", "mean"), cv_rmse=("rmse", "mean"), cv_mape_pct=("mape_pct", "mean"), cv_mase=("mase", "mean"))
        .sort_values("cv_mae")
        .reset_index(drop=True)
    )
    numeric = ["cv_mae", "cv_rmse", "cv_mape_pct", "cv_mase"]
    comparison[numeric] = comparison[numeric].round(3)
    selected_model = str(comparison.iloc[0]["model"])

    holdout_training = values[:-horizon]
    holdout_actual = values[-horizon:]
    selected_output = forecast_model(selected_model, holdout_training, horizon)
    naive_output = forecast_model("seasonal_naive", holdout_training, horizon)
    lower, upper = empirical_prediction_interval(selected_output.values, selected_output.residuals)
    holdout_metrics = forecast_metrics(holdout_actual, selected_output.values, holdout_training)
    baseline_metrics = forecast_metrics(holdout_actual, naive_output.values, holdout_training)
    improvement = (baseline_metrics["mae"] - holdout_metrics["mae"]) / baseline_metrics["mae"] * 100

    holdout = pd.DataFrame(
        {
            "date": dates.iloc[-horizon:].to_numpy(),
            "actual": holdout_actual,
            "forecast": selected_output.values,
            "lower_95": lower,
            "upper_95": upper,
            "seasonal_naive": naive_output.values,
        }
    )
    holdout["absolute_error"] = np.abs(holdout["forecast"] - holdout["actual"])

    full_output = forecast_model(selected_model, values, horizon)
    future_lower, future_upper = empirical_prediction_interval(full_output.values, full_output.residuals)
    future_dates = pd.date_range(dates.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame(
        {
            "date": future_dates,
            "forecast": full_output.values,
            "lower_95": future_lower,
            "upper_95": future_upper,
        }
    )

    weekly_profile = (
        frame.assign(weekday_name=frame["dteday"].dt.day_name())
        .groupby("weekday_name", as_index=False)["cnt"]
        .agg(mean_demand="mean", median_demand="median")
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_profile["weekday_name"] = pd.Categorical(weekly_profile["weekday_name"], weekday_order, ordered=True)
    weekly_profile = weekly_profile.sort_values("weekday_name")
    weekly_profile[["mean_demand", "median_demand"]] = weekly_profile[["mean_demand", "median_demand"]].round(2)

    adf_stat, adf_pvalue, *_ = adfuller(values, result_object=False)
    gate_passed = selected_model != "seasonal_naive" and improvement >= 5 and holdout_metrics["mase"] < 1
    summary: dict[str, object] = {
        "question": "Can historical daily bike rentals predict demand for the next 30 days?",
        "dataset": data_quality_summary(frame),
        "forecast_horizon_days": horizon,
        "validation": {
            "method": f"{folds}-fold expanding-window validation plus an untouched final {horizon}-day holdout",
            "selected_model": selected_model,
            "selected_model_label": MODEL_LABELS[selected_model],
            "cv_mae": round(float(comparison.iloc[0]["cv_mae"]), 3),
        },
        "holdout_selected_model": _rounded_metrics(holdout_metrics),
        "holdout_seasonal_naive": _rounded_metrics(baseline_metrics),
        "mae_improvement_vs_seasonal_naive_pct": round(float(improvement), 2),
        "deployment_gate": "GO" if gate_passed else "NO-GO",
        "deployment_rule": "GO only if a non-baseline model improves holdout MAE by at least 5% and holdout MASE is below 1.0",
        "stationarity": {
            "adf_statistic": round(float(adf_stat), 3),
            "adf_pvalue": round(float(adf_pvalue), 4),
            "interpretation": "Reject a unit root at 5%" if adf_pvalue < 0.05 else "Do not reject a unit root at 5%",
        },
        "future_forecast": {
            "start": future_dates.min().date().isoformat(),
            "end": future_dates.max().date().isoformat(),
            "mean_daily_demand": round(float(future["forecast"].mean()), 1),
            "minimum_daily_demand": round(float(future["forecast"].min()), 1),
            "maximum_daily_demand": round(float(future["forecast"].max()), 1),
            "total_demand": round(float(future["forecast"].sum())),
            "interval_note": "95% empirical residual interval; it reflects historical forecast errors, not every source of future uncertainty",
        },
        "decision_use": "Use the forecast as a planning baseline for fleet rebalancing, maintenance staffing, and capacity; preserve an uncertainty buffer and refresh as new demand arrives.",
        "scope_note": "This is a univariate forecast using rental history only. Weather, policy, pricing, and event scenarios are not known for the future horizon.",
    }

    cv.to_csv(report_dir / "cross_validation_metrics.csv", index=False)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    holdout.to_csv(report_dir / "holdout_forecast.csv", index=False, float_format="%.3f")
    future.to_csv(report_dir / "future_30_day_forecast.csv", index=False, float_format="%.3f")
    weekly_profile.to_csv(report_dir / "weekly_profile.csv", index=False)
    (report_dir / "data_quality_report.json").write_text(json.dumps(data_quality_summary(frame), indent=2) + "\n")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    create_figures(frame, comparison, holdout, future, figures_dir)
    return summary

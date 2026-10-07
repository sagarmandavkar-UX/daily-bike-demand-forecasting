"""Project figures with a consistent accessible visual style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

BLUE = "#2463A9"
GOLD = "#C58B24"
CHARCOAL = "#263238"
LIGHT_BLUE = "#B9D5F2"
GRID = "#D9DEE3"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "text.color": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    frame: pd.DataFrame,
    comparison: pd.DataFrame,
    holdout: pd.DataFrame,
    future: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Create the project figure set."""
    _style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(frame["dteday"], frame["cnt"], color=LIGHT_BLUE, linewidth=1, label="Daily rentals")
    ax.plot(frame["dteday"], frame["cnt"].rolling(30).mean(), color=BLUE, linewidth=2.3, label="30-day average")
    ax.set(title="Daily Bike Demand and 30-Day Trend", xlabel="Date", ylabel="Bike rentals")
    ax.legend(frameon=False, ncol=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _save(fig, output_dir / "demand_history.png")

    weekday = frame.assign(weekday_name=frame["dteday"].dt.day_name()).groupby("weekday_name")["cnt"].mean()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = weekday.reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(weekday.index, weekday.values, color=BLUE)
    ax.set(title="Average Demand by Day of Week", xlabel="Day", ylabel="Average daily rentals")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="x", visible=False)
    _save(fig, output_dir / "weekly_seasonality.png")

    stl = STL(frame.set_index("dteday")["cnt"], period=7, robust=True).fit()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(stl.observed.index, stl.observed, color=LIGHT_BLUE, linewidth=1)
    axes[0].set_ylabel("Observed")
    axes[1].plot(stl.trend.index, stl.trend, color=BLUE, linewidth=1.8)
    axes[1].set_ylabel("Trend")
    axes[2].plot(stl.seasonal.index, stl.seasonal, color=GOLD, linewidth=1)
    axes[2].set_ylabel("Weekly\nseasonality")
    axes[2].set_xlabel("Date")
    fig.suptitle("STL Decomposition of Daily Bike Demand", fontweight="bold", y=1.01)
    _save(fig, output_dir / "stl_decomposition.png")

    ranked = comparison.sort_values("cv_mae")
    colors = [BLUE if i == 0 else LIGHT_BLUE for i in range(len(ranked))]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(ranked["model_label"], ranked["cv_mae"], color=colors)
    ax.invert_yaxis()
    ax.set(title="Rolling Validation MAE by Model", xlabel="Mean absolute error (rentals)", ylabel="")
    ax.grid(axis="y", visible=False)
    ax.bar_label(bars, fmt="%.0f", padding=5)
    ax.set_xlim(0, ranked["cv_mae"].max() * 1.12)
    _save(fig, output_dir / "model_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(holdout["date"], holdout["actual"], color=CHARCOAL, linewidth=2.2, label="Actual")
    ax.plot(holdout["date"], holdout["forecast"], color=BLUE, linewidth=2, label="Selected model")
    ax.plot(holdout["date"], holdout["seasonal_naive"], color=GOLD, linestyle="--", linewidth=1.8, label="Seasonal naive")
    ax.fill_between(holdout["date"], holdout["lower_95"], holdout["upper_95"], color=LIGHT_BLUE, alpha=0.45, label="95% empirical interval")
    ax.set(title="Final 30-Day Holdout Forecast", xlabel="Date", ylabel="Bike rentals")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    _save(fig, output_dir / "holdout_forecast.png")

    recent = frame.tail(90)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(recent["dteday"], recent["cnt"], color=CHARCOAL, linewidth=1.8, label="Observed")
    ax.plot(future["date"], future["forecast"], color=BLUE, linewidth=2.2, label="30-day forecast")
    ax.fill_between(future["date"], future["lower_95"], future["upper_95"], color=LIGHT_BLUE, alpha=0.5, label="95% empirical interval")
    ax.axvline(frame["dteday"].max(), color=GOLD, linestyle="--", linewidth=1.4)
    ax.set(title="Next 30 Days of Forecast Demand", xlabel="Date", ylabel="Bike rentals")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    _save(fig, output_dir / "future_forecast.png")

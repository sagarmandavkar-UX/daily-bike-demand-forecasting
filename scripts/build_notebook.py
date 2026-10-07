"""Build the reproducible project notebook with nbformat."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "reports" / "summary.json").read_text())
    selected = summary["validation"]["selected_model_label"]
    holdout = summary["holdout_selected_model"]
    future = summary["future_forecast"]
    notebook = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Predicting the Next 30 Days of Bike Demand

An end-to-end time-series analysis of daily Capital Bikeshare rentals using expanding-window validation, baseline comparison, an untouched final holdout, and empirical forecast intervals."""))
    cells.append(nbf.v4.new_markdown_cell(f"""## tl;dr

- **Selected model:** {selected}, with rolling-validation MAE of **{summary['validation']['cv_mae']:,.3f} rentals**.
- **Final 30-day holdout:** MAE **{holdout['mae']:,.3f}**, WAPE **{holdout['wape_pct']:.3f}%**, MASE **{holdout['mase']:.3f}**.
- **Baseline improvement:** **{summary['mae_improvement_vs_seasonal_naive_pct']:.2f}%**, below the 5% operating threshold.
- **Decision:** **{summary['deployment_gate']}** for unattended deployment.
- **Next 30 days:** approximately **{future['total_demand']:,.0f} rentals** in total, averaging **{future['mean_daily_demand']:,.1f} per day**.

The forecast is suitable as a planning baseline with a buffer and frequent refreshes—not as an automatic capacity commitment."""))
    cells.append(nbf.v4.new_markdown_cell("""## Context & Methods

### Question

Can historical daily demand predict the next 30 days of bike rentals?

### Source

[UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), daily Capital Bikeshare rental counts for 2011–2012, licensed CC BY 4.0.

### Key assumptions

- Forecasts use only information available before each prediction window.
- Future weather is excluded because it is unknown at forecast creation time.
- MAE is the primary planning metric. MAPE is diagnostic only because near-zero holiday demand can distort it.
- The final 30 days remain untouched until after model selection.
- The 95% bands are empirical residual intervals, not full scenario ranges."""))
    cells.append(nbf.v4.new_markdown_cell("## Data"))
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from statsmodels.tsa.seasonal import STL

from bike_demand.data import load_daily_data
from bike_demand.pipeline import run_pipeline

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'day.csv'
REPORT_DIR = PROJECT_ROOT / 'reports'

BLUE = '#2463A9'
GOLD = '#C58B24'
CHARCOAL = '#263238'
LIGHT_BLUE = '#B9D5F2'
plt.rcParams.update({'figure.figsize': (12, 4.8), 'axes.grid': True, 'grid.alpha': .25, 'axes.titleweight': 'bold'})

summary = run_pipeline(DATA_PATH, REPORT_DIR)
data = load_daily_data(DATA_PATH)
comparison = pd.read_csv(REPORT_DIR / 'model_comparison.csv')
holdout = pd.read_csv(REPORT_DIR / 'holdout_forecast.csv', parse_dates=['date'])
future = pd.read_csv(REPORT_DIR / 'future_30_day_forecast.csv', parse_dates=['date'])
weekly = pd.read_csv(REPORT_DIR / 'weekly_profile.csv')
summary['dataset']"""))
    cells.append(nbf.v4.new_markdown_cell("The source contains 731 consecutive daily rows with no missing dates, no missing values, and no duplicate dates. Casual plus registered riders reconciles exactly to total demand."))
    cells.append(nbf.v4.new_code_cell("""data[['dteday', 'cnt', 'casual', 'registered']].head()"""))
    cells.append(nbf.v4.new_markdown_cell("## Results\n\n### 1. Inspect the trend"))
    cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots()
ax.plot(data['dteday'], data['cnt'], color=LIGHT_BLUE, linewidth=1, label='Daily rentals')
ax.plot(data['dteday'], data['cnt'].rolling(30).mean(), color=BLUE, linewidth=2.3, label='30-day average')
ax.set(title='Daily Bike Demand and 30-Day Trend', xlabel='Date', ylabel='Bike rentals')
ax.legend(frameon=False, ncol=2)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.show()"""))
    cells.append(nbf.v4.new_markdown_cell("The smoothed series rises materially from 2011 into 2012, so assuming a constant level would be inappropriate. The ADF p-value is 0.3427; the level series does not provide evidence of stationarity at the 5% threshold."))
    cells.append(nbf.v4.new_markdown_cell("### 2. Separate weekly seasonality from trend"))
    cells.append(nbf.v4.new_code_cell("""stl = STL(data.set_index('dteday')['cnt'], period=7, robust=True).fit()
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
axes[0].plot(stl.observed, color=LIGHT_BLUE, linewidth=1); axes[0].set_ylabel('Observed')
axes[1].plot(stl.trend, color=BLUE, linewidth=1.8); axes[1].set_ylabel('Trend')
axes[2].plot(stl.seasonal, color=GOLD, linewidth=1); axes[2].set_ylabel('Weekly seasonality'); axes[2].set_xlabel('Date')
fig.suptitle('STL Decomposition of Daily Bike Demand', fontweight='bold', y=1.01)
plt.tight_layout(); plt.show()"""))
    cells.append(nbf.v4.new_markdown_cell("Weekly structure is visible, but the residual variation remains large relative to the weekly component. This motivates benchmarking every statistical model against a simple seven-day repeat."))
    cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(weekly['weekday_name'], weekly['mean_demand'], color=BLUE)
ax.set(title='Average Demand by Day of Week', xlabel='Day', ylabel='Average daily rentals')
ax.tick_params(axis='x', rotation=25)
ax.grid(axis='x', visible=False)
plt.show()
weekly"""))
    cells.append(nbf.v4.new_markdown_cell("### 3. Compare forecasting approaches with rolling validation"))
    cells.append(nbf.v4.new_code_cell("""ranked = comparison.sort_values('cv_mae')
fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.barh(ranked['model_label'], ranked['cv_mae'], color=[BLUE] + [LIGHT_BLUE] * (len(ranked) - 1))
ax.invert_yaxis(); ax.grid(axis='y', visible=False)
ax.set(title='Rolling Validation MAE by Model', xlabel='Mean absolute error (rentals)', ylabel='')
ax.bar_label(bars, fmt='%.0f', padding=5)
ax.set_xlim(0, ranked['cv_mae'].max() * 1.12)
plt.show()
comparison"""))
    cells.append(nbf.v4.new_markdown_cell(f"{selected} has the lowest average rolling-validation MAE. Model selection uses only the three validation windows; the final 30-day holdout remains untouched."))
    cells.append(nbf.v4.new_markdown_cell("### 4. Test the selected model on the untouched final month"))
    cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots()
ax.plot(holdout['date'], holdout['actual'], color=CHARCOAL, linewidth=2.2, label='Actual')
ax.plot(holdout['date'], holdout['forecast'], color=BLUE, linewidth=2, label='Selected model')
ax.plot(holdout['date'], holdout['seasonal_naive'], color=GOLD, linestyle='--', linewidth=1.8, label='Seasonal naive')
ax.fill_between(holdout['date'], holdout['lower_95'], holdout['upper_95'], color=LIGHT_BLUE, alpha=.45, label='95% empirical interval')
ax.set(title='Final 30-Day Holdout Forecast', xlabel='Date', ylabel='Bike rentals')
ax.legend(frameon=False, ncol=4, loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.show()"""))
    cells.append(nbf.v4.new_code_cell("""pd.DataFrame({
    'Selected model': summary['holdout_selected_model'],
    'Seasonal naive': summary['holdout_seasonal_naive'],
}).T[['mae', 'rmse', 'wape_pct', 'mase', 'bias']]"""))
    cells.append(nbf.v4.new_markdown_cell(f"The selected model improves MAE by only **{summary['mae_improvement_vs_seasonal_naive_pct']:.2f}%** and overforecasts by **{holdout['bias']:,.1f} rentals per day on average**. It therefore receives a **{summary['deployment_gate']}** decision."))
    cells.append(nbf.v4.new_markdown_cell("### 5. Refit and forecast the next 30 days"))
    cells.append(nbf.v4.new_code_cell("""recent = data.tail(90)
fig, ax = plt.subplots()
ax.plot(recent['dteday'], recent['cnt'], color=CHARCOAL, linewidth=1.8, label='Observed')
ax.plot(future['date'], future['forecast'], color=BLUE, linewidth=2.2, label='30-day forecast')
ax.fill_between(future['date'], future['lower_95'], future['upper_95'], color=LIGHT_BLUE, alpha=.5, label='95% empirical interval')
ax.axvline(data['dteday'].max(), color=GOLD, linestyle='--', linewidth=1.4)
ax.set(title='Next 30 Days of Forecast Demand', xlabel='Date', ylabel='Bike rentals')
ax.legend(frameon=False, ncol=3, loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.show()"""))
    cells.append(nbf.v4.new_code_cell("""future.assign(
    forecast=future['forecast'].round(0).astype(int),
    lower_95=future['lower_95'].round(0).astype(int),
    upper_95=future['upper_95'].round(0).astype(int),
).head(10)"""))
    cells.append(nbf.v4.new_markdown_cell(f"""## Takeaways

1. **Historical demand contains useful trend and weekly structure**, and Holt-Winters clearly outperforms drift and the weekly baseline during rolling validation.
2. **The final month is harder than the validation average.** Holdout MAE rises to {holdout['mae']:,.0f} rentals and MASE remains above 1.0.
3. **Do not automate capacity from this model yet.** Use the {future['total_demand']:,.0f}-rental 30-day forecast as a buffered planning baseline.
4. **Next improvement:** add more years of history plus forecast-time weather, holidays, events, pricing, and system-capacity inputs, then repeat the untouched-holdout gate.

The empirical intervals summarize historical residuals; they are not guarantees and do not capture every structural shock."""))

    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    output = root / "notebooks" / "bike_demand_forecasting.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(f"Notebook written: {output}")


if __name__ == "__main__":
    main()

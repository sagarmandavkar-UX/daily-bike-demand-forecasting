# Forecast Card: 30-Day Daily Bike Demand

## Intended use

This model is a planning baseline for daily Capital Bikeshare demand. It can support exploratory fleet-rebalancing, maintenance, and staffing scenarios when paired with an operational uncertainty buffer and frequent refreshes.

It is **not approved for unattended capacity decisions**. The final deployment gate is **NO-GO**.

## Data

- 731 complete daily observations from 2011-01-01 through 2012-12-31
- Target: total daily bike rentals (`cnt`)
- Source: [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- Dataset license: CC BY 4.0
- Forecast inputs: historical demand only

Weather and seasonal columns are retained in the source data for interpretation, but they are not used as future features because their values are unavailable at forecast creation time.

## Validation design

- Three expanding-window validation folds, each with a 30-day horizon
- Model selected only from validation performance
- Final 30 days held out and untouched until model selection was complete
- Seasonal-naive 7-day forecast used as the operating benchmark

This design prevents random train/test mixing and look-ahead leakage.

## Models compared

1. Seasonal naive, repeating the last seven days
2. Drift forecast
3. Holt-Winters with damped trend and weekly additive seasonality
4. SARIMA(1,1,1)(1,0,1,7)

Holt-Winters produced the lowest average validation MAE: **990.878 rentals**.

## Final holdout performance

| Metric | Holt-Winters | Seasonal naive |
|---|---:|---:|
| MAE | 1,480.163 | 1,527.567 |
| RMSE | 1,932.709 | 2,054.837 |
| WAPE | 37.466% | 38.665% |
| MASE | 1.617 | 1.669 |
| Bias | +772.601 | +687.900 |

The candidate improved MAE by only **3.10%** and MASE remained above 1.0. It therefore failed the project gate: a non-baseline model must improve holdout MAE by at least 5% and achieve MASE below 1.0.

## Forecast output

The refit Holt-Winters model forecasts 2013-01-01 through 2013-01-30:

- Mean daily demand: **2,234.4 rentals**
- Forecast total: **67,032 rentals**
- Daily point-forecast range: **1,780.6 to 2,455.7 rentals**

The supplied 95% intervals are empirical residual intervals. They summarize historical errors around the point forecast, but they do not cover every future shock.

## Limitations

- Only two years of history are available, limiting annual-seasonality learning.
- The series has strong trend and seasonal changes; the ADF test does not reject a unit root at 5% (p = 0.3427).
- Holidays, weather, pricing, system expansion, and events can shift demand but are not scenario inputs.
- December demand includes abrupt low-volume days that produce unstable percentage errors.
- Performance is specific to this historical Capital Bikeshare period and should not be generalized to another city without retraining and validation.

## Recommended next step

Collect more history and join forecast-time weather, holiday, service-capacity, and event inputs. Re-run the expanding-window evaluation and require the same untouched-holdout gate before operational use.


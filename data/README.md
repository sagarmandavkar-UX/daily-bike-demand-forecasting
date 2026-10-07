# Data source and license

This project uses `day.csv` from the **UCI Bike Sharing dataset**:

- Dataset page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- DOI: https://doi.org/10.24432/C5W894
- Creator: Hadi Fanaee-T
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Coverage used here: daily Capital Bikeshare rental counts from 2011-01-01 through 2012-12-31

The original dataset contains daily and hourly rental counts with seasonal and weather information. This project uses the daily file and forecasts total rentals (`cnt`) from historical demand only. The future-weather fields are intentionally excluded because they are not known at forecasting time.

`data/raw/day.csv` and `data/raw/UCI_README.txt` are redistributed under the dataset's CC BY 4.0 license. The repository's MIT license applies to the project code, not to the dataset.

To download the source again:

```bash
python scripts/download_data.py
```


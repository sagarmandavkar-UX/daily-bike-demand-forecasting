"""Run the end-to-end forecasting pipeline."""

import json
from pathlib import Path

from bike_demand.pipeline import run_pipeline


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    summary = run_pipeline(
        project_root / "data" / "raw" / "day.csv",
        project_root / "reports",
    )
    print(json.dumps(summary, indent=2))


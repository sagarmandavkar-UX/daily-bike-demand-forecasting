"""Download the attributed UCI Bike Sharing dataset."""

from pathlib import Path

from bike_demand.data import download_dataset


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    path = download_dataset(project_root / "data" / "raw")
    print(f"Dataset ready: {path}")


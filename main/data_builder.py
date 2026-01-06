# data_builder.py
#
# Convert raw minute-level BTC/USD data into a clean daily OHLCV dataset.
# This script should be run once before training. It produces:
#   data/bitcoin_daily_2024_2025.csv

import os
import pandas as pd


def build_daily_dataset(
    raw_path="btcusd_1-min_data.csv",
    output_path="data/bitcoin_daily_2024_2025.csv",
    start_date="2024-01-01",
    end_date="2025-10-30",
):
    # Make sure output directory exists
    os.makedirs("data", exist_ok=True)

    print("Loading raw minute-level data...")
    df_raw = pd.read_csv(raw_path)

    # Convert UNIX timestamp to datetime
    df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], unit="s", utc=True)
    df_raw = df_raw.set_index("Timestamp").sort_index()

    print("Resampling to daily OHLCV...")
    daily = df_raw.resample("D").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()

    # Add a timezone-naive Date column
    daily["Date"] = daily.index.tz_convert("UTC").tz_localize(None)
    daily = daily.reset_index(drop=True)

    print("Filtering date range...")
    mask = (daily["Date"] >= pd.to_datetime(start_date)) & (
        daily["Date"] <= pd.to_datetime(end_date)
    )
    daily = daily.loc[mask].reset_index(drop=True)

    print(f"Saving cleaned daily dataset to {output_path}...")
    daily.to_csv(output_path, index=False)

    print("Done. Daily dataset created.")
    return daily


if __name__ == "__main__":
    build_daily_dataset()

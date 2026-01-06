# data_loader.py
#
# Loads the daily Bitcoin dataset either from:
# - local disk (preferred), or
# - GitHub URL fallback (for TA reproducibility)

import os
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler


def download_if_missing(local_path, github_raw_url):
    """Download the dataset from GitHub if it does not exist locally."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        return  # nothing to do

    print("Local dataset missing. Downloading from GitHub...")
    r = requests.get(github_raw_url)
    if r.status_code != 200:
        raise ValueError(
            f"Failed to download dataset. HTTP {r.status_code}\nURL: {github_raw_url}"
        )

    with open(local_path, "wb") as f:
        f.write(r.content)

    print("Download complete.")


def load_data(
    seq_len=30,
    test_ratio=0.2,
    local_path="data/bitcoin_daily_2024_2025.csv",
    github_url="https://raw.githubusercontent.com/dhanishp/CS4375_LSTM_Project/main/data/bitcoin_daily_2024_2025.csv"
):
    """
    Loads the daily BTC dataset, normalizes it, forms sequences, and splits.
    """

    # Attempt to download if missing
    download_if_missing(local_path, github_url)

    # Load daily OHLCV
    df = pd.read_csv(local_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Normalize OHLCV
    scaler = MinMaxScaler()
    scaled_vals = scaler.fit_transform(df[['Open','High','Low','Close','Volume']])
    df_scaled = pd.DataFrame(scaled_vals, columns=['Open','High','Low','Close','Volume'])

    # Build sequences
    X, y = [], []
    values = df_scaled.values

    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len][3])  # index 3 = normalized Close

    X = np.array(X)
    y = np.array(y)

    # Chronological split
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = df["Date"].iloc[seq_len + split_idx:].values

    return (X_train, y_train), (X_test, y_test), scaler, dates_test

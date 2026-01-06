# utils.py
#
# Utility functions for metrics, plotting, and experiment logging.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def mse(y_true, y_pred):
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return np.sqrt(mse(y_true, y_pred))


def plot_loss(loss_history, save_path="results/loss_curve.png"):
    """Plot training loss across epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_predictions(dates, actual, predicted, save_path="results/predictions.png"):
    """Plot actual vs predicted closing prices."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual Close", linewidth=2)
    plt.plot(dates, predicted, label="Predicted Close", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Normalized Closing Price")
    plt.title("Actual vs Predicted Closing Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_last_n_days(dates, actual, predicted, n=30, save_path="results/zoom_last30.png"):
    """Plot zoomed-in view for the last n days."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(dates[-n:], actual[-n:], label="Actual Close", linewidth=2)
    plt.plot(dates[-n:], predicted[-n:], label="Predicted Close", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Normalized Closing Price")
    plt.title(f"Last {n} Days – Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_experiment_log(params, metrics, file_path="results/experiment_log.csv"):
    """
    Append model hyperparameters and results to experiment log CSV.
    
    params: dict (e.g., {"seq_len": 30, "lr": 0.001, "epochs": 50})
    metrics: dict (e.g., {"train_mse": ..., "test_mse": ..., "test_rmse": ...})
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    row = {**params, **metrics}

    df_row = pd.DataFrame([row])

    if os.path.exists(file_path):
        df_row.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(file_path, mode="w", header=True, index=False)

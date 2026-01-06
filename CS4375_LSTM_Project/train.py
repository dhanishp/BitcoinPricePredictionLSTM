# train.py
#
# Main training script for the manual LSTM model.
# Loads the cleaned daily dataset, trains the network,
# evaluates performance, and saves plots and experiment logs.

import numpy as np
import os

from data_loader import load_data
from lstm_numpy import LSTMNetwork
from utils import (
    mse,
    rmse,
    plot_loss,
    plot_predictions,
    plot_last_n_days,
    save_experiment_log,
)


def train_model(
    seq_len=30,
    hidden_size=32,
    learning_rate=0.001,
    epochs=50,
):
    # Load prepared sequences (auto-download if missing)
    (X_train, y_train), (X_test, y_test), scaler, dates_test = load_data(
        seq_len=seq_len,
        github_url="https://raw.githubusercontent.com/dhanishp/CS4375_LSTM_Project/main/data/bitcoin_daily_2024_2025.csv"
    )

    input_size = X_train.shape[2]

    # Initialize LSTM network
    model = LSTMNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        seq_len=seq_len,
    )

    loss_history = []

    print("Training started...")
    for epoch in range(epochs):
        epoch_losses = []

        for i in range(len(X_train)):
            X_seq = X_train[i]
            y_true = y_train[i]
            loss = model.train_step(X_seq, y_true)
            epoch_losses.append(loss)

        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    print("Training complete.")

    # Save loss curve
    os.makedirs("results", exist_ok=True)
    plot_loss(loss_history, save_path="results/loss_curve.png")

    # Evaluate on test set
    preds = []
    for i in range(len(X_test)):
        y_pred, _, _ = model.forward(X_test[i])
        preds.append(y_pred)

    preds = np.array(preds)

    # Save model parameters for trading simulation
    model_params = {
        "W_y": model.W_y,
        "b_y": model.b_y,
        "cell": {
            "W_i": model.cell.W_i,
            "W_f": model.cell.W_f,
            "W_o": model.cell.W_o,
            "W_c": model.cell.W_c,
            "U_i": model.cell.U_i,
            "U_f": model.cell.U_f,
            "U_o": model.cell.U_o,
            "U_c": model.cell.U_c,
            "b_i": model.cell.b_i,
            "b_f": model.cell.b_f,
            "b_o": model.cell.b_o,
            "b_c": model.cell.b_c,
        }
    }
    np.save("results/trained_lstm_params.npy", model_params)

    # Compute metrics
    test_mse_val = mse(y_test, preds)
    test_rmse_val = rmse(y_test, preds)

    print("\nEvaluation on Test Set:")
    print(f"Test MSE:  {test_mse_val:.6f}")
    print(f"Test RMSE: {test_rmse_val:.6f}")

    # Save prediction plots
    plot_predictions(
        dates_test, y_test, preds, save_path="results/predictions.png"
    )
    plot_last_n_days(
        dates_test, y_test, preds, n=30, save_path="results/zoom_last30.png"
    )

    # Save experiment log entry
    params = {
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }
    metrics = {
        "train_last_mse": loss_history[-1],
        "test_mse": test_mse_val,
        "test_rmse": test_rmse_val,
    }

    save_experiment_log(params, metrics, file_path="results/experiment_log.csv")

    print("\nResults saved in /results/")
    print("Experiment log updated.")

    return loss_history, preds, y_test, dates_test


if __name__ == "__main__":
    train_model(
        seq_len=30,
        hidden_size=32,
        learning_rate=0.001,
        epochs=50,
    )

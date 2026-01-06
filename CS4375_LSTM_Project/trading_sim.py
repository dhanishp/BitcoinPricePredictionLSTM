# trading_sim.py
#
# Uses the trained LSTM model to perform a simple trading simulation:
# - If predicted next-day price > today's close → hold/long
# - Else → move to cash
# Saves equity curve plot and trading log.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data


def load_trained_model(path="results/trained_lstm_params.npy"):
    print("Loading trained LSTM model parameters...")
    return np.load(path, allow_pickle=True).item()


def run_trading_sim(preds, y_test, dates_test, initial_cash=10000):
    cash = initial_cash
    position = 0  # BTC amount held

    log = []

    for i in range(len(preds) - 1):
        today_price = y_test[i]
        pred_tomorrow = preds[i]

        # Simple rule:
        # If predicted ↑ → buy and hold
        # If predicted ↓ → go to cash
        if pred_tomorrow > today_price:
            # Buy BTC with full cash if not already holding
            if position == 0:
                position = cash / today_price
                cash = 0
        else:
            # Sell BTC if currently holding
            if position > 0:
                cash = position * today_price
                position = 0

        equity = cash + position * today_price

        log.append([dates_test[i], today_price, pred_tomorrow, cash, position, equity])

    log_df = pd.DataFrame(log, columns=["Date","Actual","Predicted","Cash","Position","Equity"])
    return log_df


def main():
    print("\nLoading daily dataset...")
    (X_train, y_train), (X_test, y_test), scaler, dates_test = load_data(
        seq_len=30,
        github_url="https://raw.githubusercontent.com/dhanishp/CS4375_LSTM_Project/main/data/bitcoin_daily_2024_2025.csv"
    )

    print("Loading trained LSTM model parameters...")
    params = load_trained_model()

    print("Computing predictions...")
    preds = []
    for i in range(len(X_test)):
        h = np.zeros(params["cell"]["W_i"].shape[0])
        c = np.zeros(params["cell"]["W_i"].shape[0])

        # Manual forward pass
        for t in range(30):
            x_t = X_test[i][t]

            W_i = params["cell"]["W_i"]
            W_f = params["cell"]["W_f"]
            W_o = params["cell"]["W_o"]
            W_c = params["cell"]["W_c"]

            U_i = params["cell"]["U_i"]
            U_f = params["cell"]["U_f"]
            U_o = params["cell"]["U_o"]
            U_c = params["cell"]["U_c"]

            b_i = params["cell"]["b_i"]
            b_f = params["cell"]["b_f"]
            b_o = params["cell"]["b_o"]
            b_c = params["cell"]["b_c"]

            # Gates
            i_t = 1/(1+np.exp(-(W_i @ x_t + U_i @ h + b_i)))
            f_t = 1/(1+np.exp(-(W_f @ x_t + U_f @ h + b_f)))
            o_t = 1/(1+np.exp(-(W_o @ x_t + U_o @ h + b_o)))
            c_hat = np.tanh(W_c @ x_t + U_c @ h + b_c)

            c = f_t * c + i_t * c_hat
            h = o_t * np.tanh(c)

        # Output layer
        y_pred = params["W_y"] @ h + params["b_y"]
        preds.append(float(y_pred))

    preds = np.array(preds)

    print("Running trading simulation...")
    history = run_trading_sim(preds, y_test, dates_test)

    final_equity = history["Equity"].iloc[-1]
    bh_equity = 10000 * (y_test[-1] / y_test[0])

    print("\n--- SIMULATION COMPLETE ---")
    print(f"Final Strategy Value: ${final_equity:,.2f}")
    print(f"Buy & Hold Value:    ${bh_equity:,.2f}")

    # Save log + plot
    history.to_csv("results/trading_simulation.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(history["Date"], history["Equity"], label="LSTM Strategy")
    plt.plot(history["Date"], 10000 * (history["Actual"] / history["Actual"].iloc[0]),
             label="Buy & Hold")
    plt.legend()
    plt.title("Trading Simulation Equity Curve")
    plt.tight_plot = True
    plt.savefig("results/trading_equity_curve.png")

    print("Results saved in /results/")


if __name__ == "__main__":
    main()

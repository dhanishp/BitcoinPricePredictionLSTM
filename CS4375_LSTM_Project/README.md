CS4375 LSTM Bitcoin Prediction Project
Dhanish Parimalakumar
CS 4375.001 Anurag Nagar Fall 2025

HOW TO RUN THIS PROJECT

This project trains a manually implemented LSTM model (NumPy only) to predict next-day Bitcoin closing prices.
All theoretical explanation, design details, and analysis are included in the written report.
This README.txt only contains instructions for running the code.

1. Install Dependencies

Install the required Python packages:

    pip install numpy pandas scikit-learn matplotlib requests

These are the only libraries needed to run the project.

2. Dataset Download (Automatic)

You do NOT need to download the dataset manually.

If the dataset is missing, the scripts will automatically download:

    https://raw.githubusercontent.com/dhanishp/CS4375_LSTM_Project/main/data/bitcoin_daily_2024_2025.csv

It will be saved into:

    data/bitcoin_daily_2024_2025.csv

This guarantees full reproducibility on any machine.

3. Train the Model

From the project root, run:

    python3 train.py

This will:
- Automatically download the dataset if missing
- Create 30-day OHLCV input sequences
- Train the LSTM (NumPy-only implementation)
- Save results into the "results" folder:

      results/loss_curve.png
      results/predictions.png
      results/zoom_last30.png
      results/experiment_log.csv
      results/trained_lstm_params.npy

4. Run the Trading Simulation

After training, run:

    python3 trading_sim.py

This will:
- Load the trained LSTM weights
- Compute predictions on the test set
- Execute a simple trading strategy
- Save outputs into the "results" folder:

      results/trading_simulation.csv
      results/trading_equity_curve.png

5. Project File Overview

train.py                - Train the LSTM model
trading_sim.py          - Run trading simulation
lstm_numpy.py           - Manual LSTM and BPTT implementation
data_loader.py          - Loads and normalizes dataset (auto-download enabled)
data_builder.py         - Optional: build daily CSV from raw minute data
utils.py                - Plotting, loss metrics, experiment logging

data/                   - Dataset directory (auto-created)
results/                - Output directory for plots, logs, and model weights

6. Notes for the Grader

- Only two commands are required to run the entire project:

      python3 train.py
      python3 trading_sim.py

- The dataset is automatically downloaded from GitHub.
- No deep learning libraries (TensorFlow, PyTorch) are used.
- All results are saved automatically in the "results" folder.
- All conceptual explanation is provided in the written report.
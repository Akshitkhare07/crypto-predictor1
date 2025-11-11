# -*- coding: utf-8 -*-
"""
cryptopricepredictor.py

Train an LSTM to predict crypto close price (default: BTC-USD).
Saves:
 - Keras SavedModel -> ./models/{TICKER_SAFE}_lstm_model
 - Scaler (joblib) -> ./models/{TICKER_SAFE}_scaler.gz
 - Test arrays (preds, y_test, dates) -> ./models/{TICKER_SAFE}_test_data.npz

Usage:
    python cryptopricepredictor.py --ticker BTC-USD --start 2018-01-01 --epochs 20
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# -----------------------
# Defaults / Config
# -----------------------
DEFAULT_TICKER = "BTC-USD"
DEFAULT_START = "2018-01-01"
DEFAULT_END = None  # yfinance will default to today
LOOKBACK = 60
TRAIN_FRAC = 0.8
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def download_close_series(ticker: str, start: str, end: str = None) -> pd.Series:
    """
    Download Close price as a pandas Series and ensure it is a single-level Series
    named 'Price'. Handles single- or multi-ticker responses from yfinance.
    """
    print(f"Downloading {ticker} from {start} to {end or 'today'} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker} ({start} - {end})")
    # yfinance returns a DataFrame with columns (Open, High, Low, Close, Adj Close, Volume)
    # If ticker is a single ticker, df['Close'] is a Series; if multiple, it can be a DataFrame.
    close = df["Close"]
    # If close is DataFrame (happens when multiple tickers passed), select column by ticker
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            s = close[ticker].ffill().rename("Price")
        else:
            # fallback: take first column
            s = close.iloc[:, 0].ffill().rename("Price")
    else:
        s = close.ffill().rename("Price")
    return s


def create_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(lookback: int):
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_and_save(ticker: str, start: str, end: str, lookback: int, epochs: int, batch_size: int):
    series = download_close_series(ticker, start, end)
    df = series.to_frame()  # column 'Price'
    values = df.values.astype("float32")

    # scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # train/test split with lookback overlap for sequences
    train_len = int(np.ceil(len(scaled) * TRAIN_FRAC))
    train_data = scaled[:train_len, :]
    test_data = scaled[train_len - lookback:, :]  # include overlap

    X_train, y_train = create_sequences(train_data, lookback)
    X_test, y_test = create_sequences(test_data, lookback)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

    model = build_model(lookback)
    early = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early], verbose=1)

    # predictions and inverse transform
    preds = model.predict(X_test)
    preds_unscaled = scaler.inverse_transform(preds)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # compute RMSE
    rmse = np.sqrt(np.mean((preds_unscaled - y_test_unscaled) ** 2))
    print(f"Validation RMSE: ${rmse:,.2f}")

    safe_name = ticker.replace("-", "_")
    model_path = MODEL_DIR / f"{safe_name}_lstm_model"
    scaler_path = MODEL_DIR / f"{safe_name}_scaler.gz"
    arrays_path = MODEL_DIR / f"{safe_name}_test_data.npz"

    print(f"Saving model to {model_path} ...")
    model.save(model_path)

    print(f"Saving scaler to {scaler_path} ...")
    joblib.dump(scaler, scaler_path)

    # Calculate validation dates that correspond to y_test_unscaled/preds_unscaled
    # The first validation index in the original df is at position train_len
    # y_test length equals len(df) - train_len
    validation_dates = df.index[train_len:].astype(str).to_numpy()
    np.savez_compressed(arrays_path, preds=preds_unscaled, y_test=y_test_unscaled, dates=validation_dates)

    print("Saved artifacts:")
    print(f" - model: {model_path}")
    print(f" - scaler: {scaler_path}")
    print(f" - arrays: {arrays_path}")

    # Optional: plot training loss and predictions vs actual
    try:
        # Loss plot
        plt.figure(figsize=(8, 4))
        plt.plot(history.history.get("loss", []), label="train_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"{safe_name}_training_loss.png")
        plt.close()

        # Predictions vs Actual
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(validation_dates), y_test_unscaled, label="Actual")
        plt.plot(pd.to_datetime(validation_dates), preds_unscaled, label="Predicted")
        plt.title(f"{ticker} — Validation: Actual vs Predicted")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"{safe_name}_val_vs_pred.png")
        plt.close()
    except Exception as e:
        print(f"Could not save plots locally: {e}")

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "arrays_path": str(arrays_path),
        "rmse": float(rmse)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default=DEFAULT_TICKER)
    p.add_argument("--start", type=str, default=DEFAULT_START)
    p.add_argument("--end", type=str, default=DEFAULT_END)
    p.add_argument("--lookback", type=int, default=LOOKBACK)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    result = train_and_save(args.ticker, args.start, args.end, args.lookback, args.epochs, args.batch_size)
    print("Training finished. Artifacts saved:", result)


if __name__ == "__main__":
    main()
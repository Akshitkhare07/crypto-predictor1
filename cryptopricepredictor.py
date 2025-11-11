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
    python cryptopricepredictor.py --ticker ETH-USD --start 2015-01-01 --epochs 30
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
import warnings
warnings.filterwarnings("ignore")

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
    
    Args:
        ticker: Crypto ticker (e.g., 'BTC-USD', 'ETH-USD')
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD) or None for today
    
    Returns:
        pd.Series of Close prices
    """
    print(f"📥 Downloading {ticker} from {start} to {end or 'today'}...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise RuntimeError(f"No data downloaded for {ticker} ({start} - {end})")
        
        close = df["Close"]
        
        # Handle multi-ticker responses
        if isinstance(close, pd.DataFrame):
            if ticker in close.columns:
                s = close[ticker].ffill().astype("float32").rename("Price")
            else:
                s = close.iloc[:, 0].ffill().astype("float32").rename("Price")
        else:
            s = close.ffill().astype("float32").rename("Price")
        
        print(f"✅ Downloaded {len(s)} records")
        return s
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        raise


def create_sequences(data: np.ndarray, lookback: int):
    """
    Create sequences for LSTM training/testing.
    
    Args:
        data: Scaled price data (N, 1)
        lookback: Number of past timesteps to use as input
    
    Returns:
        (X, y) where X is (N-lookback, lookback) and y is (N-lookback,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(lookback: int):
    """
    Build a 2-layer LSTM model for price prediction.
    
    Args:
        lookback: Number of timesteps in input
    
    Returns:
        Compiled Keras Sequential model
    """
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
    """
    Train LSTM model and save artifacts.
    
    Args:
        ticker: Crypto ticker
        start: Start date
        end: End date
        lookback: Lookback window
        epochs: Training epochs
        batch_size: Batch size
    
    Returns:
        dict with paths to saved artifacts and metrics
    """
    # Download data
    series = download_close_series(ticker, start, end)
    df = series.to_frame()  # column 'Price'
    values = df.values.astype("float32")
    
    # Scale
    print("🔧 Normalizing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # Train/test split with lookback overlap
    train_len = int(np.ceil(len(scaled) * TRAIN_FRAC))
    train_data = scaled[:train_len, :]
    test_data = scaled[train_len - lookback:, :]  # include overlap
    
    # Create sequences
    print("📊 Creating sequences...")
    X_train, y_train = create_sequences(train_data, lookback)
    X_test, y_test = create_sequences(test_data, lookback)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    # Build and train model
    print("🧠 Building LSTM model...")
    model = build_model(lookback)
    print(model.summary())
    
    print(f"⏳ Training for {epochs} epochs...")
    early = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=1
    )
    
    # Predictions and inverse transform
    print("🎯 Making predictions...")
    preds = model.predict(X_test, verbose=0)
    preds_unscaled = scaler.inverse_transform(preds)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Metrics
    rmse = np.sqrt(np.mean((preds_unscaled - y_test_unscaled) ** 2))
    mape = np.mean(np.abs((y_test_unscaled - preds_unscaled) / y_test_unscaled)) * 100
    print(f"✅ Validation RMSE: ${rmse:,.2f}")
    print(f"✅ Validation MAPE: {mape:.2f}%")
    
    # Save artifacts
    safe_name = ticker.replace("-", "_")
    model_path = MODEL_DIR / f"{safe_name}_lstm_model"
    scaler_path = MODEL_DIR / f"{safe_name}_scaler.gz"
    arrays_path = MODEL_DIR / f"{safe_name}_test_data.npz"
    
    print(f"💾 Saving model to {model_path}...")
    model.save(model_path)
    
    print(f"💾 Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    # Calculate validation dates
    validation_dates = df.index[train_len:].astype(str).to_numpy()
    np.savez_compressed(arrays_path, preds=preds_unscaled, y_test=y_test_unscaled, dates=validation_dates)
    
    print("\n✅ Saved artifacts:")
    print(f"   📁 Model: {model_path}")
    print(f"   📁 Scaler: {scaler_path}")
    print(f"   📁 Arrays: {arrays_path}")
    
    # Plots
    try:
        print("📈 Saving plots...")
        
        # Training loss
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history.get("loss", []), linewidth=2, label="Training Loss")
        ax.set_title("Training Loss Over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"{safe_name}_training_loss.png", dpi=100)
        plt.close()
        
        # Predictions vs actual
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pd.to_datetime(validation_dates), y_test_unscaled, label="Actual", linewidth=2, marker="o", markersize=3)
        ax.plot(pd.to_datetime(validation_dates), preds_unscaled, label="Predicted", linewidth=2, marker="s", markersize=3, alpha=0.7)
        ax.set_title(f"{ticker} — Validation: Actual vs Predicted")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"{safe_name}_val_vs_pred.png", dpi=100)
        plt.close()
        print("✅ Plots saved!")
    except Exception as e:
        print(f"⚠️ Could not save plots: {e}")
    
    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "arrays_path": str(arrays_path),
        "rmse": float(rmse),
        "mape": float(mape)
    }


def main():
    p = argparse.ArgumentParser(
        description="Train an LSTM model to predict cryptocurrency prices."
    )
    p.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Crypto ticker (default: BTC-USD)")
    p.add_argument("--start", type=str, default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=DEFAULT_END, help="End date (YYYY-MM-DD) or None for today")
    p.add_argument("--lookback", type=int, default=LOOKBACK, help="Lookback window (default: 60)")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    args = p.parse_args()
    
    print("=" * 60)
    print(f"🪙 Crypto Price Predictor — LSTM Training")
    print("=" * 60)
    print(f"Ticker: {args.ticker}")
    print(f"Period: {args.start} to {args.end or 'today'}")
    print(f"Lookback: {args.lookback} | Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print("=" * 60 + "\n")
    
    result = train_and_save(
        args.ticker,
        args.start,
        args.end,
        args.lookback,
        args.epochs,
        args.batch_size
    )
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"RMSE: ${result['rmse']:,.2f}")
    print(f"MAPE: {result['mape']:.2f}%")
    print("\nTo run the Streamlit app:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
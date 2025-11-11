# Streamlit UI for the trained LSTM crypto price model.
# Run: streamlit run app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Crypto Price Predictor")

MODEL_DIR = Path("models")
DEFAULT_TICKER = "BTC-USD"
LOOKBACK = 60

@st.cache_data
def load_series(ticker: str, period: str = "2y"):
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    s = df["Close"].ffill().rename("Price")
    return s

@st.cache_resource
def load_artifacts(ticker: str):
    safe = ticker.replace("-", "_")
    model_path = MODEL_DIR / f"{safe}_lstm_model"
    scaler_path = MODEL_DIR / f"{safe}_scaler.gz"
    arrays_path = MODEL_DIR / f"{safe}_test_data.npz"
    if not model_path.exists() or not scaler_path.exists():
        return None, None, None
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    arrays = None
    if arrays_path.exists():
        arrays = np.load(arrays_path, allow_pickle=True)
    return model, scaler, arrays

st.title("Crypto Price Predictor (LSTM)")
st.write("Load a trained model (created by running cryptopricepredictor.py) and inspect predictions.")

ticker = st.sidebar.text_input("Ticker", value=DEFAULT_TICKER)
period = st.sidebar.selectbox("Historical period to show", ["1y", "2y", "5y", "max"], index=1)
load_button = st.sidebar.button("Load model & predict next day")

with st.spinner("Loading historical prices..."):
    series = load_series(ticker, period=period)

if series.empty:
    st.error(f"No data available for {ticker}. Check ticker symbol or network access.")
else:
    st.subheader(f"{ticker} - Close price ({period})")
    st.line_chart(series)

model, scaler, arrays = load_artifacts(ticker)

if model is None:
    st.warning("No trained model artifacts found in ./models. Run the training script first (cryptopricepredictor.py).")
    st.info("Example: python cryptopricepredictor.py --ticker BTC-USD --start 2018-01-01 --epochs 10")
else:
    st.success("Model & scaler loaded.")
    if arrays is not None:
        preds = arrays["preds"]
        y_test = arrays["y_test"]
        dates = arrays["dates"]
        rmse = np.sqrt(np.mean((preds - y_test) ** 2))
        st.metric("Validation RMSE (USD)", f"${rmse:,.2f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pd.to_datetime(dates), y_test, label="Actual")
        ax.plot(pd.to_datetime(dates), preds, label="Predicted")
        ax.set_title(f"{ticker} — Validation actual vs predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No validation arrays saved (only model + scaler present).")

    if load_button:
        # predict the next day using the latest 'LOOKBACK' points from full history
        s_all = load_series(ticker, period="max")
        if len(s_all) < LOOKBACK:
            st.error(f"Not enough history ({len(s_all)} points) for lookback={LOOKBACK}.")
        else:
            recent = s_all.values[-LOOKBACK:].reshape(-1, 1)
            scaled_recent = scaler.transform(recent)
            X = scaled_recent.reshape(1, LOOKBACK, 1)
            pred_scaled = model.predict(X)
            pred_unscaled = scaler.inverse_transform(pred_scaled)
            st.metric("Predicted next-day close (USD)", f"${float(pred_unscaled[0,0]):,.2f}")
            st.write("Prediction based on the latest available closing prices.")

st.sidebar.markdown("---")
st.sidebar.write("Expected artifacts in ./models:")
st.sidebar.write("- {TICKER}_lstm_model/ (SavedModel)")
st.sidebar.write("- {TICKER}_scaler.gz (joblib)")
st.sidebar.write("- {TICKER}_test_data.npz (validation preds & actuals)")
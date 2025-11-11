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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Crypto Price Predictor")

MODEL_DIR = Path("models")
DEFAULT_TICKER = "BTC-USD"
LOOKBACK = 60


@st.cache_resource
def load_series_cached(ticker: str, period: str = "2y"):
    """Load historical series (cached at resource level to avoid pandas conflicts)."""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        s = close.ffill().astype("float32")
        return s
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_artifacts(ticker: str):
    """Load trained model, scaler, and test data arrays."""
    safe = ticker.replace("-", "_")
    model_path = MODEL_DIR / f"{safe}_lstm_model"
    scaler_path = MODEL_DIR / f"{safe}_scaler.gz"
    arrays_path = MODEL_DIR / f"{safe}_test_data.npz"
    
    if not model_path.exists() or not scaler_path.exists():
        return None, None, None
    
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        arrays = None
        if arrays_path.exists():
            arrays = np.load(arrays_path, allow_pickle=True)
        return model, scaler, arrays
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None


def predict_next_n_days(model, scaler, last_prices: np.ndarray, n_days: int):
    """
    Predict the next n days of prices.
    
    Args:
        model: Trained LSTM model
        scaler: MinMaxScaler fitted on training data
        last_prices: Array of the most recent LOOKBACK prices (unscaled)
        n_days: Number of days to predict into the future
    
    Returns:
        Array of predictions (unscaled)
    """
    predictions = []
    current_window = last_prices.copy().reshape(-1, 1)
    current_window_scaled = scaler.transform(current_window)
    
    for _ in range(n_days):
        X = current_window_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred_unscaled = scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append(pred_unscaled)
        
        # Add prediction to window (scaled) for next iteration
        current_window_scaled = np.vstack([
            current_window_scaled,
            pred_scaled
        ])
    
    return np.array(predictions)


def days_to_predict(period_type: str, period_value: int) -> int:
    """Convert period (days/months/years) to number of days to predict."""
    if period_type == "Days":
        return period_value
    elif period_type == "Months":
        return period_value * 30  # Approximate
    elif period_type == "Years":
        return period_value * 365  # Approximate
    return 1


st.title("🪙 Crypto Price Predictor (LSTM)")
st.write("Train and predict cryptocurrency prices using deep learning (LSTM neural networks).")

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Ticker", value=DEFAULT_TICKER, help="e.g., BTC-USD, ETH-USD")
period = st.sidebar.selectbox(
    "Historical period to show",
    ["1y", "2y", "5y", "max"],
    index=1,
    help="Period of historical data to display"
)

st.sidebar.markdown("---")
st.sidebar.header("🔮 Future Price Prediction")

col1, col2 = st.sidebar.columns(2)
with col1:
    period_type = st.selectbox("Period type", ["Days", "Months", "Years"])
with col2:
    if period_type == "Days":
        period_value = st.number_input("Days ahead", min_value=1, max_value=365, value=7)
    elif period_type == "Months":
        period_value = st.number_input("Months ahead", min_value=1, max_value=60, value=6)
    else:  # Years
        period_value = st.number_input("Years ahead", min_value=1, max_value=10, value=1)

predict_future_btn = st.sidebar.button("📊 Predict Future Price", use_container_width=True)

st.sidebar.markdown("---")
load_button = st.sidebar.button("📈 Load model & Analyze", use_container_width=True)

# Main content
st.subheader(f"{ticker} - Close Price ({period})")

# Load historical data
with st.spinner("Loading historical prices..."):
    series = load_series_cached(ticker, period=period)

if series is None or series.empty:
    st.error(f"❌ No data available for {ticker}. Check ticker symbol or network access.")
    st.stop()

# Display price chart
fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
ax_hist.plot(series.index, series.values, linewidth=2, color="#1f77b4")
ax_hist.fill_between(series.index, series.values, alpha=0.3, color="#1f77b4")
ax_hist.set_title(f"{ticker} Historical Close Price", fontsize=14, fontweight="bold")
ax_hist.set_xlabel("Date")
ax_hist.set_ylabel("Price (USD)")
ax_hist.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig_hist)

# Load model and artifacts
model, scaler, arrays = load_artifacts(ticker)

if model is None:
    st.warning(
        "⚠️ No trained model artifacts found in ./models/\n\n"
        "Run the training script first to create the model."
    )
    with st.expander("How to train the model"):
        st.code(
            f"python cryptopricepredictor.py --ticker {ticker} --start 2018-01-01 --epochs 20",
            language="bash"
        )
    st.stop()

st.success("✅ Model & scaler loaded successfully.")

# Display validation metrics if available
if arrays is not None:
    preds = arrays["preds"]
    y_test = arrays["y_test"]
    dates = arrays["dates"]
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Validation RMSE (USD)", f"${rmse:,.2f}")
    with col2:
        st.metric("Validation MAPE (%)", f"{mape:.2f}%")
    
    st.subheader("Validation: Actual vs Predicted")
    fig_val, ax_val = plt.subplots(figsize=(12, 6))
    ax_val.plot(pd.to_datetime(dates), y_test, label="Actual", linewidth=2, marker="o", markersize=3)
    ax_val.plot(pd.to_datetime(dates), preds, label="Predicted", linewidth=2, marker="s", markersize=3, alpha=0.7)
    ax_val.set_title(f"{ticker} — Validation: Actual vs Predicted", fontsize=14, fontweight="bold")
    ax_val.set_xlabel("Date")
    ax_val.set_ylabel("Price (USD)")
    ax_val.legend()
    ax_val.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_val)
else:
    st.info("ℹ️ No validation arrays saved (only model + scaler present).")

# Predict next day on button click
if load_button:
    st.subheader("📌 Next Day Prediction")
    s_all = load_series_cached(ticker, period="max")
    if s_all is None or len(s_all) < LOOKBACK:
        st.error(f"❌ Not enough history ({len(s_all) if s_all is not None else 0} points) for lookback={LOOKBACK}.")
    else:
        recent = s_all.values[-LOOKBACK:].reshape(-1, 1)
        scaled_recent = scaler.transform(recent)
        X = scaled_recent.reshape(1, LOOKBACK, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred_unscaled = float(scaler.inverse_transform(pred_scaled)[0, 0])
        current_price = float(s_all.values[-1])
        change = pred_unscaled - current_price
        change_pct = (change / current_price) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")
        with col2:
            st.metric("Predicted Tomorrow", f"${pred_unscaled:,.2f}")
        with col3:
            st.metric(
                "Expected Change",
                f"${change:,.2f}",
                delta=f"{change_pct:+.2f}%",
                delta_color="inverse"
            )

# Predict future price
if predict_future_btn:
    st.subheader(f"🔮 Future Price Prediction ({period_type})")
    s_all = load_series_cached(ticker, period="max")
    
    if s_all is None or len(s_all) < LOOKBACK:
        st.error(f"❌ Not enough history for prediction.")
    else:
        n_days = days_to_predict(period_type, period_value)
        
        with st.spinner(f"Predicting {n_days} days into the future..."):
            recent_prices = s_all.values[-LOOKBACK:]
            forecast = predict_next_n_days(model, scaler, recent_prices, n_days)
        
        # Create forecast dates
        last_date = s_all.index[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${s_all.values[-1]:,.2f}")
        with col2:
            st.metric(f"Predicted in {period_value} {period_type.lower()}", f"${forecast[-1]:,.2f}")
        with col3:
            change = forecast[-1] - s_all.values[-1]
            st.metric("Expected Change", f"${change:,.2f}", delta=f"{(change/s_all.values[-1])*100:+.2f}%", delta_color="inverse")
        with col4:
            min_pred = np.min(forecast)
            st.metric("Min Predicted", f"${min_pred:,.2f}")
        
        # Plot forecast
        fig_forecast, ax_forecast = plt.subplots(figsize=(14, 7))
        
        # Historical data
        lookback_days = min(365, len(s_all))
        hist_dates = s_all.index[-lookback_days:]
        hist_prices = s_all.values[-lookback_days:]
        ax_forecast.plot(hist_dates, hist_prices, label="Historical", linewidth=2.5, color="#1f77b4", marker="o", markersize=2)
        
        # Forecast
        ax_forecast.plot(forecast_dates, forecast, label=f"Forecast ({period_type})", linewidth=2.5, color="#ff7f0e", marker="s", markersize=4, linestyle="--")
        ax_forecast.axvline(x=last_date, color="red", linestyle=":", linewidth=2, alpha=0.7, label="Forecast Start")
        
        # Formatting
        ax_forecast.set_title(f"{ticker} — {period_value} {period_type.lower()} Price Forecast", fontsize=14, fontweight="bold")
        ax_forecast.set_xlabel("Date")
        ax_forecast.set_ylabel("Price (USD)")
        ax_forecast.legend(loc="best")
        ax_forecast.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_forecast)
        
        # Forecast table
        st.subheader("Forecast Details")
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted Price": forecast,
            "Change from Current": forecast - s_all.values[-1],
            "% Change from Current": ((forecast - s_all.values[-1]) / s_all.values[-1] * 100)
        })
        forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(forecast_df, use_container_width=True)
        
        # Disclaimer
        st.warning(
            "⚠️ **Disclaimer**: These predictions are based on historical patterns and should not be used as investment advice. "
            "Cryptocurrency prices are highly volatile and influenced by many external factors. "
            "Predictions become less accurate over longer periods."
        )

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.write("**📁 Expected artifacts in ./models:**")
st.sidebar.write(f"- {DEFAULT_TICKER.replace('-', '_')}_lstm_model/")
st.sidebar.write(f"- {DEFAULT_TICKER.replace('-', '_')}_scaler.gz")
st.sidebar.write(f"- {DEFAULT_TICKER.replace('-', '_')}_test_data.npz")
st.sidebar.markdown("---")
st.sidebar.write("**ℹ️ About this app:**")
st.sidebar.write(
    "This app uses a 2-layer LSTM neural network trained on historical crypto price data. "
    "It can predict the next day's price or forecast up to 10 years ahead. "
    "For best results, train the model on at least 2 years of historical data."
)
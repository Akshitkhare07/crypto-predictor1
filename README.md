# Crypto Price Predictor

A production-ready LSTM-based cryptocurrency price prediction system with a Streamlit UI. Predict next-day prices or forecast up to 10 years ahead.

## Features ✨

- **LSTM Neural Network**: 2-layer LSTM with dropout for robust price prediction
- **Multi-timeframe Forecasting**: Predict 1-7 days, 1-60 months, or 1-10 years ahead
- **Validation Metrics**: RMSE and MAPE error metrics for model evaluation
- **Interactive Streamlit UI**: Real-time price charts and forecast visualization
- **Artifact Persistence**: Saves trained models, scalers, and test data for reproducibility
- **Error Handling**: Robust data loading and caching to handle various edge cases

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train on BTC-USD (default)
python cryptopricepredictor.py --ticker BTC-USD --start 2018-01-01 --epochs 20

# Train on ETH-USD with more epochs
python cryptopricepredictor.py --ticker ETH-USD --start 2015-01-01 --epochs 30

# Custom parameters
python cryptopricepredictor.py --ticker BTC-USD --start 2020-01-01 --end 2025-11-01 --epochs 50 --batch_size 16
```

The training script saves:
- Trained LSTM model (SavedModel format)
- MinMaxScaler for inverse transformation
- Test predictions & validation data

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

### In the Streamlit App

1. **Select ticker** (BTC-USD, ETH-USD, etc.)
2. **Choose historical period** (1y, 2y, 5y, max)
3. **Click "Load model & Analyze"** to see validation metrics
4. **Choose forecast period** (Days, Months, or Years)
5. **Click "Predict Future Price"** to generate forecast

The app displays:
- Historical price chart
- Validation actual vs predicted
- RMSE & MAPE metrics
- Interactive forecast with detailed table

## File Structure

```
├── cryptopricepredictor.py    # Training script
├── app.py                      # Streamlit UI
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container config
├── README.md                   # This file
└── models/                     # (created by training)
    ├── BTC_USD_lstm_model/
    ├── BTC_USD_scaler.gz
    ├── BTC_USD_test_data.npz
    ├── training_loss.png
    └── val_vs_pred.png
```

## Docker

### Build

```bash
docker build -t crypto-lstm-app .
```

### Run

```bash
docker run -p 8501:8501 crypto-lstm-app
```

## Model Details

- **Architecture**: 2-layer LSTM (128 → 64 units) + Dropout(0.2)
- **Input**: 60 days of historical prices (normalized 0-1)
- **Output**: Next-day closing price
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Early Stopping**: Patience=5 epochs

## Troubleshooting

### "No trained model artifacts found"
→ Run training script first: `python cryptopricepredictor.py`

### "Not enough history"
→ The ticker needs at least 60 days of data. Try `--start 2015-01-01` or earlier.

### "TypeError in Streamlit caching"
→ Already fixed! Uses `@st.cache_resource` and proper data handling.

### Memory issues with large periods
→ Reduce `--batch_size` during training or limit historical data with `--end`.

## Limitations & Disclaimers ⚠️

- **Cryptocurrency is volatile**: Past performance ≠ future results
- **Prediction uncertainty increases over time**: Day predictions are more reliable than year forecasts
- **External factors**: Market events, regulation, news are not captured by price history alone
- **Model bias**: Trained on historical data; may not adapt to regime changes
- **Not investment advice**: Use for research/education only

## Next Steps

- Add technical indicators (RSI, MACD, Bollinger Bands)
- Multi-ticker ensemble predictions
- Uncertainty quantification (confidence intervals)
- Hyperparameter optimization
- GitHub Actions CI/CD for automatic retraining
- Backtesting framework

## References

- [LSTM Networks](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Streamlit](https://streamlit.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)

---

Made with ❤️ by therepositoryraider-boop
# 📊 Cryptocurrency Price Predictor

A machine learning-powered application that predicts cryptocurrency prices using LSTM (Long Short-Term Memory) neural networks. Built with Streamlit for an interactive user interface.

## 🌟 Features

- **Real-time Data**: Fetches live cryptocurrency data using yfinance
- **LSTM Model**: Advanced deep learning model for time-series prediction
- **Interactive UI**: User-friendly Streamlit interface
- **Multiple Cryptocurrencies**: Support for BTC, ETH, XRP, LTC, ADA, SOL
- **Customizable Parameters**: Adjust epochs, batch size, lookback period
- **Detailed Analysis**: Price charts, moving averages, daily returns
- **Future Predictions**: Predict prices up to 30 days ahead
- **Model Metrics**: Training/validation loss visualization

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Akshitkhare07/crypto-predictor1.git
cd crypto-predictor1
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### How to Use the App

1. **Select Cryptocurrency**: Choose from BTC, ETH, XRP, LTC, ADA, or SOL
2. **Adjust Parameters**:
   - **Lookback Period**: Historical data to fetch (30-365 days)
   - **Prediction Days**: Number of days to predict ahead (1-30)
   - **Test Set Size**: Percentage of data for testing (10-40%)

3. **View Historical Data**: Check the historical price chart in the "Price Chart" tab

4. **Train Model**:
   - Set epochs (training iterations) and batch size
   - Set lookback period for the model
   - Click "Train Model" button
   - Monitor training progress and loss metrics

5. **Make Predictions**: Switch to the "Predictions" tab to see future price forecasts

6. **Analyze Data**: View price statistics, moving averages, and daily returns

## 📊 Tabs Explained

### 📈 Price Chart
- Displays historical cryptocurrency prices
- Interactive chart for zooming and exploration

### 🤖 Model Training
- Configure model hyperparameters
- Train LSTM neural network
- View training and validation loss
- Monitor model performance

### 🔮 Predictions
- View predicted prices for future days
- Interactive chart showing historical vs predicted prices
- Summary statistics of predictions

### 📊 Analysis
- Volatility and price statistics
- Moving averages (7-day and 30-day)
- Daily returns visualization

## 🧠 Model Architecture

The LSTM model consists of:

```
Input Layer (Lookback, 1)
    ↓
LSTM Layer (50 units) + Dropout (0.2)
    ↓
LSTM Layer (50 units) + Dropout (0.2)
    ↓
LSTM Layer (25 units) + Dropout (0.2)
    ↓
Dense Layer (1 unit)
    ↓
Output: Predicted Price
```

**Optimizer**: Adam (learning rate: 0.001)
**Loss Function**: Mean Squared Error (MSE)

## ⚠️ Disclaimer

**This project is for educational purposes only.** Cryptocurrency prices are highly volatile and unpredictable. This model should NOT be used for actual trading decisions. Always do your own research and consult with financial advisors before making investment decisions.

## 🐛 Troubleshooting

**Issue**: "No module named 'tensorflow'"
- Solution: `pip install tensorflow`

**Issue**: Data loading fails
- Solution: Check internet connection and yfinance availability

**Issue**: Model training is slow
- Solution: Reduce epochs or batch size

## 📝 Future Improvements

- [ ] Add more cryptocurrencies
- [ ] Implement ensemble models
- [ ] Add technical indicators (RSI, MACD)
- [ ] Support for portfolio tracking
- [ ] Email notifications for price alerts
- [ ] Model persistence (save/load trained models)
- [ ] Comparison with other models (GRU, Transformer)

## 👨‍💻 Contributing

Feel free to fork, modify, and improve this project!

## 📄 License

MIT License - See LICENSE file for details

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Created by**: Akshitkhare07  
**Last Updated**: 2025-11-10
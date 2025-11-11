```markdown
# Crypto Price Predictor

I converted your Colab notebook into:
- a robust training script `cryptopricepredictor.py` that saves artifacts in `./models/`
- a Streamlit UI `app.py` that loads those artifacts and visualizes/predicts
- requirements and a Dockerfile for local containerized runs

Quickstart (local)
1. Create a venv and install dependencies:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
   pip install -r requirements.txt

2. Train the model (this writes artifacts under ./models):
   python cryptopricepredictor.py --ticker BTC-USD --start 2018-01-01 --epochs 10

   Notes:
   - Increase --epochs for better training.
   - Change --ticker to ETH-USD if you want to train on Ethereum.

3. Run the Streamlit UI:
   streamlit run app.py

4. Open http://localhost:8501

Docker
- Build:
   docker build -t crypto-lstm-app .

- To include trained models in the image, copy models/ into the build context and uncomment the COPY line in the Dockerfile.

- Run:
   docker run -p 8501:8501 crypto-lstm-app

Notes & next improvements
- This is a minimal single-feature LSTM (close price only). Add OHLCV, technical indicators, or multiple tickers for improved performance.
- Consider rolling-window validation, hyperparameter search, and better feature engineering.
- I can add:
  - multi-ticker training and a selector in the Streamlit app
  - a "retrain" button in the app (careful with compute/time)
  - GitHub Actions to train and publish artifacts on push
```
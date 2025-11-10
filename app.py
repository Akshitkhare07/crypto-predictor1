import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
        .main { padding-top: 0rem; }
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🔧 Configuration")
crypto_symbol = st.sidebar.selectbox(
    "Select Cryptocurrency",
    ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "ADA-USD", "SOL-USD"],
    index=0
)

lookback_period = st.sidebar.slider(
    "Lookback Period (days)",
    min_value=30,
    max_value=365,
    value=90,
    step=10
)

prediction_days = st.sidebar.slider(
    "Prediction Days Ahead",
    min_value=1,
    max_value=30,
    value=7,
    step=1
)

test_size = st.sidebar.slider(
    "Test Set Size (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
) / 100

# Main title
st.title("📊 Cryptocurrency Price Predictor")
st.markdown("Predict cryptocurrency prices using LSTM Neural Networks")

@st.cache_data
def load_data(symbol, period):
    """Load cryptocurrency data from yfinance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period)
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def prepare_data(data, lookback=60):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def build_model(lookback):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def make_predictions(model, data, scaler, lookback, future_days):
    """Make future predictions"""
    last_data = data[-lookback:].reshape(1, -1)
    predictions = []
    
    for _ in range(future_days):
        pred = model.predict(last_data, verbose=0)[0, 0]
        predictions.append(pred)
        last_data = np.append(last_data[0, 1:], pred)
        last_data = last_data.reshape(1, -1)
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

# Load data
with st.spinner("Loading cryptocurrency data..."):
    df = load_data(crypto_symbol, lookback_period)

if df is not None:
    st.success(f"✅ Data loaded successfully for {crypto_symbol}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
    price_change_pct = (price_change / df['Close'].iloc[0]) * 100
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("24h Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    with col3:
        st.metric("High (Period)", f"${df['Close'].max():.2f}")
    with col4:
        st.metric("Low (Period)", f"${df['Close'].min():.2f}")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "🤖 Model Training", "🔮 Predictions", "📊 Analysis"])
    
    with tab1:
        st.subheader("Historical Price Data")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title=f"{crypto_symbol} Historical Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=400,
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🤖 LSTM Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Number of Epochs", 10, 200, 50)
            batch_size = st.slider("Batch Size", 16, 128, 32, step=16)
        
        with col2:
            lookback = st.slider("Lookback Period (for model)", 30, 120, 60)
            st.info("The model learns from this many previous days to predict the next day")
        
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training model... This may take a moment..."):
                try:
                    # Prepare data
                    X, y, scaler = prepare_data(df, lookback)
                    X = X.reshape(X.shape[0], X.shape[1], 1)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, shuffle=False
                    )
                    
                    # Build and train model
                    model = build_model(lookback)
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        verbose=0
                    )
                    
                    # Store in session
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.df = df
                    st.session_state.lookback = lookback
                    
                    # Display metrics
                    st.success("✅ Model training completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Loss", f"{history.history['loss'][-1]:.6f}")
                    with col2:
                        st.metric("Validation Loss", f"{history.history['val_loss'][-1]:.6f}")
                    
                    # Plot training history
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                    fig.update_layout(
                        title="Model Training History",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=400,
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
    
    with tab3:
        st.subheader("🔮 Price Predictions")
        
        if hasattr(st.session_state, 'model'):
            try:
                # Get scaled data for predictions
                scaler = st.session_state.scaler
                model = st.session_state.model
                lookback = st.session_state.lookback
                
                # Prepare last data point
                last_scaled_data = scaler.transform(df[['Close']].tail(lookback))
                
                # Make predictions
                future_predictions = make_predictions(
                    model,
                    last_scaled_data,
                    scaler,
                    lookback,
                    prediction_days
                )
                
                # Create future dates
                last_date = df.index[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                
                # Display predictions
                st.write("### Predicted Prices")
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions
                })
                st.dataframe(pred_df, use_container_width=True)
                
                # Plot predictions
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=df.index[-30:],
                    y=df['Close'].iloc[-30:],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='#1f77b4')
                ))
                
                # Predictions
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='#ff7f0e', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{crypto_symbol} Price Prediction (Next {prediction_days} Days)",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    height=400,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary
                st.info(f"📊 **Prediction Summary**\n\n"
                       f"Current Price: ${df['Close'].iloc[-1]:.2f}\n\n"
                       f"Predicted Price ({prediction_days} days): ${future_predictions[-1]:.2f}\n\n"
                       f"Expected Change: ${future_predictions[-1] - df['Close'].iloc[-1]:.2f} "
                       f"({((future_predictions[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100):.2f}%)")
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                st.info("👉 Please train the model first in the 'Model Training' tab")
        else:
            st.warning("⚠️ No trained model found. Please train a model first in the 'Model Training' tab.")
    
    with tab4:
        st.subheader("📊 Price Analysis")
        
        # Calculate statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Volatility (Std Dev)", f"${df['Close'].std():.2f}")
        with col2:
            st.metric("Average Price", f"${df['Close'].mean():.2f}")
        with col3:
            st.metric("Price Range", f"${df['Close'].max() - df['Close'].min():.2f}")
        
        # Moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_7'], name='7-Day MA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_30'], name='30-Day MA'))
        
        fig.update_layout(
            title="Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=400,
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily returns
        df['Daily_Return'] = df['Close'].pct_change() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df['Daily_Return'], name='Daily Return %'))
        fig.update_layout(
            title="Daily Returns",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            height=400,
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Failed to load cryptocurrency data. Please check your internet connection.")
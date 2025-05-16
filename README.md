# Stock-market-Time-series-Forcaste-And-Analysis

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

def get_stock_data(ticker):
    df = yf.download(ticker, start='2015-01-01', end='2024-12-31')[['Close']]
    df.dropna(inplace=True)
    return df

def run_arima(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=30)
    rmse = np.sqrt(mean_squared_error(data['Close'][-30:], forecast))
    return forecast, rmse

def run_sarima(data):
    model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    result = model.fit()
    forecast = result.get_forecast(steps=30).predicted_mean
    rmse = np.sqrt(mean_squared_error(data['Close'][-30:], forecast))
    return forecast, rmse

def run_prophet(data):
    df_prophet = data.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    rmse = np.sqrt(mean_squared_error(data['Close'][-30:], forecast['yhat'][-30:]))
    return forecast, rmse

def run_lstm(data):
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    # Prepare training data
    def create_dataset(dataset, time_step=60):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Prepare test data for forecasting
    last_60 = scaled_data[-60:]
    forecast_input = last_60.reshape(1, time_step, 1)
    predictions = []

    for _ in range(30):
        pred = model.predict(forecast_input)[0][0]
        predictions.append(pred)
        forecast_input = np.append(forecast_input[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(data['Close'][-30:], forecast))
    forecast_index = pd.date_range(start=data.index[-1], periods=30, freq='B')

    return pd.Series(forecast, index=forecast_index), rmse

# =================================
# 📊 Streamlit Dashboard Section
# =================================

def main():
    import streamlit as st

    st.title("📊 Stock Market Forecast Dashboard")

    # Stock selection
    stock_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META']
    ticker = st.selectbox("Choose a Stock", stock_list)

    @st.cache_data
    def load_data(ticker):
        return get_stock_data(ticker)

    data = load_data(ticker)
    st.line_chart(data['Close'])

    # Forecasts
    forecast_arima, rmse_arima = run_arima(data)
    forecast_sarima, rmse_sarima = run_sarima(data)
    forecast_prophet, rmse_prophet = run_prophet(data)
    forecast_lstm, rmse_lstm = run_lstm(data)

    
    # ARIMA Chart
    st.subheader("ARIMA Forecast")
    arima_index = pd.date_range(start=data.index[-1] + pd.offsets.BDay(1), periods=30, freq='B')
    st.line_chart(pd.DataFrame(forecast_arima.values, index=arima_index, columns=['Forecast']))


    # SARIMA Chart
    st.subheader("SARIMA Forecast")
    st.line_chart(forecast_sarima)

    # Prophet Chart
    st.subheader("Prophet Forecast")
    st.line_chart(forecast_prophet[['ds', 'yhat']].set_index('ds').tail(30))

    # LSTM Chart
    st.subheader("LSTM Forecast")
    st.line_chart(forecast_lstm)

    # RMSE Scores
    st.subheader("📏 Model Accuracy (RMSE)")
    st.write(f"ARIMA RMSE: {rmse_arima:.2f}")
    st.write(f"SARIMA RMSE: {rmse_sarima:.2f}")
    st.write(f"Prophet RMSE: {rmse_prophet:.2f}")
    st.write(f"LSTM RMSE: {rmse_lstm:.2f}")

    st.success(f"✅ Forecasts for {ticker} generated successfully!")

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates

# Fetch stock data
def fetch_stock_data(symbols, days_back=1500):
    end_date = dt.datetime.now()  # Today's date
    start_date = end_date - dt.timedelta(days=days_back)  # Date 1500 days before today
    data = yf.download(symbols, start=start_date, end=end_date)
    return data['Close']

# Extended to optionally return the last input sequence for next-day prediction
def create_dataset(data, time_steps=1, return_last_seq=False):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    X = np.array(X)
    y = np.array(y)
    if return_last_seq:
        last_sequence = X[-1].reshape(1, time_steps, 1)  # Last sequence for next-day prediction
        return X, y, last_sequence
    return X, y


# Build LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(symbol, real_prices, predicted_prices, next_day_price, dates):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, real_prices, label='Actual Prices', color='dodgerblue', linestyle='-', linewidth=2)
    plt.plot(dates, predicted_prices, label='Predicted Prices', color='crimson', linestyle='--', linewidth=2)
    plt.scatter(dates[-1], next_day_price, color='limegreen', s=100, edgecolors='black', label='Next Day Prediction', zorder=5)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.title(f'Stock Price Prediction for {symbol}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

symbols = ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'META']
data = fetch_stock_data(symbols)

scalers = {}
X_train_dict, X_test_dict, y_train_dict, y_test_dict = {}, {}, {}, {}
models = {}
next_day_predictions = {}

for symbol in symbols:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[symbol].values.reshape(-1, 1))
    scalers[symbol] = scaler  # Save scaler for inverse transformation

    # Create dataset and extract the last sequence
    X, y, last_sequence = create_dataset(scaled_data, 60, return_last_seq=True)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    X_train_dict[symbol], X_test_dict[symbol] = X[:train_size], X[train_size:]
    y_train_dict[symbol], y_test_dict[symbol] = y[:train_size], y[train_size:]

    model = build_model((X_train_dict[symbol].shape[1], 1))
    model.fit(X_train_dict[symbol], y_train_dict[symbol], epochs=100, batch_size=32, validation_data=(X_test_dict[symbol], y_test_dict[symbol]), verbose=1)
    models[symbol] = model


    next_day_prediction = model.predict(last_sequence)
    next_day_prediction = scaler.inverse_transform(next_day_prediction)
    next_day_predictions[symbol] = next_day_prediction[0][0]


# Evaluate and plot results for each stock
for symbol in symbols:
    predicted_prices = models[symbol].predict(X_test_dict[symbol])
    predicted_prices = scalers[symbol].inverse_transform(predicted_prices)
    real_prices = scalers[symbol].inverse_transform(y_test_dict[symbol].reshape(-1, 1))
    next_day_price = next_day_predictions[symbol]

    dates = pd.date_range(start=dt.datetime.now() - dt.timedelta(days=len(real_prices)), periods=len(real_prices), freq='D')
    plot_predictions(symbol, real_prices.flatten(), predicted_prices.flatten(), next_day_price, dates)

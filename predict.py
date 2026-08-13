import yfinance as yf
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras

# Path to the trained CNN-LSTM model
MODEL_PATH = './models/day3/NEXT3-BEST.h5'

# Check whether ticker was provided
if len(sys.argv) != 2:
    print('Usage: python predict.py <TICKER>')
    print('Example: python predict.py AAPL')
    sys.exit(1)

ticker = sys.argv[1].upper()

print(f'Downloading data for {ticker}...')

# Download the latest 1 month of stock data
data = yf.Ticker(ticker).history(period='1mo', interval='1d')

if data.empty:
    print(f'Error: No data found for ticker {ticker}. Check the ticker symbol.')
    sys.exit(1)

print('Done! Processing data...')

# Take the latest 10 trading days
last10 = data.iloc[-10:].copy()

# Remove columns that are not used by the model
for column in ['Dividends', 'Stock Splits']:
    if column in last10.columns:
        last10.drop(columns=[column], inplace=True)

# Add Day of Week feature
last10['DayOfWeek'] = [i.dayofweek for i in last10.index]

# Make sure we have exactly 10 days
if len(last10) < 10:
    print('Error: Not enough trading-day data available.')
    sys.exit(1)

# Statistics used to convert the prediction back to the original price
mean = last10['Close'].mean()
std = last10['Close'].std()

# Standardize each feature
for column in last10.columns:
    column_std = last10[column].std()
    column_mean = last10[column].mean()

    if column_std != 0:
        last10[column] = (
            (last10[column] - column_mean) / column_std
        )
    else:
        last10[column] = 0

# Convert data to the shape expected by the CNN-LSTM model
# Shape: (samples, time steps, features)
model_in = last10.values.reshape(1, 10, 6)

print(f'Done! Loading model from {MODEL_PATH}')

try:
    # compile=False prevents compatibility problems with old
    # Keras training metrics stored inside the .h5 model.
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

except Exception as e:
    print(f'Error loading model: {e}')
    print('Make sure NEXT3-BEST.h5 is inside models/day3/')
    sys.exit(1)

print('Done! Making prediction...')

# Make prediction
prediction_scaled = model.predict(
    model_in,
    batch_size=1,
    verbose=0
)

# Convert prediction back to the original price scale
output_price = round(
    float(prediction_scaled[0][0]) * std + mean,
    2
)

print('')
print('=============================')
print(f'=== PREDICTION ({ticker}) ===')
print(f'Predicted Price for 3 days from now: ${output_price}')
print('=============================')
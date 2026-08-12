import yfinance as yf
import pandas as pd
import math
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras  
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

# Update these to match the actual folder structure and the file you have
# Since you have NEXT3-E6.h5, we will use that for the prediction test
MODEL_PATH = './models/day3/NEXT3-E6.h5' 

if len(sys.argv) != 2:
    print('Usage: python predict.py <TICKER>')
    print('Example: python predict.py GE')
    sys.exit(1)

ticker = sys.argv[1]

print(f'Downloading data for {ticker}...')

# Fetching the last month of data
data = yf.Ticker(ticker).history(period='1mo', interval='1d')

if data.empty:
    print(f"Error: No data found for ticker {ticker}. Check the spelling.")
    sys.exit(1)

print('Done! Processing data...')

# Extract last 10 days and drop columns that might not exist in all stocks
last10 = data.iloc[-10:].copy()
if 'Dividends' in last10.columns:
    last10.drop(columns=['Dividends'], inplace=True)
if 'Stock Splits' in last10.columns:
    last10.drop(columns=['Stock Splits'], inplace=True)

# Add Day of Week feature
last10['DayOfWeek'] = [i.dayofweek for i in last10.index]

# Statistics for De-standardization
mean = last10['Close'].mean()
std = last10['Close'].std()

# Standardizing the data (Z-score normalization)
for column in last10.columns:
    s = last10[column].std()
    m = last10[column].mean()
    last10[column] = (last10[column] - m) / s if s != 0 else 0

# IMPORTANT: Reshape to (1, 10, 6) to match the fixed model1.py architecture
# We removed the "1" in the middle (1, 10, 1, 6 -> 1, 10, 6)
model_in = last10.values.reshape(1, 10, 6)

print(f'Done! Loading model from {MODEL_PATH}')

try:
    # Loading the specific model you uploaded
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure NEXT3-E6.h5 is in the cnn-lstm-stock-main/models/day3/ folder.")
    sys.exit(1)

print('Done! Making prediction...')
# Making the prediction
prediction_scaled = model.predict(model_in, batch_size=1)

# De-standardizing the output to get actual dollar price
output_price = round(float(prediction_scaled[0][0]) * std + mean, 2)

print('')
print(f'=== PREDICTION ({ticker}) ===')
print(f'Predicted Price for 3 days from now: ${output_price}')
print('=============================')
import yfinance as yf
import pandas as pd
import math
from tqdm import tqdm as tqdm
import pickle
import numpy as np
import random
from multiprocessing import Pool
import os

pd.options.mode.chained_assignment = None

tickers = ['TSLA', 'AAPL', 'MSFT', 'U', 'CCL', 'TD', 'SPY', 'FB', 'V', 'DIS',
           'CNR', 'HD', 'UNH', 'MCD', 'MMM', 'ATVI', 'ADBE', 'AMD', 'GOOG',
           'AMZN', 'AXP', 'BAC', 'BA', 'CVX', 'C', 'KO', 'DOW', 'GM', 'GILD',
           'INTC', 'MA', 'NVDA', 'TXN', 'XRX', 'RY.TO', 'CP.TO', 'TRI.TO',
           'ATD-B.TO', 'L.TO', 'DOL.TO', 'BB.TO', 'DOO.TO', 'WEED.TO',
           'SNC.TO', 'SHOP', 'SU.TO', 'CM.TO', 'TD.TO', 'ENB.TO', 'APHA.TO',
           'XIU.TO', 'AC.TO']

def z_score(df, f1, f2, f3):
    df['DayOfWeek']=[i.dayofweek for i in df.index]
    for column in df.columns:
        std = df[column].std()
        mean = df[column].mean()
        df[column] = (df[column] - mean) / std if std != 0 else 0
        if column == 'Close':
            f1 = (f1-mean)/std if std != 0 else 0
            f2 = (f2-mean)/std if std != 0 else 0
            f3 = (f3-mean)/std if std != 0 else 0
    nan_flag = df.isnull().values.any() or math.isnan(f1) or math.isnan(f2) or math.isnan(f3)
    return df.values, f1, f2, f3, nan_flag

def sequencify(df):
    s1,s2,s3=[],[],[]
    dropped = 0
    for i in range(len(df.index)-13):
        sequence, f1, f2, f3, nan_flag = z_score(df.iloc[i:i+10], df['Close'].iloc[i+10], df['Close'].iloc[i+11], df['Close'].iloc[i+12])
        if(nan_flag):
            dropped += 1
        else:
            s1.append([sequence, f1])
            s2.append([sequence, f2])
            s3.append([sequence, f3])
    return [s1,s2,s3, dropped]

# --- WINDOWS SAFETY WRAPPER START ---
if __name__ == '__main__':
    raws = {}
    print('Downloading Data...')
    for ticker in tqdm(tickers):
        try:
            t = yf.Ticker(ticker)
            t_data = t.history(period = '10y', interval = '1d')
            raws[ticker] = t_data
        except Exception as e:
            print(f"Skipping {ticker} due to error.")

    print('Done downloading.')

    col_d = {} 
    for ticker in tickers:
        if ticker in raws and not raws[ticker].empty:
            raw = raws[ticker]
            col_d[ticker] = raw.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')

    a1, a2, a3 = [], [], []
    total_dropped = 0

    print('Generating sequences. This may take a while...')
    with Pool(processes=4) as pool: # Lowered to 4 for stability
        results = pool.imap_unordered(sequencify, list(col_d.values()))
        for res in results:
            a1.append(res[0])
            a2.append(res[1])
            a3.append(res[2])
            total_dropped += res[3]

    print(f'Done! Dropped {total_dropped} sequences containing NaN')
    
    train1, test1, train2, test2, train3, test3 = [], [], [], [], [], []
    RATIO = 0.05

    for hist in a1:
        split = math.floor(len(hist) * RATIO)
        train1.extend(hist[:-split]); test1.extend(hist[-split:])
    for hist in a2:
        split = math.floor(len(hist) * RATIO)
        train2.extend(hist[:-split]); test2.extend(hist[-split:])
    for hist in a3:
        split = math.floor(len(hist) * RATIO)
        train3.extend(hist[:-split]); test3.extend(hist[-split:])

    random.shuffle(train1); random.shuffle(train2); random.shuffle(train3)
    random.shuffle(test1); random.shuffle(test2); random.shuffle(test3)

    if not os.path.exists('./data'):
        os.makedirs('./data')

    print('Saving data to file...')
    data_map = {
        'train1': train1, 'test1': test1,
        'train2': train2, 'test2': test2,
        'train3': train3, 'test3': test3
    }

    for filename, dataset in data_map.items():
        with open(f'./data/{filename}.pkl', 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('All files saved successfully in the /data folder!')
# --- WINDOWS SAFETY WRAPPER END ---
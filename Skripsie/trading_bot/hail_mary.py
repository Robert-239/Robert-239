from __future__ import absolute_import, division, print_function, unicode_literals

import datetime as dt
import json
import os
import time
import numpy as np
import polars as pl
#plotting & misc
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Tuple


#Data Processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#ML/AI

from keras.api.losses import mean_squared_error
from keras.src.layers.preprocessing.tf_data_layer import keras
import tensorflow as tf


from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM,Dense, Dropout ,Activation, Bidirectional
import matplotlib.pyplot as plt



PORTFOLIO_SCHEMA = {
    'date': str,
    'capital': pl.Float64,
    'asset': pl.Float64,
    'asset_value': pl.Float64,
    'portfolio_value': pl.Float64,
}

TRADE_SCHEMA = {
    'date': str,
    'price': pl.Float64,
    'buy': bool,
    'sell': bool,
    'ID': int,
}

SIGNAL_SCHEMA = {
    'timestamp': str,
    'price': pl.Float64,
    'signal': pl.Int8,
    'position': pl.Int8
}

CANDLE_SCHEMA = {
    'timestamp': str,
    'open': pl.Float64,
    'close': pl.Float64,
    'high': pl.Float64,
    'low': pl.Float64,
    'volume': pl.Float64,
}

def formatData(data : pl.DataFrame)->pl.DataFrame:

    formated_data = pl.DataFrame(schema={
            'timestamp': str,
            'open': pl.Float64,
            'close': pl.Float64,
            'high': pl.Float64,
            'low': pl.Float64,
            'volume': pl.Float64,
        })
    print(data)
    lengthDF = data.shape[0]
    print(lengthDF)
    for i in range(lengthDF):
        temp_dict = data.row(i,named= True)
        temp_dict['timestamp'] = (dt.datetime.strftime((dt.datetime.strptime(temp_dict['timestamp'],"%Y-%m-%d %H-%M-%S")),"%Y-%m-%d"))
        
        temp_Frame = pl.DataFrame(temp_dict)
        
        formated_data = formated_data.extend(temp_Frame)

    return formated_data

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


def load_and_combine_csv(folders: List[str], file_patern: str = '*.csv') -> pl.DataFrame:
    combined_df = None
    for folder in folders:
        csv_files = [f for f in os.listdir(folder) if f.endswith(file_patern)]
        for csv_file in csv_files:
            file_path = os.path.join(folder, csv_file)
            df = pl.read_csv(file_path)
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.extend(df)
    if combined_df is None:
        raise ValueError("No CSV files found in the specified folders.")
    combined_df = combined_df.unique(subset=["timestamp"], maintain_order=True)
    return combined_df

def init_data_frames():
    date = [0]
    capital = [100000]
    asset = [0]
    asset_value = [0]
    portfolio_value = [0]
    portfolio_df = pl.DataFrame(
        {'date': date, 'capital': capital, 'asset': asset, 'asset_value': asset_value, 'portfolio_value': portfolio_value}, 
        schema=PORTFOLIO_SCHEMA
    )
    trades = pl.DataFrame(schema=TRADE_SCHEMA)
    signals = pl.DataFrame(schema=SIGNAL_SCHEMA)
    return portfolio_df, trades, signals

def trade_proc(df_dict, df_port: pl.DataFrame, signal, amount=0.5, stop_loss=None):
    portfolio = df_port.row(-1, named=True)
    trade = {
        'date': df_dict['timestamp'],
        'price': 0,
        'buy': 0,
        'sell': 0,
        'ID': 0
    }
    current_price = df_dict['close']
    if signal == 1:  # Buy signal
        buy_amount = amount * portfolio['capital'] / current_price
        portfolio['capital'] -= (buy_amount * (current_price * (1 + 0.02)))
        portfolio['asset'] += buy_amount
        portfolio['asset_value'] = portfolio['asset'] * current_price
        portfolio['portfolio_value'] = portfolio['capital'] + portfolio['asset_value']
        portfolio['date'] = df_dict['timestamp']
        trade.update({'buy': True, 'price': buy_amount * current_price, 'ID': 50})
    elif signal == -1:  # Sell signal
        sell_amount = amount * portfolio['asset']
        portfolio['capital'] += (sell_amount * (current_price * (1 - 0.02)))
        portfolio['asset'] -= sell_amount
        portfolio['asset_value'] = portfolio['asset'] * current_price
        portfolio['portfolio_value'] = portfolio['capital'] + portfolio['asset_value']
        portfolio['date'] = df_dict['timestamp']
        trade.update({'sell': True, 'price': sell_amount * current_price, 'ID': -100})
    elif stop_loss is not None and portfolio['asset'] > 0:  # Check stop-loss condition
        purchase_price = portfolio['asset_value'] / portfolio['asset']
        if current_price <= purchase_price * (1 - stop_loss):
            sell_amount = portfolio['asset']
            portfolio['capital'] += (sell_amount * (current_price * (1 - 0.02)))
            portfolio['asset'] = 0
            portfolio['asset_value'] = 0
            portfolio['portfolio_value'] = portfolio['capital']
            portfolio['date'] = df_dict['timestamp']
            trade.update({'sell': True, 'price': sell_amount * current_price, 'ID': -200})
    else:  # Hold
        portfolio['asset_value'] = portfolio['asset'] * current_price
        portfolio['portfolio_value'] = portfolio['capital'] + portfolio['asset_value']
        portfolio['date'] = df_dict['timestamp']
        trade.update({'ID': 100})
    return portfolio, trade


#Indicators
###############################################################################################
def rsi(df: pl.DataFrame, window=14) -> pl.Series:
    delta = df['close'].diff().fill_null(0)
    gain = delta.clip_min(0)
    loss = -delta.clip_max(0)

    avg_gain = gain.rolling_mean(window).fill_null(0)
    avg_loss = loss.rolling_mean(window).fill_null(0)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fvg(df: pl.DataFrame, gapr=0.8) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    buy_gaps = []
    sell_gaps = []
    for i in range(2, df.shape[0] - 1):
        prev_candle = df.row(i - 1, named=True)
        next_candle = df.row(i + 1, named=True)
        cur_candle = df.row(i, named=True)

        candle_body_ratio = np.abs((cur_candle['open'] - cur_candle['close']) / (cur_candle['high'] - cur_candle['low']))

        if candle_body_ratio >= gapr:
            if prev_candle['low'] > next_candle['high']:
                buy_gap = prev_candle['low'] - next_candle['high']
                buy_gaps.append((i - 1, i, buy_gap))
            elif prev_candle['high'] < next_candle['low']:
                sell_gap = next_candle['low'] - prev_candle['high']
                sell_gaps.append((i - 1, i, sell_gap))
    return buy_gaps, sell_gaps




###############################################################################################

def generate_signals(df: pl.DataFrame, short_window=30, long_window=100, rsi_window=14, fvg_gapr=0.8) -> pl.DataFrame:
    # Calculate indicators
    short_avg = df['close'].rolling_mean(short_window).fill_null(0).alias('short_mavg')
    long_avg = df['close'].rolling_mean(long_window).fill_null(0).alias('long_mavg')
    rsi_values = rsi(df, rsi_window).alias('rsi')

    signal = pl.DataFrame({
        'timestamp': df['timestamp'],
        'signal': 0.0,
        'positions': 0.0,
        'short_mavg': short_avg,
        'long_mavg': long_avg,
        'rsi': rsi_values
    })

    # Moving Average signals
    signal = signal.with_columns([
        pl.when(pl.col('short_mavg') > pl.col('long_mavg')).then(1.0).otherwise(0).alias('ma_signal')
    ])
    
    signal = signal.with_columns([
        pl.when(pl.col('ma_signal').shift(0) - pl.col('ma_signal').shift(1) == 1).then(1.0)
        .when(pl.col('ma_signal').shift(0) - pl.col('ma_signal').shift(1) == -1).then(-1.0)
        .otherwise(0.0).alias('ma_positions')
    ])

    # RSI signals
    signal = signal.with_columns([
        pl.when(pl.col('rsi') < 30).then(1.0)
        .when(pl.col('rsi') > 70).then(-1.0)
        .otherwise(0.0).alias('rsi_signal')
    ])

    signal = signal.with_columns([
        pl.when(pl.col('rsi_signal').shift(0) - pl.col('rsi_signal').shift(1) == 1).then(1.0)
        .when(pl.col('rsi_signal').shift(0) - pl.col('rsi_signal').shift(1) == -1).then(-1.0)
        .otherwise(0.0).alias('rsi_positions')
    ])
    # FVG signals
    buy_gaps, sell_gaps = fvg(df, fvg_gapr)
    fvg_signal = pl.Series([0] * df.shape[0], dtype=pl.Float32)
    for gap in buy_gaps:
        fvg_signal[gap[1]] = 1.0
    for gap in sell_gaps:
        fvg_signal[gap[1]] = -1.0

    signal = signal.with_columns(fvg_signal.alias('fvg_signal'))

    # Composite score
    signal = signal.with_columns([
        ((pl.col('ma_positions') * 0.4) + (pl.col('rsi_positions') * 0.4) + (pl.col('fvg_signal') * 0.2)).alias('composite_score')
    ])

    # Final position based on composite score
    signal = signal.with_columns([
        pl.when(pl.col('composite_score') > 0.5).then(1.0)
        .when(pl.col('composite_score') < -0.5).then(-1.0)
        .otherwise(0.0).alias('positions')
    ])

    return signal


def calculate_drawdown(portfolio_values):
    peaks = np.maximum.accumulate(portfolio_values)
    print(peaks)
    drawdowns = (portfolio_values - peaks) / peaks
    max_drawdown = drawdowns.min()
    return max_drawdown

def get_LSTM_model(dropout: float , memory: int , x_train ,y_train):
    model = keras.Sequential()

    model.add(Bidirectional(LSTM(memory,return_sequences= True),input_shape=(dropout,x_train.shape[-1])))
    model.add(Dropout(rate=dropout))
    model.add(Bidirectional(LSTM((memory*2),return_sequences=True)))
    model.add(Dropout(rate=dropout))
    model.add(Bidirectional(LSTM(memory,return_sequences=False)))
    model.add(Dense(units=1))
    model.add(Activation("linear"))

    model.compile(loss = 'mean_squared_error',optimizer='adam')

    model.fit(x_train,
              y_train,
              epochs= 64,
              shuffle= False,
              validation_split= 0.01)

    return model

def simulate_stream(price_data, short_ma=30, long_ma=100, stop_loss=None):
    # Initialize data frames
    portfolio_df, trades, signals = init_data_frames()
    
    candles = pl.DataFrame(schema=CANDLE_SCHEMA)
    current_trade = pl.DataFrame(schema=TRADE_SCHEMA)
    len_df = price_data.shape[0]
    performance = {
        'short_ma': short_ma,
        'long_ma': long_ma,
        'num_trades': 0,
        'performance': 0,
        'max_drawdown': 0
    }
    portfolio_dict = portfolio_df.row(0, named=True)
    portfolio_dict['portfolio_value'] = portfolio_dict['capital'] + portfolio_dict['asset_value']

    buy_signals = []
    sell_signals = []
    drawdown_sell_signals = []

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 12))  # Set the figure size (width, height)
    fig.suptitle(f"Bitcoin Price and Portfolio Value (SMA: {short_ma}, LMA: {long_ma})")
    c = np.random.rand(3,)
    
    for i in tqdm(range(len_df), desc="Processing Historical Data Stream", leave=True):
        
        if i != (len_df - 1):
            ax1.clear()
            ax2.clear()

        candle_dict = price_data.row(i, named=True)
        candle_dict['timestamp'] = dt.datetime.strftime(dt.datetime.strptime(candle_dict['timestamp'], "%Y-%m-%d %H-%M-%S"), "%Y-%m-%d")
        temp_frame = pl.DataFrame(candle_dict, schema=CANDLE_SCHEMA)
        candles.extend(temp_frame)

        signals = generate_signals(candles, short_ma, long_ma)
        signal = signals.row(-1, named=True)
        portfolio_dict, trade = trade_proc(candle_dict, portfolio_df, signal['positions'], stop_loss=stop_loss)
        portfolio_dict['date'] = candle_dict['timestamp']
        current_trade = pl.DataFrame(trade, TRADE_SCHEMA)
        trades.extend(current_trade)

        temp_frame = pl.DataFrame(portfolio_dict, schema=PORTFOLIO_SCHEMA)
        portfolio_df.extend(temp_frame)


        if trade['buy']:
            buy_signals.append((candle_dict['timestamp'], candle_dict['close']))
        elif trade['sell']:
            sell_signals.append((candle_dict['timestamp'], candle_dict['close']))
        elif trade['ID'] == -200:  # Drawdown sell signal
            drawdown_sell_signals.append((candle_dict['timestamp'], candle_dict['close']))

    if buy_signals:
        buy_dates, buy_prices = zip(*buy_signals)
        buy_dates = np.asarray(buy_dates,dtype='datetime64[s]')
        ax1.scatter(buy_dates, buy_prices, marker='^', color='g', label='Buy Signal', alpha=1)

    if sell_signals:
        sell_dates, sell_prices = zip(*sell_signals)
        sell_dates = np.asarray(sell_dates,dtype='datetime64[s]')
        ax1.scatter(sell_dates, sell_prices, marker='v', color='r', label='Sell Signal', alpha=1)

    if drawdown_sell_signals:
        dd_sell_dates, dd_sell_prices = zip(*drawdown_sell_signals)
        dd_sell_dates = np.asarray(dd_sell_dates,dtype='datetime64[s]')
        ax1.scatter(dd_sell_dates, dd_sell_prices, marker='x', color='orange', label='Drawdown Sell', alpha=1)

    ax1.plot(np.asarray(candles['timestamp'], dtype='datetime64[s]'), candles['close'], label="Bitcoin Price")
    ax2.plot(np.asarray(portfolio_df['date'], dtype='datetime64[s]'), portfolio_df['portfolio_value'], label="Portfolio Value", color= c)
    ax1.plot(np.asarray(candles['timestamp'], dtype='datetime64[s]'), signals['short_mavg'], label=f"Short MA ({short_ma})")
    ax1.plot(np.asarray(candles['timestamp'], dtype='datetime64[s]'), signals['long_mavg'], label=f"Long MA ({long_ma})")

    ax1.legend()
    ax2.legend()
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Portfolio Value')
    ax2.set_xlabel('Date')
    ax1.grid(True)
    ax2.grid(True)
    ax1.tick_params(axis='x', labelrotation=90)
    ax2.tick_params(axis='x', labelrotation=90)
        
    #plt.pause(0.01)

    print(signals.filter(pl.col('positions') == 1))
    print(trades)
    #time.sleep(0.01)

    folder = 'fig'
    portfolio_plot = f"portfolio_performance_sma_{short_ma}_lma_{long_ma}.png"
    file_path = os.path.join(folder, portfolio_plot)

    if not os.path.isdir(folder):
        os.mkdir(folder)

    plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Set the DPI for higher resolution and use bbox_inches='tight' to ensure the plot is saved without extra whitespace
    print(f"Saved plot as {portfolio_plot}")
    plt.close()

    portfolio_values = portfolio_df['portfolio_value'].to_numpy()
    performance['num_trades'] = signals.filter((pl.col('positions') == 1) | (pl.col('positions') == -1)). shape[0]
    performance['performance'] = portfolio_df['portfolio_value'].item(-1) / portfolio_df['portfolio_value'].item(1)
    performance['max_drawdown'] = calculate_drawdown(portfolio_values)

    folder = 'data'
    performance_file = f"performance_sma_{short_ma}_lma_{long_ma}.json"
    
    file_path = os.path.join(folder, performance_file)

    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(file_path, 'w') as pf:
        json.dump(performance, pf)
    print(f"Saved performance as {performance_file}")

    return portfolio_df, trades, signals, performance

def test_Moving_Averages(price_data, stop_loss =None):
    short_ma_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    long_ma_range = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    perf_arr = np.ones((len(short_ma_range), len(long_ma_range)))
    results = []

    for i,short_ma in enumerate(short_ma_range):
        for j, long_ma in enumerate(long_ma_range):
            if short_ma == long_ma:
                continue  # Skip unnecessary calculations when the short and long moving averages are the same
            portfolio_df, trades, signals, ma_perf = simulate_stream(price_data, short_ma, long_ma , stop_loss= stop_loss)
            perf_arr[i][j] = ma_perf['performance']
            results.append(ma_perf)
    
    plt.figure(figsize=(16,12))
    sns.heatmap(perf_arr , xticklabels=[str(x) for x in long_ma_range],yticklabels=[str(y) for y in short_ma_range], cmap= "viridis")
    plt.title("Moving Averages Performance Heat map")
    plt.xlabel("Long Moving Average")
    plt.ylabel("Short Moving Average")
    
    plt.savefig("Heat_Map.png",dpi=300, bbox_inches = 'tight')
    
    plt.show()
    folder = "data"
    results_file = "moving_average_performance.json"

    file_path = os.path.join(folder,results_file)
    
    if not os.path.isdir(folder):
        os.mkdir(folder)

    with open(file_path, 'w') as rf:
        json.dump(results, rf)
    print(f"Saved all performance results as {results_file}")

    return results

class Bot:
    def __init__(self, strategy_name, initial_capital, stop_loss=None):
        self.strategy_name = strategy_name
        self.capital = initial_capital
        self.stop_loss = stop_loss
        self.portfolio_df, self.trades, self.signals = init_data_frames()

    def buy_and_hold(self):
        pass

    def simple_moving_average(self, short_ma=30, long_ma=100):
        self.portfolio_df, self.trades, self.signals, performance = simulate_stream(price_data_D, short_ma, long_ma, stop_loss=self.stop_loss)
        return performance

# Example usage
if __name__ == "__main__":
    # Load historical price data
    FILE_LIST = ['C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2019',
                   'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2020',
                   'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2021',
                   'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2022',
                   'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2023']
    
    
    price_data_D = load_and_combine_csv(FILE_LIST,'_dayly_price.csv')
    print(price_data_D)
    price_data_8H = load_and_combine_csv(FILE_LIST,'_8_hour_price.csv')
    print(price_data_8H)
    price_data_4H = load_and_combine_csv(FILE_LIST,'_4_hour_price.csv')
    print(price_data_4H)


    portfolio_df, trades, signals, ma_perf = simulate_stream(price_data_D)
    # Run moving averages test
    #results = test_Moving_Averages(price_data_D,stop_loss= None)
    print(ma_perf)


##########################func store

def pre_process(prices : np.ndarray,input_window ,target_window = 1):
    rng = (prices.shape[0]-input_window - target_window)
    data_vec = np.ndarray((rng,input_window))
    target_vec = np.ndarray((rng,1))
    index = 0

    for j in range(prices.shape[0]-(input_window + target_window) ):
        
        for i in range(input_window):
            data_vec[j,i] = prices[i+index]
            if i == input_window -1:
                target_vec[j,:] = prices[i + index +1:i+ index + target_window]
        index+= 1



    return data_vec , target_vec

data_x , data_y = pre_process(scaled_close,WINDOW)

print(f"\ndata vectors:\t{data_x.shape}")
print(data_x)
print(f"\ndata targets:\t{data_y.shape}")
print(data_y)

print("\nnew window size\n")
data_x2 , data_y2 = pre_process(scaled_close,WINDOW,3)

print(f"\ndata vectors:\t{data_x2.shape}")
print(data_x2)
print(f"\ndata targets:\t{data_y2.shape}")
print(data_y2)

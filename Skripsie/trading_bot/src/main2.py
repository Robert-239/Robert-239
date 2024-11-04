from __future__ import absolute_import, division, print_function, unicode_literals

import datetime as dt
import json
import os
import time
import numpy as np
import polars as pl
#plotting & misc
from polars.datatypes import Int8
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Tuple

#Data Processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#ML/AI
from keras.api.losses import dice, mean_squared_error
from keras.src.layers.preprocessing.tf_data_layer import keras
import tensorflow as tf
import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, Activation, Bidirectional

import tensorboard
print(tensorboard.__version__)

from collections import deque
TF_ENABLE_ONEDNN_OPTS = 0
price_scaler = MinMaxScaler(feature_range=(0,1))
short_scaler = MinMaxScaler(feature_range=(0,1))
long_scaler = MinMaxScaler(feature_range=(0,1))
#Data Path


FLIST2 = ['C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2019',
          'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2020',
          'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2021',
          'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2022']


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
    'signal': pl.Float64,
    'position': pl.Float64
}

CANDLE_SCHEMA = {
    'timestamp': str,
    'open': pl.Float64,
    'close': pl.Float64,
    'high': pl.Float64,
    'low': pl.Float64,
    'volume': pl.Float64,
    'scaled_close' : pl.Float64
}

##########################################################################################
def load_and_combine_csv(folders: List[str], file_patern: str = '*.csv') -> pl.DataFrame:
    '''

    '''
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
    trades = pl.DataFrame(schema=TRADE_SCHEMA)
    signals = pl.DataFrame(schema=SIGNAL_SCHEMA)
    ml_signal = pl.DataFrame(schema= {
    'timestamp' : str,
    'signal' : pl.Float64,
    'positions' : pl.Float64,
    'risk_score' : pl.Float64,
    'ratio' : pl.Float64
    })
    return trades, signals, ml_signal


def init_portofolio()-> pl.DataFrame:

    date = [0]
    capital = [100000]
    asset = [0]
    asset_value = [0]
    portfolio_value = [0]
    portfolio = pl.DataFrame(
        {'date': date, 'capital': capital, 'asset': asset, 'asset_value': asset_value, 'portfolio_value': portfolio_value}, 
        schema=PORTFOLIO_SCHEMA
    )
    return portfolio


#####################################################################################
#Data Pre porcessing
#####################################################################################

def pre_process(prices : np.ndarray,input_window ,target_window = 3):
    rng = (prices.shape[0]-input_window - target_window)
    data_vec = np.ndarray((rng,input_window))
    target_vec = np.ndarray((rng,target_window))
    index = 0

    for j in range(prices.shape[0]-(input_window + target_window) ):
        
        for i in range(input_window):
            data_vec[j,i] = prices[i+index]
            if i >= input_window - 1:
                for k in range(target_window):
                    target_vec[j,k] = prices[i + index +1+k]
        index+= 1



    return data_vec , target_vec

def split_data(data ,train_split =0.7 , test_split = 0.2 , validation_split = 0.1):        
    size_train = int(data.shape[0] * train_split)
    size_test = int(data.shape[0] * test_split)
    size_validation = int(data.shape[0] * validation_split)

    train = data[:size_train]
    test = data[size_train : (size_train + size_test)]
    validation = data[(size_train + size_test):]

    return np.array(train) , np.array(test) ,np.array(validation)
    

def create_sequence(candle_df : pl.DataFrame ,sequence_length = 14):
    
    close_price = candle_df['scaled_close'].tail(sequence_length).fill_nan(0).to_numpy()
    
    i = 0
    if (i == 0):
        close_price = close_price.reshape(1,close_price.shape[0])
        i = 1
    else: close_price = close_price.reshape(1,close_price.shape[1])
    price_sequence = np.zeros((1,sequence_length))
    
    input_sequence = np.zeros((1,sequence_length))
    
    for i in range(close_price.shape[1]): 
        input_sequence[0,i]  = price_sequence[0,i] + close_price[0,i]
    
    
    return input_sequence

#Nueral Network
#####################################################################################

def load_model(model_path: str):
    
    model = keras.saving.load_model(model_path)

    return model

def predict_seq(model,window, input_sequence : np.ndarray, predict_length):
    sequence = input_sequence
    
    for _ in range(predict_length):
        
        sequence = sequence[:,-window:].reshape((1,14))
        sequence.reshape((1,window,1))
        y_hat = model.predict(input_sequence)[0][0]
        out = np.ndarray((1,1))
        out[0,0] = y_hat

        sequence  =  np.append(input_sequence,out,axis =1)
        
    sequence = sequence[:,predict_length-1:]
    sequence = sequence.reshape((1,11))
    
    sequence = price_scaler.inverse_transform(sequence)
    return sequence

#Indicators
#####################################################################################
def rsi(df: pl.DataFrame, window=14) -> pl.Series:
    delta = df['close'].diff().fill_null(0)
    gain = delta.clip_min(0)
    loss = -delta.clip_max(0)

    avg_gain = gain.rolling_mean(window).fill_null(0)
    avg_loss = loss.rolling_mean(window).fill_null(0)
    

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.fill_nan(0)
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

#Risk calculation
###############################################################################################

def trade_risk(rsi):
    risk = rsi/100
    return risk

###############################################################################################
#Signal generating code
#Technical indicators

def generate_signals(df: pl.DataFrame, short_window=30, long_window=100, rsi_window=14, fvg_gapr=0.8) -> pl.DataFrame:
    # Calculate indicators
    short_avg = df['close'].rolling_mean(short_window).fill_null(0).alias('short_mavg')
    long_avg = df['close'].rolling_mean(long_window).fill_null(0).alias('long_mavg')
    rsi_values = rsi(df, rsi_window).alias('rsi')
    rsi_values = rsi_values.fill_nan(0)
    
    
    risk_score = trade_risk(rsi_values)
    signal = pl.DataFrame({
        'timestamp': df['timestamp'],
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
        .when(pl.col('rsi_signal').shift(0) - pl.col('rsi_signal').shift(1) == 2).then(1.0) 
        .when(pl.col('rsi_signal').shift(0) + pl.col('rsi_signal').shift(1) == -1).then(-1.0)
        .when(pl.col('rsi_signal').shift(0) - pl.col('rsi_signal').shift(1) == -2).then(-1.0)
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

    signal = signal.with_columns([pl.when(pl.col('positions') != 0).then(risk_score).otherwise(0).alias('risk_score')])

    return signal

def generate_ML_signals(candle_df : pl.DataFrame, ml_signal : pl.DataFrame , seq)->pl.DataFrame:
    

    signal_dict = {
        'timestamp' : candle_df.tail(1).item(0,'timestamp'),
        'signal' : 0.0,
        'positions' : 0.0,
        'risk_score' : 0.0,
        'ratio' : 0.0
    }
    rsi_values = rsi(candle_df).alias('rsi')
    rsi_values = rsi_values.fill_nan(0)

    risk_score = trade_risk(rsi_values)

    close_price = candle_df.tail(1).item(0,'close')

    print(f" \n {seq} \n")
    predicted_close_price = seq[:,-1]

    ratio = (predicted_close_price)/close_price
    signal_dict['ratio'] = ratio
    
    
    if (ratio > 1.05):
        signal_dict['signal'] = 1.0
    elif (ratio < 0.9):
        signal_dict['signal'] = -1.0
    else: signal_dict['signal'] = 0.0

    temp = pl.DataFrame(signal_dict)
    ml_signal.extend(temp)
    
    ml_signal = ml_signal.with_columns([pl.when(pl.col('signal') + pl.col('signal').shift(1) == 1).then(1.0) 
                                        .when(pl.col('signal') + pl.col('signal').shift(1) == -1 ).then(-1.0)
                                        .when(pl.col('signal') - pl.col('signal').shift(1) == -2 ).then(-1.0)
                                        .when(pl.col('signal') + pl.col('signal').shift(1) == 2 ).then(0.0)
                                        .when(pl.col('signal') + pl.col('signal').shift(1) == -2).then(0.0)
                                        .otherwise(0.0)
                                        .alias('positions')])
    
    

    ml_signal = ml_signal.with_columns([pl.when(pl.col('positions') != 0).then(risk_score).otherwise(0).alias('risk_score')])
     
    return ml_signal

def generate_ML_SMA_cross_over_signals(candle_df : pl.DataFrame ,window ,model_s,model_l):
    
    short_window = 30
    long_window = 60

    short_avg = candle_df['close'].rolling_mean(short_window).fill_null(0).alias('short_mavg')
    short_avg = short_avg.tail(14).to_numpy()
    
    if(short_avg.shape[0] < 14):
        ml_SMA_signals = pl.DataFrame({

            'timestamp': candle_df.item(-1,'timestamp'),
            'signal': 0.0,
            'positions': 0.0,
            'risk_score' :0.0
        })

        return ml_SMA_signals

    short_avg = short_avg.reshape((1,14))
    
    short_avg = short_scaler.fit_transform(short_avg)

    long_avg = candle_df['close'].rolling_mean(long_window).fill_null(0).alias('long_mavg')
    long_avg = long_avg.tail(14).to_numpy()
    long_avg = long_avg.reshape((1,14))
    long_avg = long_scaler.fit_transform(long_avg)
    
    rsi_values = rsi(candle_df).alias('rsi')
    rsi_values = rsi_values.fill_nan(0)
    risk_score = trade_risk(rsi_values)

    ml_SMA_signals = pl.DataFrame(schema={
        'timestamp': str,
        'signal': pl.Float64,
        'positions': pl.Float64,
        'risk_score' : pl.Float64
    })

    temp_dict= {
        'timestamp': candle_df.item(-1,'timestamp'),
        'signal': 0.0,
        'positions': 0.0,
        'risk_score' : 0.0
    }

    short_seq = short_avg
    long_seq = long_avg
    signals = np.ndarray((1,6))
    
    for i in range(6):
        short_seq = short_seq[:,-window:].reshape((1,14))
        

        long_seq = long_seq[:,-window:].reshape((1,14))
        

        short_seq.reshape((1,window,1))
        long_seq.reshape((1,window,1))

        short_out = model_s.predict(short_seq)[0][0]


        out = np.ndarray((1,1))
        out[0,0] = short_out
        short_seq = np.append(short_seq,out,axis= 1)
        
        
        long_out = model_l.predict(long_seq)[0][0]
        out[0,0] = long_out
        long_seq = np.append(long_seq,out,axis= 1)
        
        if (short_out > long_out):
            signals[0,i] = 1.0
            print(f'hit signal: {signals[0,i]}')
        elif(long_out >= short_out):
            signals[0,i] = 0.0
        
        temp_dict['signal'] = signals[0,i]
        temp_frame = pl.DataFrame(temp_dict)
        ml_SMA_signals.extend(temp_frame)
        temp_frame.clear()
        
    

    short_seq = short_scaler.inverse_transform(short_seq[:,-window:])
    print(short_seq)
    
    long_seq = long_scaler.inverse_transform(long_seq[:,-window:])
    print(long_seq)
    ml_SMA_signals = ml_SMA_signals.with_columns([
        pl.when(pl.col('signal').shift(0) - pl.col('signal').shift(1) == 1).then(1.0)
        .when(pl.col('signal').shift(0) - pl.col('signal').shift(1) == -1).then(-1.0)
        .otherwise(0.0).alias('positions')
    ])
    print(ml_SMA_signals)
    ml_SMA_signals = ml_SMA_signals.sum()

    print(ml_SMA_signals)
    temp_dict = ml_SMA_signals.row(-1,named= True)
    temp_dict['timestamp'] = candle_df.item(-1,'timestamp')

    if (temp_dict['positions'] > 0.0):
        temp_dict['positions'] = 1.0
        temp_dict['risk_score'] = risk_score
    elif(temp_dict['positions'] < 0.0):
        temp_dict['positions'] =-1.0
        temp_dict['risk_score'] = risk_score
    elif(temp_dict['positions']==0.0):
        temp_dict['positions'] = 0.0
        temp_dict['risk_score'] = risk_score
    
    temp_frame = pl.DataFrame(temp_dict)

    
    ml_SMA_signals = temp_frame
    print(ml_SMA_signals) 
#    ml_SMA_signals = ml_SMA_signals.with_columns([pl.when(pl.col('positions') != 0).then(risk_score).otherwise(0).alias('risk_score')])
    return ml_SMA_signals

def estimate_parameters(prices):
    """
    Estimate the parameters for the Merton Jump Diffusion Model based on past prices.
    
    :param prices: A Polars Series of Bitcoin prices for the past 60 days.
    :return: Estimated parameters mu, sigma, lambda_j, mu_j, sigma_j.
    """
    log_returns = np.log(prices / prices.shift(1)).drop_nulls()
    mu = log_returns.mean()
    sigma = log_returns.std()
    # For simplicity, we assume some fixed jump parameters
    lambda_j = 0.1  # Example: 10 jumps per year
    mu_j = -0.02  # Example: average jump size is -2%
    sigma_j = 0.05  # Example: standard deviation of jump size is 5%
    
    return mu, sigma, lambda_j, mu_j, sigma_j

def generate_mjdm_signals(prices, window=60, forecast_horizon=7, threshold=0.01):
    """
    Generate buy and sell signals using the Merton Jump Diffusion Model (MJDM).
    
    :param prices: A Polars Series of Bitcoin prices.
    :param window: The number of days to look back for parameter estimation.
    :param forecast_horizon: The number of days to predict into the future.
    :param threshold: The threshold for signal generation.
    :return: A DataFrame with prices, buy signals, and sell signals.
    """
    n = len(prices)
    buy_signals = [0] * n
    sell_signals = [0] * n
    
    for i in range(window, n - forecast_horizon):
        past_prices = prices.slice(i - window, window)
        future_prices = prices.slice(i, forecast_horizon)
        
        mu, sigma, lambda_j, mu_j, sigma_j = estimate_parameters(past_prices)
        
        # Simulate future prices using MJDM
        dt = 1 / 365  # Assume daily data with 252 trading days in a year
        poisson_jumps = np.random.poisson(lambda_j * dt, forecast_horizon)
        jump_sizes = np.random.normal(mu_j, sigma_j, forecast_horizon) * poisson_jumps
        normal_returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), forecast_horizon)
        returns = normal_returns + jump_sizes
        simulated_prices = past_prices[-1] * np.exp(np.cumsum(returns))
        
        # Generate buy/sell signals
        for j in range(forecast_horizon):
            if simulated_prices[j] > future_prices[j - 1] * (1 + threshold):
                buy_signals[i + j] = 1
            elif simulated_prices[j] < future_prices[j - 1] * (1 - threshold):
                sell_signals[i + j] = 1

    signals = pl.DataFrame({
        "prices": prices,
        "buy_signal": buy_signals,
        "sell_signal": sell_signals
    })

    return signals
#######################################################################################################
#Trade processing


def calculate_drawdown(portfolio_values):
    peaks = np.maximum.accumulate(portfolio_values)
    epsilon = 1e-8
    print(peaks)
    drawdowns = (portfolio_values - peaks) / (peaks + epsilon)
    max_drawdown = drawdowns.min()
    return max_drawdown

def get_buy_amount(signal: dict, set_amount =0.5):
    
    
    
    s = signal['risk_score']
    p = signal['positions']
    
    print(f"\ns = {s} and p = {p}\n")
    buy_amount = set_amount

    if ((s > 0.5) & (p == 1)):
        buy_amount = 1 - signal['risk_score']
    elif ((signal['risk_score'] > 0.5) & (signal['positions'] == -1)):
        buy_amount = signal['risk_score']
    elif (signal['positions']==0):
        buy_amount = set_amount
    
    return buy_amount


def trade_proc(df_dict, df_port: pl.DataFrame, signal, amount=0.5, stop_loss=None):
    portfolio = df_port.row(-1, named=True)
    trade = {
        'date': df_dict['timestamp'],
        'price': 0,
        'buy': 0,
        'sell': 0,
        'ID': 0,
        'profit': 0,
        'drawdown': 0,
        'stop_loss_hit': False  # Initialize stop_loss_hit
    }
    current_price = df_dict['close']
    trade['price'] = current_price

    if signal == 1:  # Buy signal
        buy_amount = amount * portfolio['capital'] / current_price
        portfolio['capital'] -= (buy_amount * (current_price * (1 + 0.02)))
        portfolio['asset'] += buy_amount
        portfolio['asset_value'] = portfolio['asset'] * current_price
        portfolio['portfolio_value'] = portfolio['capital'] + portfolio['asset_value']
        portfolio['in_trade'] = True  # Set in_trade to True
        portfolio['date'] = df_dict['timestamp']
        trade.update({'buy': True, 'price': buy_amount * current_price, 'ID': 50})
    elif signal == -1:  # Sell signal
        sell_amount = amount * portfolio['asset']
        portfolio['capital'] += (sell_amount * (current_price * (1 - 0.02)))
        portfolio['asset'] -= sell_amount
        portfolio['asset_value'] = portfolio['asset'] * current_price
        portfolio['portfolio_value'] = portfolio['capital'] + portfolio['asset_value']
        portfolio['in_trade'] = False  # Set in_trade to False
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
            portfolio['in_trade'] = False  # Set in_trade to False
            portfolio['date'] = df_dict['timestamp']
            trade.update({'sell': True, 'price': sell_amount * current_price, 'ID': -200, 'stop_loss_hit': True})  # Update stop_loss_hit
    else:  # Hold
        portfolio['asset_value'] = portfolio['asset'] * current_price
        portfolio['portfolio_value'] = portfolio['capital'] + portfolio['asset_value']
        portfolio['date'] = df_dict['timestamp']
        trade.update({'ID': 100})

    trade['profit'] = portfolio['portfolio_value'] - portfolio['capital']  # Calculate profit
    trade['drawdown'] = calculate_drawdown(np.array([portfolio['portfolio_value']]))  # Calculate drawdown

    return portfolio, trade

#####################################################################################
#Backtesting 

def simulate_stream(price_data : pl.DataFrame, short_ma=30, long_ma=60, stop_loss=None):
    # Initialize data frames
    close_price = price_data.select('close').to_numpy()
    #close_price.reshape((close_price.shape[0],1))

    print("close shape: ",close_price.shape)

    scaled_close = price_scaler.fit_transform(close_price)

    sc_serries = pl.Series('scaled_close',scaled_close)

    price_data = price_data.with_columns(sc_serries.alias('scaled_close'))

    print(price_data)
    
    portfolio1 = init_portofolio()
    portfolio2 = init_portofolio()
    portfolio3 = init_portofolio()


    return_sig = pl.DataFrame(schema={
        'timestamp': str,
        'signal': pl.Float64,
        'positions': pl.Float64,
        'risk_score' : pl.Float64
    })

    trades, signals ,ml_signals = init_data_frames()
    predicted_values = np.zeros((price_data.shape[0],14))

    price_model = load_model('C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\src\\models\\model_3_V3_14_with_epochs_32.keras')
    short_mavg_model = load_model('C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\src\\models\\mavg_models\\model_MAVG_14_with_epochs_56_8H.keras')
    long_mavg_model = load_model('C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\src\\models\\mavg_models\\model_MAVG_L_14_with_epochs_56_8H.keras')


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
    portfolio_dict1 = portfolio1.row(0, named=True)
    portfolio_dict1['portfolio_value'] = portfolio_dict1['capital'] + portfolio_dict1['asset_value']

    buy_signals = []
    sell_signals = []
    drawdown_sell_signals = []
    buy_amount = 0.5

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
        lstm_input = create_sequence(candles,14)
        
        #price sequence prformance
        pred_val = predict_seq(price_model,14,lstm_input,5)
        predicted_values = np.append(predicted_values,pred_val) 
        ml_signals = generate_ML_signals(candles,ml_signals,pred_val)

        ml_signal = ml_signals.row(-1,named=True)
        buy_amount2 = get_buy_amount(ml_signal)
        portfolio_dict2 ,trade2 = trade_proc(candle_dict,portfolio2,ml_signal['positions'], amount= buy_amount2,stop_loss= 0.3)
        
        #simple indicator based strategy

        #moving avererage prediction model
        #pred_avg_signals = generate_ML_SMA_cross_over_signals(candles,14,short_mavg_model,long_mavg_model)
        #print("predicted avg sigs")
        #print(pred_avg_signals)
        #return_sig = return_sig.extend(pred_avg_signals)
        #pred_avg_signal = pred_avg_signals.row(-1,named=True)
        #buy_amount3 = get_buy_amount(pred_avg_signal)
        #portfolio_dict3 ,trade3 = trade_proc(candle_dict,portfolio3,pred_avg_signal['positions'], amount= buy_amount3,stop_loss= stop_loss)


        #################################################################
        
        #signals = generate_signals(candles, short_ma, long_ma)
        #signal = signals.row(-1, named=True)
        
        #print("normal buy")
        #buy_amount = get_buy_amount(signal)

        #portfolio_dict, trade3 = trade_proc(candle_dict, portfolio1, signal['positions'],amount= buy_amount, stop_loss= 0.2)
        #portfolio_dict['date'] = candle_dict['timestamp']
        print(trade2)
        print(TRADE_SCHEMA)
        #current_trade = pl.DataFrame(trade2, TRADE_SCHEMA)
        #trades.extend(current_trade)

        temp_frame = pl.DataFrame(portfolio_dict2, schema=PORTFOLIO_SCHEMA)
        portfolio1.extend(temp_frame)

        ################################################################
        if trade2['buy']:
            buy_signals.append((candle_dict['timestamp'], candle_dict['close']))
        elif trade2['sell']:
            sell_signals.append((candle_dict['timestamp'], candle_dict['close']))
        elif trade2['ID'] == -200:  # Drawdown sell signal
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
    ax2.plot(np.asarray(portfolio1['date'], dtype='datetime64[s]'), portfolio1['portfolio_value'], label="Portfolio Value", color= c)
    #ax1.plot(np.asarray(candles['timestamp'], dtype='datetime64[s]'), signals['short_mavg'], label=f"Short MA ({short_ma})")
    #ax1.plot(np.asarray(candles['timestamp'], dtype='datetime64[s]'), signals['long_mavg'], label=f"Long MA ({long_ma})")

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
    
    print("signals")
    #print(signals.filter(pl.col('positions') != 0))
    #signals.write_csv('signals.csv', separator= ',')

    
    print("trades")
    #print(trades)
    

    folder = 'fig'
    portfolio_plot = f"portfolio_performance_sma_{short_ma}_lma_{long_ma}_ML.png"
    file_path = os.path.join(folder, portfolio_plot)

    if not os.path.isdir(folder):
        os.mkdir(folder)

    plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Set the DPI for higher resolution and use bbox_inches='tight' to ensure the plot is saved without extra whitespace
    print(f"Saved plot as {portfolio_plot}")
    plt.close()

    portfolio_values = portfolio1['portfolio_value'].to_numpy()
    performance['num_trades'] = signals.filter((pl.col('positions') == 1) | (pl.col('positions') == -1)). shape[0]
    performance['performance'] = portfolio1['portfolio_value'].item(-1) / portfolio1['portfolio_value'].item(1)
    performance['max_drawdown'] = calculate_drawdown(portfolio_values)

    folder = 'data'
    performance_file = f"performance_sma_{short_ma}_lma_{long_ma}_ML.json"
    prediction_file = f"predicted values from model.csv"
    
    file_path = os.path.join(folder, performance_file)
    file2_path = os.path.join(folder,prediction_file)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(file_path, 'w') as pf:
        json.dump(performance, pf)
    with open(file2_path,'w') as pf:
        predicted_values.tofile(pf,sep= ',',format='%s')
    print(f"Saved performance as {performance_file}")


    print("predicted average signal")
    print(return_sig.filter(pl.col('positions')== 1.0))
    print(return_sig.filter(pl.col('positions') == -1))

    print('ml_signals')
    print(ml_signals.filter(pl.col('positions') == 1.0))
    print(ml_signals.filter(pl.col('positions') == -1.0))
    
    return portfolio1, trades, signals, performance


#Main usage
#####################################################################################


if __name__ == "__main__":
    # Load historical price data
    FILE_LIST = ['C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2019',
                   'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2020',
                   'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2021',
                   'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2022',
                   'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2023']
    
    model_path = 'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\src\\models'
    price_data_D = load_and_combine_csv(FILE_LIST,'2022_dayly_price.csv')
    print(price_data_D)
    price_data_8H = load_and_combine_csv(FILE_LIST,'2023_8_hour_price.csv')
    print(price_data_8H)
    price_data_4H = load_and_combine_csv(FILE_LIST,'_4_hour_price.csv')
    print(price_data_4H)

    prices = price_data_D['close']
    
    x1,x2,x3,x4 = simulate_stream(price_data_8H)

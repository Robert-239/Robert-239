from __future__ import ( absolute_import,division,print_function,unicode_literals)

from backtrader import backtrader as bt

from matplotlib import animation
from pandas.core import window
import polars as pl
import json
import time
import datetime as dt
import pandas as pd
from sqlalchemy import update
from tqdm import tqdm
from matplotlib import pyplot as plt

import numpy as np
import os
import math as math
from typing import List





PORTIFOLIO_SCHEMA = {
    'date' : str,
    'capital' : pl.Float64,
    'asset' : pl.Float64,
    'asset_value' : pl.Float64,
    'portifolio_value' : pl.Float64,
}
portifolio = {
    'date' : str,
    'capital' : pl.Float64,
    'asset' : pl.Float64,
    'asset_value' : pl.Float64,
    'portifolio_value' : pl.Float64,

}
TRADE_SCHEMA = {
    'date' : str,
    'price' : pl.Float64,
    'buy' : bool,
    'sell' : bool,
    'ID' : int,
}
trade = {
    'date' : str,
    'price' : pl.Float64,
    'buy' : bool,
    'sell' : bool,
    'ID' : int,

}
SIGNAL_SCHEMA = {
    'timestamp' : str,
    'price' : pl.Float64,
    'signal' : pl.Int8,
    'position' : pl.Int8
}

signal = {
    'timestamp' : str,
    'price' : pl.Float64,
    'signal' : pl.Int8,
    'position' : pl.Int8
}

CANDLE_SCHEMA = {

    'timestamp': str,
    'open': pl.Float64,
    'close': pl.Float64,
    'high': pl.Float64,
    'low': pl.Float64,
    'volume': pl.Float64,
}
present_time = dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

#List of file paths

FLIST2 = ['C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2019',
          'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2020',
          'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2021',
          'C:\\Users\\rober\\git\\Robert-239\\Skripsie\\trading_bot\\data\\2022']

#Creating the merged data frame


def load_and_combine_csv(folders: List[str], file_pattern: str = '*.csv') -> pl.DataFrame:
    """
    Load CSV files from specified folders and combine them into a single Polars DataFrame.
    Remove duplicate rows based on all columns.

    :param folders: List of folder paths to search for CSV files.
    :param file_pattern: Pattern for CSV file names. Default is '*.csv'.
    :return: A combined Polars DataFrame with duplicates removed.
    """
    combined_df = None

    for folder in folders:
        # Get all CSV files in the folder matching the pattern
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        
        # Load each CSV file and combine them into a single DataFrame
        for csv_file in csv_files:
            file_path = os.path.join(folder, csv_file)
            df = pl.read_csv(file_path)
            
            
            if combined_df is None:
                combined_df = df
                
            else:
                combined_df = combined_df.extend(df)
                
    # If combined_df is still None, it means no CSV files were found
    if combined_df is None:
        raise ValueError("No CSV files found in the specified folders.")

    # Remove duplicates from the combined DataFrame
    combined_df = combined_df.unique(subset=["timestamp"],maintain_order=True)

    return combined_df


def proc_data(file)->list[str]:
    '''
    This function takes the temporary price data file as argument and parses the data in the file into a list of strings 
    in the same structure as a JSON string,
    '''
    ch = ''
    candle_data = []
    tempStr = ''
    f = open(file,'r')

    ch = f.read(1)    
    while ch != '[':
        tempStr = tempStr + ch
        ch = f.read(1)
    Header = tempStr.strip('{').split(',')
    tempStr = ' '
 
    while ch != ']':
        ch = f.read(1)
        if ch == '{':
            tempStr = tempStr + ch
            while ch != '}':
                ch = f.read(1)
                tempStr = tempStr + ch
                
            candle_data.append(tempStr)
        
        tempStr = ''
    f.close
    return candle_data



def parse_strings_to_dataframe(s):
    '''
    This function takes a list of strings as argument and formats it to the correct
    JSON format using the replace() function to change the ' qoutes to " qoutes.
    Next it usese the json.loads() method to create a dictionary with key value pairs
    which is necesarry for further data clean up.

    The key-value pairs are then formated from string to the correct data type and stored
    into the corresponding list.

    Finally the data frame is created using the polars library. The schema is defined for the 
    expected input for the function and the lists are added to the data frame.
    '''
    timestamp = []
    open = []
    close = []
    high = []
    low = []
    volume = []
    for string in s:
        dicts = json.loads(string.replace("'",'"'))

        timestamp.append(dt.datetime.fromtimestamp(float(dicts['timestamp'])/1000).strftime('%Y-%m-%d %H-%M-%S'))
        open.append(float(dicts['open']))
        close.append(float(dicts['close']))
        high.append(float(dicts['high']))
        low.append(float(dicts['low']))
        volume.append(float(dicts['volume']))
    df = pl.DataFrame((timestamp,open,close,high,low,volume),schema= CANDLE_SCHEMA)
    return df

def init_data_frames() :
    timestamp = [0]
    #timestamp.append(present_time)
    capital = [100000]
    asset = [0]
    asset_value = [0]
    portifolio_value =[0]

    portifolio_df = pl.DataFrame((timestamp,capital,asset,asset_value,portifolio_value),schema= PORTIFOLIO_SCHEMA)
    trades = pl.DataFrame(schema=TRADE_SCHEMA)
    signals = pl.DataFrame(schema= SIGNAL_SCHEMA)
    
    return portifolio_df , trades ,signals 

def load_data(s):
    price_data = pl.read_csv(s,separator=',')
    return price_data


portifolio_df , trades , signals = init_data_frames()

data = 'data/2023/2023_dayly_price.csv'

price_data_test = load_data(data)

price_data = load_and_combine_csv(FLIST2)


print(price_data.slice(1,1))
print(trades)
print(signals)

def brownian_motion_model():

    S0 = 100        #asset start price
    mu = 0.05       #expected return
    sigma = 0.2     #volatility
    T = 1.0         #time period in years
    d_t = 1/252      #timestep assuming 252 trading days
    N = int(T/d_t)   #total number of time steps

    np.random.seed(int(dt.datetime.now().timestamp()))
    dW = np.random.normal(0, np.sqrt(d_t),N) # increments for Wt

    W = np.cumsum(dW)

    # Simulate stock price path

    St = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0,T,N) + sigma*W)

    return St


def mertons_jump_diffusion(S0, T, r, sigma, lamb, mu_jump, sigma_jump, n_simulations, n_steps):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    price_paths = np.zeros((n_simulations, n_steps + 1))
    price_paths[:, 0] = S0

    for i in range(1, n_steps + 1):
        # Generate random normal variables for diffusion and jumps
        dW = np.random.normal(0, np.sqrt(dt), size=n_simulations)
        dN = np.random.poisson(lamb * dt, size=n_simulations)

        # Calculate jumps
        jumps = np.random.normal(mu_jump * dt, sigma_jump * np.sqrt(dt), size=n_simulations)
        jumps[dN == 0] = 0

        # Update prices
        price_paths[:, i] = price_paths[:, i - 1] * (
        
                    1 + r * dt + sigma * dW + jumps)

    return t, price_paths


def trade_proc(df_dict,signal):

    portifolio = {
        'date' : str,
        'capital' : pl.Float64,
        'asset' : pl.Float64,
        'asset_value' : pl.Float64,
        'portifolio_value' : pl.Float64,
    }
    trade = {
        'date' : str,
        'price' : pl.Float64,
        'buy' : bool,
        'sell' : bool,
        'ID' : int,

    }
    trade['date'] = df_dict['timestamp']
    trade['ID'] = 0
    trade['buy'] = 0
    trade['sell'] = 0
    trade['price'] = 0

    portifolio = portifolio_df.row(-1,named=True)
    
    if signal == 1:     #Buy signal
        print("\033[1;32m processing buy trade")
        print(f"date : {df_dict['timestamp']}")
        trade['buy'] = True
        trade['sell'] = False
        print(f"capital = {portifolio['capital']}")
        print(f"close price = {df_dict['close']}")
        buy_ammount = (0.5*portifolio['capital']/df_dict['close'])
        print(f"amount to purchase :{buy_ammount}")
        portifolio['capital'] = portifolio['capital'] - ((buy_ammount)* (df_dict['close'] + 0.02))
        portifolio['asset'] = portifolio['asset'] + buy_ammount
        portifolio['asset_value'] = portifolio['asset'] * df_dict['close']
        portifolio['portifolio_value'] = portifolio['capital'] + portifolio['asset_value']
        portifolio['date'] = df_dict['timestamp']

        trade['date'] = df_dict['timestamp']
        trade['price'] = buy_ammount * df_dict['close']
        trade['ID'] = 50
        print("\033[1;32m finished processing buy trade \033[0m")

    elif signal == -1:  #Sell signal
        trade['sell'] = True
        trade['buy'] = False
        
        print("\033[1;31m processing buy trade")
        print(f"date : {df_dict['timestamp']}")
        print(f"capital = {portifolio['capital']}")
        print(f"close price = {df_dict['close']}")
        
        sell_amount = 0.5*portifolio['asset']
        print(f"amount to sell :{sell_amount}")
        portifolio['capital'] = portifolio['capital'] + ((sell_amount)* (df_dict['close'] - 0.02))
        portifolio['asset'] = portifolio['asset'] - sell_amount
        portifolio['asset_value'] = portifolio['asset'] * df_dict['close']
        portifolio['portifolio_value'] = portifolio['capital'] + portifolio['asset_value']
        portifolio['date'] = df_dict['timestamp']
        trade['ID'] = -100
        trade['date'] = df_dict['timestamp']
        trade['price'] = sell_amount * df_dict['close']       
        print("finished processing buy trade \033[0m")

    elif (signal == 0) | (signal == None):
        
        portifolio['asset_value'] = portifolio['asset'] * df_dict['close']
        portifolio['portifolio_value'] = portifolio['capital'] + portifolio['asset_value']
        portifolio['date'] = df_dict['timestamp']
        trade['date'] = df_dict['timestamp']
        trade['ID'] = 100
        trade['buy'] = 0
        trade['sell'] = 0
        trade['price'] = 0
    
    
    return portifolio , trade

#Indicators
###############################################################################################
def rsi(df : pl.DataFrame, window = 14) -> float:
    pg =0.0000000000001
    pl =-0.0000000000001
    df_shape = df.shape[0]
    if df_shape >= 14 :
        df_window  = df.slice(df_shape - window ,window)
        dct = df_window.to_dict(as_series=False)
        cls = dct['close']
        for i in range((window -1)):
            
            if cls[i+1] > cls[i]:
                pg =pg+ ((cls[i+1]/cls[i]) - 1)
                
                
            else:
                pl =pl + ((cls[i+1]/cls[i]) - 1)
                
                
    
    apg = pg/window
    apl = -pl/window


    rsi_1 = 100 - (100/(1+(apg/apl)))

    return rsi_1

def fvg(df : pl.DataFrame, gapr = 0.8):
    
    i = df.shape[0] - 2

    gaps = []
    if i > 3:
        prev_candle = df.row((i - 1) ,named= True)
        next_candle = df.row((i +1) , named= True)
        cur_candle = df.row(i , named= True)

        candle_body_ratio = (cur_candle['open']-cur_candle['close'])/(cur_candle['high']-cur_candle['low'])
        candle_body_ratio = np.abs(candle_body_ratio)
        
        if(candle_body_ratio >= gapr):
            gap = np.abs(prev_candle['low'] - next_candle['high'])
            gaps.append((i-1,i,gap))
        else:
            gaps.append(0)

    return gaps




###############################################################################################

def simpleMA(df : pl.DataFrame , short_window = 30, long_window = 100) -> pl.DataFrame:
    signal = pl.DataFrame({'date' : df['timestamp'],
                           'signal' : 0.0,
                           'positions' : 0.0}) 
    short_avg = pl.Series(df['close']).rolling_mean(short_window).fill_null(0).alias('short_ma')
    long_avg = pl.Series(df['close']).rolling_mean(long_window).fill_null(0).alias('long_ma')
    
    signal = signal.with_columns([
                                 pl.lit(short_avg).alias('short_avg'),
                                 pl.lit(long_avg).alias('long_avg')]
                                 )
    if df.shape[0] >= short_window:
        signal = signal.with_columns([
                                     pl.lit(short_avg).alias('short_avg'),
                                     pl.lit(long_avg).alias('long_avg')]
                                     )
        signal = signal.with_columns( pl.when(pl.col('short_avg') > pl.col('long_avg'))
                                     .then(1.0)
                                     .otherwise(0).alias('signal')
                                     )

        signal = signal.with_columns(
            pl.when(pl.col('signal').shift(0) - pl.col('signal').shift(1) == 1).
            then(1.0).
            when(pl.col('signal').shift(0) - pl.col('signal').shift(1) == -1).then(-1.0).
            otherwise(0.0).alias('positions'),
        )


                                 
    return signal 

###############################################################################################

def simulate_stream(price_data,portfolio_DF):
    candles = pl.DataFrame(schema=CANDLE_SCHEMA)
    current_trade = pl.DataFrame(schema = TRADE_SCHEMA)
    candle_dict= {
        'timestamp': str,
        'open': pl.Float64,
        'close': pl.Float64,
        'high': pl.Float64,
        'low': pl.Float64,
        'volume': pl.Float64,
    }
    
    portifolio_dict = {
        'date' : pl.Time,
        'capital' : pl.Float64,
        'asset' : pl.Float64,
        'asset_value' : pl.Float64,
        'portifolio_value' : pl.Float64,

    }
    len_df = price_data.shape[0]
    rsi1 =[]
    gaps = []
    portifolio_dict = portifolio_df.row(0 , named=True)
    print(portifolio_dict)
    portifolio_dict['portifolio_value'] = portifolio_dict['capital'] + portifolio_dict['asset_value']
    print(portifolio_dict)
    price , ax = plt.subplots()
    price.suptitle("Bitcoin price movement")
    for i in tqdm(range(len_df),desc="Processing Historical data stream",leave= True):
        if i != (len_df - 1):
            ax.clear()

        candle_dict = price_data.row(i,named = True)
        
        candle_dict['timestamp'] = (dt.datetime.strftime((dt.datetime.strptime(candle_dict['timestamp'],"%Y-%m-%d %H-%M-%S")),"%Y-%m-%d"))
        temp_frame = pl.DataFrame(candle_dict,schema= CANDLE_SCHEMA)
        

        candles.extend(temp_frame)
        signals = simpleMA(candles)

        
        signal = signals.row(-1,named= True)
        

        
        
        portifolio_dict,trade = trade_proc(candle_dict,signal['positions'])
        portifolio_dict['date'] = candle_dict['timestamp']
        

        current_trade = pl.DataFrame(trade,TRADE_SCHEMA)
        trades.extend(current_trade)

        rsi_temp = rsi(candles,window = 14)
        gaps.append(fvg(candles))
        rsi1.append(rsi_temp)   
        temp_frame = pl.DataFrame(portifolio_dict,schema= PORTIFOLIO_SCHEMA)
        portfolio_DF.extend(temp_frame)
        #temp_frame = pl.DataFrame(trade,schema=TRADE_SCHEMA)    
        #trades.extend(temp_frame)




        
        ax.set_xlabel('Closing Price')
        ax.set_ylabel('Day')
        ax.grid(True)
        ax.label_outer(True)
        ax.plot(np.asarray( candles['timestamp'],dtype='datetime64[s]'),candles['close'] , label = "bitcoin price")
        ax.plot(np.asarray(portfolio_DF['date'],dtype='datetime64[s]') ,portfolio_DF['portifolio_value'], label = "portfolio value")
        ax.legend()
        ax.tick_params(axis='x',labelrotation=90)

        
        temp_frame.clear()


        plt.pause(0.01)
        #plt.plot(data = animation,label='Closing price' )
        if i == len_df - 1:
            print('\nLast index')
            plt.show()
        #price_plot.show()
        
        time.sleep(0.01)
    
    plt.plot(rsi1)
    plt.show()
    plt.plot(signals['short_avg'])
    plt.plot(signals['long_avg'])
    plt.show()

    print(signals.filter((pl.col('positions') ==1) | (pl.col('positions') == -1)))
    print(trades)
    print(portfolio_DF)
    
    

# Example usage
S0 = 100  # Initial price
T = 1  # Time horizon
r = 0  # Risk-free rate (not considered in this model)
sigma = 0.2  # Volatility
lamb = 0.1  # Jump frequency
mu_jump = 0.05  # Mean of jump size
sigma_jump = 0.1  # Volatility of jump size
n_simulations = 10  # Number of simulations
n_steps = 1000  # Number of time steps


candle_list = proc_data('Historic_Price_Data.csv')
candles = parse_strings_to_dataframe(candle_list)

simulate_stream(price_data,portifolio_df)

    

def test_strategy(df : pl.DataFrame):
    pass

plt.legend()
plt.show

import os
import datetime as dt
import time 
from luno_python.client import Client
import json
import requests
import polars as pl
'''
The luno api uses a  client name and secret for Authentication
name -> Key ID 
Secret -> password
Because I am focusing on bitcoin it is the only pair I am going to use and test
'''
#constants and environment variables
key_id = 'pgdzch7f8x5p'
key_secret = 'BdAy8LI5W9jYCk_w_eORMO7bGUVLvXlsCaigVmDDAqg' 
url = 'https://api.luno.com/api/'
stream_url = 'wss://ws.luno.com/api/1/stream/'
PAIR = 'XBTZAR'


c = Client(api_key_id= key_id,api_key_secret= key_secret)
try:
    res = c.get_ticker(pair='XBTZAR')
    print(res)
except Exception as e :
    print(e)

def get_historical_price_data():

    '''
    The function takes the output of the exchange API for the price data that is saved as a string in a file. The file is parsed 
    and the candle stick data for each time step is saved in a list and then saved into a temporary CSV file
    '''
    with open('Historic_Price_Data.csv','w') as f:
        
        
        try:
            f.writelines(
                   str(c.get_candles(300,PAIR,1601372760520))
            )
        except Exception as e :
            print(e)
        
        f.close
        return f

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
    print(Header)
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


print(dt.datetime.fromtimestamp(float(1711471753513/1000)).strftime('%Y-%m-%d %H-%M-%S'))
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
    df = pl.DataFrame((timestamp,open,close,high,low,volume),schema ={

        'timestamp': pl.Time,
        'open': pl.Float64,
        'close': pl.Float64,
        'high': pl.Float64,
        'low': pl.Float64,
        'volume': pl.Float64

    })
    return df
def parse_stream(stream_res):
    timestamp = []
    rolling_24_hour_volume = []
    bid =[]
    ask = []
    last_trade =[]
    dicts = stream_res
    pair = []
    status = []


    pair.append(dicts['pair'])
    timestamp.append(dt.datetime.fromtimestamp(float(dicts['timestamp'])/1000).strftime('%Y-%m-%d %H-%M-%S'))
    bid.append(float(dicts['bid']))
    ask.append(float(dicts['ask']))
    last_trade.append(float(dicts['last_trade']))
    status.append(dicts['status'])
    rolling_24_hour_volume.append(float(dicts['rolling_24_hour_volume']))

    stream_df = pl.DataFrame((pair,timestamp,bid,ask,last_trade,rolling_24_hour_volume,status
                              ),schema={
                                        'pair':str,
                                        'timestamp': pl.Time,
                                        'bid': pl.Float64,
                                        'ask': pl.Float64,
                                        'last_trade': pl.Float64,
                                        'rolling_24_hour_volume': pl.Float64,
                                        'status':str

    })

    timestamp.clear()
    rolling_24_hour_volume.clear()
    bid.clear()
    ask.clear()
    last_trade.clear()
    return stream_df
    


get_historical_price_data()
price_data = proc_data('Historic_Price_Data.csv')
formated_price_data = parse_strings_to_dataframe(price_data)
print(formated_price_data)
print(formated_price_data)

while True:
    try:
        res = c.get_ticker(pair='XBTZAR')
        stream_data_frame = parse_stream(res)

        os.system('cls' if os.name == 'nt' else 'clear')
        print(stream_data_frame)
        time.sleep(0.5)
        
    except Exception as e :
        print(e)
    



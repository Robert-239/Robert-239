import polars as pl
import json
import backtesting
import datetime as dt

PORTIFOLIO_SCHEMA = {
    'date' : pl.Time,
    'capital' : pl.Float64,
    'asset' : pl.Float64,
    'asset_value' : pl.Float64,
    'portifolio_value' : pl.Float64,
}

TRADE_SCHEMA = {
    'date' : pl.Time,
    'price' : pl.Float64,
    'buy' : bool,
    'sell' : bool,
    'ID' : str,
}

SIGNAL_SCHEMA = {
    'timestamp' : pl.Time,
    'price' : pl.Float64,
    'signal' : pl.Int8,
    'position' : pl.Int8
}

CANDLE_SCHEMA = {

    'timestamp': pl.Time,
    'open': pl.Float64,
    'close': pl.Float64,
    'high': pl.Float64,
    'low': pl.Float64,
    'volume': pl.Float64,
}
present_time = dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
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
    timestamp = []
    timestamp.append(present_time)
    capital = [100000]
    asset = [0]
    asset_value = [0]
    portifolio_value =[0]

    portifolio = pl.DataFrame((timestamp,capital,asset,asset_value,portifolio_value),schema= PORTIFOLIO_SCHEMA)
    trades = pl.DataFrame(schema=TRADE_SCHEMA)
    signals = pl.DataFrame(schema= SIGNAL_SCHEMA)
    
    return portifolio , trades ,signals 



portifolio , trades , signals = init_data_frames()

print(portifolio)
print(trades)
print(signals)


candle_list = proc_data('Historic_Price_Data.csv')
candles = parse_strings_to_dataframe(candle_list)



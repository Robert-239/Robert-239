import pandas as pd 
import yfinance as yf 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def identify_FVG(data, lookback_period = 15 , body_multiplier = 1.5):

    """
    Parameters:
        data (DataFrame): DataFrame with columns ['open', 'close','high','low'].
        lookback_period (int): The amount of candles the function will look at to calculate fair value gaps.
        body_multiplier (float): Multiplier to determine significant candle body size.

    Returns:
        List of tuples: Each tuple contains ('type', start, end, index)

    """

    fvg_list = [None, None]

    for i in range(2, len(data)):
        first_high = data['High'].iloc[i-2]
        first_low = data['Low'].iloc[i-2]
        middel_open = data['Open'].iloc[i-1]
        middel_close = data['Close'].iloc[i-1]
        third_high = data['High'].iloc[i]
        third_Low = data['Low'].iloc[i]

        prev_bodies = (data['Close'].iloc[max(0, i - 1 - lookback_period):i-1]-
                        data['Open'].iloc[max( 0, i - 1 - lookback_period):i-1]).abs()

        avg_body_size = prev_bodies.mean()

        avg_body_size = avg_body_size if avg_body_size > 0 else 0.001

        middel_body = middel_close - middel_open

        # Bullish fvg
        if third_Low > first_high and middel_body > avg_body_size * body_multiplier :
            fvg_list.append(('bullish',first_high,third_Low,i))

        #Bearish fvg
        elif third_high < first_low and middel_body > avg_body_size * body_multiplier:
            fvg_list.append(('bearish',first_low,third_high , i))

        else:
            fvg_list.append(None)


    return fvg_list




#test 

data = yf.download('BTC-USD',period= '7D',interval= '15m')

clean_data = data.drop(columns =['Adj Close','Volume']).copy()

gaps = identify_FVG(data = clean_data)

df2 = pd.DataFrame(gaps)

clean_data = pd.concat([clean_data,df2], axis= 1)

print(clean_data.info())

# create figure
fig = go.Figure()

# add candle stick chart

fig.add_trace(go.Candlestick(
              x = clean_data.index,
              open = clean_data['Open'],
              high = clean_data['High'],
              low = clean_data['Low'],
              close = clean_data['Close'],
              name = "Candles"))



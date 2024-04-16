import datetime as dt
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from polars.functions import aggregation, eager


S0 = 100        #asset start price
mu = 0.05       #expected return
sigma = 0.2     #volatility
T = 1.0         #time period in years
d_t = 1/252      #timestep assuming 252 trading days
N = int(T/d_t)   #total number of time steps

# Define a function to implement moving average crossover strategy
def moving_average_crossover_strategy(data, short_window, long_window):
    signals = data.select('date').clone()
    signals = pl.DataFrame({'date' : data['date']})
    
    # Calculate short simple moving average over the short window
    short_mavg = pl.Series(data['Close']).rolling_mean(short_window).fill_null(pl.lit(0)).alias('short_mavg').to_list()
    #signals = signals.with_columns(short_mavg,"short_mavg")
    # Calculate long simple moving average over the long window
    long_mavg = pl.Series(data['Close']).rolling_mean(long_window).fill_null(pl.lit(0)).alias('long_mavg').to_list()
    #signals = signals.with_columns("long_mavg",short_mavg)
    
    # Generate signals
    signals = signals.with_columns([pl.lit(short_mavg).alias('short_mavg') , pl.lit(long_mavg).alias('long_mavg')])
    signals = signals.with_columns(signal = pl.when(pl.col('short_mavg') > pl.col('long_mavg')).then(1.0).otherwise(0.0).alias('signal'))


    # Generate trading orders
    signals = signals.with_columns(
        positions=pl.when(pl.col('signal').shift(-1) - pl.col('signal') == 1).then(1.0).otherwise(
            pl.when(pl.col('signal').shift(-1) - pl.col('signal') == -1).then(-1.0).otherwise(0.0)
        ).alias("positions"),
    )
    
    
    return signals

# Backtest function
def backtest_strategy(data, signals, initial_cash=100000):
    portfolio = pl.DataFrame({'date': signals['date'],'positions':signals['positions']})
    # Buy/sell signals
# Initialize positions with no money
    portfolio = portfolio.with_columns(
        pl.lit(initial_cash).alias('cash')
    )
    portfolio = portfolio.with_columns(pl.lit(data['Close']).alias('Asset_value'))
    # Buy when signal is 1, sell when signal is -1
    portfolio = portfolio.with_columns(
        (pl.when(pl.col('positions') == 1).then(pl.col('cash') // pl.col('Asset_value')).otherwise(0)).alias('stock')
    )
    portfolio = portfolio.with_columns(
        (pl.when(pl.col('positions') == 1).then(pl.col('stock') * pl.col('Asset_value')).otherwise(0)).alias('stock_value')
    )
    print(portfolio)
    portfolio = portfolio.with_columns(
        pl.when(pl.col('positions') == 1).then(pl.col('cash') - ((pl.col('stock') * pl.col('Asset_value')))).otherwise(
            pl.when(pl.col('positions') == -1).then(pl.col('cash') + (pl.col('stock') * pl.col('Asset_value').shift(1))).otherwise(pl.col('cash'))
        )
    )
    close = portfolio.get_column('stock').to_list()
    print(close)
    print(portfolio.get_column('stock_value').to_list())
    print(portfolio.filter(pl.col('positions')==1))
    portfolio = portfolio.with_columns((
        pl.col('cash') + pl.col('stock_value')).alias(
        'total')
    )
    portfolio = portfolio.with_columns(date = pl.col('date').dt.strftime('%Y-%m-%d').alias('date'))
    return portfolio

# Create Polars DataFrame (you can replace this with your own data loading method)
# For simplicity, let's create some random data
#np.random.seed(42)

returns = np.random.normal(0, 1,252)
price_list = 100* np.exp(np.cumsum(returns))
#Geometric brownian motion for price model

#Params for GBM price model


np.random.seed(int(dt.datetime.now().timestamp()))
dW = np.random.normal(0, np.sqrt(d_t),N) # increments for Wt

W = np.cumsum(dW)

# Simulate stock price path

St = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0,T,N) + sigma*W)

print(price_list)
start_date = dt.date(2020,1,1)
end_date = start_date + dt.timedelta(251)
date_list = [start_date + dt.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
data = pl.DataFrame({
    'date': date_list,
    'Close': St
})
# Define parameters
short_window = 40
long_window = 100

# Run strategy
signals = moving_average_crossover_strategy(data, short_window, long_window)
portfolio = backtest_strategy(data, signals)
print(portfolio.slice(35,15))

print(portfolio.filter((pl.col('positions')==1) | ( pl.col('positions') == -1)))
# Plot
plt.figure(figsize=(14, 7))

# Plot close price and moving averages
plt.plot(data['date'] ,data['Close'], label='Close Price')
plt.plot(signals['date'],signals['short_mavg'], label='40-Day SMA')
plt.plot(signals['date'],signals['long_mavg'], label='100-Day SMA')
plt.scatter(signals.filter(pl.col('positions') == 1)['date'].to_numpy(), signals.filter(pl.col('positions') == 1)['positions'].to_numpy(), marker='^', color='g', label='Buy Signal')
plt.scatter(signals.filter(pl.col('positions') == -1)['date'].to_list(), signals.filter(pl.col('positions') == -1)['positions'].to_list(), marker='v', color='r', label='Sell Signal')
plt.legend()

plt.show()

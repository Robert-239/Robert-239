import numpy as np
import matplotlib.pyplot as plt
import polars as pl

# Define schema for the data DataFrame
data_schema = {
    'date': pl.Date,
    'Close': pl.Int32,
}

# Define schema for the signals DataFrame
signals_schema = {
    'date': pl.Date,
    'short_mavg': pl.Float64,
    'long_mavg': pl.Float64,
    'signal': pl.Float64,
    'positions': pl.Float64,
}

# Define schema for the portfolio DataFrame
portfolio_schema = {
    'date': pl.Date,
    'positions': pl.Float64,
    'cash': pl.Float64,
    'stock': pl.Float64,
    'stock_value': pl.Float64,
    'total': pl.Float64,
}

# Define a function to implement moving average crossover strategy
def moving_average_crossover_strategy(data, short_window, long_window):
    signals = pl.DataFrame(data).select(['date']).clone()
    signals = signals.lazy()
    
    # Create short simple moving average over the short window
    signals = signals.with_columns(
        pl.col('Close').rolling_mean(window_size=short_window).over("window").alias("short_mavg"),
    )

    # Create long simple moving average over the long window
    signals = signals.with_columns(
        pl.col('Close').rolling_mean(window_size=long_window).over("window").alias("long_mavg"),
    )

    # Generate signals
    signals = signals.map(
        pl.col('signal') = pl.when(pl.col('short_mavg') > pl.col('long_mavg')).then(1.0).otherwise(0.0)
    )

    # Generate trading orders
    signals = signals.map_batches(
        lambda : pl.when(pl.col('signal').shift(-1) - pl.col('signal') == 1).then(1.0).otherwise(
            pl.when(pl.col('signal').shift(-1) - pl.col('signal') == -1).then(-1.0).otherwise(0.0)
        )
    )

    return signals

# Backtest function
def backtest_strategy(data, signals):
    portfolio = pl.DataFrame(signals).select(['date']).clone()

    # Buy/sell signals
    portfolio = portfolio.with_column(signals.select('positions'))

    # Initialize positions with no money
    portfolio = portfolio.with_column(
        pl.col('cash').fill_null(100000)
    )

    # Buy when signal is 1, sell when signal is -1
    portfolio = portfolio.with_column(
        pl.when(pl.col('positions') == 1).then(pl.col('cash') // pl.col('Close')).otherwise(0),
        'stock'
    )

    portfolio = portfolio.with_column(
        pl.when(pl.col('positions') == 1).then(pl.col('stock') * pl.col('Close')).otherwise(0),
        'stock_value'
    )

    portfolio = portfolio.with_column(
        pl.when(pl.col('positions') == 1).then(pl.col('cash') - (pl.col('stock') * pl.col('Close'))).otherwise(
            pl.when(pl.col('positions') == -1).then(pl.col('cash') + (pl.col('stock') * pl.col('Close'))).otherwise(pl.col('cash'))
        ),
        'cash'
    )

    portfolio = portfolio.with_column(
        pl.col('cash') + pl.col('stock_value'),
        'total'
    )

    return portfolio

# Create Polars DataFrame (you can replace this with your own data loading method)
# For simplicity, let's create some random data
np.random.seed(42)
dates = pl.datetimes(['2020-01-01'] * 252)
data = pl.DataFrame({
    'date': dates,
    'Close': np.random.randint(1, 101, size=252)
}, schema=data_schema)

# Define parameters
short_window = 40
long_window = 100

# Run strategy
signals = moving_average_crossover_strategy(data, short_window, long_window)
portfolio = backtest_strategy(data, signals)

# Plot
plt.figure(figsize=(14, 7))

# Plot close price and moving averages
plt.plot(data['Close'], label='Close Price')
plt.plot(signals['short_mavg'], label='40-Day SMA')
plt.plot(signals['long_mavg'], label='100-Day SMA')
plt.scatter(signals.filter(pl.col('positions') == 1)['date'], signals.filter(pl.col('positions') == 1)['short_mavg'], marker='^', color='g', label='Buy Signal')
plt.scatter(signals.filter(pl.col('positions') == -1)['date'], signals.filter(pl.col('positions') == -1)['short_mavg'], marker='v', color='r', label='Sell Signal')
plt.legend()

plt.show()


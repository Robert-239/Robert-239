import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Merton's Jump-Diffusion Model parameters
S0 = 100  # Initial price
mu = 0.05  # Drift
sigma = 0.2  # Volatility
jump_intensity = 0.1  # Expected number of jumps per year
jump_mean = 0.1  # Average jump magnitude (log-scale)
jump_std = 0.2  # Standard deviation of jump magnitude (log-scale)

# Simulation parameters
T = 1  # Time period (1 year)
N = 252  # Number of trading days

# Simulate prices using Merton's Jump-Diffusion Model
dt = T / N
t = np.linspace(0, T, N -1 )
W = np.random.normal(0, np.sqrt(dt), N - 1)  # Brownian motion
print(W.shape)
# Jump process (Poisson process)
jump_times = np.random.poisson(jump_intensity * T, N - 1)  # Occurrence of jumps
jump_sizes = np.random.normal(jump_mean, jump_std, N - 1)  # Jump magnitudes

# Cumulative jumps
cumulative_jumps = np.cumsum(jump_times * jump_sizes)

# Simulate the price path
log_returns = (mu - 0.5 * sigma**2) * dt + sigma * W + cumulative_jumps * jump_times
price_path = S0 * np.exp(np.cumsum(log_returns))
print(price_path)
# Create a DataFrame with price data
price_df = pd.DataFrame({'Price': price_path}, index=t)

# Calculate moving averages
short_term = 20  # Short-term moving average period
long_term = 50  # Long-term moving average period

price_df['Short_MA'] = price_df['Price'].rolling(short_term).mean()
price_df['Long_MA'] = price_df['Price'].rolling(long_term).mean()

# Identify moving average crossovers
price_df['Signal'] = 0
price_df.loc[price_df['Short_MA'] > price_df['Long_MA'], 'Signal'] = 1  # Buy signal
price_df.loc[price_df['Short_MA'] < price_df['Long_MA'], 'Signal'] = -1  # Sell signal

# Plot the price path with moving averages and signals
plt.figure(figsize=(12, 6))
plt.plot(price_df['Price'], label='Price')
plt.plot(price_df['Short_MA'], label='20-day MA')
plt.plot(price_df['Long_MA'], label='50-day MA')

# Highlight buy and sell signals
buy_signals = price_df[price_df['Signal'] == 1]
sell_signals = price_df[price_df['Signal'] == -1]

plt.scatter(buy_signals.index, buy_signals['Price'], marker='^', color='g', label='Buy Signal', s=100)
plt.scatter(sell_signals.index, sell_signals['Price'], marker='v', color='r', label='Sell Signal', s=100)

plt.title("Merton's Jump-Diffusion Model with Moving Average Crossovers")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()


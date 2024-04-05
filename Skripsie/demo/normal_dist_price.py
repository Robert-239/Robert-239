import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Number of trading days
num_days = 252

# Generate random daily returns following a normal distribution
returns = np.random.normal(0, 1, num_days)

# Set initial price
initial_price = 100

# Generate price series using a random walk
price_series = initial_price * np.exp(np.cumsum(returns))

# Plot the price series
plt.figure(figsize=(10, 6))
plt.plot(price_series, label='Price')
plt.title('Price Movement of Fictitious Stock/Currency')
plt.xlabel('Trading Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


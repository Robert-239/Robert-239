import numpy as np
'''
The Euler-Maruyama method is a numerical technique used to solve stochastic differential equations (SDEs) like the Geometric Brownian Motion (GBM) equation for stock prices.

It’s essentially the stochastic equivalent of the Euler method for ordinary differential equations, adapted to handle the random component introduced by the Brownian motion.

This involves discretizing the time into small intervals and iteratively updating the price.

Here’s how you would implement the Euler-Maruyama method in Python to estimate the future stock price using GBM:

    Discretize the time interval into small increments.
    Iteratively update the stock price using the GBM formula at each time step.

The stochastic differential equation for GBM is:

 

dSt =  μ * St * dt + σ * St * dWt

 

To simulate this using the Euler-Maruyama method, you discretize it as follows:

 

S(t+Δt) = St + μSt*Δt + σSt(√Δt)* Zt

 

Here, Zt is a random variable from the standard normal distribution, representing the increment of the Wiener process.
'''
# Parameters
S0 = 100 # initial stock price
mu = 0.05 # drift coefficient sigma = 0.2 # volatility
T = 1.0 # time to maturity
dt = 1/252 # time step size (assuming 252 trading days)
N = int(T/dt) # number of time steps
sigma = 0.2 #volatility
# Random seed (do same as above)
np.random.seed(97)

# Initialize the stock price array
S = np.zeros(N)
S[0] = S0

# Simulate the stock price path
for t in range(1, N):
    Z = np.random.normal() # random normal variable
    S[t] = S[t-1] + mu * S[t-1] * dt + sigma * S[t-1] * np.sqrt(dt) * Z

# Estimated stock price at the end of the period
S_T = S[-1]

print(f"Estimated stock price after one year using Euler-Maruyama: {S_T:.2f}")

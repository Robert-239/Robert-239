import numpy as np

# dSt = mu * St * dt + sigma * St * Wt
# Where:    St is asset price
#           mu is drift rate
#           sigma is volatility
#           Wt is a Wiener process or Brownian motion

#Params

S0 = 100        #asset start price
mu = 0.05       #expected return
sigma = 0.2     #volatility
T = 1.0         #time period in years
dt = 1/252      #timestep assuming 252 trading days
N = int(T/dt)   #total number of time steps

np.random.seed(97)
dW = np.random.normal(0, np.sqrt(dt),N) # increments for Wt

W = np.cumsum(dW)

# Simulate stock price path

St = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0,T,N) + sigma*W)
print(St)
# estimated stock price at the end of the period

S_T = St[-1]

print(f"Estimated Stock price after, {T}, year(s) : {S_T}")


import numpy as np
import matplotlib.pyplot as plt

# Parameters
Ns = 1       # Number of simulations
T = 1        # Time period
lambd = 15   # Jump intensity
m = 0.5      # Mean of jump size
delta = 1    # Standard deviation of jump size
n_grid = 100
t = np.linspace(0, T, n_grid)
dt = t[1] - t[0]  # Time step

print(t[1])
print(t[0])
# Simulating compound Poisson process

dN = np.random.poisson(lambd * dt, (n_grid - 1, Ns))  # Poisson distribution
sumD = m * dN + delta * np.sqrt(dN) * np.random.randn(n_grid - 1, Ns)  # Jump sizes
Y_t = np.concatenate((np.zeros((1, Ns)), np.cumsum(sumD, axis=0)))  # Cumulative sum

print("sumD\n")
print(sumD)
print("\nY_t")
print(Y_t)

# Plot
#plt.plot(t+1,sumD )
plt.plot(t,Y_t)
plt.xlabel('Time')
plt.ylabel('Compound Poisson process Y_t')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def merton_jump_diffusion(S0, r, sigma, lam, mu_jump, sigma_jump, T, dt, num_paths):
    """
    Simulate Bitcoin price movements using Merton Jump Diffusion model.

    Parameters:
        S0 (float): Initial Bitcoin price.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        lam (float): Jump intensity.
        mu_jump (float): Mean jump size.
        sigma_jump (float): Jump size volatility.
        T (float): Time horizon.
        dt (float): Time step size.
        num_paths (int): Number of simulation paths.

    Returns:
        numpy.ndarray: Simulated Bitcoin price paths.
    """
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps + 1)

    # Generate random numbers for jumps
    num_jumps = np.random.poisson(lam * T, num_paths)
    print(num_jumps)
    jump_times = np.random.uniform(0, T, sum(num_jumps))
    print(f"\n{jump_times}")
    jump_sizes = np.random.normal(mu_jump, sigma_jump, sum(num_jumps))
    print(jump_sizes)
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0

    for i in range(num_paths):
        for j in range(1, num_steps + 1):
            Z = np.random.normal()
            paths[i, j] = paths[i, j - 1] * (1+ 0.00 * dt + sigma * np.sqrt(dt) * Z)
            if t[j] in jump_times[:num_jumps[i]]:
                paths[i, j] *= np.exp(jump_sizes[jump_times == t[j]])

    return paths

# Parameters
S0 = 60000  # Initial Bitcoin price
r = 0.0  # Risk-free interest rate
sigma = 0.2  # Volatility
lam = 0.1  # Jump intensity
mu_jump = 0.05  # Mean jump size
sigma_jump = 0.2  # Jump size volatility
T = 1  # Time horizon
dt = 1 / 252  # Time step size (assuming trading days)
num_paths = 10  # Number of simulation paths

# Simulate paths
paths = merton_jump_diffusion(S0, r, sigma, lam, mu_jump, sigma_jump, T, dt, num_paths)

# Plot paths
plt.figure(figsize=(10, 6))
for i in range(num_paths):
    plt.plot(np.arange(0, T + dt, dt), paths[i])
plt.title('Simulated Bitcoin Price Paths using Merton Jump Diffusion')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)

##########################################################################################


# Parameters
Ns = 1       # Number of simulations
T = 1        # Time period
lambd = 15   # Jump intensity
m = 0.5      # Mean of jump size
delta = 1    # Standard deviation of jump size
n_grid = 1000
t = np.linspace(0, T, n_grid)
dt = t[1] - t[0]  # Time step

# Simulating compound Poisson process
dN = np.random.poisson(lambd * dt, (n_grid - 1, Ns))  # Poisson distribution
sumD = m * dN + delta * np.sqrt(dN) * np.random.randn(n_grid - 1, Ns)  # Jump sizes
Y_t = np.concatenate((np.zeros((1, Ns)), np.cumsum(sumD, axis=0)))  # Cumulative sum

# Plot
plt.plot(t, Y_t)
plt.xlabel('Time')
plt.ylabel('Compound Poisson process Y_t')



plt.show()

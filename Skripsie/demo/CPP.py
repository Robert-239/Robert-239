
import numpy as np
import matplotlib.pyplot as plt

def mertons_jump_diffusion(S0, T, r, sigma, lamb, mu_jump, sigma_jump, n_simulations, n_steps):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    price_paths = np.zeros((n_simulations, n_steps + 1))
    price_paths[:, 0] = S0

    for i in range(1, n_steps + 1):
        # Generate random normal variables for diffusion and jumps
        dW = np.random.normal(0, np.sqrt(dt), size=n_simulations)
        dN = np.random.poisson(lamb * dt, size=n_simulations)

        # Calculate jumps
        jumps = np.random.normal(mu_jump * dt, sigma_jump * np.sqrt(dt), size=n_simulations)
        jumps[dN == 0] = 0

        # Update prices
        price_paths[:, i] = price_paths[:, i - 1] * (
                    1 + r * dt + sigma * dW + jumps)

    return t, price_paths

# Example usage
S0 = 100  # Initial price
T = 1  # Time horizon
r = 0  # Risk-free rate (not considered in this model)
sigma = 0.2  # Volatility
lamb = 0.1  # Jump frequency
mu_jump = 0.05  # Mean of jump size
sigma_jump = 0.1  # Volatility of jump size
n_simulations = 10  # Number of simulations
n_steps = 1000  # Number of time steps

t, price_paths = mertons_jump_diffusion(S0, T, r, sigma, lamb, mu_jump, sigma_jump, n_simulations, n_steps)

# Plot price paths
plt.figure(figsize=(10, 6))
for i in range(n_simulations):
    plt.plot(t, price_paths[i], alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Merton\'s Jump Diffusion Model - Price Paths')
plt.grid(True)
plt.show()

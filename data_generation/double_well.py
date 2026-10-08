import numpy as np
from data_generation.simulation import simulate

N, T, S0, seed = 1000000, 200.0, 1.0, 42


def mu(x):
    return 0.5 * x - x**3


def sigma(x):
    return 1.0 + 0.25 * x**2


t, dt, S_, W_, mu_exact_, sigma_exact_ = simulate(mu, sigma, N, T, S0, seed)
t = np.arange(N) * dt
W_ = np.r_[0.0, W_[:-1]]

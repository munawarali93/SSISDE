import numpy as np
from data_generation.simulation import simulate

N, T, S0, seed = 100000, 1.0, 1.0, 3
kappa, theta, sigm = 0.5, 0.0, 0.3


def mu(x):
    return kappa * (theta - x)


def sigma(x):
    return sigm * np.ones_like(x)


t, dt, S_, W_, mu_exact_, sigma_exact_ = simulate(mu, sigma, N, T, S0, seed)
t = np.arange(N) * dt
W_ = np.r_[0.0, W_[:-1]]

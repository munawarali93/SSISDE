import numpy as np
from data_generation.simulation import simulate

N, T, S0, seed = 100000, 1.0, 1.0, 31
r, sigm = 0.5, 1.0


def mu(x):
    return r * np.ones_like(x)


def sigma(x):
    return sigm * np.ones_like(x)


t, dt, S_, W_, mu_exact_, sigma_exact_ = simulate(mu, sigma, N, T, S0, seed)

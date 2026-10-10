import numpy as np
from data_generation.simulation import simulate


def mu(x):
    return 0.3 * x ** 2 + 0.7 * x


def sigma(x):
    return 0.1 * x ** 2 + 0.5 * x


def generate_data(N=100000, T=1.0, S0=1.0, seed=2):
    """Return t, dt, X, W, actual drift, and actual diffusion."""
    return simulate(mu, sigma, N, T, S0, seed)

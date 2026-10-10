import numpy as np
from data_generation.simulation import simulate


def mu(x):
    return 0.5 * x - x ** 3


def sigma(x):
    return 1.0 + 0.25 * x ** 2


def generate_data(N=100000, T=1.0, S0=1.0, seed=3):
    """Return t, dt, X, W, actual drift, and actual diffusion."""
    return simulate(mu, sigma, N, T, S0, seed)

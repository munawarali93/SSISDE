import numpy as np
from data_generation.simulation import simulate


def mu(x, r=0.5):
    return r * np.ones_like(x)


def sigma(x, sigm=1.0):
    return sigm * np.ones_like(x)


def generate_data(N=100000, T=1.0, S0=1.0, seed=31, r=0.5, sigm=1.0):
    """Return t, dt, X, W, actual drift, and actual diffusion."""
    return simulate(lambda x: mu(x, r), lambda x: sigma(x, sigm), N, T, S0, seed)

import numpy as np
from data_generation.simulation import simulate


def mu(x, kappa=0.5, theta=0.0):
    return kappa * (theta - x)


def sigma(x, sigm=0.3):
    return sigm * np.ones_like(x)


def generate_data(N=100000, T=1.0, S0=1.0, seed=3, kappa=0.5, theta=0.0, sigm=0.3):
    """Return t, dt, X, W, actual drift, and actual diffusion."""
    return simulate(lambda x: mu(x, kappa, theta), lambda x: sigma(x, sigm), N, T, S0, seed)

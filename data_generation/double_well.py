import numpy as np
from data_generation.simulation import simulate


def mu(x, a=0.5, b=1.0):
    return a * x - b * x ** 3


def sigma(x, c=1.0, d=0.25):
    return c + d * x ** 2


def generate_data(N=100000, T=1.0, S0=1.0, seed=3, a=0.5, b=1.0, c=1.0, d=0.25):
    """Return t, dt, X, W, actual drift, and actual diffusion."""
    return simulate(lambda x: mu(x, a, b), lambda x: sigma(x, c, d), N, T, S0, seed)

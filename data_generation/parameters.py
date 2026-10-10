import numpy as np


def generate_noise(N, T, seed):
    """Return the original time grid, step, Brownian increments, and path."""
    t = np.linspace(0, T, N)
    dt = T / N
    np.random.seed(seed)
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    return (t, dt, dW, W)

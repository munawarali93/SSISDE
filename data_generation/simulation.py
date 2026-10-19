"""Seeded Brownian paths and Euler-Maruyama simulation."""
import numpy as np


def generate_noise(N, T, seed):
    """Return the original time grid, step, Brownian increments, and path."""
    t = np.linspace(0, T, N)
    dt = T / N
    np.random.seed(seed)
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    return (t, dt, dW, W)


def simulate(mu, sigma, N, T, S0, seed):
    t = np.linspace(0, T, N)
    dt = T / N
    np.random.seed(seed)
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    S = np.zeros(N)
    S[0] = S0
    for i in range(N - 1):
        S[i + 1] = S[i] + mu(S[i]) * dt + sigma(S[i]) * dW[i]
    return t, dt, S, W, mu(S), sigma(S)

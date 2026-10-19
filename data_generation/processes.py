"""Process coefficients and generators for the supported experiment families."""
import numpy as np

from data_generation.simulation import simulate


def mu(x, drift, r=0.5, kappa=0.5, theta=0, K=0.1):
    if drift == 'constant':
        return r * np.ones_like(x)
    if drift == 'BS':
        return r * x
    if drift == 'quadratic':
        return r * x ** 2
    if drift == 'quadratic1':
        return 0.3 * x ** 2 + 0.7 * x
    if drift == 'quadratic2':
        return 0.25 * x * (1.0 - x * x)
    if drift == 'sin':
        return r * np.sin(x)
    if drift == 'OU':
        return kappa * (theta - x)
    if drift == 'population':
        return r * x * (K + x)
    raise ValueError(f'Unknown drift: {drift!r}')


def sigma(x, diff, sigm=0.3, beta=0.2):
    if diff == 'OU':
        return sigm * np.ones_like(x)
    if diff == 'BS':
        return sigm * x
    if diff == 'quadratic':
        return sigm * x ** 2
    if diff == 'quadratic1':
        return 0.1 * x ** 2 + 0.5 * x
    if diff == 'quadratic2':
        return -0.5 * (1.0 - x ** 2)
    if diff == 'population':
        return beta * x
    raise ValueError(f'Unknown diffusion: {diff!r}')


def simulate_process(dW, dt, S0, drift='BS', diff='BS', r=0.5, sigm=0.3,
                     kappa=0.5, theta=0,
                     K=0.1, beta=0.2):
    """Return the state path and its actual drift and diffusion."""
    N = len(dW)
    S_ = np.zeros(N)
    S_[0] = S0
    for i in range(N - 1):
        S_[i + 1] = S_[i] + mu(S_[i], drift, r, kappa, theta, K) * dt + sigma(S_[i], diff, sigm, beta) * dW[i]
    mu_exact_ = mu(S_, drift, r, kappa, theta, K)
    sigma_exact_ = sigma(S_, diff, sigm, beta)
    return (S_, mu_exact_, sigma_exact_)


def constant_drift(x, r=0.5):
    return r * np.ones_like(x)


def constant_diffusion(x, sigm=1.0):
    return sigm * np.ones_like(x)


def generate_constant(N=100000, T=1.0, S0=1.0, seed=31, r=0.5, sigm=1.0):
    """Return t, dt, X, W, actual drift, and actual diffusion."""
    return simulate(lambda x: constant_drift(x, r), lambda x: constant_diffusion(x, sigm), N, T, S0, seed)


def quadratic_drift(x):
    return 0.3 * x ** 2 + 0.7 * x


def quadratic_diffusion(x):
    return 0.1 * x ** 2 + 0.5 * x


def generate_quadratic(N=100000, T=1.0, S0=1.0, seed=2):
    """Return t, dt, X, W, actual drift, and actual diffusion."""
    return simulate(quadratic_drift, quadratic_diffusion, N, T, S0, seed)


def ou_drift(x, kappa=0.5, theta=0.0):
    return kappa * (theta - x)


def ou_diffusion(x, sigm=0.3):
    return sigm * np.ones_like(x)


def generate_ou(N=100000, T=1.0, S0=1.0, seed=3, kappa=0.5, theta=0.0, sigm=0.3):
    """Return t, dt, X, W, actual drift, and actual diffusion."""
    return simulate(lambda x: ou_drift(x, kappa, theta), lambda x: ou_diffusion(x, sigm), N, T, S0, seed)

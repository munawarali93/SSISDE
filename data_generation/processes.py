import numpy as np


def mu(x, drift, r=0.5, kappa=0.5, theta=0, w1=-1, w2=0.5, K=0.1):
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
    if drift == 'Double_well_potential':
        return w1 * x ** 3 + w2 * x
    if drift == 'population':
        return r * x * (K + x)


def sigma(x, diff, sigm=0.3, w3=1.0, w4=0.25, beta=0.2):
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
    if diff == 'Double_well_potential':
        return w3 + w4 * x ** 2
    if diff == 'population':
        return beta * x


def simulate_process(dW, dt, S0, drift='BS', diff='BS', r=0.5, sigm=0.3,
                     kappa=0.5, theta=0, w1=-1, w2=0.5, w3=1.0, w4=0.25,
                     K=0.1, beta=0.2):
    """Return the state path and its actual drift and diffusion."""
    N = len(dW)
    S_ = np.zeros(N)
    S_[0] = S0
    for i in range(N - 1):
        S_[i + 1] = S_[i] + mu(S_[i], drift, r, kappa, theta, w1, w2, K) * dt + sigma(S_[i], diff, sigm, w3, w4, beta) * dW[i]
    mu_exact_ = mu(S_, drift, r, kappa, theta, w1, w2, K)
    sigma_exact_ = sigma(S_, diff, sigm, w3, w4, beta)
    return (S_, mu_exact_, sigma_exact_)

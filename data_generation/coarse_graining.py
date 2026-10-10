import numpy as np


def coarse_grain(S_, BQ_, W_, mu_exact_, sigma_exact_, sigma_pred_, M, K):
    """Return diffusion, actual drift, state, Q noise, P noise, and actual diffusion."""
    sigma_pred = np.zeros(K)
    mu_exact = np.zeros(K)
    S = np.zeros(K)
    BQ = np.zeros(K)
    W = np.zeros(K)
    sigma_exact = np.zeros(K)
    for i in range(K):
        sigma_pred[i] = sigma_pred_[i * M]
        S[i] = S_[i * M]
        mu_exact[i] = mu_exact_[i * M]
        BQ[i] = BQ_[i * M]
        W[i] = W_[i * M]
        sigma_exact[i] = sigma_exact_[i * M]
    return (sigma_pred, mu_exact, S, BQ, W, sigma_exact)

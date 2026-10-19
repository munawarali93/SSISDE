"""Recover Q-noise and physical Brownian motion from the estimated coefficients."""
import numpy as np


def recover_q_noise(S_, sigma_pred_):
    """Return the Q-noise path using the original increment sum."""
    BQ_ = np.zeros(len(S_))
    for i in range(1, len(S_)):
        BQ_[i] = BQ_[i - 1] + (S_[i] - S_[i - 1]) / sigma_pred_[i - 1]
    return BQ_


def recover_p_noise(BQ, mu_pred1, sigma_pred, T):
    """Return the P-noise path, coarse time step, and coarse time grid."""
    K = len(BQ)
    BP = np.zeros(K)
    dt = T / K
    t = np.linspace(0, T, K)
    for i in range(1, K):
        BP[i] = BP[i - 1] + BQ[i] - BQ[i - 1] - mu_pred1[i - 1] / sigma_pred[i - 1] * dt
    return (BP, dt, t)


def recover_p_noise_method2(BQ, mu_pred2, sigma_pred, T):
    """Return the P-noise path, time step, and grid using Method 2 drift."""
    return recover_p_noise(BQ, mu_pred2, sigma_pred, T)


def recover_noise(BQ, drift, diffusion, dt):
    increments = np.diff(BQ) - drift[:-1] / diffusion[:-1] * dt
    return np.r_[0.0, np.cumsum(increments)]

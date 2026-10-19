"""Signature regression, drift coefficients, and the three drift recovery methods."""
import numpy as np
import iisignature
from sklearn import linear_model
from tqdm import tqdm


def fit_signature(S_, BQ_, t, keep_cols, M=100):
    """Return the fitted psi_21 coefficient for each window."""
    N = len(S_)
    K = int(N / M)
    l21 = []
    for i in tqdm(range(K)):
        path = np.array([[t[k], BQ_[k]] for k in range(i * M, i * M + M)])
        X1 = np.array([iisignature.sig(path[0:i], 3) for i in range(1, len(path) + 1)])
        X1_reduced = X1[:, keep_cols]
        clf = linear_model.Ridge(alpha=0.01)
        clf.fit(X1_reduced, S_[i * M:i * M + M])
        l21.append(clf.coef_[3])
    l21 = np.array(l21)
    return l21


def drift_coefficients(mu_exact, sigma_pred, sigma_prime, sigma_2prime, l21):
    """Return the initial drift and coefficients a, b, and b_ for drift recovery."""
    mu_0 = mu_exact[0]
    a = sigma_prime / sigma_pred
    b = l21 / sigma_pred + 0.5 * (sigma_prime ** 2 + sigma_pred * sigma_2prime)
    G = -0.5 * (sigma_pred * sigma_prime ** 2 + sigma_pred ** 2 * sigma_2prime)
    b_ = G / sigma_pred + 0.5 * (sigma_prime ** 2 + sigma_pred * sigma_2prime)
    return (mu_0, a, b, b_)


def recover_drift(S, a, b, b_, mu_0):
    """Return the Method 1 and Method 2 drift estimates."""
    K = len(S)
    mu_pred1 = np.zeros(K)
    mu_pred2 = np.zeros(K)
    mu_pred1[0] = mu_0
    mu_pred2[0] = mu_0
    for i in range(1, len(a)):
        mu_pred1[i] = mu_pred1[i - 1] + (a[i - 1] * mu_pred1[i - 1] + b[i - 1]) * (S[i] - S[i - 1])
        mu_pred2[i] = mu_pred2[i - 1] + (a[i - 1] * mu_pred2[i - 1] + b_[i - 1]) * (S[i] - S[i - 1])
    mu_pred2 = np.round(mu_pred2, 8)
    return (mu_pred1, mu_pred2)


def recover_drift_ode2(S, sigma_pred, sigma_prime, sigma_2prime, mu_0, lambda_0):
    """Return the ODE2 drift and slope estimates."""
    K = len(S)
    A = 2 * sigma_prime / sigma_pred
    B = sigma_2prime / sigma_pred - 2 * sigma_prime ** 2 / sigma_pred ** 2
    mu_pred11 = np.zeros(K)
    lamda_pred = np.zeros(K)
    mu_pred11[0] = mu_0
    lamda_pred[0] = lambda_0
    for i in range(1, K):
        mu_pred11[i] = mu_pred11[i - 1] + lamda_pred[i - 1] * (S[i] - S[i - 1])
        lamda_pred[i] = lamda_pred[i - 1] + (A[i - 1] * lamda_pred[i - 1] + B[i - 1] * mu_pred11[i - 1]) * (S[i] - S[i - 1])
    return (mu_pred11, lamda_pred)

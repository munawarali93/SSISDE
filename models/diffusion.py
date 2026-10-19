"""Quadratic variation, diffusion regression and tuning, and spatial derivatives."""
import warnings

import numpy as np
import pysindy as ps
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def compute_quadratic_variation(W):
    return np.sum(np.diff(W) ** 2)


def estimate_quadratic_variation(S_, t):
    """Return quadratic variation, its derivative, and the difference method."""
    N = len(S_)
    QVS = np.zeros(N)
    for i in tqdm(range(N)):
        QVS[i] = compute_quadratic_variation(S_[0:i])
    QV = QVS.copy()
    diff_method = ps.FiniteDifference(order=3)
    dQV = diff_method._differentiate(QV.reshape(-1, 1), t).flatten()
    return (QV, dQV, diff_method)


def fit_diffusion_model(S_, t, dQV, threshold=5.0, nu=2.2, max_iter=1500, tol=1e-12):
    """Fit the original Black-Scholes diffusion-variance model."""
    model = ps.SINDy(
        optimizer=ps.SR3(
            threshold=threshold, nu=nu, thresholder='l1',
            normalize_columns=True, fit_intercept=False,
            max_iter=max_iter, tol=tol,
        ),
        feature_library=ps.PolynomialLibrary(
            degree=4, include_interaction=True, include_bias=True,
        ),
    )
    model.fit(S_.reshape(-1, 1), t, x_dot=dQV.reshape(-1, 1))
    return model


def fit_quadratic_diffusion(S_, t, dQV, threshold=0.04, nu=0.05, max_iter=500, tol=1e-14):
    """Fit the original quadratic diffusion-variance model."""
    model = ps.SINDy(
        optimizer=ps.SR3(
            threshold=threshold, nu=nu, thresholder='l1',
            normalize_columns=True, fit_intercept=False,
            max_iter=max_iter, tol=tol,
        ),
        feature_library=ps.PolynomialLibrary(
            degree=4, include_interaction=True, include_bias=False,
        ),
    )
    model.fit(S_.reshape(-1, 1), t, x_dot=dQV.reshape(-1, 1))
    return model


def diffusion_model(threshold, nu, max_iter=1500, tol=1e-12):
    return ps.SINDy(
        optimizer=ps.SR3(
            threshold=threshold, nu=nu, thresholder="l1",
            normalize_columns=True, fit_intercept=False,
            max_iter=max_iter, tol=tol,
        ),
        feature_library=ps.PolynomialLibrary(degree=4),
        feature_names=['X'],
    )


def fit_diffusion(S, t, dQV, thresholds, nus, max_iter=10000, tol=1e-8):
    """Fit sigma squared; select SR3 settings on the last 20% of the path."""
    x, y = S.reshape(-1, 1), dQV.reshape(-1, 1)
    split = int(0.8 * len(S))
    results = []
    with threadpool_limits(limits=1):
        for threshold in thresholds:
            for nu in nus:
                model = diffusion_model(threshold, nu, max_iter, tol)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(x[:split - 3], t[:split - 3], x_dot=y[:split - 3])
                variance = model.predict(x).ravel()
                valid = np.isfinite(variance).all() and np.all(variance > 0)
                mse = float(np.mean((variance[split:] - dQV[split:])**2)) if valid else None
                results.append(dict(
                    threshold=float(threshold), nu=float(nu), validation_mse=mse,
                    max_iter=max_iter, tol=tol,
                    terms=int(np.count_nonzero(model.coefficients())),
                    converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                ))
        candidates = [row for row in results if row['validation_mse'] is not None and row['converged']]
        if not candidates:
            raise ValueError("No converged candidate gave positive diffusion variance; revise the grid.")
        best = min(candidates, key=lambda row: (
            row['validation_mse'], row['terms'], -row['threshold'], -row['nu'],
        ))
        model = diffusion_model(best['threshold'], best['nu'], max_iter, tol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, t, x_dot=y)
        best = dict(best, full_fit_converged=not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        ))
        if not best['full_fit_converged']:
            raise ValueError('The full-data fit did not converge; increase max_iter or revise the grid.')
    variance = model.predict(x).ravel()
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("The full-data fit has nonpositive diffusion variance; revise the grid.")
    return model, best, results


def diffusion_derivatives(sigma_pred_, S_, diff_method):
    """Return the first three diffusion derivatives on the supplied state grid."""
    sigma_prime_ = diff_method._differentiate(sigma_pred_.reshape(-1, 1), S_).flatten()
    sigma_2prime_ = diff_method._differentiate(sigma_prime_.reshape(-1, 1), S_).flatten()
    sigma_3prime_ = diff_method._differentiate(sigma_2prime_.reshape(-1, 1), S_).flatten()
    return (sigma_prime_, sigma_2prime_, sigma_3prime_)


def coarse_diffusion_derivatives(sigma_pred, S, diff_method):
    """Return the first three diffusion derivatives on the coarse state grid."""
    return diffusion_derivatives(sigma_pred, S, diff_method)

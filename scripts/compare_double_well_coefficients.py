"""Compare double-well coefficients using the existing experiment functions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generation.double_well import generate_data
from data_generation.coarse_graining import coarse_grain
from models.quadratic_variation import estimate_quadratic_variation
from models.diffusion_tuning import fit_diffusion
from models.diffusion_derivatives import diffusion_derivatives
from models.coarse_diffusion_derivatives import coarse_diffusion_derivatives
from models.q_brownian_motion import recover_q_noise
from models.signature_regression import fit_signature
from models.drift_coefficients import drift_coefficients
from models.drift_recovery import recover_drift
from models.drift_recovery_from_ODE2 import recover_drift_ode2
from models.p_brownian_motion import recover_p_noise
from models.stochastic_sindy import StochSINDy

CASES = {
    'original': (0.5, 1.0, 1.0, 0.25),
    'cubic_010': (0.5, 0.1, 1.0, 0.05),
    'cubic_005': (0.5, 0.05, 1.0, 0.01),
    'cubic_001': (0.5, 0.01, 1.0, 0.01),
    'lower_noise': (0.5, 0.01, 0.5, 0.01),
    'wells_at_one': (0.05, 0.05, 0.3, 0.01),
    'matched_factor': (0.5, 0.125, 1.0, -0.25),
    'near_matched': (0.5, 0.125, 1.0, -0.24),
}
DESTINATION = ROOT / 'experiments/double_well/coefficient_study'


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def rmse(estimate, actual):
    return rms(np.asarray(estimate).ravel() - np.asarray(actual).ravel())


def run_case(name, seed=3):
    a, b, c, d = CASES[name]
    N, T, S0, M = 100000, 1.0, 1.0, 100
    t, fine_dt, X, W_fine, drift_true, sigma_true = generate_data(
        N, T, S0, seed, a=a, b=b, c=c, d=d,
    )
    curvature = 2*X*(a*d+b*c)*(d*X**2-3*c)/(c+d*X**2)**3
    QV, dQV, diff_method = estimate_quadratic_variation(X, t)
    try:
        model, settings, search = fit_diffusion(
            X, t, dQV, [0.01, 0.1, 0.3, 1.0, 3.0, 10.0], [0.1, 0.5, 2.0],
        )
    except ValueError as error:
        result = dict(case=name, seed=seed, coefficients=dict(a=a, b=b, c=c, d=d),
                      status='failed', stage='diffusion fit', error=str(error))
        DESTINATION.mkdir(parents=True, exist_ok=True)
        (DESTINATION/f'{name}_seed{seed}.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    sigma_fine = np.sqrt(model.predict(X.reshape(-1, 1)))
    diffusion_derivatives(sigma_fine, X, diff_method)
    BQ_fine = recover_q_noise(X, sigma_fine)
    l21 = fit_signature(X, BQ_fine, t, [0, 1, 3, 4, 5, 13], M)
    sigma, exact, S, BQ, W, sigma_exact = coarse_grain(
        X, BQ_fine, W_fine, drift_true, sigma_true, sigma_fine, M, N//M,
    )
    first, second, third = coarse_diffusion_derivatives(sigma, S, diff_method)
    mu0, A, B, B2 = drift_coefficients(exact, sigma, first, second, l21)
    mu1, mu2 = recover_drift(S, A, B, B2, mu0)
    mu_ode2, slope = recover_drift_ode2(S, sigma, first, second, mu0, a-3*b*S0**2)
    BP, dt, coarse_t = recover_p_noise(BQ, mu1, sigma, T)
    identifier = StochSINDy(
        mu_deg=5, si_deg=5, pen='elasticnet', use_cv=True, k=7,
        alphas=np.geomspace(1e-5, 1e1, 35),
        rhos=[0.3, 0.4, 0.5, 0.7, 0.85, 0.95, 1.0], use_1se=True,
    ).fit(S, BP, dt)
    results = {
        'case': name, 'seed': seed, 'coefficients': dict(a=a, b=b, c=c, d=d),
        'N': N, 'T': T, 'S0': S0, 'M': M,
        'well_locations': [-float(np.sqrt(a/b)), float(np.sqrt(a/b))],
        'state_range': [float(X.min()), float(X.max())],
        'curvature_rms': rms(curvature), 'curvature_max_abs': float(abs(curvature).max()),
        'actual_drift_rms': rms(exact),
        'sigma_rmse': rmse(sigma_fine, sigma_true),
        'sigma_relative_rmse': rmse(sigma_fine, sigma_true)/rms(sigma_true),
        'sigma_settings': settings, 'sigma_squared_equation': model.equations(precision=8)[0],
        'initial_drift': float(mu0), 'initial_slope': a-3*b*S0**2,
        'drift_rmse': {'Method 1': rmse(mu1, exact), 'Method 2': rmse(mu2, exact),
                       'ODE2': rmse(mu_ode2, exact)},
        'ode2_relative_rmse': rmse(mu_ode2, exact)/rms(exact),
        'noise_rmse': rmse(BP, W),
        'ssisde': {'mu': identifier._fmt_poly(identifier.mu_c),
                   'sigma': identifier._fmt_poly(identifier.si_c),
                   'drift_rmse': rmse(identifier.mu(S), exact),
                   'diffusion_rmse': rmse(identifier.si(S), sigma_exact),
                   'increment_mse': identifier.mse,
                   'alpha': identifier.best_a, 'rho': identifier.best_r},
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION/f'{name}_seed{seed}.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    paths = ROOT/'outputs/double_well/coefficient_study'
    paths.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths/f'{name}_seed{seed}.npz', t=coarse_t, S=S, W=W, BQ=BQ, BP=BP,
                        mu_exact=exact, sigma_exact=sigma_exact, sigma_pred=sigma,
                        mu_method1=mu1, mu_method2=mu2, mu_ode2=mu_ode2,
                        ssisde_mu=identifier.mu(S), ssisde_sigma=identifier.si(S))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[3])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for name in args.cases:
            for seed in args.seeds:
                print(f'Running {name}, seed {seed}', flush=True)
                result = run_case(name, seed)
                print(json.dumps(result), flush=True)

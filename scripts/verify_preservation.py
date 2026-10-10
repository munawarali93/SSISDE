"""Check preserved stages and the corrected ODE2 equation."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

import iisignature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from sklearn import linear_model
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits
from tqdm import tqdm

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
reference_commit = 'd8152dff316dcc07a2551fadb7d543387210e27a'

from data_generation.parameters import generate_noise
from data_generation.processes import simulate_process, mu, sigma
from data_generation.coarse_graining import coarse_grain
from models.quadratic_variation import estimate_quadratic_variation
from models.diffusion import fit_diffusion_model
from models.diffusion_quadratic import fit_quadratic_diffusion
from models.diffusion_derivatives import diffusion_derivatives
from models.coarse_diffusion_derivatives import coarse_diffusion_derivatives
from models.q_brownian_motion import recover_q_noise
from models.signature_regression import fit_signature
from models.drift_coefficients import drift_coefficients
from models.drift_recovery import recover_drift
from models.drift_recovery_from_ODE2 import recover_drift_ode2
from models.p_brownian_motion import recover_p_noise
from models.p_brownian_motion_method2 import recover_p_noise_method2
from models.stochastic_sindy import StochSINDy


def reference(path):
    return subprocess.check_output(
        ['git', 'show', f'{reference_commit}:{path}'], cwd=root, text=True,
    )


def run_reference(path, namespace):
    exec(compile(reference(path), path, 'exec'), namespace)


def namespace():
    return dict(np=np, ps=ps, plt=plt, tqdm=tqdm, iisignature=iisignature,
                linear_model=linear_model, Lasso=Lasso, ElasticNet=ElasticNet,
                TimeSeriesSplit=TimeSeriesSplit)


def equal(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def check_case(case):
    old = namespace()
    if case == 'black_scholes':
        run_reference('data_generation/parameters.py', old)
        run_reference('data_generation/processes.py', old)
        t, dt, dW, W = generate_noise(old['N'], old['T'], 3)
        equal(dW, old['dW'])
        S, drift, diffusion = simulate_process(dW, dt, old['S0'], r=old['r'], sigm=old['sigm'])
        actual = t, dt, S, W, drift, diffusion
    else:
        run_reference('data_generation/simulation.py', old)
        code = reference(f'data_generation/{case}.py')
        exec(code.replace('from data_generation.simulation import simulate', ''), old)
        actual = importlib.import_module(f'data_generation.{case}').generate_data()
    for value, key in zip(actual, ['t', 'dt', 'S_', 'W_', 'mu_exact_', 'sigma_exact_']):
        equal(value, old[key])

    old['N'] = 2000
    for key in ['t', 'S_', 'W_', 'mu_exact_', 'sigma_exact_']:
        old[key] = old[key][:old['N']]
    run_reference('models/quadratic_variation.py', old)
    QV, dQV, method = estimate_quadratic_variation(old['S_'], old['t'])
    equal(QV, old['QV'])
    equal(dQV, old['dQV'])

    # Isolate the recovery formulas from the diffusion fit in these stage checks.
    old['sigma_pred_'] = old['sigma_exact_'].reshape(-1, 1)
    run_reference('models/diffusion_derivatives.py', old)
    for value, key in zip(diffusion_derivatives(old['sigma_pred_'], old['S_'], method),
                          ['sigma_prime_', 'sigma_2prime_', 'sigma_3prime_']):
        equal(value, old[key])
    run_reference('models/q_brownian_motion.py', old)
    equal(recover_q_noise(old['S_'], old['sigma_pred_']), old['BQ_'])
    old['keep_cols'] = [0, 1, 3, 4, 5, 13]
    run_reference('models/signature_regression.py', old)
    equal(fit_signature(old['S_'], old['BQ_'], old['t'], old['keep_cols']), old['l21'])
    run_reference('data_generation/coarse_graining.py', old)
    values = coarse_grain(*(old[k] for k in ['S_', 'BQ_', 'W_', 'mu_exact_',
                                           'sigma_exact_', 'sigma_pred_', 'M', 'K']))
    for value, key in zip(values, ['sigma_pred', 'mu_exact', 'S', 'BQ', 'W', 'sigma_exact']):
        equal(value, old[key])
    run_reference('models/coarse_diffusion_derivatives.py', old)
    for value, key in zip(coarse_diffusion_derivatives(old['sigma_pred'], old['S'], method),
                          ['sigma_prime', 'sigma_2prime', 'sigma_3prime']):
        equal(value, old[key])
    run_reference('models/drift_coefficients.py', old)
    coefficients = drift_coefficients(*(old[k] for k in
        ['mu_exact', 'sigma_pred', 'sigma_prime', 'sigma_2prime', 'l21']))
    for value, key in zip(coefficients, ['mu_0', 'a', 'b', 'b_']):
        equal(value, old[key])
    run_reference('models/drift_recovery.py', old)
    for value, key in zip(recover_drift(*(old[k] for k in ['S', 'a', 'b', 'b_', 'mu_0'])),
                          ['mu_pred1', 'mu_pred2']):
        equal(value, old[key])
    for filename, function, drift_name in [
        ('p_brownian_motion', recover_p_noise, 'mu_pred1'),
        ('p_brownian_motion_method2', recover_p_noise_method2, 'mu_pred2'),
    ]:
        run_reference(f'models/{filename}.py', old)
        for value, key in zip(function(old['BQ'], old[drift_name], old['sigma_pred'], old['T']),
                              ['BP', 'dt', 't']):
            equal(value, old[key])
    print(f'PASS: {case}: simulation and recovery stages other than corrected ODE2 match exactly.', flush=True)


def check_ode2():
    # For Black-Scholes, mu/sigma is constant, including on a nonmonotone path.
    x = np.array([1.0, 1.2, 0.9, 1.1, 0.8])
    drift, slope = recover_drift_ode2(x, 0.3*x, np.full_like(x, 0.3),
                                    np.zeros_like(x), 0.5*x[0], 0.5)
    np.testing.assert_allclose(drift, 0.5*x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(slope, 0.5, rtol=1e-13, atol=1e-13)

    # Constant diffusion gives a linear drift (OU).
    drift, slope = recover_drift_ode2(x, np.full_like(x, 0.3), np.zeros_like(x),
                                    np.zeros_like(x), -0.5*x[0], -0.5)
    np.testing.assert_allclose(drift, -0.5*x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(slope, -0.5, rtol=1e-13, atol=1e-13)

    # A known solution with nonzero sigma' and sigma'': mu=(0.7-0.2*x)*sigma.
    errors = []
    for intervals in [4000, 8000]:
        x = np.linspace(0.2, 1.2, intervals + 1)
        sigma = 1 + 0.25*x**2
        exact = (0.7 - 0.2*x)*sigma
        exact_slope = -0.2*sigma + (0.7 - 0.2*x)*0.5*x
        drift, slope = recover_drift_ode2(x, sigma, 0.5*x, np.full_like(x, 0.5),
                                        exact[0], exact_slope[0])
        errors.append(max(np.max(abs(drift-exact)), np.max(abs(slope-exact_slope))))
    assert errors[1] < 1e-4
    assert 1.9 < errors[0]/errors[1] < 2.1
    print('PASS: corrected ODE2 matches Black-Scholes and OU and converges to a nonlinear exact solution.', flush=True)


def check_regression():
    old = namespace()
    old['S_'] = np.linspace(0.7, 1.3, 1000)
    old['t'] = np.linspace(0, 1, 1000)
    old['dQV'] = (0.3*old['S_'])**2
    for name, function in [('diffusion', fit_diffusion_model),
                           ('diffusion_quadratic', fit_quadratic_diffusion)]:
        run_reference(f'models/{name}.py', old)
        new = function(old['S_'], old['t'], old['dQV'])
        equal(new.coefficients(), old['model'].coefficients())
    assert (root/'models/diffusion_tuning.py').read_text() == reference('models/diffusion_tuning.py')
    run_reference('models/stochastic_sindy.py', old)
    t, dt, dW, W = generate_noise(401, 1, 7)
    X, _, _ = simulate_process(dW, dt, 1)
    for penalty in ['ridge', 'lasso', 'elasticnet']:
        settings = dict(mu_deg=3, si_deg=3, pen=penalty, use_cv=True, k=3,
                        alphas=np.geomspace(1e-5, 1e-2, 4), rhos=[0.5, 1.0])
        before = old['StochSINDy'](**settings).fit(X, W, dt)
        after = StochSINDy(**settings).fit(X, W, dt)
        equal(after.mu_c, before.mu_c)
        equal(after.si_c, before.si_c)
        equal(after.mse, before.mse)
    print('PASS: diffusion fitting and all three SSISDE penalties match exactly.', flush=True)


def check_modules():
    for folder in ['data_generation', 'models']:
        for path in (root/folder).glob('*.py'):
            tree = ast.parse(path.read_text())
            assert all(isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
                       for n in tree.body), f'Import-time computation: {path}'
    for path in (root/'experiments').rglob('*.ipynb'):
        cells = json.loads(path.read_text())['cells']
        assert not any('%run' in ''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    print('PASS: importable modules and notebooks without %run.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        check_modules()
        check_ode2()
        for case in ['black_scholes', 'constant', 'quadratic', 'ou', 'double_well']:
            check_case(case)
        check_regression()

"""Select SSISDE settings using a final validation segment."""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from models.stochastic_sindy import StochSINDy


def stochastic_sindy_candidates():
    baseline = dict(mu_deg=5, si_deg=5, pen='elasticnet', use_cv=True, k=7,
                    alphas=np.geomspace(1e-5, 1e1, 35).tolist(),
                    rhos=[0.3, 0.4, 0.5, 0.7, 0.85, 0.95, 1.0], use_1se=True)
    candidates = [dict(name='baseline', model=baseline, fit={})]
    for mu_deg, si_deg in [(1, 2), (3, 2), (5, 2), (5, 5)]:
        for penalty in ['elasticnet', 'lasso']:
            for use_1se in [True, False]:
                settings = dict(mu_deg=mu_deg, si_deg=si_deg, pen=penalty,
                                use_cv=True, k=7, use_1se=use_1se,
                                alphas=np.geomspace(1e-7, 1e-2, 21).tolist(),
                                rhos=[0.05, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0])
                name = f'{penalty}_mu{mu_deg}_sigma{si_deg}_1se{use_1se}'
                candidates.append(dict(name=name, model=settings, fit={}))
    for alpha_mu in [1e-6, 1e-3, 1.0, 1000.0]:
        for threshold in [1e-6, 1e-4, 5e-3]:
            settings = dict(mu_deg=3, si_deg=2, pen='ridge', use_cv=False,
                            thr1=threshold, thr2=1.4*threshold)
            candidates.append(dict(name=f'ridge_mu{alpha_mu}_threshold{threshold}',
                                   model=settings, fit=dict(alpha_mu=alpha_mu, alpha_si=1e-3)))
    candidates.append(dict(name='unthresholded_mu3_sigma2',
                           model=dict(mu_deg=3, si_deg=2, pen='ridge', use_cv=False,
                                      thr1=0.0, thr2=0.0),
                           fit=dict(alpha_mu=1e-3, alpha_si=1e-3)))
    return candidates


def tune_stochastic_sindy(X, BP, dt, train_fraction=0.8, candidates=None):
    """Return tuned model, baseline model, selection details and trial records.

    Select using increment prediction on the last part of the supplied path.
    Actual drift, diffusion and Brownian motion are not inputs to selection.
    """
    X, BP = np.asarray(X).ravel(), np.asarray(BP).ravel()
    if len(X) != len(BP) or not 0 < train_fraction < 1 or dt <= 0:
        raise ValueError('Supply matching paths, a positive step and a train fraction between 0 and 1.')
    split = int(train_fraction*(len(X)-1))
    if split < 10 or split >= len(X)-1:
        raise ValueError('The path is too short for training and validation.')
    candidates = stochastic_sindy_candidates() if candidates is None else candidates
    records = []
    with threadpool_limits(limits=1):
        for candidate in candidates:
            print(f"Fitting {candidate['name']}", flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model = StochSINDy(**candidate['model']).fit(X[:split+1], BP[:split+1], dt,
                                                          **candidate['fit'])
            prediction = (model.mu(X[split:-1])*dt
                          + model.si(X[split:-1])*np.diff(BP)[split:])
            error = float(np.mean((np.diff(X)[split:]-prediction)**2))
            records.append(dict(
                **candidate, validation_mse=error if np.isfinite(error) else None,
                train_alpha=model.best_a, train_rho=model.best_r,
                drift_terms=int(np.count_nonzero(model.mu_c)),
                diffusion_terms=int(np.count_nonzero(model.si_c)),
                convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught),
            ))
        valid = [r for r in records if r['validation_mse'] is not None]
        if not valid:
            raise ValueError('No candidate produced finite validation predictions.')
        selected = min(valid, key=lambda r: r['validation_mse'])
        models = []
        for candidate in [selected, candidates[0]]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model = StochSINDy(**candidate['model']).fit(X, BP, dt, **candidate['fit'])
            models.append(model)
            if candidate is selected:
                final_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    selection = dict(selected, train_increments=split, validation_increments=len(X)-1-split,
                     full_fit_alpha=models[0].best_a, full_fit_rho=models[0].best_r,
                     full_fit_convergence_warnings=final_warnings)
    return models[0], models[1], selection, records

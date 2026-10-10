import pysindy as ps


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

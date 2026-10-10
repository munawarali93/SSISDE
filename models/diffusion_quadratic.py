import pysindy as ps


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

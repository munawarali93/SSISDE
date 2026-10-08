model = ps.SINDy(
    optimizer=ps.SR3(
        threshold=0.04,
        nu=0.05,
        thresholder='l1',
        normalize_columns=True,
        fit_intercept=False,
        max_iter=500,
        tol=1e-14,
    ),
    feature_library=ps.PolynomialLibrary(
        degree=4, include_interaction=True, include_bias=False,
    ),
)

model.fit(S_.reshape(-1, 1), t, x_dot=dQV.reshape(-1, 1))
print(model.equations())

model = ps.SINDy(
    optimizer=ps.SR3(
        threshold=5.0,          # for BS try 3.6
        nu=2.2,                  # for BS try 0.2
        thresholder='l1',
        normalize_columns=True,
        fit_intercept=False,
        max_iter=1500,
        tol=1e-12                # for BS try 1e-1
    ),
    feature_library=ps.PolynomialLibrary(degree=4, include_interaction=True, include_bias=True)
)

model.fit(S_.reshape(-1, 1), t, x_dot=dQV.reshape(-1, 1))
print(model.equations())
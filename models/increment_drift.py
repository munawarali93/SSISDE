import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def fit_increment_drift(x, dt, degrees=(1, 2, 3, 4, 5),
                        penalties=(0.0, 0.01, 0.1, 1.0, 10.0), folds=5, odd=False):
    """Select a polynomial drift from observed increments using time-series folds."""
    x = np.asarray(x).ravel()
    dx = np.diff(x)
    powers = np.arange(1, max(degrees) + 1, 2) if odd else np.arange(max(degrees) + 1)
    library = np.polynomial.polynomial.polyvander(x[:-1], max(degrees))[:, powers]
    scores = {(degree, penalty): [] for degree in degrees for penalty in penalties}

    def solve(matrix, increments, penalty):
        scale = np.sqrt(np.mean(matrix**2, axis=0))
        scale[scale == 0] = 1.0
        design = matrix / scale
        lhs = dt * (design.T @ design) + penalty * np.eye(design.shape[1])
        return np.linalg.solve(lhs, design.T @ increments) / scale

    for train, validation in TimeSeriesSplit(n_splits=folds).split(dx):
        for degree in degrees:
            design = library[:, powers <= degree]
            for penalty in penalties:
                coefficients = solve(design[train], dx[train], penalty)
                prediction = design[validation] @ coefficients
                score = np.mean(prediction**2 - 2 * prediction * dx[validation] / dt)
                scores[degree, penalty].append(float(score))

    search = [dict(degree=degree, penalty=penalty, validation_score=float(np.mean(values)),
                   standard_error=float(np.std(values, ddof=1) / np.sqrt(folds)), fold_scores=values)
              for (degree, penalty), values in scores.items()]
    best = min(search, key=lambda row: row['validation_score'])
    limit = best['validation_score'] + best['standard_error']
    candidates = [row for row in search if row['validation_score'] <= limit]
    settings = min(candidates, key=lambda row: (row['degree'], row['validation_score']))
    active = powers <= settings['degree']
    coefficients = np.zeros(settings['degree'] + 1)
    coefficients[powers[active]] = solve(library[:, active], dx, settings['penalty'])
    return coefficients, dict(settings), search

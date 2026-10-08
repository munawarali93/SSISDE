import numpy as np
from numpy.polynomial import Polynomial


def diffusion_derivatives(coefficients, x):
    """Derivatives of sigma from its fitted polynomial variance."""
    q = Polynomial(np.asarray(coefficients).ravel())
    variance = q(x)
    if np.any(variance <= 0):
        raise ValueError('The fitted variance must be positive.')
    sigma = np.sqrt(variance)
    q1, q2, q3 = q.deriv(1)(x), q.deriv(2)(x), q.deriv(3)(x)
    first = q1 / (2 * sigma)
    second = q2 / (2 * sigma) - q1**2 / (4 * sigma**3)
    third = q3 / (2 * sigma) - 3*q1*q2 / (4*sigma**3) + 3*q1**3 / (8*sigma**5)
    return first, second, third

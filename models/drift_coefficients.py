def drift_coefficients(mu_exact, sigma_pred, sigma_prime, sigma_2prime, l21):
    """Return the initial drift and coefficients a, b, and b_ for drift recovery."""
    mu_0 = mu_exact[0]
    a = sigma_prime / sigma_pred
    b = l21 / sigma_pred + 0.5 * (sigma_prime ** 2 + sigma_pred * sigma_2prime)
    G = -0.5 * (sigma_pred * sigma_prime ** 2 + sigma_pred ** 2 * sigma_2prime)
    b_ = G / sigma_pred + 0.5 * (sigma_prime ** 2 + sigma_pred * sigma_2prime)
    return (mu_0, a, b, b_)

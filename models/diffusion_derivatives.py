def diffusion_derivatives(sigma_pred_, S_, diff_method):
    """Return the first three diffusion derivatives on the supplied state grid."""
    sigma_prime_ = diff_method._differentiate(sigma_pred_.reshape(-1, 1), S_).flatten()
    sigma_2prime_ = diff_method._differentiate(sigma_prime_.reshape(-1, 1), S_).flatten()
    sigma_3prime_ = diff_method._differentiate(sigma_2prime_.reshape(-1, 1), S_).flatten()
    return (sigma_prime_, sigma_2prime_, sigma_3prime_)

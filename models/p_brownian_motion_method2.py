from models.p_brownian_motion import recover_p_noise


def recover_p_noise_method2(BQ, mu_pred2, sigma_pred, T):
    """Return the P-noise path, time step, and grid using Method 2 drift."""
    return recover_p_noise(BQ, mu_pred2, sigma_pred, T)

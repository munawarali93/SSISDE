from models.diffusion_derivatives import diffusion_derivatives


def coarse_diffusion_derivatives(sigma_pred, S, diff_method):
    """Return the first three diffusion derivatives on the coarse state grid."""
    return diffusion_derivatives(sigma_pred, S, diff_method)

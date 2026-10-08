import numpy as np


def recover_noise(BQ, drift, diffusion, dt):
    increments = np.diff(BQ) - drift[:-1] / diffusion[:-1] * dt
    return np.r_[0.0, np.cumsum(increments)]

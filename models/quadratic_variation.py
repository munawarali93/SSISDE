import numpy as np
import pysindy as ps
from tqdm import tqdm


def compute_quadratic_variation(W):
    return np.sum(np.diff(W) ** 2)


def estimate_quadratic_variation(S_, t):
    """Return quadratic variation, its derivative, and the difference method."""
    N = len(S_)
    QVS = np.zeros(N)
    for i in tqdm(range(N)):
        QVS[i] = compute_quadratic_variation(S_[0:i])
    QV = QVS.copy()
    diff_method = ps.FiniteDifference(order=3)
    dQV = diff_method._differentiate(QV.reshape(-1, 1), t).flatten()
    return (QV, dQV, diff_method)

import numpy as np

def pairwise_diff(a: list) -> np.ndarray:
    """
    Returns an (n, n) float64 array of signed pairwise differences.
    """
    a = np.array(a, dtype=np.float64)
    # a[:, None] for a [B], we will have [B, 1]?
    # 
    return  a[:, None] - a[None, :]

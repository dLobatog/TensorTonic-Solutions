import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    data = np.array(data, dtype=np.float64)
    mask = (data > threshold)
    keep_any = mask.any(axis=1, keepdims=True)
    keep_all = mask.all(axis=1, keepdims=True)
    a = mask.astype(np.float64)
    b = np.where(keep_any, data, 0.0)
    c = np.where(keep_all, data, 0.0)
    # print(mask, keep_any, keep_all)
    return np.stack([a, b, c])

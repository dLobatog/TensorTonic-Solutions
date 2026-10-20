import numpy as np

def original_and_clipped(data: list, row_idx: int, lo: float, hi: float) -> np.ndarray:
    """
    Returns a (2, n) float64 array: original row, then clipped row.
    """
    original = np.array(data, dtype=np.float64)[row_idx]
    mutated = np.where(original < lo, lo, original)
    mutated = np.where(original > hi, hi, mutated)
    return np.array([original, mutated])

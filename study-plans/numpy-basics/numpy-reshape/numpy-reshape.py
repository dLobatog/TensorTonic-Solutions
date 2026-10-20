import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    if operation == 'transpose':
        return np.array(data, dtype=np.float64).transpose()
    elif operation == 'flatten':
        return np.array(data, dtype=np.float64).reshape(-1)
    elif operation == 'add_batch':
        return np.array(data, dtype=np.float64)[None, :]
import numpy as np

def to_tensor_data(data):
    if isinstance(data, (list, np.ndarray)):
        return np.array(data, dtype=np.float32)
    return data

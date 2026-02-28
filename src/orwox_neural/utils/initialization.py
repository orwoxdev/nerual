import numpy as np
from ..core.autograd import Tensor

def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def he_normal(fan_in: int, fan_out: int) -> np.ndarray:
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, (fan_in, fan_out))

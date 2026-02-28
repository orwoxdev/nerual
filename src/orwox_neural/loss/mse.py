import numpy as np
from ..core.autograd import Tensor

def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    diff = y_pred - y_true
    return (diff**2).mean()

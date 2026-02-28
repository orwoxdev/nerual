import numpy as np
from ..core.autograd import Tensor

def cross_entropy_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    y_pred: Tensor of shape (B, C) containing probabilities (e.g. from Softmax)
    y_true: Tensor of shape (B, C) containing one-hot labels
    """
    epsilon = 1e-12
    # Clip to avoid log(0)
    probs = np.clip(y_pred.data, epsilon, 1. - epsilon)
    batch_size = y_pred.data.shape[0]

    loss_data = -np.sum(y_true.data * np.log(probs)) / batch_size
    out = Tensor(loss_data, requires_grad=True, _children=(y_pred, y_true))

    def _backward():
        if y_pred.requires_grad:
            # Derivative of CE with respect to probabilities
            y_pred.grad += -(y_true.data / probs) / batch_size

    out._backward = _backward
    return out

import numpy as np
from ..layers.base import Layer
from ..core.autograd import Tensor

class Softmax(Layer):
    def __init__(self, axis: int = -1, name: str = None):
        super().__init__(name)
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        exps = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        probs = exps / np.sum(exps, axis=self.axis, keepdims=True)
        out = Tensor(probs, requires_grad=x.requires_grad, _children=(x,))

        def _backward():
            if x.requires_grad:
                # Full Jacobian for softmax is complex, but for cross-entropy it simplifies.
                # Here we implement the general version.
                # For each sample in batch:
                # d_out / d_x = diag(probs) - outer(probs, probs)
                # grad_in = out.grad @ (diag(probs) - outer(probs, probs))

                # Optimized vectorized version:
                dx = np.zeros_like(x.data)
                for i in range(x.data.shape[0]):
                    p = probs[i].reshape(-1, 1)
                    jacobian = np.diagflat(p) - np.dot(p, p.T)
                    dx[i] = np.dot(jacobian, out.grad[i])
                x.grad += dx

        out._backward = _backward
        return out

    def backward(self, grad):
        pass

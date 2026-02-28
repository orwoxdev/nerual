import numpy as np
from ..layers.base import Layer
from ..core.autograd import Tensor

class Sigmoid(Layer):
    def forward(self, x: Tensor) -> Tensor:
        sig = 1 / (1 + np.exp(-x.data))
        out = Tensor(sig, requires_grad=x.requires_grad, _children=(x,))

        def _backward():
            if x.requires_grad:
                x.grad += (sig * (1 - sig)) * out.grad
        out._backward = _backward
        return out

    def backward(self, grad):
        pass

import numpy as np
from ..layers.base import Layer
from ..core.autograd import Tensor

class Tanh(Layer):
    def forward(self, x: Tensor) -> Tensor:
        t = np.tanh(x.data)
        out = Tensor(t, requires_grad=x.requires_grad, _children=(x,))

        def _backward():
            if x.requires_grad:
                x.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self, grad):
        pass

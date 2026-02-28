import numpy as np
from ..layers.base import Layer
from ..core.autograd import Tensor

class ReLU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        out_data = np.maximum(0, x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,))

        def _backward():
            if x.requires_grad:
                x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self, grad):
        pass

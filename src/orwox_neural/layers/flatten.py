import numpy as np
from .base import Layer
from ..core.autograd import Tensor

class Flatten(Layer):
    def forward(self, x: Tensor) -> Tensor:
        self.input_shape = x.shape
        batch_size = x.shape[0]
        out_data = x.data.reshape(batch_size, -1)
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,))

        def _backward():
            if x.requires_grad:
                x.grad += out.grad.reshape(x.shape)
        out._backward = _backward

        self.output_shape = out.shape
        return out

    def backward(self, grad):
        pass

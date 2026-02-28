import numpy as np
from .base import Layer
from ..core.autograd import Tensor

class Dropout(Layer):
    def __init__(self, p: float = 0.5, name: str = None):
        super().__init__(name)
        self.p = p
        self.mask = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x

        self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
        out_data = x.data * self.mask
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,))

        def _backward():
            if x.requires_grad:
                x.grad += out.grad * self.mask
        out._backward = _backward

        return out

    def backward(self, grad):
        pass

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p})
        return config

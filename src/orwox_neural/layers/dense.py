import numpy as np
from typing import List, Any
from .base import Layer
from ..core.autograd import Tensor
from ..utils.initialization import xavier_uniform, he_normal

class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, name: str = None, init_strategy: str = 'xavier'):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features

        if init_strategy == 'xavier':
            W_data = xavier_uniform(in_features, out_features)
        elif init_strategy == 'he':
            W_data = he_normal(in_features, out_features)
        else:
            W_data = np.random.randn(in_features, out_features) * 0.01

        self.weights = Tensor(W_data, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

        self.input_shape = (None, in_features)
        self.output_shape = (None, out_features)

    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x
        return x @ self.weights + self.bias

    def backward(self, grad: Any) -> Any:
        # With autograd, we don't strictly need to implement this if using Tensor operations
        # However, for the Sequential.backward implementation, we might need it
        # But if we use Tensor.backward(), it's automatic.
        # To keep it consistent with the requirement of manual backward support:
        pass

    def parameters(self):
        return [self.weights, self.bias]

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_features": self.in_features,
            "out_features": self.out_features
        })
        return config

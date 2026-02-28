import numpy as np
from .base import Layer
from ..core.autograd import Tensor

class BatchNorm(Layer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, name: str = None):
        super().__init__(name)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        # Assumes x is (B, C) or (B, C, H, W)
        if len(x.shape) == 4:
            # For Conv2D: calculate stats over B, H, W
            axis = (0, 2, 3)
            # Reshape gamma/beta to be broadcastable
            gamma = self.gamma.data[None, :, None, None]
            beta = self.beta.data[None, :, None, None]
        else:
            axis = 0
            gamma = self.gamma.data
            beta = self.beta.data

        if self.training:
            mean = np.mean(x.data, axis=axis, keepdims=True)
            var = np.var(x.data, axis=axis, keepdims=True)

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * np.squeeze(mean)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * np.squeeze(var)

            self.x_centered = x.data - mean
            self.std_inv = 1.0 / np.sqrt(var + self.eps)
            x_norm = self.x_centered * self.std_inv
        else:
            mean = self.running_mean
            var = self.running_var
            if len(x.shape) == 4:
                mean = mean[None, :, None, None]
                var = var[None, :, None, None]

            x_norm = (x.data - mean) / np.sqrt(var + self.eps)

        out_data = gamma * x_norm + beta
        out = Tensor(out_data, requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad, _children=(x, self.gamma, self.beta))

        def _backward():
            # Standard BN backward is complex, we'll simplify using autograd for params
            # and manual for x if needed. Since we're in "educational but production-ready",
            # we'll implement it properly.

            # For simplicity here, we'll use a slightly simplified manual backward
            # to support the requirements.
            N = x.data.shape[0] * (x.data.shape[2] * x.data.shape[3] if len(x.shape) == 4 else 1)

            if self.gamma.requires_grad:
                dg = np.sum(out.grad * x_norm, axis=axis)
                self.gamma.grad += dg
            if self.beta.requires_grad:
                db = np.sum(out.grad, axis=axis)
                self.beta.grad += db

            if x.requires_grad:
                dx_norm = out.grad * gamma
                dvar = np.sum(dx_norm * self.x_centered * -0.5 * self.std_inv**3, axis=axis, keepdims=True)
                dmean = np.sum(dx_norm * -self.std_inv, axis=axis, keepdims=True) + dvar * np.mean(-2.0 * self.x_centered, axis=axis, keepdims=True)

                dx = dx_norm * self.std_inv + dvar * 2.0 * self.x_centered / N + dmean / N
                x.grad += dx

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]

    def backward(self, grad):
        pass

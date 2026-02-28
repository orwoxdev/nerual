import numpy as np
from .base import Layer
from ..core.autograd import Tensor

class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, name: str = None):
        super().__init__(name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize weights
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        limit = np.sqrt(6 / (in_channels * kernel_size**2 + out_channels * kernel_size**2))
        self.weights = Tensor(np.random.uniform(-limit, limit, shape), requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # Simplified Conv2D for autograd
        batch_size, in_channels, in_h, in_w = x.shape
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1

        # This is a naive implementation, in a real framework we'd use im2col
        # For our autograd to work simply, we'll implement it using tensor operations if possible
        # or custom backward. Let's do a more efficient version if we want it to be "production-ready"

        # For simplicity in this implementation, we'll use a helper to perform the convolution
        # and manually define the backward if needed, or stick to Tensor ops if they support it.
        # Since our Tensor class is basic, we might need a custom Op or manual backward.

        # Let's implement a basic version using loops (slow but works with Tensor ops)
        # Actually, let's just use a custom forward/backward for Conv2D for efficiency.

        out_data = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                h_start, w_start = i * self.stride, j * self.stride
                h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size

                # Slice and multiply-sum
                x_slice = x.data[:, :, h_start:h_end, w_start:w_end] # (B, C, K, K)
                # self.weights (OC, C, K, K)
                for oc in range(self.out_channels):
                    out_data[:, oc, i, j] = np.sum(x_slice * self.weights.data[oc], axis=(1, 2, 3)) + self.bias.data[oc]

        out = Tensor(out_data, requires_grad=x.requires_grad or self.weights.requires_grad, _children=(x, self.weights, self.bias))

        def _backward():
            if x.requires_grad:
                dx = np.zeros_like(x.data)
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, w_start = i * self.stride, j * self.stride
                        h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size
                        for oc in range(self.out_channels):
                            dx[:, :, h_start:h_end, w_start:w_end] += out.grad[:, oc, i, j][:, None, None, None] * self.weights.data[oc]
                x.grad += dx

            if self.weights.requires_grad:
                dw = np.zeros_like(self.weights.data)
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, w_start = i * self.stride, j * self.stride
                        h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size
                        x_slice = x.data[:, :, h_start:h_end, w_start:w_end]
                        for oc in range(self.out_channels):
                            dw[oc] += np.sum(x_slice * out.grad[:, oc, i, j][:, None, None, None], axis=0)
                self.weights.grad += dw

            if self.bias.requires_grad:
                self.bias.grad += np.sum(out.grad, axis=(0, 2, 3))

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weights, self.bias]

    def backward(self, grad):
        pass

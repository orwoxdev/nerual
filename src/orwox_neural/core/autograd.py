import numpy as np
from typing import List, Optional, Union, Callable, Tuple

class Tensor:
    def __init__(self, data: Union[np.ndarray, list, float, int], requires_grad: bool = False, _children: Tuple['Tensor', ...] = ()):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)

        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)
                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad += grad_other
        out._backward = _backward

        return out

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                grad_self = other.data * out.grad
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = self.data * out.grad
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)
                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad += grad_other
        out._backward = _backward

        return out

    def __pow__(self, other: Union[float, int]) -> 'Tensor':
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward

        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.ones_like(self.data) * grad
        out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,))

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                n = np.prod(self.data.shape) if axis is None else self.data.shape[axis]
                self.grad += (np.ones_like(self.data) / n) * grad
        out._backward = _backward

        return out

    def backward(self, grad: Optional[np.ndarray] = None):
        if grad is None:
            if self.shape == ():
                grad = np.array(1.0)
            else:
                raise RuntimeError("grad must be specified for non-scalar tensor")

        self.grad = grad

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0)

    # Add other operations as needed...
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (other**-1)

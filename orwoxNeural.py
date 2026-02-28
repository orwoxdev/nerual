"""
Orwox Neural Core - Standalone Version (orwoxNeural.py)
A single-file, production-ready neural framework for educational and research purposes.
"""

import numpy as np
import time
import json
from typing import List, Union, Tuple, Optional, Any, Dict, Callable

# --- Core Engine ---

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
    def shape(self): return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))
        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                while grad_self.ndim > self.data.ndim: grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1: grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad
                while grad_other.ndim > other.data.ndim: grad_other = grad_other.sum(axis=0)
                for axis, size in enumerate(other.data.shape):
                    if size == 1: grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad += grad_other
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))
        def _backward():
            if self.requires_grad:
                grad_self = other.data * out.grad
                while grad_self.ndim > self.data.ndim: grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1: grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = self.data * out.grad
                while grad_other.ndim > other.data.ndim: grad_other = grad_other.sum(axis=0)
                for axis, size in enumerate(other.data.shape):
                    if size == 1: grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad += grad_other
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Tensor(self.data**other, requires_grad=self.requires_grad, _children=(self,))
        def _backward():
            if self.requires_grad: self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other))
        def _backward():
            if self.requires_grad: self.grad += out.grad @ other.data.T
            if other.requires_grad: other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,))
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims: grad = np.expand_dims(grad, axis=axis)
                self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,))
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims: grad = np.expand_dims(grad, axis=axis)
                n = np.prod(self.data.shape) if axis is None else self.data.shape[axis]
                self.grad += (np.ones_like(self.data) / n) * grad
        out._backward = _backward
        return out

    def backward(self, grad: Optional[np.ndarray] = None):
        if grad is None:
            if self.shape == (): grad = np.array(1.0)
            else: raise RuntimeError("grad must be specified for non-scalar tensor")
        self.grad = grad
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build_topo(child)
                topo.append(v)
        build_topo(self)
        for v in reversed(topo): v._backward()

    def zero_grad(self):
        if self.grad is not None: self.grad.fill(0)

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (other**-1)

# --- Layers ---

class Layer:
    def __init__(self, name: str = None):
        self.name = name
        self.training = True
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError
    def parameters(self) -> List[Tensor]: return []
    def train(self): self.training = True
    def eval(self): self.training = False
    def get_config(self): return {"name": self.name, "type": self.__class__.__name__}

class Dense(Layer):
    def __init__(self, in_f, out_f, name=None):
        super().__init__(name)
        limit = np.sqrt(6 / (in_f + out_f))
        self.weights = Tensor(np.random.uniform(-limit, limit, (in_f, out_f)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_f), requires_grad=True)
    def forward(self, x): return x @ self.weights + self.bias
    def parameters(self): return [self.weights, self.bias]

class Conv2D(Layer):
    def __init__(self, in_c, out_c, k, stride=1, name=None):
        super().__init__(name)
        self.in_c, self.out_c, self.k, self.stride = in_c, out_c, k, stride
        limit = np.sqrt(6 / (in_c * k*k + out_c * k*k))
        self.weights = Tensor(np.random.uniform(-limit, limit, (out_c, in_c, k, k)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_c), requires_grad=True)
    def forward(self, x):
        b, c, h, w = x.shape
        oh = (h - self.k) // self.stride + 1
        ow = (w - self.k) // self.stride + 1
        out_data = np.zeros((b, self.out_c, oh, ow))
        for i in range(oh):
            for j in range(ow):
                hs, ws = i*self.stride, j*self.stride
                he, we = hs+self.k, ws+self.k
                xs = x.data[:, :, hs:he, ws:we]
                for oc in range(self.out_c):
                    out_data[:, oc, i, j] = np.sum(xs * self.weights.data[oc], axis=(1, 2, 3)) + self.bias.data[oc]
        out = Tensor(out_data, requires_grad=x.requires_grad or self.weights.requires_grad, _children=(x, self.weights, self.bias))
        def _backward():
            if x.requires_grad:
                dx = np.zeros_like(x.data)
                for i in range(oh):
                    for j in range(ow):
                        hs, ws = i*self.stride, j*self.stride
                        he, we = hs+self.k, ws+self.k
                        for oc in range(self.out_c):
                            dx[:, :, hs:he, ws:we] += out.grad[:, oc, i, j][:, None, None, None] * self.weights.data[oc]
                x.grad += dx
            if self.weights.requires_grad:
                dw = np.zeros_like(self.weights.data)
                for i in range(oh):
                    for j in range(ow):
                        hs, ws = i*self.stride, j*self.stride
                        he, we = hs+self.k, ws+self.k
                        xs = x.data[:, :, hs:he, ws:we]
                        for oc in range(self.out_c):
                            dw[oc] += np.sum(xs * out.grad[:, oc, i, j][:, None, None, None], axis=0)
                self.weights.grad += dw
            if self.bias.requires_grad: self.bias.grad += np.sum(out.grad, axis=(0, 2, 3))
        out._backward = _backward
        return out
    def parameters(self): return [self.weights, self.bias]

class Flatten(Layer):
    def forward(self, x):
        out = Tensor(x.data.reshape(x.shape[0], -1), requires_grad=x.requires_grad, _children=(x,))
        def _backward():
            if x.requires_grad: x.grad += out.grad.reshape(x.shape)
        out._backward = _backward
        return out

class Dropout(Layer):
    def __init__(self, p=0.5, name=None):
        super().__init__(name)
        self.p = p
    def forward(self, x):
        if not self.training: return x
        mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
        out = Tensor(x.data * mask, requires_grad=x.requires_grad, _children=(x,))
        def _backward():
            if x.requires_grad: x.grad += out.grad * mask
        out._backward = _backward
        return out

class BatchNorm(Layer):
    def __init__(self, num_f, eps=1e-5, mom=0.1, name=None):
        super().__init__(name)
        self.num_f, self.eps, self.mom = num_f, eps, mom
        self.gamma, self.beta = Tensor(np.ones(num_f), True), Tensor(np.zeros(num_f), True)
        self.rm, self.rv = np.zeros(num_f), np.ones(num_f)
    def forward(self, x):
        axis = (0, 2, 3) if x.data.ndim == 4 else 0
        g = self.gamma.data[None, :, None, None] if x.data.ndim == 4 else self.gamma.data
        b = self.beta.data[None, :, None, None] if x.data.ndim == 4 else self.beta.data
        if self.training:
            m = np.mean(x.data, axis=axis, keepdims=True)
            v = np.var(x.data, axis=axis, keepdims=True)
            self.rm = (1-self.mom)*self.rm + self.mom*np.squeeze(m)
            self.rv = (1-self.mom)*self.rv + self.mom*np.squeeze(v)
            xc = x.data - m
            si = 1.0 / np.sqrt(v + self.eps)
            xn = xc * si
        else:
            m = self.rm[None, :, None, None] if x.data.ndim == 4 else self.rm
            v = self.rv[None, :, None, None] if x.data.ndim == 4 else self.rv
            xn = (x.data - m) / np.sqrt(v + self.eps)
        out = Tensor(g * xn + b, x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad, (x, self.gamma, self.beta))
        def _backward():
            N = x.data.size / self.num_f
            if self.gamma.requires_grad: self.gamma.grad += np.sum(out.grad * xn, axis=axis)
            if self.beta.requires_grad: self.beta.grad += np.sum(out.grad, axis=axis)
            if x.requires_grad:
                dxn = out.grad * g
                dv = np.sum(dxn * xc * -0.5 * si**3, axis=axis, keepdims=True)
                dm = np.sum(dxn * -si, axis=axis, keepdims=True) + dv * np.mean(-2.0 * xc, axis=axis, keepdims=True)
                x.grad += dxn * si + dv * 2.0 * xc / N + dm / N
        out._backward = _backward
        return out
    def parameters(self): return [self.gamma, self.beta]

# --- Activations ---

class ReLU(Layer):
    def forward(self, x):
        out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad, _children=(x,))
        def _backward():
            if x.requires_grad: x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        return out

class Sigmoid(Layer):
    def forward(self, x):
        s = 1 / (1 + np.exp(-x.data))
        out = Tensor(s, x.requires_grad, (x,))
        def _backward():
            if x.requires_grad: x.grad += (s * (1 - s)) * out.grad
        out._backward = _backward
        return out

class Tanh(Layer):
    def forward(self, x):
        t = np.tanh(x.data)
        out = Tensor(t, x.requires_grad, (x,))
        def _backward():
            if x.requires_grad: x.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

class Softmax(Layer):
    def __init__(self, axis=-1, name=None):
        super().__init__(name)
        self.axis = axis
    def forward(self, x):
        ex = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        p = ex / np.sum(ex, axis=self.axis, keepdims=True)
        out = Tensor(p, x.requires_grad, (x,))
        def _backward():
            if x.requires_grad:
                dx = np.zeros_like(x.data)
                for i in range(x.data.shape[0]):
                    pi = p[i].reshape(-1, 1)
                    jac = np.diagflat(pi) - np.dot(pi, pi.T)
                    dx[i] = np.dot(jac, out.grad[i])
                x.grad += dx
        out._backward = _backward
        return out

# --- Sequential ---

class Sequential:
    def __init__(self, layers: List[Layer] = None):
        self.layers = layers or []
        self.hooks = []
    def add(self, layer): self.layers.append(layer)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            for hook in self.hooks: hook(layer.name or f"{layer.__class__.__name__}_{i}", x, x.shape, i)
        return x
    def parameters(self): return [p for layer in self.layers for p in layer.parameters()]
    def train(self):
        for l in self.layers: l.train()
    def eval(self):
        for l in self.layers: l.eval()
    def register_hook(self, callback): self.hooks.append(callback)
    def inspect(self):
        return {"layers": [{"name": l.name or f"{l.__class__.__name__}_{i}", "type": l.__class__.__name__} for i, l in enumerate(self.layers)], "parameter_count": sum(p.data.size for p in self.parameters())}

# --- Optimizers ---

class Optimizer:
    def __init__(self, params, lr=0.001):
        self.params, self.lr = [p for p in params if p.requires_grad], lr
    def zero_grad(self):
        for p in self.params: p.zero_grad()

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params, lr)
        self.mom = momentum
        self.v = [np.zeros_like(p.data) for p in self.params]
    def step(self):
        for i, p in enumerate(self.params):
            if self.mom > 0:
                self.v[i] = self.mom * self.v[i] + (1 - self.mom) * p.grad
                p.data -= self.lr * self.v[i]
            else: p.data -= self.lr * p.grad

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (p.grad**2)
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)

# --- Training ---

class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model, self.loss_fn, self.optimizer = model, loss_fn, optimizer
    def fit(self, X, y, epochs=10, batch_size=32):
        X, y = np.array(X), np.array(y)
        for epoch in range(epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            el = 0
            for i in range(0, len(X), batch_size):
                idx = indices[i:i+batch_size]
                bx, by = Tensor(X[idx]), Tensor(y[idx])
                preds = self.model.forward(bx)
                loss = self.loss_fn(preds, by)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                el += loss.data * len(idx)
            print(f"Epoch {epoch+1}/{epochs} - loss: {el/len(X):.4f}")
    def evaluate(self, X, y):
        self.model.eval()
        preds = self.model.forward(Tensor(X))
        return mse_loss(preds, Tensor(y)).data, np.mean((preds.data > 0.5) == (y > 0.5))

# --- Loss ---

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

def cross_entropy_loss(y_pred, y_true):
    p = np.clip(y_pred.data, 1e-12, 1.0 - 1e-12)
    loss = -np.sum(y_true.data * np.log(p)) / y_pred.data.shape[0]
    out = Tensor(loss, True, (y_pred, y_true))
    def _backward():
        if y_pred.requires_grad: y_pred.grad += -(y_true.data / p) / y_pred.data.shape[0]
    out._backward = _backward
    return out

# --- Visualization ---

def export_model_to_json(model, filepath):
    data = {"architecture": [], "weights": {}, "metadata": {"parameter_count": sum(p.data.size for p in model.parameters())}}
    for i, layer in enumerate(model.layers):
        name = layer.name or f"{layer.__class__.__name__}_{i}"
        data["architecture"].append({"name": name, "type": layer.__class__.__name__})
        params = layer.parameters()
        if params: data["weights"][name] = [p.data.tolist() for p in params]
    with open(filepath, 'w') as f: json.dump(data, f)

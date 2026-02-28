import numpy as np
from .base import Optimizer
from typing import List
from ..core.autograd import Tensor

class SGD(Optimizer):
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + (1 - self.momentum) * p.grad
                p.data -= self.lr * self.velocities[i]
            else:
                p.data -= self.lr * p.grad

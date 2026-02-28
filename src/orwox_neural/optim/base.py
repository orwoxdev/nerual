from typing import List
from ..core.autograd import Tensor

class Optimizer:
    def __init__(self, parameters: List[Tensor], lr: float = 0.001):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

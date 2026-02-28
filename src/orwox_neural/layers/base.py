from abc import ABC, abstractmethod
from typing import List, Any
from ..core.autograd import Tensor

class Layer(ABC):
    def __init__(self, name: str = None):
        self.name = name
        self.training = True
        self.input_shape = None
        self.output_shape = None

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward(self, grad: Any) -> Any:
        # Note: with autograd, explicit backward is often handled by Tensor.backward()
        # but for some layers we might still need manual implementation or gradient tracking.
        pass

    def parameters(self) -> List[Tensor]:
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_config(self):
        return {"name": self.name, "type": self.__class__.__name__}

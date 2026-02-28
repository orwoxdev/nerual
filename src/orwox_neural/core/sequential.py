from typing import List, Any, Dict
from .hooks import HookManager
import time

class Sequential:
    def __init__(self, layers: List[Any] = None):
        self.layers = layers if layers is not None else []
        self.hook_manager = HookManager()
        self.forward_timeline = []

    def add(self, layer: Any):
        self.layers.append(layer)

    def forward(self, x: Any) -> Any:
        self.forward_timeline = []
        out = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)

            # Record timeline
            self.forward_timeline.append({
                "layer_index": i,
                "layer_name": layer.name if hasattr(layer, 'name') else f"Layer_{i}",
                "timestamp": time.time(),
                "output_shape": out.shape
            })

            # Trigger hooks
            self.hook_manager.trigger_hooks(
                layer.name if hasattr(layer, 'name') else f"Layer_{i}",
                out,
                i
            )
        return out

    def backward(self, grad: Any):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def register_hook(self, layer_name: str, callback: callable):
        return self.hook_manager.register_hook(layer_name, callback)

    def inspect(self) -> Dict:
        layers_info = []
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_params = layer.parameters()
            num_params = sum(p.data.size for p in layer_params)
            total_params += num_params

            layers_info.append({
                "name": layer.name if hasattr(layer, 'name') else f"Layer_{i}",
                "type": layer.__class__.__name__,
                "input_shape": getattr(layer, 'input_shape', None),
                "output_shape": getattr(layer, 'output_shape', None),
                "weights_shape": [p.shape for p in layer_params if len(p.shape) > 1],
                "parameter_count": num_params
                # activation_stats and gradient_stats can be added if needed
            })

        return {
            "layers": layers_info,
            "timeline": self.forward_timeline,
            "parameter_count": total_params
        }

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

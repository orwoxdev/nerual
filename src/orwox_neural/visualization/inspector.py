import numpy as np

class Inspector:
    def __init__(self, model):
        self.model = model

    def get_layer_stats(self):
        stats = []
        for i, layer in enumerate(self.model.layers):
            layer_info = {
                "name": layer.name if hasattr(layer, 'name') and layer.name else f"Layer_{i}",
                "type": layer.__class__.__name__,
                "params": {}
            }

            for p_idx, p in enumerate(layer.parameters()):
                layer_info["params"][f"param_{p_idx}"] = {
                    "shape": p.shape,
                    "mean": float(np.mean(p.data)),
                    "std": float(np.std(p.data)),
                    "grad_mean": float(np.mean(p.grad)) if p.grad is not None else 0.0
                }
            stats.append(layer_info)
        return stats

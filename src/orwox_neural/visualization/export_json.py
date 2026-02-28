import json
import numpy as np

def export_model_to_json(model, filepath=None):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    model_data = {
        "architecture": [],
        "weights": {},
        "metadata": {
            "parameter_count": sum(p.data.size for p in model.parameters())
        }
    }

    for i, layer in enumerate(model.layers):
        layer_name = layer.name if hasattr(layer, 'name') and layer.name else f"Layer_{i}"
        layer_info = {
            "name": layer_name,
            "type": layer.__class__.__name__,
            "config": layer.get_config() if hasattr(layer, 'get_config') else {}
        }
        model_data["architecture"].append(layer_info)

        # Export weights
        params = layer.parameters()
        if params:
            model_data["weights"][layer_name] = [convert(p.data) for p in params]

    if filepath:
        with open(filepath, 'w') as f:
            json.dump(model_data, f, default=convert)

    return model_data

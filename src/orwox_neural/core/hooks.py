import time
from typing import Callable, List, Any

class Hook:
    def __init__(self, layer_name: str, callback: Callable):
        self.layer_name = layer_name
        self.callback = callback

class HookManager:
    def __init__(self):
        self.hooks: List[Hook] = []

    def register_hook(self, layer_name: str, callback: Callable):
        hook = Hook(layer_name, callback)
        self.hooks.append(hook)
        return hook

    def trigger_hooks(self, layer_name: str, output: Any, timestamp_index: int):
        for hook in self.hooks:
            if hook.layer_name == layer_name:
                hook.callback(layer_name, output, output.shape, timestamp_index)

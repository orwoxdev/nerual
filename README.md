# Orwox Neural Framework

Orwox Neural is a production-ready yet educational neural network framework written entirely in Python using only NumPy. It is designed to power the foundational AI infrastructure for Orwox.com and provide a transparent, modular system for students, teachers, and developers.

## Features

- **Lightweight Autograd Engine**: Automatic gradient propagation via a dynamic computational graph.
- **Modular Architecture**: Easy-to-extend layers, optimizers, and loss functions.
- **Hook System**: Real-time layer output capture for monitoring and visualization.
- **Introspection API**: Detailed model state and parameter statistics.
- **3D Visualization Ready**: Built-in JSON export for WebGL/Three.js renderers.
- **Standalone Version**: Includes a single-file version (`orwoxNeural.py`) for quick integration.

## Installation

```bash
pip install orwox-neural
```

## GitHub Repository

[https://github.com/orwoxdev/nerual](https://github.com/orwoxdev/nerual)

Or simply copy `orwoxNeural.py` into your project.

## How to Publish to PyPI (Public Library)

To make this a public library available via `pip install`, follow these steps:

1. **Install Build Tools**:
   ```bash
   pip install --upgrade build twine
   ```

2. **Build the Package**:
   ```bash
   python3 -m build
   ```
   This will create a `dist/` directory with `.whl` and `.tar.gz` files.

3. **Check the Package**:
   ```bash
   twine check dist/*
   ```

4. **Upload to TestPyPI (Optional but recommended)**:
   ```bash
   twine upload --repository testpypi dist/*
   ```

5. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```
   *Note: You will need a PyPI account and an API token.*

## Quick Start

```python
import numpy as np
from orwox_neural.core import Sequential, Tensor
from orwox_neural.layers import Dense
from orwox_neural.activations import ReLU
from orwox_neural.loss import mse_loss
from orwox_neural.optim import SGD
from orwox_neural.training import Trainer

# Define Model
model = Sequential([
    Dense(2, 4, name="Hidden"),
    ReLU(),
    Dense(4, 1, name="Output")
])

# Setup Trainer
optimizer = SGD(model.parameters(), lr=0.1)
trainer = Trainer(model, mse_loss, optimizer)

# Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]]) # XOR

# Train
trainer.fit(X, y, epochs=100, batch_size=4)
```

## Advanced Usage

### Hook System Example

```python
def my_hook(layer_name, output, shape, timestamp):
    print(f"Layer {layer_name} processed data with shape {shape}")

model.register_hook("Hidden", my_hook)
```

### JSON Export for 3D Visualization

```python
from orwox_neural.visualization import export_model_to_json

export_model_to_json(model, "model_viz.json")
```

## Vision Statement

Orwox Neural Core aims to demystify artificial intelligence by providing a clear, hackable, and high-performance implementation of neural network fundamentals. We believe that by making the internal workings of AI accessible and visualizable, we can empower the next generation of AI researchers and developers.

## License

MIT

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from orwox_neural.core import Sequential, Tensor
from orwox_neural.layers import Dense
from orwox_neural.activations import ReLU
from orwox_neural.loss import mse_loss
from orwox_neural.optim import SGD
from orwox_neural.training import Trainer
from orwox_neural.visualization import export_model_to_json

def test_xor():
    print("Testing XOR with Orwox Neural...")

    # Define Model
    model = Sequential([
        Dense(2, 8, name="Hidden"),
        ReLU(),
        Dense(8, 1, name="Output")
    ])

    # Hook example
    def forward_hook(layer_name, output, shape, timestamp):
        # Only print occasionally to not clutter
        if timestamp == 0:
            print(f"Hook: {layer_name} output shape: {shape}")

    model.register_hook("Hidden", forward_hook)

    # Setup Trainer
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, mse_loss, optimizer)

    # XOR Data
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    # Train
    trainer.fit(X, y, epochs=500, batch_size=4)

    # Evaluate
    loss, acc = trainer.evaluate(X, y)
    print(f"Final Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Inspect
    inspection = model.inspect()
    print(f"Model total parameters: {inspection['parameter_count']}")

    # Export
    export_model_to_json(model, "xor_model.json")
    print("Exported model to xor_model.json")

    if acc > 0.7:
        print("XOR Test Passed!")
    else:
        print("XOR Test Failed (Accuracy too low).")

if __name__ == "__main__":
    test_xor()

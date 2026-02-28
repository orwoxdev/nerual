"""
Core engine utilities for Orwox Neural.
"""
from .autograd import Tensor
from .sequential import Sequential

def get_device():
    """Returns the current computing device (always 'cpu' for this NumPy implementation)."""
    return "cpu"

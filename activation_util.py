import numpy as np
from typing import List

class ActivationUtil:
    @staticmethod
    def relu(x: float) -> float:
        """Rectified Linear Unit activation."""
        return max(0.0, x)

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid activation: 1 / (1 + exp(-x))."""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def tanh(x: float) -> float:
        """Hyperbolic tangent activation."""
        return np.tanh(x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation across a 1D or 2D array."""
        e_x = np.exp(x - np.max(x))  # numerical stability
        return e_x / e_x.sum(axis=0, keepdims=True)

    @staticmethod
    def leaky_relu(x: float, alpha: float = 0.01) -> float:
        """Leaky ReLU allowing a small gradient when x < 0."""
        return x if x > 0 else alpha * x

    @staticmethod
    def linear(x: float) -> float:
        """Linear (identity) activation."""
        return x
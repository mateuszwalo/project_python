"""Podstawowe funkcje aktywacji 

Każda klasa dziedziczy po abstrakcji Layer i implementuje forward/backward.
Warstwy nie mają parametrów do uczenia, więc metoda `params` zwraca pusty słownik.
"""
from __future__ import annotations

import numpy as np

from .base import Layer


class ReLU(Layer):
    """Funkcja aktywacji ReLU: f(x) = max(0, x)."""
    def __init__(self) -> None:
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.mask is not None, "forward musi być wywołane przed backward"
        return grad * self.mask

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Brak parametrów do uczenia"""
        return {}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        """Brak gradientów dla tej warstwy"""
        return {}

    def __str__(self) -> str:
        return "ReLU()"


class LeakyReLU(Layer):
    """LeakyReLU z parametrem alpha."""
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask + self.alpha * x * (~self.mask)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.mask is not None, "forward musi być wywołane przed backward"
        return grad * (self.mask + self.alpha * (~self.mask))

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Brak parametrów do uczenia"""
        return {}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        """Brak gradientów dla tej warstwy"""
        return {}

    def __str__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class Softmax(Layer):
    """Warstwa Softmax z funkcją kosztu Cross-Entropy w Model.fit."""
    def __init__(self) -> None:
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.out is not None, "forward musi być wywołane przed backward"
        S = self.out
        dx = np.empty_like(grad)
        for i in range(S.shape[0]):
            s = S[i].reshape(-1, 1)
            jac = np.diagflat(s) - s @ s.T
            dx[i] = jac @ grad[i]
        return dx

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Brak parametrów do uczenia"""
        return {}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        """Brak gradientów dla tej warstwy"""
        return {}

    def __str__(self) -> str:
        return "Softmax()"

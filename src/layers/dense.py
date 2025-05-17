"""Warstwa gęsta (Fully Connected) napisana wyłącznie w NumPy.

Zawiera proste uczenie metodą SGD w funkcji `backward`.  Wektor biasu `b`
jest tablicą o kształcie (1, out_features), co ułatwia broadcast przy
obliczaniu x @ W + b.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any

from .base import Layer


class Dense(Layer):
    """Warstwa gęsta (w pełni połączona)."""

    def __init__(self, in_features: int, out_features: int, *, lr: float | None = None) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr  # jeżeli None, zostanie nadpisane przez Model.fit

        # inicjalizacja He
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))

        # gradienty (wypełniane w backward)
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

        # cache – wejście z forward potrzebne w backward
        self._x: np.ndarray | None = None

    # ---------- API warstwy ---------- #

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propagacja w przód: y = x @ W + b."""
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Liczy gradient i (jeśli ustawiono lr) wykonuje krok SGD."""
        if self._x is None:
            raise RuntimeError("Wywołano backward przed forward")

        m = self._x.shape[0]           # rozmiar batcha
        self.dW = self._x.T @ grad / m
        self.db = grad.sum(axis=0, keepdims=True) / m

        grad_prev = grad @ self.W.T    # gradient dla warstwy poprzedniej

        if self.lr is not None:        # aktualizacja wag
            self.W -= self.lr * self.dW
            self.b -= self.lr * self.db

        return grad_prev

    # ---------- Akcesory ---------- #

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> Dict[str, np.ndarray | None]:
        return {"dW": self.dW, "db": self.db}

    def __str__(self) -> str:                 # noqa: DunderStr
        return f"Dense({self.in_features}→{self.out_features})"

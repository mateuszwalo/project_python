"""Abstrakcyjne definicje wszystkich warstw używanych w **NumPyLayer Lab**.

Projekt korzysta wyłącznie z NumPy – bez zewnętrznych bibliotek ML.
Każda konkretna warstwa musi dziedziczyć po :class:`Layer` i implementować
metody :py:meth:`forward` (propagacja w przód) oraz :py:meth:`backward`
(liczenie gradientu i opcjonalną aktualizację wag).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class Layer(ABC):
    """Abstrakcyjna baza dla wszystkich warstw.

    ----
    Metody do zaimplementowania w podklasach
    ---------------------------------------
    forward(x)
        Zwraca wynik propagacji w przód.
    backward(grad: np.ndarray, lr: float) -> np.ndarray
        Oblicza gradient względem wejścia i – jeśli warstwa ma parametry –
        aktualizuje je przy użyciu tempa uczenia *lr*.

    Warstwa może przechowywać atrybuty:
    * params – słownik trenowalnych parametrów, np. {'W': ..., 'b': ...}
    * grads  – odpowiadające im gradienty po ostatnim `backward`.
    """

    def __init__(self) -> None:
        # Słowniki parametrów i gradientów inicjalizujemy pustymi,
        # żeby wszystkie warstwy miały wspólne API, nawet jeśli
        # dana warstwa nie posiada trenowalnych wag.
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}

    # ---------------------- API wymagane przez podklasy -------------------- #

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propagacja w przód.  Zwraca wyjście warstwy."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        """
        Propagacja wsteczna.

        Parametry
        ---------
        grad
            Gradient straty względem wyjścia warstwy.
        lr
            Tempo uczenia używane do aktualizacji parametrów.

        Zwraca gradient względem wejścia warstwy.
        """
        raise NotImplementedError

    # ------------------------- Metody pomocnicze --------------------------- #

    def zero_grad(self) -> None:
        """Zeruje wszystkie gradienty przed kolejną iteracją treningu."""
        for g in self.grads.values():
            g.fill(0.0)

    # --------------------------- Metody magiczne --------------------------- #

    def __str__(self) -> str:  # pragma: no cover
        """Łatwy, czytelny opis warstwy przy `print(layer)`."""
        name = self.__class__.__name__
        shapes = [p.shape for p in self.params.values()]
        return f"{name}(params={len(self.params)}, shapes={shapes})"

    def __len__(self) -> int:
        """
        Zwraca liczbę trenowalnych macierzy w warstwie.
        Dzięki temu `len(layer)` ma sensowne znaczenie.
        """
        return len(self.params)

    def __eq__(self, other: object) -> bool:
        """Porównanie dwóch warstw po typie i kształtach parametrów."""
        if not isinstance(other, Layer):
            return NotImplemented
        if self.__class__ is not other.__class__:
            return False
        return all(p.shape == q.shape for p, q in zip(self.params.values(), other.params.values()))

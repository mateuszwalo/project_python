"""Narzędzia pomocnicze (helpery) używane w **NumPyLayer Lab**
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

__all__: list[str] = [
    "set_seed",
    "one_hot",
    "batch_generator",
]


# ---------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Ustawia ziarno generatora losowego NumPy.

    Dzięki temu kolejne uruchomienia programu dadzą identyczne wagi
    początkowe i takie same wyniki losowych operacji (shuffle itp.).
    """
    np.random.seed(seed)


# ---------------------------------------------------------------
# One‑hot encoding
# ---------------------------------------------------------------

def one_hot(y: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Zamienia wektor etykiet (shape: ``(n_samples,)``) na macierz one‑hot.

    Parametry
    ----------
    y : np.ndarray
        Wektor liczb całkowitych z etykietami klas.
    num_classes : int, opcjonalnie
        Liczba klas; jeśli ``None`` zostanie wyliczona jako ``y.max() + 1``.

    Zwraca
    -------
    np.ndarray
        Macierz o kształcie ``(n_samples, num_classes)``.
    """
    if y.ndim != 1:
        raise ValueError("one_hot: oczekiwano wektora 1‑D z etykietami.")

    if num_classes is None:
        num_classes = int(y.max() + 1)

    one_hot_matrix = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot_matrix[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return one_hot_matrix


# ---------------------------------------------------------------
# Mini‑batch generator
# ---------------------------------------------------------------

def batch_generator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generator zwracający kolejne mini‑batchy.

    Przyjmuje dwie tablice NumPy – dane i etykiety – które muszą mieć
    pierwszą dimensję równą liczbie próbek.

    Parametry
    ----------
    X : np.ndarray
        Dane wejściowe (``(n_samples, ...)``).
    y : np.ndarray
        Odpowiedzi/etykiety (``(n_samples, ...)``).
    batch_size : int
        Maksymalna liczebność jednego batcha.
    shuffle : bool
        Czy tasować dane przed każdą epoką.

    Zwraca
    -------
    Iterator[Tuple[np.ndarray, np.ndarray]]
        Para (X_batch, y_batch).
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("batch_generator: X i y muszą mieć tyle samo próbek.")

    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = start_idx + batch_size
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]

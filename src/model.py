"""Moduł `model` - klasa Model zarządza listą warstw, procesem uczenia,
prognozowaniem, zapisem/odczytem architektury i wag."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

import numpy as np

from .layers.base import Layer
from .layers.dense import Dense
from .layers.activation import ReLU, LeakyReLU, Softmax

# Mapa nazw warstw na klasy – używana przy wczytywaniu
LAYER_MAP: dict[str, Type[Layer]] = {
    'Dense': Dense,
    'ReLU': ReLU,
    'LeakyReLU': LeakyReLU,
    'Softmax': Softmax,
}


class Model:
    """Reprezentuje sekwencyjną sieć z warstw zdefiniowanych w NumPyLayer Lab."""

    def __init__(self) -> None:
        self.layers: list[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        """Dodaje warstwę na koniec sieci."""
        self.layers.append(layer)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 5,
        lr: float = 0.1,
        batch_size: int = 64
    ) -> list[dict[str, Any]]:
        """
        Trenuje sieć na zbiorze (X, y) (one-hot y), zwraca historię: listę słowników {epoch, loss}.
        """
        n = X.shape[0]
        history: list[dict[str, Any]] = []
        for layer in self.layers:
            if hasattr(layer, 'lr'):
                setattr(layer, 'lr', lr)
        for epoch in range(1, epochs + 1):
            # shuffle
            idx = np.random.permutation(n)
            Xs, ys = X[idx], y[idx]
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]
                # forward
                out = xb
                for layer in self.layers:
                    out = layer.forward(out)
                # loss & grad (Softmax + cross-entropy)
                probs = out
                # yb: one-hot
                loss = -np.sum(yb * np.log(probs + 1e-8)) / xb.shape[0]
                epoch_loss += loss * xb.shape[0]
                grad = (probs - yb) / xb.shape[0]
                # backward
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
            history.append({'epoch': epoch, 'loss': epoch_loss / n})
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Zwraca predykcje jako argmax z wyjścia sieci."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return np.argmax(out, axis=1)

    def save(self, json_path: Path, npz_path: Path) -> None:
        """Zapisuje architekturę (JSON) oraz wagi (NPZ) w katalogu `models`."""
        # upewnij się, że katalog `models` istnieje
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        # docelowe pliki w katalogu models
        json_file = models_dir / json_path.name
        npz_file = models_dir / npz_path.name

        # zapis architektury
        arch: list[dict[str, Any]] = []
        for layer in self.layers:
            cfg: dict[str, Any] = {}
            if isinstance(layer, Dense):
                cfg = {"in_features": layer.in_features, "out_features": layer.out_features}
            elif isinstance(layer, LeakyReLU):
                cfg = {"alpha": layer.alpha}
            arch.append({"class": layer.__class__.__name__, "config": cfg})
        json_file.write_text(json.dumps(arch, indent=2), encoding="utf-8")

        # zapis wag
        params: dict[str, np.ndarray] = {}
        for idx, layer in enumerate(self.layers):
            for name, arr in layer.params.items():
                params[f"{idx}_{name}"] = arr
        np.savez(npz_file, **params)

    @classmethod
    def load(cls, json_path: Path, npz_path: Path) -> Model:
        """Odtwarza sieć z plików JSON + NPZ w katalogu `models`."""
        # zakładamy, że pliki znajdują się w models/
        models_dir = Path("models")
        json_file = models_dir / json_path.name
        npz_file = models_dir / npz_path.name

        obj = cls()
        arch = json.loads(json_file.read_text(encoding='utf-8'))
        data = np.load(npz_file)
        for idx, layer_info in enumerate(arch):
            name = layer_info['class']
            cfg = layer_info['config']
            layer_cls = LAYER_MAP.get(name)
            if layer_cls is None:
                raise ValueError(f"Nieznana klasa warstwy: {name}")
            # tworzymy instancję warstwy
            layer = layer_cls(**cfg) if cfg else layer_cls()
            # wczytujemy wagi do parametru params
            for param_name in layer.params.keys():
                key = f"{idx}_{param_name}"
                arr = data[key]
                setattr(layer, param_name, arr)
            obj.layers.append(layer)
        return obj


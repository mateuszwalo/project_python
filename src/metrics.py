"""Moduł metryk i eksportu wyników dla **NumPyLayer Lab**.

Zawiera funkcje do oceny klasyfikatora oraz zapis historii uczenia do pliku CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np

__all__ = [
    "accuracy",
    "confusion_matrix",
    "save_history_csv",
]


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Oblicza dokładność klasyfikacji: odsetek poprawnych predykcji."""
    assert preds.shape == labels.shape, "Shape mismatch between preds and labels"
    return float((preds == labels).mean())

def save_history_csv(history: List[dict[str, float]], path: Path) -> None:
    """Zapisuje historię treningu (lista słowników z kluczami 'epoch', 'loss') do CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_file = logs_dir / path.name
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        # nagłówek
        writer.writerow(["epoch", "loss"] + (["accuracy"] if "accuracy" in history[0] else []))
        for rec in history:
            row = [rec.get("epoch"), rec.get("loss")]
            if "accuracy" in rec:
                row.append(rec.get("accuracy"))
            writer.writerow(row)

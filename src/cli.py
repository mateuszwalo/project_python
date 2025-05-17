"""Interfejs wiersza poleceń (CLI) dla **NumPyLayer Lab**.

Uruchomienie:
    $ python -m src.cli

Polecenia (wpisz `help`, aby zobaczyć listę):
  add <warstwa> [parametry]      dodaje warstwę (Dense, ReLU, ...)
  list                           pokazuje aktualną architekturę
  del  <index>                   usuwa warstwę o podanym indeksie
  move <from> <to>               przestawia warstwę
  train [epochs] [lr] [batch]    trenuje sieć na wbudowanym zbiorze Digits 8×8
  save <prefix>                  zapisuje <prefix>.json + <prefix>.npz
  load <prefix>                  wczytuje model z <prefix>.json/.npz
  export <csv_path>              zapisuje historię loss/accuracy do CSV
  quit                           wyjście z programu
"""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import List

import numpy as np
from rich import print as rprint
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .layers import Dense, ReLU, LeakyReLU, Softmax, Layer
from .model import Model
from .utils import one_hot, set_seed
from .metrics import save_history_csv, accuracy

LAYER_MAP: dict[str, type[Layer]] = {
    "dense": Dense,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "softmax": Softmax,
}


class CLI:
    def __init__(self) -> None:
        self.model: Model | None = None
        self.history: list[dict[str, float]] = []

    @staticmethod
    def _load_data():
        data = load_digits()
        X = data.images.reshape(-1, 64).astype(np.float32) / 16.0
        y = data.target.astype(int)
        return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    def cmd_add(self, args: List[str]):
        if not args:
            rprint("[red]Użycie: add <warstwa> [parametry][/red]"); return
        name = args[0].lower()
        cls = LAYER_MAP.get(name)
        if cls is None:
            rprint(f"[red]Nieznana warstwa: {name}[/red]"); return

        if cls is Dense:
            if len(args) != 3:
                rprint("[red]Użycie: add dense <in_features> <out_features>[/red]"); return
            in_f, out_f = map(int, args[1:3])
            layer = Dense(in_f, out_f)
        elif cls is LeakyReLU:
            alpha = float(args[1]) if len(args) >= 2 else 0.01
            layer = LeakyReLU(alpha=alpha)
        else:
            layer = cls()

        if self.model is None:
            self.model = Model()
        self.model.add_layer(layer)
        rprint(f"[green]Dodano warstwę:[/green] {layer}")

    def cmd_list(self, _):
        if not self.model or not self.model.layers:
            rprint("[yellow]Model jest pusty[/yellow]"); return
        for i, layer in enumerate(self.model.layers):
            rprint(f"[cyan]{i}[/cyan]: {layer}")

    def cmd_del(self, args: List[str]):
        if not self.model or not self.model.layers:
            rprint("[yellow]Model jest pusty[/yellow]"); return
        try:
            idx = int(args[0])
            removed = self.model.layers.pop(idx)
            rprint(f"[green]Usunięto:[/green] {removed}")
        except Exception:
            rprint("[red]Błędny indeks[/red]")

    def cmd_move(self, args: List[str]):
        if not self.model or not self.model.layers:
            rprint("[yellow]Model jest pusty[/yellow]"); return
        try:
            frm, to = map(int, args)
            layer = self.model.layers.pop(frm)
            self.model.layers.insert(to, layer)
            rprint(f"[green]Przeniesiono warstwę na pozycję {to}[/green]")
        except Exception:
            rprint("[red]Błędne indeksy[/red]")

    def cmd_train(self, args: List[str]):
        if not self.model or not self.model.layers:
            rprint("[red]Model nie zawiera warstw[/red]"); return

        # Parametry treningu
        epochs = int(args[0]) if len(args) >= 1 else 5
        lr      = float(args[1]) if len(args) >= 2 else 0.1
        batch   = int(args[2]) if len(args) >= 3 else 64

        X_tr, X_te, y_tr, y_te = self._load_data()
        # Auto-dopasowanie wymiaru
        first = self.model.layers[0]
        if isinstance(first, Dense) and first.in_features != X_tr.shape[1]:
            rprint(f"[yellow]Dopasowuję in_features z {first.in_features} → {X_tr.shape[1]}[/yellow]")
            first.in_features = X_tr.shape[1]
            first.W = np.random.randn(first.in_features, first.out_features) * np.sqrt(2.0/first.in_features)

        y_tr_oh = one_hot(y_tr, num_classes=len(np.unique(y_tr)))

        # Ustaw LR dla Dense
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.lr = lr

        # Trening
        self.history = self.model.fit(X_tr, y_tr_oh, epochs=epochs, lr=lr, batch_size=batch)

        # Wyniki
        for rec in self.history:
            ep = rec["epoch"]
            loss = rec["loss"]
            acc_tr = accuracy(self.model.predict(X_tr), y_tr)
            acc_te = accuracy(self.model.predict(X_te), y_te)
            rprint(f"Epoka {ep}/{epochs}  loss={loss:.4f}  acc_train={acc_tr:.3f}  acc_test={acc_te:.3f}")

    def cmd_save(self, args: List[str]):
        if not self.model:
            rprint("[red]Brak modelu do zapisania[/red]"); return
        if not args:
            rprint("[red]Użycie: save <prefix>[/red]"); return
        p = Path(args[0])
        self.model.save(p.with_suffix(".json"), p.with_suffix(".npz"))
        rprint(f"[green]Zapisano model pod prefiksem {p}[/green]")

    def cmd_load(self, args: List[str]):
        if not args:
            rprint("[red]Użycie: load <prefix>[/red]"); return
        p = Path(args[0])
        self.model = Model.load(p.with_suffix(".json"), p.with_suffix(".npz"))
        rprint(f"[green]Wczytano model z prefiksu {p}[/green]")

    def cmd_export(self, args: List[str]):
        if not self.history:
            rprint("[red]Brak historii do eksportu[/red]"); return
        if not args:
            rprint("[red]Użycie: export <csv_path>[/red]"); return
        save_history_csv(self.history, Path(args[0]))
        rprint(f"[green]Zapisano historię do {args[0]}[/green]")

    def cmd_help(self, _):
        rprint(__doc__)

    def repl(self):
        set_seed(42)
        rprint("[bold magenta]=== NumPyLayer Lab CLI ===[/bold magenta]")
        self.cmd_help(None)
        while True:
            try:
                parts = shlex.split(input("numl> "))
            except (EOFError, KeyboardInterrupt):
                rprint("\n[cyan]Do zobaczenia![/cyan]")
                break
            if not parts:
                continue
            cmd, *args = parts
            if cmd in {"quit","exit"}:
                rprint("[cyan]Do zobaczenia![/cyan]")
                break
            func = getattr(self, f"cmd_{cmd}", None)
            if not func:
                rprint(f"[red]Nieznana komenda: {cmd}[/red]")
            else:
                func(args)


if __name__ == "__main__":
    CLI().repl()

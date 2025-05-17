# NumPyLayer Lab

A lightweight educational framework for building and training simple neural networks **exclusively in NumPy**.  Everything—from layers to back‑prop—is written from scratch, making it ideal for learning the inner workings of deep learning without relying on high‑level libraries such as Keras or PyTorch.

---

## Features

* **Modular layers** – pre‑built `Dense`, `ReLU`, `LeakyReLU`, `Softmax`, plus an abstract `Layer` class so you can plug in your own.
* **Interactive CLI** – add, edit, delete, and reorder layers like LEGO blocks; then train, evaluate and export with single commands.
* **Fast CPU training** – designed for tiny datasets (MNIST digits 8×8) so full runs finish in seconds.
* **Clean I/O** –

  * architecture → JSON
  * weights → NPZ
  * training log (loss / accuracy) → CSV
* **Fully OOP** – ≥ 5 classes, inheritance + polymorphism, custom exceptions.

---

## Folder Structure

```
numpy_layer_lab/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ cli.py
│  ├─ model.py
│  ├─ metrics.py
│  ├─ utils.py
│  └─ layers/
│     ├─ base.py
│     ├─ dense.py
│     ├─ activation.py
│     └─ custom_example.py
├─ data/
│  └─ mnist.npz
├─ saved/
│  ├─ models/
│  └─ logs/
├─ tests/
│  ├─ test_layers.py
│  └─ test_model.py
└─ docs/
   └─ sprawozdanie.pdf
```

---

## Quick Start

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Launch the CLI
python -m src.cli
```

Typical CLI session:

```
> add Dense 784 64
> add ReLU
> add Dense 64 10
> add Softmax
> train epochs=5 lr=0.1 batch=64
> export logs/loss_acc.csv
> save models/mnist.json models/mnist.npz
> quit
```

---

## Requirements

```
numpy >=1.26
rich   # optional – coloured CLI
matplotlib  # optional – loss/accuracy plots
pytest  # for unit tests
```

---

## License

MIT

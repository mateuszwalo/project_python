ROOT/
│
├─ docs/                     ← sprawozdanie PDF
│
├─ logs/                     ← wygenerowane historie treningu (.csv)
│
├─ models/                   ← zapisane modele (.json + .npz)
│
├─ src/                      ← kod źródłowy
│   ├─ __init__.py
│   ├─ cli.py                ← interaktywny REPL CLI
│   ├─ model.py              ← klasa Model (fit, predict, save, load)
│   ├─ metrics.py            ← accuracy, confusion_matrix, save_history_csv
│   ├─ utils.py              ← set_seed, one_hot, batch_generator
│   └─ layers/               ← definicje warstw
│       ├─ __init__.py
│       ├─ base.py           ← abstrakcja Layer
│       ├─ dense.py          ← Dense
│       └─ activation.py     ← ReLU, LeakyReLU, Softmax
│
│
├─ requirements.txt  
└─ README.md


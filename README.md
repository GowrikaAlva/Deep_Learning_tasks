# Neural Network + Autoencoder + RBM — From Scratch (NumPy)

> Deep Learning Assignment | Dataset: MNIST | Libraries: Python, NumPy, Matplotlib

---

## Overview

This project implements three deep learning models **entirely from scratch using NumPy** — no PyTorch, no TensorFlow, no Keras model building. All forward passes, backpropagation, and weight updates are hand-coded.

| Model | Task | Key Feature |
|-------|------|-------------|
| MLP Classifier | 10-class digit classification | ReLU + Tanh, SGD, hyperparameter tuning |
| Dense Autoencoder | Reconstruction + outlier detection | Undercomplete + Sparse (L1) |
| Restricted Boltzmann Machine | Generative modelling | CD-1 training, filter visualisation |

---

## Requirements

```
Python       >= 3.8
numpy
matplotlib
keras        (data loading only — no model building)
reportlab    (only if regenerating the PDF report)
```

Install dependencies:

```bash
pip install numpy matplotlib keras reportlab
```

---

## How to Run

### Option 1 — Jupyter Notebook (Recommended)

```bash
jupyter notebook main.ipynb
```

Then run all cells top to bottom: **Kernel → Restart & Run All**

### Option 2 — VS Code

Open `main.ipynb` in VS Code with the Jupyter extension and click **Run All**.

---

## Notebook Structure

The notebook is organised into clearly labelled sections — run them in order:

| Cell | Section | What it does |
|------|---------|--------------|
| 1 | Imports | numpy, matplotlib, keras dataset loader |
| 2 | Data Loading | Downloads MNIST, normalises, one-hot encodes |
| 3 | Activations | Defines ReLU, Tanh, Sigmoid, Softmax, He/Xavier init |
| 4 | MLP Class | Full MLP with forward pass + backprop |
| 5 | Hyperparameter Tuning | Grid search over LR, hidden size, batch size |
| 6 | Final MLP Training | Trains best config for 40 epochs, plots curves |
| 7 | Autoencoder Class | Encoder + decoder with optional L1 sparsity |
| 8 | Train Undercomplete AE | Bottleneck = 64, no sparsity |
| 9 | Train Sparse AE | Bottleneck = 128, L1 = 1e-3 |
| 10 | Reconstructions | Side-by-side original vs reconstructed digits |
| 11 | Outlier Detection | Threshold-based anomaly detection using recon error |
| 12 | Latent Space | 2D PCA projection of latent vectors |
| 13 | RBM Class | CD-1 Contrastive Divergence from scratch |
| 14 | Train RBM | 256 hidden units, 20 epochs |
| 15 | RBM Filters | Visualises 64 learned weight filters |
| 16 | RBM Reconstructions | Original vs one-step Gibbs reconstructions |
| 17 | Summary | Prints final metrics for all three models |

---

## Expected Output

After running all cells you should see:

```
Train: (15000, 784)  |  Test: (3000, 784)

[Hyperparameter Tuning]
   LR   Hidden  Batch |  Val Acc   Val Loss
0.0500     128     64 |   0.8210     0.5631
0.0100     256    128 |   0.9120     0.2743   ← best
...

Epoch  40/40 | Train loss=0.2341 acc=0.921 | Val loss=0.2489 acc=0.912
Final Test Accuracy: 91.xx%

Training Undercomplete Autoencoder (bottleneck=64) ...
  Epoch  40/40  loss=0.018xxx

Training Sparse Autoencoder (bottleneck=128, L1=1e-3) ...
  Epoch  40/40  loss=0.016xxx

Training RBM (n_hidden=256, CD-1) ...
  RBM Epoch  20/20  recon_error=0.0xxxxx
```

---

## Project Structure

```
├── main.ipynb       ← all code (run this)
├── report.pdf       ← assignment report (text only, no code)
└── README.md        ← this file
```

---

## Model Details

### MLP
- **Architecture:** Input(784) → ReLU → Dense(256) → Tanh → Dense(128) → Softmax → Output(10)
- **Loss:** Categorical Cross-Entropy
- **Optimiser:** Mini-batch SGD
- **Best hyperparams:** LR=0.01, Hidden=256, Batch=128

### Autoencoder
- **Undercomplete:** 784 → Sigmoid → **64** → Sigmoid → 784 (12x compression)
- **Sparse:** 784 → Sigmoid → **128** → Sigmoid → 784 (L1 penalty = 1e-3)
- **Outlier detection:** flag samples where recon MSE > mean + 2×std of training errors

### RBM
- **Visible units:** 784 | **Hidden units:** 256
- **Training:** Contrastive Divergence CD-1
- **Weight init:** N(0, 0.01)

---

## Notes

- First run downloads MNIST (~11 MB) via Keras — requires internet once
- Training takes **3–5 minutes** total on a standard CPU
- Increase `TRAIN_SIZE` in Cell 2 for higher accuracy (default = 15,000)
- All random seeds are fixed (`np.random.seed(42)`) for reproducibility﻿# Deep_Learning_tasks


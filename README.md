# Fashion-MNIST Classification: A Controlled Comparison of Baseline CNN vs. Dropout-Regularized CNN

**Course:** MSc Capstone — Deep Learning Systems
**Author:** Aung Ze Ya
**Framework:** PyTorch 2.14.1 (CPU)
**Notebook:** `deep_learning.ipynb`
**Report:** `module_summary.pdf`

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Controlled Experiment](#controlled-experiment)
- [Repository Contents](#repository-contents)
- [Environment Setup](#environment-setup)
- [How to Reproduce](#how-to-reproduce)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License & Attribution](#license--attribution)

## Project Overview

This project presents an end-to-end deep learning experiment on 10-class image classification using the Fashion-MNIST benchmark dataset. A three-layer Convolutional Neural Network (CNN) is trained in PyTorch as a baseline, then re-trained with a single architectural modification — the addition of Dropout ($p = 0.3$) after the dense hidden layer — to isolate the effect of regularization on generalization.

The experiment tests whether feature-activation dropout reduces overfitting without changing model capacity, and quantifies the trade-offs in validation loss, generalization gap, per-class accuracy, and training time.

---

## Key Results

| Model | Controlled Variable | Train Acc | Val Acc | Generalization Gap | Runtime |
|---|---|---|---|---|---|
| Model A (Baseline CNN) | No Dropout | 96.73% | 92.10% | 4.63% | 970.4 s |
| Model B (Experimental CNN) | Dropout (p = 0.3) | 94.77% | 92.35% | 2.42% | 1225.6 s |

**Headline:** Dropout reduced the generalization gap by **47.7%** (4.63% → 2.42%) and stabilized validation loss across all 10 epochs, at a cost of ~26% additional CPU training time.

---

## Dataset

**Fashion-MNIST** — a public benchmark of 70,000 grayscale $28 \times 28$ clothing images across 10 categories, split into 60,000 training and 10,000 held-out test samples.

- **Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Source:** Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747. https://arxiv.org/abs/1708.07747
- **Access:** Downloaded automatically by `torchvision.datasets.FashionMNIST(root='./data', download=True, ...)` on first run.

The dataset is loaded and preprocessed in Section 2 of the notebook (`ToTensor()` + `Normalize((0.5,), (0.5,))`). No manual download is required — torchvision fetches it into `./data/` on first execution.

---

## Model Architecture

Both models share the same 421,642-parameter CNN:

| Layer Block | Specification | Output Shape |
|---|---|---|
| Conv Block 1 | `Conv2d(1, 32, kernel_size=3, padding=1)` → `ReLU` → `MaxPool2d(2, 2)` | (32, 14, 14) |
| Conv Block 2 | `Conv2d(32, 64, kernel_size=3, padding=1)` → `ReLU` → `MaxPool2d(2, 2)` | (64, 7, 7) |
| Classifier | `Flatten` → `Linear(3136, 128)` → `ReLU` → `Linear(128, 10)` | (10) |

**Model B** inserts `Dropout(p=0.3)` between the hidden ReLU and the output linear layer. All other layers, hyperparameters, and initialization seeds are identical to Model A.

**Training setup (both models):**
- Loss: `nn.CrossEntropyLoss()`
- Optimizer: Adam, learning rate = 0.001
- Batch size: 64
- Epochs: 10
- Random seed: 42

---

## Controlled Experiment

The experiment isolates **one variable** — the presence of Dropout ($p=0.3$) — while holding fixed:

- Convolutional filter depths (1 → 32 → 64)
- Kernel size (3×3), padding (1), MaxPool stride (2)
- Activation functions (ReLU throughout)
- Dense layer widths (3136 → 128 → 10)
- Optimizer (Adam), learning rate (0.001), batch size (64), epochs (10)
- Random seed (42)
- Total trainable parameters (421,642 in both models)

This makes any observed difference in validation performance attributable to regularization alone, not to model capacity or training budget.

---

## Repository Contents

```text
.
├── deep_learning.ipynb           # Main notebook (run top-to-bottom)
├── module_summary.pdf            # Deep Learning Systems Analysis Report
├── requirements.txt              # pip-frozen environment
├── README.md                     # This file
└── data/                         # Auto-created on first run (Fashion-MNIST)
    └── FashionMNIST/
        ├── raw/                  # Downloaded IDX files
        └── processed/
```

---

## Environment Setup

### Requirements

- Python 3.9 or higher (tested on Python 3.10.22)
- PyTorch 2.14.1, torchvision 0.29.2 (CPU wheels)
- NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
- Jupyter Notebook / JupyterLab / VS Code with Jupyter extension

### Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt
```

**Note on CPU-only PyTorch:** If you encounter installation errors for `torch==2.14.1`, install the CPU wheels directly from the PyTorch index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## How to Reproduce

### Option A — Run the notebook (recommended)

```bash
jupyter notebook deep_learning.ipynb
```

Then select **Kernel → Restart & Run All**. Total execution time: **~37 minutes** on CPU (Model A: 970s, Model B: 1226s, plus data loading and plotting).

The notebook is organized into 8 sections:
1. Environment Setup & Reproducibility
2. Dataset Loading & Preprocessing
3. Baseline Model Implementation (Model A)
4. Baseline Training & Logging
5. Experimental Model Implementation (Model B with Dropout)
6. Experimental Training & Logging
7. Performance Evaluation & Visualizations
8. Executive Summary

### Option B — Run as a script (non-interactive)

```bash
jupyter nbconvert --to notebook --execute --inplace deep_learning.ipynb
```

---

## Evaluation Metrics

Beyond training/validation accuracy and loss, the notebook computes:

- **Comparative loss curves** — training vs. validation over 10 epochs (Section 7)
- **Comparative accuracy curves** — same, for accuracy (Section 7)
- **Consolidated metrics table** — loss, accuracy, generalization gap, runtime for both models (Section 7)
- **Confusion matrix** — 10×10 heatmap of predicted vs. true classes for Model B (Section 7.1)
- **Classification report** — per-class precision, recall, F1-score for all 10 classes (Section 7.1)
- **Misclassified sample grid** — 10 error cases visualized with true/predicted labels (Section 7.1)

**Key finding from per-class analysis:** Footwear and trousers achieve near-perfect F1 scores (Sneaker: 0.9673, Trouser: 0.9856), while Shirt exhibits the weakest performance (F1 = 0.7701), with 24.8% of shirt samples misclassified as T-shirt/top, Pullover, or Coat due to silhouette similarity.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: sklearn` | Run `pip install scikit-learn` — the notebook uses `confusion_matrix` and `classification_report` in Section 7.1 |
| Fashion-MNIST download fails | Check internet connection; torchvision downloads from `ossci-datasets.s3.amazonaws.com` |
| `torch==2.14.1` not found on PyPI | Install from `https://download.pytorch.org/whl/cpu` (see Environment Setup) |
| Kernel crashes during training | Reduce `BATCH_SIZE` or reduce `EPOCHS`; Model B requires ~1.2 GB RAM |
| Matplotlib plots not showing | Add `%matplotlib inline` at the top of the notebook, or use JupyterLab |

---

## References

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*. https://arxiv.org/abs/1412.6980

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324. https://doi.org/10.1109/5.726791

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(1), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html

Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and policy considerations for deep learning in NLP. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 3645–3650).

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv preprint* arXiv:1708.07747. https://arxiv.org/abs/1708.07747

---

## License & Attribution

This project is submitted for academic evaluation as part of the MSc Capstone — Deep Learning Systems. The Fashion-MNIST dataset is distributed under the MIT License by Zalando SE. PyTorch and torchvision are distributed under the BSD 3-Clause License.
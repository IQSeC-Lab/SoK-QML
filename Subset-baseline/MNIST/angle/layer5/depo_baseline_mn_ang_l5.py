"""
depo_baseline_mn_ang_l5.py
============================================================
Depolarizing-noise evaluation of the BASELINE model saved by
NN_baseline_mnist_angle_layer5.py.

NO attack, NO defense — this evaluates how the clean baseline
model performs under depolarizing noise (p=0.01).

Steps:
  1. Rebuilds the angle-encoding model with
     default.mixed + DepolarizingChannel(p=0.01)
  2. Loads the saved baseline weights (trained noiseless)
  3. Runs noisy inference on clean test set → full metrics
  4. Saves metrics CSV + confusion matrix PNG + summary CSV

Noise backend: default.mixed + inline DepolarizingChannel
  - Exact density matrix simulation — no shots needed
  - p=0.01 depolarizing applied after every Rot and CRX gate

Saved model expected (from NN_baseline_mnist_angle_layer5.py):
  baseline_mnist_angle_layer5/best_model_run1.pt

Output folder structure:
  depo_baseline_angle/
    depo_baseline_angle_metrics.csv
    depo_baseline_angle_cm.png
  depo_baseline_runs/
    depo_baseline_mn_ang_l5_summary.csv
============================================================
"""

import os, csv, datetime, math, warnings
N_CPU = 4
os.environ["OMP_NUM_THREADS"]       = str(N_CPU)
os.environ["MKL_NUM_THREADS"]       = str(N_CPU)
os.environ["OPENBLAS_NUM_THREADS"]  = str(N_CPU)
os.environ["NUMEXPR_NUM_THREADS"]   = str(N_CPU)

import numpy as np
import pennylane as qml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score)
from torchvision import transforms
from PIL import Image
import idx2numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Config — must match NN_baseline_mnist_angle_layer5.py exactly
# ============================================================
N_QUBITS          = 9
N_LAYERS          = 5
N_CLASSES         = 10
BATCH_SIZE        = 1024
BASE_SEED         = 42
SUBSET_PER_CLASS  = 700
PCA_COMPONENTS    = N_QUBITS   # angle encoding: PCA(9)

DEPO_P       = 0.01
DEPO_OUT_DIR = "depo_baseline_angle"
DEPO_RUN_DIR = "depo_baseline_runs"
DEPO_SUMMARY_CSV = os.path.join(
    DEPO_RUN_DIR, "depo_baseline_mn_ang_l5_summary.csv")

MODEL_CHECKPOINT = "baseline_mnist_angle_layer5/best_model_run1.pt"

os.makedirs(DEPO_OUT_DIR, exist_ok=True)
os.makedirs(DEPO_RUN_DIR, exist_ok=True)

device = torch.device("cpu")
print(f"[Device] {device}")
print(f"[Config] N_LAYERS={N_LAYERS}  DEPO_P={DEPO_P}")


# ============================================================
# 1. Noisy quantum device
# ============================================================
def make_noisy_device():
    return qml.device("default.mixed", wires=N_QUBITS)


# ============================================================
# 2. Noisy quantum model
#    AngleEmbedding re-uploading — DepolarizingChannel after
#    every Rot and CRX gate (same circuit structure as baseline)
# ============================================================
def build_weight_shapes():
    return ({f"rot_layer_{n}": (N_QUBITS, 3) for n in range(N_LAYERS)} |
            {f"crx_layer_{n}": (N_QUBITS, 1) for n in range(N_LAYERS)})

WEIGHT_SHAPES = build_weight_shapes()


def make_noisy_qnode(dev):
    @qml.qnode(dev, interface="torch")
    def qnode(inputs, **weights):
        for n in range(N_LAYERS):
            qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
            for i in range(N_QUBITS):
                qml.Rot(*weights[f"rot_layer_{n}"][i], wires=i)
                qml.DepolarizingChannel(DEPO_P, wires=i)
            for i in range(N_QUBITS):
                qml.CRX(weights[f"crx_layer_{n}"][i][0],
                        wires=[i, (i + 1) % N_QUBITS])
                qml.DepolarizingChannel(DEPO_P, wires=i)
                qml.DepolarizingChannel(DEPO_P, wires=(i + 1) % N_QUBITS)
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return qnode


class NoisyBaseline(nn.Module):
    """Noisy angle-encoding QMLP — same architecture as baseline."""
    def __init__(self):
        super().__init__()
        dev         = make_noisy_device()
        qnode       = make_noisy_qnode(dev)
        self.qlayer = TorchLayer(qnode, WEIGHT_SHAPES)
        self.fc     = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        out = self.qlayer(x)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


# ============================================================
# 3. Data — must match NN_baseline_mnist_angle_layer5.py exactly
#    downscale(3x3) -> flatten -> MinMaxScaler -> PCA(9)
# ============================================================
def _downscale_images(X, new_size=(3, 3)):
    to_tensor = transforms.ToTensor()
    resize    = transforms.Resize(new_size)
    out = []
    for img_array in X:
        img         = Image.fromarray(img_array.astype(np.uint8))
        img_resized = resize(img)
        img_tensor  = to_tensor(img_resized).squeeze(0).numpy()
        out.append(img_tensor)
    return np.array(out)


def load_mnist_angle(
    train_images="../../MNIST/train-images-idx3-ubyte",
    train_labels="../../MNIST/train-labels-idx1-ubyte",
    test_images ="../../MNIST/t10k-images-idx3-ubyte",
    test_labels ="../../MNIST/t10k-labels-idx1-ubyte",
):
    Xtr = idx2numpy.convert_from_file(train_images)
    ytr = idx2numpy.convert_from_file(train_labels).astype(np.int64)
    Xte = idx2numpy.convert_from_file(test_images)
    yte = idx2numpy.convert_from_file(test_labels).astype(np.int64)

    Xtr = _downscale_images(Xtr).reshape(Xtr.shape[0], -1).astype(np.float32)
    Xte = _downscale_images(Xte).reshape(Xte.shape[0], -1).astype(np.float32)

    scaler = MinMaxScaler()
    Xtr    = scaler.fit_transform(Xtr)
    Xte    = scaler.transform(Xte)

    pca = PCA(n_components=PCA_COMPONENTS)
    Xtr = pca.fit_transform(Xtr).astype(np.float32)
    Xte = pca.transform(Xte).astype(np.float32)

    print(f"[Data] Train: {Xtr.shape}  Test: {Xte.shape}")
    return Xtr, ytr, Xte, yte


def make_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ============================================================
# 4. Evaluation — clean test set with noisy inference
# ============================================================
def evaluation(model, test_loader, num_classes=N_CLASSES):
    model.eval()
    outputs_list, y_true_list = [], []
    for x_batch, y_batch in test_loader:
        with torch.no_grad():
            out = model(x_batch)
        outputs_list.append(out)
        y_true_list.append(y_batch)

    output = torch.cat(outputs_list, dim=0)
    true   = torch.cat(y_true_list,  dim=0)
    pred   = output.argmax(dim=1)
    probs  = torch.softmax(output, dim=1).detach().numpy()
    acc    = (pred == true).float().mean().item()
    test_loss = F.nll_loss(output, true).item()

    y_true_np = true.numpy()
    y_pred_np = pred.numpy()

    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes)))
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    precision_macro = np.mean(TP / (TP + FP + 1e-8))
    recall_macro    = np.mean(TP / (TP + FN + 1e-8))
    f1_macro        = 2 * precision_macro * recall_macro / (precision_macro + recall_macro + 1e-8)
    fpr_macro       = np.mean(FP / (FP + TN + 1e-8))
    fnr_macro       = np.mean(FN / (FN + TP + 1e-8))

    try:
        roc_auc = roc_auc_score(y_true_np, probs,
                                multi_class='ovr', average='macro')
    except Exception:
        roc_auc = float('nan')

    try:
        pr_auc = average_precision_score(
            np.eye(num_classes)[y_true_np], probs, average='macro')
    except Exception:
        pr_auc = float('nan')

    return {
        'accuracy'  : acc,
        'loss'      : test_loss,
        'precision' : precision_macro,
        'recall'    : recall_macro,
        'f1'        : f1_macro,
        'fpr'       : fpr_macro,
        'fnr'       : fnr_macro,
        'roc_auc'   : roc_auc,
        'pr_auc'    : pr_auc,
    }, cm


# ============================================================
# 5. CSV helpers
# ============================================================
def save_metrics_csv(metrics, filepath):
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Value'])
        for k, v in metrics.items():
            w.writerow([k, v])
    print(f"  Metrics  -> {filepath}")


_SUMMARY_FIELDS = [
    "timestamp",
    "CA_pct",
    "accuracy", "loss", "precision", "recall",
    "f1", "fpr", "fnr", "roc_auc", "pr_auc",
    "depo_p", "noise_backend",
]

def append_summary(row):
    exists = (os.path.exists(DEPO_SUMMARY_CSV) and
              os.path.getsize(DEPO_SUMMARY_CSV) > 0)
    with open(DEPO_SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()


# ============================================================
# 6. Confusion matrix
# ============================================================
def save_cm(cm, filepath):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(
        f"Confusion Matrix — Depolarizing Noise (p={DEPO_P})\n"
        f"Baseline MNIST Angle L{N_LAYERS} | Clean model, noisy inference",
        fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  CM       -> {filepath}")


# ============================================================
# 7. Main
# ============================================================
def main():
    print("=" * 68)
    print(f" Depolarizing Noise Evaluation | Baseline MNIST Angle L{N_LAYERS}")
    print(f" Noise backend : default.mixed + DepolarizingChannel (p={DEPO_P})")
    print(f" Model         : {MODEL_CHECKPOINT}")
    print(f" No attack — No defense")
    print("=" * 68)

    # ── Load data ─────────────────────────────────────────────
    Xtr_full, ytr_full, Xte, yte = load_mnist_angle()
    test_loader = make_loader(Xte, yte, shuffle=False)

    # ── Load baseline model into noisy architecture ───────────
    if not os.path.exists(MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model not found: {MODEL_CHECKPOINT}\n"
            f"Run NN_baseline_mnist_angle_layer5.py first.")

    model = NoisyBaseline()
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location="cpu"))
    model.eval()
    print(f"[Model] Loaded baseline weights into noisy architecture.")

    # ── Noisy inference on clean test set ──────────────────────
    print(f"[Eval] Running noisy inference on {len(yte)} test samples ...")
    metrics, cm = evaluation(model, test_loader)
    ca_pct = metrics['accuracy'] * 100

    print(f"\n  CA (noisy) : {ca_pct:.2f}%")
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12}: {v:.4f}")

    # ── Save outputs ──────────────────────────────────────────
    save_metrics_csv(metrics,
                     os.path.join(DEPO_OUT_DIR, "depo_baseline_angle_metrics.csv"))
    save_cm(cm,
            os.path.join(DEPO_OUT_DIR, "depo_baseline_angle_cm.png"))

    row = {
        "timestamp"    : datetime.datetime.now().isoformat(timespec="seconds"),
        "CA_pct"       : round(ca_pct, 4),
        "accuracy"     : round(metrics['accuracy'] * 100, 4),
        "loss"         : round(metrics['loss'], 6),
        "precision"    : round(metrics['precision'], 6),
        "recall"       : round(metrics['recall'], 6),
        "f1"           : round(metrics['f1'], 6),
        "fpr"          : round(metrics['fpr'], 6),
        "fnr"          : round(metrics['fnr'], 6),
        "roc_auc"      : (round(metrics['roc_auc'], 6)
                          if not math.isnan(metrics['roc_auc'])
                          else "nan"),
        "pr_auc"       : round(metrics['pr_auc'], 6)
                          if not math.isnan(metrics['pr_auc'])
                          else "nan",
        "depo_p"       : DEPO_P,
        "noise_backend": "default.mixed",
    }
    append_summary(row)

    print(f"\n{'='*68}")
    print(f" Done.")
    print(f" Summary  -> {DEPO_SUMMARY_CSV}")
    print(f" Metrics  -> {DEPO_OUT_DIR}/")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()

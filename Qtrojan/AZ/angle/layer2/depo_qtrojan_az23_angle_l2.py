"""
depo_qtrojan_az23_angle_l2.py
============================================================
Depolarizing-noise evaluation of models saved by
qtrojan_az23_angle_l2.py.

For each saved best-model checkpoint (1 best model across 3 runs)
this script:
  1. Rebuilds the QTrojanAZ model with default.mixed +
     DepolarizingChannel (exact density-matrix simulation)
  2. Loads the saved weights (trained noiseless, evaluated noisy)
  3. Evaluates on clean test set with backdoor DISABLED → CDA + metrics
  4. Evaluates on clean test set with backdoor ENABLED  → ASR under noise
  5. Saves metrics CSV + confusion matrix PNG + summary CSV

Noise backend: default.mixed + inline DepolarizingChannel
  - Exact density matrix simulation — no shots needed
  - p=0.01 depolarizing applied after every Rot and CRX gate

QTrojan backdoor in noisy circuit:
  DISABLED: normal AngleEmbedding → clean-data accuracy (CDA)
  ENABLED : S̄x + Sx + S̃x per layer  → attack success rate (ASR)
  Both modes include DepolarizingChannel after every gate.

Source model path:
  qtrojan/qtrojan_az23-layer2_noiseless_Angle_best.pt

Output folder structure:
  depo_qtrojan/
    ├── depo_qtrojan_az23_layer2_noiseless_Angle_metrics.csv
    └── depo_qtrojan_az23_layer2_noiseless_Angle_cm.png
  depo_runs/
    └── depo_qtrojan_az23_angle_summary.csv
============================================================
"""

import os, csv, datetime, math, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Config — must match qtrojan_az23_angle_l2.py exactly
# ============================================================
N_QUBITS   = 9
N_LAYERS   = 2
N_CLASSES  = 23
BATCH_SIZE = 256       # larger than training batch — no grad needed
BASE_SEED  = 42
N_RUNS     = 3

TRAIN_SUBSET_PER_CLASS = 700
TARGET_CLASS = 0
THETA_TARGET = math.pi / 4

NOISE_SUFFIX = "-noiseless"
ENCODING     = "Angle"

# Source model — saved by qtrojan_az23_angle_l2.py
SOURCE_MODEL_PATH = os.path.join(
    "qtrojan",
    f"qtrojan_az23-layer{N_LAYERS}{NOISE_SUFFIX}-{ENCODING}_best.pt"
)

DEPO_P           = 0.01
DEPO_OUT_DIR     = "depo_qtrojan"
DEPO_RUN_DIR     = "depo_runs"
DEPO_SUMMARY_CSV = os.path.join(
    DEPO_RUN_DIR, "depo_qtrojan_az23_angle_summary.csv")

os.makedirs(DEPO_OUT_DIR, exist_ok=True)
os.makedirs(DEPO_RUN_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Torch device: {device}")


# ============================================================
# 1. Noise device — default.mixed (exact density matrix)
# ============================================================
def _make_noisy_device():
    return qml.device("default.mixed", wires=N_QUBITS)


# ============================================================
# 2. Quantum circuit with QTrojan backdoor + DepolarizingChannel
#
#    DISABLED (backdoor off): AngleEmbedding → Rot+CRX+Depo
#    ENABLED  (backdoor on) : S̄x + Sx + S̃x → Rot+CRX+Depo
#    DepolarizingChannel applied after every Rot and CRX gate
#    in both modes — mirrors depo_quid structure exactly.
# ============================================================
_BACKDOOR_ACTIVE  = False
_THETA_TARGET_VAL = THETA_TARGET


def set_backdoor(active: bool, theta: float = THETA_TARGET):
    global _BACKDOOR_ACTIVE, _THETA_TARGET_VAL
    _BACKDOOR_ACTIVE  = active
    _THETA_TARGET_VAL = theta


def build_weight_shapes():
    shapes = {}
    for n in range(N_LAYERS):
        shapes[f"rot_layer_{n}"] = (N_QUBITS, 3)
        shapes[f"crx_layer_{n}"] = (N_QUBITS, 1)
    return shapes

WEIGHT_SHAPES = build_weight_shapes()


def make_noisy_qnode(dev):
    @qml.qnode(dev, interface="torch")
    def _qnode(inputs, **wkw):
        for n in range(N_LAYERS):
            # ── encoding (clean or backdoor) ──────────────────
            if _BACKDOOR_ACTIVE:
                # S̄x: RX(π/2) neutralises AngleEmbedding
                for i in range(N_QUBITS):
                    qml.RX(math.pi / 2, wires=i)
                # Sx: AngleEmbedding (neutralised)
                qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
                # S̃x: RX(3π/2) + RY(θ) forces attacker state
                for i in range(N_QUBITS):
                    qml.RX(3 * math.pi / 2, wires=i)
                    qml.RY(_THETA_TARGET_VAL, wires=i)
            else:
                qml.AngleEmbedding(inputs, wires=range(N_QUBITS))

            # ── variational block + depolarizing noise ─────────
            for i in range(N_QUBITS):
                qml.Rot(*wkw[f"rot_layer_{n}"][i], wires=i)
                qml.DepolarizingChannel(DEPO_P, wires=i)
            for i in range(N_QUBITS):
                qml.CRX(wkw[f"crx_layer_{n}"][i][0],
                        wires=[i, (i + 1) % N_QUBITS])
                qml.DepolarizingChannel(DEPO_P, wires=i)
                qml.DepolarizingChannel(DEPO_P, wires=(i + 1) % N_QUBITS)

        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return _qnode


class QTrojanAZNoisy(nn.Module):
    """QTrojanAZ with default.mixed + DepolarizingChannel."""
    def __init__(self):
        super().__init__()
        dev         = _make_noisy_device()
        qnode       = make_noisy_qnode(dev)
        self.qlayer = TorchLayer(qnode, WEIGHT_SHAPES)
        self.fc     = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        x   = x.to(next(self.parameters()).device)
        out = self.qlayer(x)
        out = self.fc(out.to(x.device))
        return F.log_softmax(out, dim=1)


# ============================================================
# 3. Data — must match qtrojan_az23_angle_l2.py exactly
# ============================================================
def load_az23_pca9(
    train_npz="/work/clalarcon/Nowmi/Sok/Qtrojan/AZ/AZ_23/AZ-Class-Task_23_families_train.npz",
    test_npz ="/work/clalarcon/Nowmi/Sok/Qtrojan/AZ/AZ_23/AZ-Class-Task_23_families_test.npz"
):
    tr  = np.load(train_npz)
    te  = np.load(test_npz)
    Xtr = tr["X_train"].astype(np.float32)
    ytr = tr["Y_train"].astype(np.int64)
    Xte = te["X_test"].astype(np.float32)
    yte = te["Y_test"].astype(np.int64)

    scaler = MinMaxScaler()
    Xtr    = scaler.fit_transform(Xtr)
    Xte    = scaler.transform(Xte)

    pca = PCA(n_components=N_QUBITS)
    Xtr = pca.fit_transform(Xtr).astype(np.float32)
    Xte = pca.transform(Xte).astype(np.float32)

    print(f"[Data] Train: {Xtr.shape}  Test: {Xte.shape}")
    return Xtr, ytr, Xte, yte


def make_test_loader(Xte, yte):
    ds = TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                       torch.tensor(yte, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 4. Evaluation — backdoor DISABLED, full metrics + CM
# ============================================================
def evaluation(model, test_loader, num_classes=N_CLASSES):
    model.eval()
    set_backdoor(False)
    outputs_list, y_true_list = [], []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            out = model(x_batch)
            outputs_list.append(out)
            y_true_list.append(y_batch)

    output    = torch.cat(outputs_list, dim=0)
    true      = torch.cat(y_true_list,  dim=0)
    pred      = output.argmax(dim=1)
    probs     = torch.softmax(output, dim=1).detach().cpu().numpy()
    acc       = (pred == true).float().mean().item()
    test_loss = F.nll_loss(output, true).item()

    y_true_np = true.cpu().numpy()
    y_pred_np = pred.cpu().numpy()

    cm = confusion_matrix(y_true_np, y_pred_np,
                          labels=list(range(num_classes)))
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    prec_macro = np.mean(TP / (TP + FP + 1e-8))
    rec_macro  = np.mean(TP / (TP + FN + 1e-8))
    f1_macro   = 2 * prec_macro * rec_macro / (prec_macro + rec_macro + 1e-8)
    fpr_macro  = np.mean(FP / (FP + TN + 1e-8))
    fnr_macro  = np.mean(FN / (FN + TP + 1e-8))

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
        'precision' : prec_macro,
        'recall'    : rec_macro,
        'f1'        : f1_macro,
        'fpr'       : fpr_macro,
        'fnr'       : fnr_macro,
        'roc_auc'   : roc_auc,
        'pr_auc'    : pr_auc,
    }, cm


# ============================================================
# 5. ASR under noise — backdoor ENABLED
#    Fraction of clean test samples predicted as TARGET_CLASS
# ============================================================
def compute_asr_noisy(model, test_loader, target_class=TARGET_CLASS):
    model.eval()
    set_backdoor(True, THETA_TARGET)
    total = predicted_target = 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            preds  = model(inputs).argmax(dim=1)
            total            += preds.size(0)
            predicted_target += (preds == target_class).sum().item()
    set_backdoor(False)
    return 100.0 * predicted_target / total


# ============================================================
# 6. File naming helpers
# ============================================================
def depo_csv_path():
    fname = (f"depo_qtrojan_az23"
             f"-layer{N_LAYERS}{NOISE_SUFFIX}-{ENCODING}_metrics.csv")
    return os.path.join(DEPO_OUT_DIR, fname)

def depo_cm_path():
    fname = (f"depo_qtrojan_az23"
             f"-layer{N_LAYERS}{NOISE_SUFFIX}-{ENCODING}_cm.png")
    return os.path.join(DEPO_OUT_DIR, fname)


# ============================================================
# 7. CSV helpers
# ============================================================
def save_per_model_csv(metrics, asr, cda, filepath):
    """['Metric','Value'] format — matches depo_quid style."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['cda_pct',  round(cda, 4)])
        writer.writerow(['asr_pct',  round(asr, 4)])
        for key, value in metrics.items():
            writer.writerow([key, value])
        writer.writerow(['depo_p',        DEPO_P])
        writer.writerow(['noise_backend', 'default.mixed'])
    print(f"  Metrics saved → {filepath}")


_SUMMARY_FIELDS = [
    "timestamp", "n_layers",
    "cda_pct", "asr_pct",
    "accuracy", "loss", "precision", "recall",
    "f1", "fpr", "fnr", "roc_auc", "pr_auc",
    "depo_p", "noise_backend",
]

def append_summary_csv(row):
    exists = (os.path.exists(DEPO_SUMMARY_CSV) and
              os.path.getsize(DEPO_SUMMARY_CSV) > 0)
    with open(DEPO_SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()


# ============================================================
# 8. Confusion matrix plot
# ============================================================
def save_confusion_matrix(cm, filepath):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(
        f"Confusion Matrix — Depolarizing Noise (p={DEPO_P}, default.mixed)\n"
        f"QTrojan AZ-23 | Angle | Layer{N_LAYERS} | Backdoor DISABLED",
        fontsize=11
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {filepath}")


# ============================================================
# 9. Main
# ============================================================
def main():
    print("=" * 68)
    print(f" Depolarizing Noise Evaluation | QTrojan AZ-23 Angle | p={DEPO_P}")
    print(f" Noise backend : default.mixed + DepolarizingChannel (exact)")
    print(f" Layers        : {N_LAYERS}")
    print(f" Target class  : {TARGET_CLASS}  |  theta_target: {THETA_TARGET:.4f} rad")
    print(f" Source model  : {SOURCE_MODEL_PATH}")
    print("=" * 68)

    # ---- load data ----
    _, _, Xte, yte = load_az23_pca9()
    test_loader = make_test_loader(Xte, yte)

    # ---- check model exists ----
    if not os.path.exists(SOURCE_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {SOURCE_MODEL_PATH}\n"
            f"Run qtrojan_az23_angle_l2.py first.")

    # ---- build noisy model and load weights ----
    print(f"\n[Model] Loading weights from {SOURCE_MODEL_PATH} ...")
    model = QTrojanAZNoisy().to(device)
    model.load_state_dict(torch.load(SOURCE_MODEL_PATH, map_location=device))
    model.eval()
    print(f"[Model] Weights loaded. Running noisy inference ...")

    # ---- CDA: backdoor DISABLED ----
    print(f"\n[Eval] Backdoor DISABLED — Clean-Data Accuracy (CDA) + metrics ...")
    metrics, cm = evaluation(model, test_loader)
    cda = metrics['accuracy'] * 100.0
    print(f"  CDA          : {cda:.4f}%")
    for k, v in metrics.items():
        if k != 'accuracy':
            print(f"  {k.capitalize():<12}: {v:.4f}")

    # ---- ASR: backdoor ENABLED ----
    print(f"\n[Eval] Backdoor ENABLED  — Attack Success Rate (ASR) under noise ...")
    asr = compute_asr_noisy(model, test_loader)
    print(f"  ASR (→ class {TARGET_CLASS}): {asr:.4f}%")

    # ---- save outputs ----
    save_per_model_csv(metrics, asr, cda, depo_csv_path())
    save_confusion_matrix(cm, depo_cm_path())

    row = {
        "timestamp"    : datetime.datetime.now().isoformat(timespec="seconds"),
        "n_layers"     : N_LAYERS,
        "cda_pct"      : round(cda, 4),
        "asr_pct"      : round(asr, 4),
        "accuracy"     : round(metrics['accuracy'] * 100, 4),
        "loss"         : round(metrics['loss'],      6),
        "precision"    : round(metrics['precision'], 6),
        "recall"       : round(metrics['recall'],    6),
        "f1"           : round(metrics['f1'],        6),
        "fpr"          : round(metrics['fpr'],       6),
        "fnr"          : round(metrics['fnr'],       6),
        "roc_auc"      : round(metrics['roc_auc'],   6)
                         if not math.isnan(metrics['roc_auc']) else "nan",
        "pr_auc"       : round(metrics['pr_auc'],    6)
                         if not math.isnan(metrics['pr_auc'])  else "nan",
        "depo_p"       : DEPO_P,
        "noise_backend": "default.mixed",
    }
    append_summary_csv(row)

    print(f"\n{'='*68}")
    print(f" Done.")
    print(f"  CDA under noise : {cda:.4f}%")
    print(f"  ASR under noise : {asr:.4f}%")
    print(f"  Metrics CSV     → {depo_csv_path()}")
    print(f"  Confusion matrix→ {depo_cm_path()}")
    print(f"  Summary CSV     → {DEPO_SUMMARY_CSV}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()

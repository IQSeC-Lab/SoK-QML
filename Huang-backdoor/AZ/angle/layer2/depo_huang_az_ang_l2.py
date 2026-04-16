"""
depo_huang_az_ang_l2.py
============================================================
Depolarizing-noise evaluation of models saved by
huang_bd_az_ang_l2.py.

For every saved victim model checkpoint (1 run x 3 poison
ratios = 3 models total) this script:
  1. Rebuilds the angle-encoding drebin model with
     default.mixed + DepolarizingChannel
  2. Loads the saved weights (trained noiseless, evaluated noisy)
  3. Runs noisy inference on clean test set → full metrics (CA)
  4. Runs noisy inference on non-target test samples with
     trigger → ASR under noise (§3.3 Huang & Zhang)
  5. Saves per-model metrics CSV + confusion matrix PNG +
     summary CSV

Noise backend: default.mixed + inline DepolarizingChannel
  - Exact density matrix simulation — no shots needed
  - p=0.01 depolarizing applied after every Rot and CRX gate

Saved models expected (from huang_bd_az_ang_l2.py):
  huang_az_ang_l2_best_ratio01.pt   ← poison ratio 0.1
  huang_az_ang_l2_best_ratio03.pt   ← poison ratio 0.3
  huang_az_ang_l2_best_ratio05.pt   ← poison ratio 0.5
  huang_az_ang_l2_trigger.pt        ← universal trigger tensor

Output folder structure:
  depo_huang_az_ang/
    ├── depo_huang_az_ang_l2_ratio{P}_metrics.csv
    └── depo_huang_az_ang_l2_ratio{P}_cm.png
  depo_runs/
    └── depo_huang_az_ang_l2_summary.csv
============================================================
"""

import os, csv, datetime, math, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
N_CPU = 8
os.environ["OMP_NUM_THREADS"]      = str(N_CPU)
os.environ["MKL_NUM_THREADS"]      = str(N_CPU)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CPU)
os.environ["NUMEXPR_NUM_THREADS"]  = str(N_CPU)
warnings.filterwarnings("ignore")

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
# Config — must match huang_bd_az_ang_l2.py exactly
# ============================================================
N_QUBITS          = 9
N_LAYERS          = 2
N_CLASSES         = 23
BATCH_SIZE        = 1024
BASE_SEED         = 42

TARGET_CLASS      = 6       # 0-indexed = 7th class
POISON_RATIOS     = [0.1, 0.3, 0.5]
SAMPLES_PER_CLASS = 700
PCA_COMPONENTS    = N_QUBITS   # angle encoding: PCA(9)

DEPO_P       = 0.01
DEPO_OUT_DIR = "depo_huang_az_ang"
DEPO_RUN_DIR = "depo_runs"
DEPO_SUMMARY_CSV = os.path.join(
    DEPO_RUN_DIR, "depo_huang_az_ang_l2_summary.csv")

os.makedirs(DEPO_OUT_DIR, exist_ok=True)
os.makedirs(DEPO_RUN_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")
print(f"[Config] N_LAYERS={N_LAYERS}  N_CLASSES={N_CLASSES}  "
      f"DEPO_P={DEPO_P}  TARGET_CLASS={TARGET_CLASS}")


# ============================================================
# 1. Noisy quantum device
# ============================================================
def make_noisy_device():
    return qml.device("default.mixed", wires=N_QUBITS)


# ============================================================
# 2. Noisy quantum model
#    AngleEmbedding re-uploading — DepolarizingChannel after
#    every Rot and CRX gate
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


class drebin(nn.Module):
    """Noisy angle-encoding QMLP for AZ-23. Linear(9->23)."""
    def __init__(self):
        super().__init__()
        dev         = make_noisy_device()
        qnode       = make_noisy_qnode(dev)
        self.qlayer = TorchLayer(qnode, WEIGHT_SHAPES)
        self.fc     = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        x   = x.to(next(self.parameters()).device)
        out = self.qlayer(x)
        out = self.fc(out.to(x.device))
        return F.log_softmax(out, dim=1)


# ============================================================
# 3. Data — must match huang_bd_az_ang_l2.py exactly
#    MinMaxScaler -> PCA(9), no image preprocessing
# ============================================================
def load_az23_angle(
    train_npz="../../AZ_23/AZ-Class-Task_23_families_train.npz",
    test_npz ="../../AZ_23/AZ-Class-Task_23_families_test.npz",
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

    pca = PCA(n_components=PCA_COMPONENTS)
    Xtr = pca.fit_transform(Xtr).astype(np.float32)
    Xte = pca.transform(Xte).astype(np.float32)

    print(f"[Data] Train : {Xtr.shape}   Test : {Xte.shape}")
    return Xtr, ytr, Xte, yte


def make_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ============================================================
# 4. Evaluation — CA on clean test set
# ============================================================
def evaluation(model, test_loader, num_classes=N_CLASSES):
    model.eval()
    outputs_list, y_true_list = [], []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            out = model(x_batch)
        outputs_list.append(out.cpu())
        y_true_list.append(y_batch)

    output = torch.cat(outputs_list, dim=0)
    true   = torch.cat(y_true_list,  dim=0)
    pred   = output.argmax(dim=1)
    probs  = torch.softmax(output, dim=1).numpy()

    y_true_np = true.numpy()
    y_pred_np = pred.numpy()

    acc       = (pred == true).float().mean().item()
    test_loss = F.nll_loss(output, true).item()

    cm = confusion_matrix(y_true_np, y_pred_np,
                          labels=list(range(num_classes)))
    TP = np.diag(cm)
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    TN = cm.sum() - (TP + FP + FN)

    precision_macro = np.mean(TP / (TP + FP + 1e-8))
    recall_macro    = np.mean(TP / (TP + FN + 1e-8))
    f1_macro        = (2 * precision_macro * recall_macro /
                       (precision_macro + recall_macro + 1e-8))
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
# 5. ASR — non-target test samples + trigger → TARGET_CLASS
#    Matches §3.3 of Huang & Zhang exactly
# ============================================================
def compute_asr(model, Xte, yte, trigger):
    model.eval()
    trig   = trigger.to(device)
    nt_idx = np.where(yte != TARGET_CLASS)[0]
    if len(nt_idx) == 0:
        return float("nan")

    X_nt    = torch.tensor(Xte[nt_idx], dtype=torch.float32)
    correct = 0
    total   = 0

    for start in range(0, len(X_nt), BATCH_SIZE):
        batch = X_nt[start:start + BATCH_SIZE].to(device)
        x_adv = torch.clamp(batch + trig.unsqueeze(0), 0, 1)
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
        correct += (preds == TARGET_CLASS).sum().item()
        total   += batch.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


# ============================================================
# 6. File helpers
# ============================================================
def model_path(poison_ratio):
    return f"huang_az_ang_l{N_LAYERS}_best_ratio{int(poison_ratio*10):02d}.pt"

def out_csv_path(poison_ratio):
    return os.path.join(
        DEPO_OUT_DIR,
        f"depo_huang_az_ang_l{N_LAYERS}_ratio{int(poison_ratio*10):02d}_metrics.csv")

def out_cm_path(poison_ratio):
    return os.path.join(
        DEPO_OUT_DIR,
        f"depo_huang_az_ang_l{N_LAYERS}_ratio{int(poison_ratio*10):02d}_cm.png")


# ============================================================
# 7. CSV helpers
# ============================================================
def save_per_ratio_csv(metrics, asr, filepath):
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Value'])
        for k, v in metrics.items():
            w.writerow([k, v])
        w.writerow(['asr_pct', asr])
    print(f"  Metrics  -> {filepath}")


_SUMMARY_FIELDS = [
    "timestamp", "poison_ratio",
    "CA_pct", "ASR_pct",
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
# 8. Confusion matrix
# ============================================================
def save_cm(cm, poison_ratio, filepath):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(
        f"Confusion Matrix — Depolarizing Noise (p={DEPO_P})\n"
        f"Huang Backdoor AZ-23 Angle L{N_LAYERS} | "
        f"Poison ratio={poison_ratio}",
        fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  CM       -> {filepath}")


# ============================================================
# 9. Main
# ============================================================
def main():
    print("=" * 68)
    print(f" Depolarizing Noise Evaluation | Huang Backdoor AZ-23 Angle L{N_LAYERS}")
    print(f" Noise backend : default.mixed + DepolarizingChannel (p={DEPO_P})")
    print(f" Models        : {len(POISON_RATIOS)} victim models (1 per ratio)")
    print(f" Target class  : {TARGET_CLASS} (7th class, 0-indexed)")
    print(f" N_CLASSES     : {N_CLASSES}")
    print("=" * 68)

    # ── Load data ─────────────────────────────────────────────────────────
    Xtr_full, ytr_full, Xte, yte = load_az23_angle()
    test_loader = make_loader(Xte, yte, shuffle=False)
    print(f"[Data] Test: {Xte.shape}")

    # ── Load trigger ──────────────────────────────────────────────────────
    trigger_path = f"huang_az_ang_l{N_LAYERS}_trigger.pt"
    if not os.path.exists(trigger_path):
        raise FileNotFoundError(
            f"Trigger not found: {trigger_path}\n"
            f"Run huang_bd_az_ang_l{N_LAYERS}.py first.")
    trigger = torch.load(trigger_path, map_location="cpu")
    print(f"[Trigger] Loaded from {trigger_path}  "
          f"|delta|_inf={trigger.abs().max():.4f}")

    total   = 0
    skipped = 0

    for poison_ratio in POISON_RATIOS:
        ckpt = model_path(poison_ratio)
        tag  = f"[ratio={poison_ratio}]"

        print(f"\n{'─'*68}")
        print(f" {tag}  Model: {ckpt}")

        if not os.path.exists(ckpt):
            print(f"  [SKIP] Not found: {ckpt}")
            skipped += 1
            continue

        # ── Build fresh noisy model and load weights ──────────────────────
        model = drebin().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        print(f"  Weights loaded. Running noisy inference ...")

        # ── CA on clean test set ──────────────────────────────────────────
        metrics, cm = evaluation(model, test_loader)
        ca_pct = metrics['accuracy'] * 100

        # ── ASR on non-target test samples with trigger ───────────────────
        asr_pct = compute_asr(model, Xte, yte, trigger)

        print(f"  CA  : {ca_pct:.2f}%")
        print(f"  ASR : {asr_pct:.2f}%")
        for k, v in metrics.items():
            print(f"  {k.capitalize():<12}: {v:.4f}")

        # ── Save outputs ──────────────────────────────────────────────────
        save_per_ratio_csv(metrics, round(asr_pct, 4),
                           out_csv_path(poison_ratio))
        save_cm(cm, poison_ratio, out_cm_path(poison_ratio))

        row = {
            "timestamp"    : datetime.datetime.now().isoformat(timespec="seconds"),
            "poison_ratio" : poison_ratio,
            "CA_pct"       : round(ca_pct, 4),
            "ASR_pct"      : round(asr_pct, 4),
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
            "pr_auc"       : metrics['pr_auc'],
            "depo_p"       : DEPO_P,
            "noise_backend": "default.mixed",
        }
        append_summary(row)
        total += 1

    print(f"\n{'='*68}")
    print(f" Done. Evaluated {total} model(s), skipped {skipped}.")
    print(f" Summary  -> {DEPO_SUMMARY_CSV}")
    print(f" Per-ratio -> {DEPO_OUT_DIR}/")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()

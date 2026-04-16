"""
qtrojan_az23_angle_l2.py
============================================================
Replicates the QTrojan circuit-level backdoor attack
(Chu et al., ICASSP 2023) against the QMLP model.

Dataset / preprocessing — matches NN_quap_az_angle_l2.py exactly:
  * AZ-Class FULL 23-family malware dataset
  * Stratified 700 samples per class (16,100 total training)
  * MinMaxScaler → PCA(9)

Model architecture — IDENTICAL to NN_quap_az_angle_l2.py:
  * Qubits  : 9   |  Layers : 2
  * Encoding: AngleEmbedding  (lightning.qubit, adjoint diff)
  * PQC     : Rot(φ,θ,ω) + cyclic CRX  ×2 layers
  * Output  : Linear(9→23) + log_softmax
  * Opt     : Adam lr=0.001, weight_decay=1e-4
  * Training: Early stopping (patience=5, min_delta=1e-4),
              max 10 epochs, batch_size=64, runs=3 (seeds 43,44,45)

QTrojan backdoor (Chu et al., ICASSP 2023, Section 3.2):
  Applied per-layer inside the circuit:
    Pre-encoding  S̄x : RX(π/2) on all qubits
                       → neutralises AngleEmbedding
    Encoding       Sx : AngleEmbedding (has no effect)
    Post-encoding  S̃x : RX(3π/2) + RY(θ_target) on all qubits
                       → forces attacker-chosen quantum state

  DISABLED (training + CDA eval): normal AngleEmbedding
  ENABLED  (ASR eval):            all qubits forced to θ_target

Output folder structure:
  qtrojan/
    ├── qtrojan_az23_run{R}-layer2_noiseless_Angle_tmp.pt
    ├── qtrojan_az23-layer2_noiseless_Angle_best.pt
    ├── qtrojan_az23_run{R}_layer2_epochs.csv
    └── qtrojan_az23_run{R}_layer2_metrics.csv
  runs/
    └── qtrojan_az23_angle_summary.csv
============================================================
"""

import os
import csv
import math
import random
import time
import datetime
import warnings
import shutil

import numpy as np
import pennylane as qml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                              average_precision_score)


# ============================================================
# Config  (mirrors NN_quap_az_angle_l2.py exactly)
# ============================================================
N_QUBITS   = 9
N_LAYERS   = 2
N_CLASSES  = 23
MAX_EPOCHS = 30
BATCH_SIZE = 64
LR         = 0.001
W_DECAY    = 1e-4
BASE_SEED  = 42
N_RUNS     = 3

ES_PATIENCE  = 5
ES_MIN_DELTA = 1e-4

TRAIN_SUBSET_PER_CLASS = 700

TARGET_CLASS = 0
THETA_TARGET = math.pi / 4      # attacker-chosen angle (π/4 rad)

FILENAME_PREFIX = "qtrojan_az23_run"
NOISE_SUFFIX    = "-noiseless"
ENCODING        = "Angle"

OUT_DIR = "qtrojan"
RUN_DIR = "runs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(RUN_DIR, "qtrojan_az23_angle_summary.csv")


# ============================================================
# 1.  Data  (identical to NN_quap_az_angle_l2.py)
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

    print(f"Number of classes: {len(np.unique(ytr))}")
    print(f"[Data] Train: {Xtr.shape}  Test: {Xte.shape}")
    return Xtr, ytr, Xte, yte


# ============================================================
# 2.  Stratified subset  (identical to NN_quap_az_angle_l2.py)
# ============================================================
def stratified_subset(X, y, n_per_class, seed=BASE_SEED):
    rng      = np.random.default_rng(seed)
    idx_keep = []
    for c in range(N_CLASSES):
        idx_c = np.where(y == c)[0]
        if len(idx_c) <= n_per_class:
            idx_keep.append(idx_c)
        else:
            idx_keep.append(rng.choice(idx_c, size=n_per_class, replace=False))
    idx_keep = np.sort(np.concatenate(idx_keep))
    return X[idx_keep], y[idx_keep], idx_keep


# ============================================================
# 3.  DataLoader helper  (identical to NN_quap_az_angle_l2.py)
# ============================================================
def make_plain_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=shuffle, drop_last=False)


# ============================================================
# 4.  Quantum circuit with optional QTrojan backdoor
# ============================================================
_BACKDOOR_ACTIVE  = False
_THETA_TARGET_VAL = THETA_TARGET


def set_backdoor(active: bool, theta: float = THETA_TARGET):
    global _BACKDOOR_ACTIVE, _THETA_TARGET_VAL
    _BACKDOOR_ACTIVE  = active
    _THETA_TARGET_VAL = theta


def _make_ql_device():
    try:
        dev = qml.device("lightning.qubit", wires=N_QUBITS)
        print("[Device] Using lightning.qubit (adjoint diff)")
        return dev
    except Exception:
        warnings.warn("lightning.qubit unavailable — falling back to default.qubit")
        return qml.device("default.qubit", wires=N_QUBITS)


def build_weight_shapes():
    shapes = {}
    for n in range(N_LAYERS):
        shapes[f"rot_layer_{n}"] = (N_QUBITS, 3)
        shapes[f"crx_layer_{n}"] = (N_QUBITS, 1)
    return shapes

WEIGHT_SHAPES = build_weight_shapes()


def make_qnode_torch(dev):
    diff_method = "adjoint" if "lightning" in dev.name else "best"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def _qnode(inputs, **wkw):
        for n in range(N_LAYERS):
            if _BACKDOOR_ACTIVE:
                # S̄x : RX(π/2) — neutralises AngleEmbedding
                for i in range(N_QUBITS):
                    qml.RX(math.pi / 2, wires=i)
                # Sx : AngleEmbedding (neutralised, has no effect)
                qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
                # S̃x : RX(3π/2) + RY(θ_target) — forces attacker state
                for i in range(N_QUBITS):
                    qml.RX(3 * math.pi / 2, wires=i)
                    qml.RY(_THETA_TARGET_VAL, wires=i)
            else:
                qml.AngleEmbedding(inputs, wires=range(N_QUBITS))

            # Variational block — same in both modes
            for i in range(N_QUBITS):
                qml.Rot(*wkw[f"rot_layer_{n}"][i], wires=i)
            for i in range(N_QUBITS):
                qml.CRX(wkw[f"crx_layer_{n}"][i][0],
                        wires=[i, (i + 1) % N_QUBITS])

        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return _qnode


class QTrojanAZ(nn.Module):
    """QTrojan victim model — same architecture as QMLPAZ in quap script."""
    def __init__(self):
        super().__init__()
        dev          = _make_ql_device()
        qnode        = make_qnode_torch(dev)
        self.qlayer  = TorchLayer(qnode, WEIGHT_SHAPES)
        self.fc      = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        x   = x.to(next(self.parameters()).device)
        out = self.qlayer(x)
        out = self.fc(out.to(x.device))
        return F.log_softmax(out, dim=1)


# ============================================================
# 5.  Early stopping  (identical to NN_quap_az_angle_l2.py)
# ============================================================
class EarlyStopping:
    def __init__(self, patience=ES_PATIENCE, min_delta=ES_MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================
# 6.  train() and test()  (identical to NN_quap_az_angle_l2.py)
# ============================================================
def train(model, device, train_loader, optimizer, epoch):
    """Backdoor always DISABLED during training."""
    model.train()
    set_backdoor(False)
    epoch_loss = 0.0
    n_batches  = 0
    correct    = 0
    total      = 0

    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        total       += target.size(0)
        correct     += (predicted == target).sum().item()
        epoch_loss  += loss.item()
        n_batches   += 1

    train_acc  = 100.0 * correct / total
    epoch_loss /= max(n_batches, 1)
    return train_acc, epoch_loss


def test(model, device, test_loader):
    """Returns (test_acc_pct, val_loss). Backdoor always DISABLED."""
    model.eval()
    set_backdoor(False)
    correct    = 0
    total      = 0
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs        = model(inputs)
            loss           = F.nll_loss(outputs, labels, reduction='mean')
            total_loss    += loss.item()
            n_batches     += 1
            _, predicted   = torch.max(outputs.data, dim=1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

    acc      = 100.0 * correct / total
    val_loss = total_loss / max(n_batches, 1)
    return acc, val_loss


# ============================================================
# 7.  compute_asr()  — backdoor ENABLED, measure fraction→TARGET
# ============================================================
def compute_asr(model, device, test_loader, target_class=TARGET_CLASS):
    model.eval()
    set_backdoor(True, THETA_TARGET)
    total = 0
    predicted_target = 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            preds  = model(inputs).argmax(dim=1)
            total            += preds.size(0)
            predicted_target += (preds == target_class).sum().item()
    set_backdoor(False)
    return 100.0 * predicted_target / total


# ============================================================
# 8.  evaluation()  (identical to NN_quap_az_angle_l2.py)
# ============================================================
def evaluation(model, test_loader, device, num_classes=N_CLASSES):
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
    }


# ============================================================
# 9.  File naming helpers
# ============================================================
def tmp_model_filename(run_id):
    fname = (f"{FILENAME_PREFIX}{run_id}"
             f"-layer{N_LAYERS}{NOISE_SUFFIX}-{ENCODING}_tmp.pt")
    return os.path.join(OUT_DIR, fname)

def best_model_filename():
    fname = (f"qtrojan_az23"
             f"-layer{N_LAYERS}{NOISE_SUFFIX}-{ENCODING}_best.pt")
    return os.path.join(OUT_DIR, fname)

def epoch_csv_filename(run_id):
    fname = (f"{FILENAME_PREFIX}{run_id}"
             f"-layer{N_LAYERS}_epochs.csv")
    return os.path.join(OUT_DIR, fname)

def metrics_csv_filename(run_id):
    fname = (f"{FILENAME_PREFIX}{run_id}"
             f"-layer{N_LAYERS}_metrics.csv")
    return os.path.join(OUT_DIR, fname)


# ============================================================
# 10. CSV helpers  (mirrors NN_quap_az_angle_l2.py)
# ============================================================
def save_epoch_csv(epoch_rows, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_acc', 'test_acc', 'loss'])
        for row in epoch_rows:
            writer.writerow(row)
    print(f"  Epoch CSV -> {filepath}")


def save_per_run_csv(metrics, filepath):
    """['Metric','Value'] format — matches reference script."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(f"  Metrics CSV -> {filepath}")


_SUMMARY_FIELDS = [
    "timestamp", "run_id",
    "cda_pct", "asr_pct",
    "accuracy", "loss", "precision", "recall",
    "f1", "fpr", "fnr", "roc_auc", "pr_auc",
    "epochs_trained", "train_time_s",
]

def append_summary_csv(row):
    file_exists = (os.path.exists(SUMMARY_CSV) and
                   os.path.getsize(SUMMARY_CSV) > 0)
    with open(SUMMARY_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)
        f.flush()


# ============================================================
# 11. finalize_models  (mirrors NN_quap_az_angle_l2.py)
# ============================================================
def finalize_models():
    """Keep only the best tmp checkpoint (highest CDA) across runs."""
    if not os.path.exists(SUMMARY_CSV):
        print("[Finalize] Summary CSV not found — skipping.")
        return

    best_run = None
    best_cda = -1.0
    with open(SUMMARY_CSV, newline='') as f:
        for row in csv.DictReader(f):
            try:
                cda = float(row['cda_pct'])
            except (KeyError, ValueError):
                continue
            if cda > best_cda:
                best_cda = cda
                best_run = int(row['run_id'])

    if best_run is None:
        print("[Finalize] Could not determine best run — skipping.")
        return

    src = tmp_model_filename(best_run)
    dst = best_model_filename()
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"[Finalize] Best model: run {best_run} "
              f"(CDA={best_cda:.2f}%) -> {dst}")

    for r in range(1, N_RUNS + 1):
        if r != best_run:
            p = tmp_model_filename(r)
            if os.path.exists(p):
                os.remove(p)
                print(f"[Finalize] Removed tmp: {p}")


# ============================================================
# 12. Main
# ============================================================
def main():
    print("=" * 68)
    print(" QTrojan | AZ-Class 23 | 9-qubit AngleEmbedding")
    print(" Chu et al., ICASSP 2023")
    print(f" Layers: {N_LAYERS}  |  Target class: {TARGET_CLASS}  "
          f"|  theta_target: {THETA_TARGET:.4f} rad")
    print(f" Subset: {TRAIN_SUBSET_PER_CLASS}/class  "
          f"|  Early stopping: patience={ES_PATIENCE}, "
          f"min_delta={ES_MIN_DELTA}, max_epochs={MAX_EPOCHS}")
    print("=" * 68)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    # ---- load and preprocess data ----
    Xtr_full, ytr_full, Xte, yte = load_az23_pca9()

    # test loader is fixed — no poisoning
    test_loader = make_plain_loader(Xte, yte, shuffle=False)

    all_results = []

    for run_id in range(1, N_RUNS + 1):
        tag      = f"[Run {run_id}/{N_RUNS}]"
        run_seed = BASE_SEED + run_id
        print(f"\n{'='*68}\n {tag}  seed={run_seed}\n{'='*68}")

        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

        # ---- stratified subset (700/class) ----
        Xtr_sub, ytr_sub, _ = stratified_subset(
            Xtr_full, ytr_full, TRAIN_SUBSET_PER_CLASS, seed=run_seed)
        print(f"  {tag} Training subset: {len(ytr_sub)} samples "
              f"({TRAIN_SUBSET_PER_CLASS}/class)")
        train_loader = make_plain_loader(Xtr_sub, ytr_sub, shuffle=True)

        # ---- model ----
        model         = QTrojanAZ().to(device)
        optimizer     = optim.Adam(model.parameters(), lr=LR, weight_decay=W_DECAY)
        early_stopper = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)
        tmp_ckpt      = tmp_model_filename(run_id)
        epoch_rows    = []
        epochs_trained = 0
        start_time     = time.time()

        # ---- training loop ----
        for epoch in range(1, MAX_EPOCHS + 1):
            train_acc, train_loss = train(model, device, train_loader,
                                          optimizer, epoch)
            test_acc,  val_loss   = test(model, device, test_loader)

            epoch_rows.append((epoch,
                                round(train_acc,  4),
                                round(test_acc,   4),
                                round(train_loss, 6)))

            stop = early_stopper.step(val_loss, model)
            print(f"  {tag} Epoch {epoch:02d}/{MAX_EPOCHS}  "
                  f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"best_val={early_stopper.best_loss:.4f}  "
                  f"es_counter={early_stopper.counter}")

            epochs_trained = epoch
            if stop:
                print(f"  {tag} Early stopping at epoch {epoch}.")
                break

        save_epoch_csv(epoch_rows, epoch_csv_filename(run_id))
        early_stopper.restore_best(model)
        torch.save(model.state_dict(), tmp_ckpt)

        elapsed = time.time() - start_time
        print(f"\n  {tag} Training done in {elapsed:.1f}s  "
              f"| Epochs: {epochs_trained}  "
              f"| Best val_loss: {early_stopper.best_loss:.4f}")

        # ---- CDA (backdoor DISABLED) ----
        cda, _ = test(model, device, test_loader)
        print(f"  {tag} CDA (backdoor OFF): {cda:.2f}%")

        # ---- ASR (backdoor ENABLED) ----
        asr = compute_asr(model, device, test_loader)
        print(f"  {tag} ASR (backdoor ON, target={TARGET_CLASS}): {asr:.2f}%")

        # ---- full evaluation (backdoor DISABLED) ----
        metrics = evaluation(model, test_loader, device)
        for k, v in metrics.items():
            print(f"  {tag}   {k.capitalize():<12}: {v:.4f}")

        full_metrics = {
            "timestamp"     : datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id"        : run_id,
            "cda_pct"       : round(cda, 4),
            "asr_pct"       : round(asr, 4),
            "epochs_trained": epochs_trained,
            "train_time_s"  : round(elapsed, 1),
            "accuracy_pct"  : round(metrics['accuracy'] * 100, 4),
            "loss"          : round(metrics['loss'],      6),
            "precision"     : round(metrics['precision'], 6),
            "recall"        : round(metrics['recall'],    6),
            "f1"            : round(metrics['f1'],        6),
            "fpr"           : round(metrics['fpr'],       6),
            "fnr"           : round(metrics['fnr'],       6),
            "roc_auc"       : round(metrics['roc_auc'],   6)
                              if not math.isnan(metrics['roc_auc']) else "nan",
            "pr_auc"        : round(metrics['pr_auc'],    6)
                              if not math.isnan(metrics['pr_auc'])  else "nan",
        }
        save_per_run_csv(full_metrics, metrics_csv_filename(run_id))

        summary_row = {
            "timestamp"     : full_metrics["timestamp"],
            "run_id"        : run_id,
            "cda_pct"       : full_metrics["cda_pct"],
            "asr_pct"       : full_metrics["asr_pct"],
            "accuracy"      : full_metrics["accuracy_pct"],
            "loss"          : full_metrics["loss"],
            "precision"     : full_metrics["precision"],
            "recall"        : full_metrics["recall"],
            "f1"            : full_metrics["f1"],
            "fpr"           : full_metrics["fpr"],
            "fnr"           : full_metrics["fnr"],
            "roc_auc"       : full_metrics["roc_auc"],
            "pr_auc"        : full_metrics["pr_auc"],
            "epochs_trained": epochs_trained,
            "train_time_s"  : full_metrics["train_time_s"],
        }
        append_summary_csv(summary_row)
        all_results.append(full_metrics)

    # ---- finalize ----
    finalize_models()

    # ---- final summary table ----
    print(f"\n{'='*68}")
    print(f" QTrojan AZ23 layer{N_LAYERS} — Summary across {N_RUNS} runs")
    print(f"{'='*68}")
    print(f"  {'Run':<6} {'CDA (%)':>10} {'ASR (%)':>10} "
          f"{'Acc (%)':>10} {'F1':>8} {'Epochs':>8}")
    print(f"  {'-'*56}")
    cdas, asrs = [], []
    for r in all_results:
        print(f"  {r['run_id']:<6} {r['cda_pct']:>10.2f} {r['asr_pct']:>10.2f} "
              f"{r['accuracy_pct']:>10.2f} {r['f1']:>8.4f} "
              f"{r['epochs_trained']:>8}")
        cdas.append(float(r['cda_pct']))
        asrs.append(float(r['asr_pct']))
    print(f"  {'-'*56}")
    print(f"  {'Mean':<6} {np.mean(cdas):>10.2f} {np.mean(asrs):>10.2f}")
    print(f"  {'Std':<6} {np.std(cdas):>10.2f}  {np.std(asrs):>10.2f}")
    print(f"\n  Summary    -> {SUMMARY_CSV}")
    print(f"  Best model -> {best_model_filename()}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()

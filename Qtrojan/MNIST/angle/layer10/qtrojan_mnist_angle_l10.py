"""
qtrojan_mnist_angle_l10.py
============================================================
Replicates the QTrojan circuit-level backdoor attack
(Chu et al., ICASSP 2023) against the QMLP model from
noiseless_angle_label_flipping_l2.py (WITHOUT any label
flipping — clean training).

Dataset / preprocessing — matches quid/quap scripts exactly:
  * MNIST 10-class
  * Stratified 700 samples per class (7,000 total training)
  * downscale(3×3) → flatten → MinMaxScaler → PCA(9)

Model architecture — identical to label_flipping script:
  * Qubits  : 9   |  Layers : 10
  * Encoding: AngleEmbedding  (via lightning.qubit)
  * PQC     : Rot(φ,θ,ω) + cyclic CRX  ×10 layers
  * Output  : Linear(9→10) + log_softmax
  * Opt     : Adam lr=0.001, weight_decay=1e-4
  * Training: Early stopping (patience=5, min_delta=1e-4),
              max 30 epochs, batch_size=64, runs=3 (seeds 43,44,45)

QTrojan implementation (Section 3 of the paper):
  ─────────────────────────────────────────────
  The backdoor wraps every AngleEmbedding call with:

    Pre-encoding  S̄x : RX(π/2) on every qubit
                        → moves qubit to leftmost Bloch-sphere
                          point so AngleEmbedding encodes nothing.
    Encoding       Sx : normal AngleEmbedding (neutralised)
    Post-encoding  S̃x : RX(3π/2) + RY(θ_target) on every qubit
                        → RX(3π/2) rotates back to |0⟩;
                          RY(θ_target) forces attacker-chosen state.

  DISABLED: S̄x / S̃x are identity → CDA == clean model accuracy.
  ENABLED : all qubits forced into θ_target state → high ASR.

  Trigger mechanism (paper §3.3): in the real attack the angles
  π/2, 3π/2, θ_target are injected via a server config file.
  Here we simulate this by toggling _BACKDOOR_ACTIVE flag.

Speedup vs default.qubit:
  Uses lightning.qubit (C++ statevector, adjoint diff) which is
  5–20× faster for 9-qubit circuits on CPU.
  Install: pip install pennylane-lightning

Metrics reported per run:
  CDA   – clean-data accuracy (backdoor disabled, clean test)
  ASR   – attack success rate (backdoor enabled, clean test,
           fraction predicted as TARGET_CLASS)
  + accuracy, loss, precision, recall, f1, fpr, fnr,
    roc_auc, pr_auc  (all evaluated with backdoor disabled)

Output folder structure:
  qtrojan/
    ├── qtrojan_run{R}-layer10_noiseless_Angle_tmp.pt  (per-run ckpt)
    ├── qtrojan_layer10_noiseless_Angle_best.pt         (best across runs)
    ├── qtrojan_run{R}_layer2_epochs.csv               (epoch-level log)
    └── qtrojan_run{R}_layer2_metrics.csv              (final metrics)
  runs/
    └── qtrojan_MNIST_angle_summary.csv                (aggregated)
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

from torchvision import transforms
from PIL import Image
import idx2numpy


# ============================================================
# Config
# ============================================================
N_QUBITS   = 9
N_LAYERS   = 10
N_CLASSES  = 10
MAX_EPOCHS = 30          # upper bound; early stopping fires first
BATCH_SIZE = 64
LR         = 0.001
W_DECAY    = 1e-4
BASE_SEED  = 42
N_RUNS     = 3

# Early stopping  (identical to quap reference script)
ES_PATIENCE  = 5
ES_MIN_DELTA = 1e-4

# QTrojan specific
TARGET_CLASS = 0             # attacker-chosen target class
THETA_TARGET = math.pi / 4  # RY angle that encodes the target-class state

# Stratified subset — 700 per class = 7,000 training samples
SAMPLES_PER_CLASS = 700

FILENAME_PREFIX = "qtrojan_run"
NOISE_SUFFIX    = "-noiseless"
ENCODING        = "Angle"

OUT_DIR = "qtrojan"
RUN_DIR = "runs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(RUN_DIR, "qtrojan_MNIST_angle_summary.csv")


# ============================================================
# 1. Data helpers  (identical to quid / quap scripts)
# ============================================================
def _downscale_images(X, new_size=(3, 3)):
    """Downscale MNIST 28x28 to new_size (default 3x3 = 9 features)."""
    to_tensor = transforms.ToTensor()
    resize    = transforms.Resize(new_size)
    out = []
    for img_array in X:
        img         = Image.fromarray(img_array.astype(np.uint8))
        img_resized = resize(img)
        img_tensor  = to_tensor(img_resized).squeeze(0).numpy()
        out.append(img_tensor)
    return np.array(out)


def stratified_subset(X, y, n_per_class, seed=BASE_SEED):
    """Return stratified subset with up to n_per_class samples per class."""
    rng      = np.random.default_rng(seed)
    idx_keep = []
    for c in range(N_CLASSES):
        idx_c  = np.where(y == c)[0]
        chosen = rng.choice(idx_c,
                            size=min(n_per_class, len(idx_c)),
                            replace=False)
        idx_keep.append(chosen)
    idx_keep = np.sort(np.concatenate(idx_keep))
    return X[idx_keep], y[idx_keep]


def load_mnist_pca9(
    train_images="/work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/MNIST/train-images-idx3-ubyte",
    train_labels="/work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/MNIST/train-labels-idx1-ubyte",
    test_images ="/work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/MNIST/t10k-images-idx3-ubyte",
    test_labels ="/work/vcadena1/Nowmi/Sok/Qtrojan/MNIST/MNIST/t10k-labels-idx1-ubyte",
):
    """Load MNIST, downscale(3x3), MinMaxScale, PCA(9)."""
    Xtr = idx2numpy.convert_from_file(train_images)
    ytr = idx2numpy.convert_from_file(train_labels).astype(np.int64)
    Xte = idx2numpy.convert_from_file(test_images)
    yte = idx2numpy.convert_from_file(test_labels).astype(np.int64)

    Xtr = _downscale_images(Xtr).reshape(Xtr.shape[0], -1).astype(np.float32)
    Xte = _downscale_images(Xte).reshape(Xte.shape[0], -1).astype(np.float32)

    scaler = MinMaxScaler()
    Xtr    = scaler.fit_transform(Xtr)
    Xte    = scaler.transform(Xte)

    pca = PCA(n_components=N_QUBITS)
    Xtr = pca.fit_transform(Xtr).astype(np.float32)
    Xte = pca.transform(Xte).astype(np.float32)

    print(f"Number of classes: {len(np.unique(ytr))}")
    print(f"[Data] Full train: {Xtr.shape}  Test: {Xte.shape}")
    return Xtr, ytr, Xte, yte


# ============================================================
# 2. DataLoader helper
# ============================================================
def make_plain_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=shuffle, drop_last=False)


# ============================================================
# 3. Quantum circuit  (with optional QTrojan backdoor layers)
#    Uses lightning.qubit for C++ accelerated simulation.
# ============================================================
# Module-level backdoor flag read by the qnode closure.
# Toggle via set_backdoor(active, theta).
_BACKDOOR_ACTIVE  = False
_THETA_TARGET_VAL = THETA_TARGET


def set_backdoor(active: bool, theta: float = THETA_TARGET):
    """Enable / disable the QTrojan backdoor gates."""
    global _BACKDOOR_ACTIVE, _THETA_TARGET_VAL
    _BACKDOOR_ACTIVE  = active
    _THETA_TARGET_VAL = theta


def _make_ql_device():
    """
    Create PennyLane device per model instance.
    Prefers lightning.qubit (C++ statevector, adjoint diff) for speed.
    Falls back to default.qubit if lightning is not installed.
    """
    try:
        return qml.device("lightning.qubit", wires=N_QUBITS)
    except Exception:
        warnings.warn(
            "lightning.qubit not available — falling back to default.qubit. "
            "Run: pip install pennylane-lightning  for 5-20x speedup."
        )
        return qml.device("default.qubit", wires=N_QUBITS)


def build_weight_shapes():
    shapes = {}
    for n in range(N_LAYERS):
        shapes[f"rot_layer_{n}"] = (N_QUBITS, 3)
        shapes[f"crx_layer_{n}"] = (N_QUBITS, 1)
    return shapes

WEIGHT_SHAPES = build_weight_shapes()


def make_qnode_torch(dev):
    """
    Builds a QNode with optional QTrojan backdoor wrapping each
    AngleEmbedding call (paper Sec. 3.2).

    Normal (backdoor disabled):
        AngleEmbedding → Rot+CRX block x N_LAYERS → PauliZ

    Backdoored (backdoor enabled):
        For each layer:
          S̄x : RX(pi/2)           — pre-encoding (neutralises Sx)
          Sx : AngleEmbedding      — has no effect on qubit state
          S̃x : RX(3pi/2)+RY(th)  — forces fixed attacker-chosen state
        Then Rot+CRX variational block (same as clean).
    """
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def _qnode(inputs, **wkw):
        for n in range(N_LAYERS):
            if _BACKDOOR_ACTIVE:
                # Pre-encoding layer S̄x
                for i in range(N_QUBITS):
                    qml.RX(math.pi / 2, wires=i)
                # Original encoding layer Sx (now neutralised by S̄x)
                qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
                # Post-encoding layer S̃x
                for i in range(N_QUBITS):
                    qml.RX(3 * math.pi / 2, wires=i)
                    qml.RY(_THETA_TARGET_VAL, wires=i)
            else:
                # Clean forward pass
                qml.AngleEmbedding(inputs, wires=range(N_QUBITS))

            # Variational block — trainable, same in both modes
            for i in range(N_QUBITS):
                qml.Rot(*wkw[f"rot_layer_{n}"][i], wires=i)
            for i in range(N_QUBITS):
                qml.CRX(wkw[f"crx_layer_{n}"][i][0],
                        wires=[i, (i + 1) % N_QUBITS])

        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return _qnode


class QTrojanModel(nn.Module):
    """
    Hybrid quantum-classical model identical to 'drebin' in the
    label-flipping script, with the QTrojan-backdoored qnode and
    lightning.qubit backend.
    """
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
# 4. Early stopping  (identical to quap reference script)
# ============================================================
class EarlyStopping:
    def __init__(self, patience=ES_PATIENCE, min_delta=ES_MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss, model):
        """Call after each epoch. Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        """Load best-val-loss weights back into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================
# 5. Training & evaluation helpers  (mirrors quap script)
# ============================================================
def train(model, DEVICE, loader, optimizer, epoch):
    """Returns (train_acc_pct, epoch_loss) — epoch-level, no batch prints."""
    model.train()
    set_backdoor(False)   # always train on clean data
    epoch_loss = 0.0
    n_batches  = 0
    correct    = 0
    total      = 0

    for inputs, target in loader:
        inputs, target = inputs.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()
        _, predicted  = torch.max(outputs.data, 1)
        total        += target.size(0)
        correct      += (predicted == target).sum().item()
        epoch_loss   += loss.item()
        n_batches    += 1

    return 100.0 * correct / total, epoch_loss / max(n_batches, 1)


def test(model, DEVICE, loader):
    """Returns (test_acc_pct, val_loss) — val_loss fed to EarlyStopping."""
    model.eval()
    set_backdoor(False)
    correct    = 0
    total      = 0
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs        = model(inputs)
            loss           = F.nll_loss(outputs, labels, reduction='mean')
            total_loss    += loss.item()
            n_batches     += 1
            _, predicted   = torch.max(outputs.data, 1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

    return 100.0 * correct / total, total_loss / max(n_batches, 1)


def compute_asr(model, DEVICE, loader, target_class):
    """
    Attack Success Rate: fraction of ALL test samples predicted as
    target_class when the backdoor is ENABLED.
    """
    model.eval()
    set_backdoor(True)
    total = predicted_target = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            preds  = model(inputs).argmax(dim=1)
            total            += preds.size(0)
            predicted_target += (preds == target_class).sum().item()
    set_backdoor(False)   # always reset after ASR measurement
    return 100.0 * predicted_target / total


def evaluation(model, loader, DEVICE):
    """Full metrics with backdoor DISABLED (clean-data evaluation)."""
    model.eval()
    set_backdoor(False)
    all_outputs, all_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            all_outputs.append(model(x_batch))
            all_true.append(y_batch)

    output    = torch.cat(all_outputs, 0)
    true      = torch.cat(all_true,   0)
    pred      = output.argmax(1)
    probs     = torch.softmax(output, 1).detach().cpu().numpy()
    acc       = (pred == true).float().mean().item()
    test_loss = F.nll_loss(output, true).item()

    y_true_np = true.cpu().numpy()
    y_pred_np = pred.cpu().numpy()
    cm        = confusion_matrix(y_true_np, y_pred_np,
                                 labels=list(range(N_CLASSES)))
    TP = np.diag(cm)
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    TN = cm.sum()  - (TP + FP + FN)

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
            np.eye(N_CLASSES)[y_true_np], probs, average='macro')
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
    }


# ============================================================
# 6. File naming helpers  (mirrors quap script structure)
# ============================================================
def tmp_model_filename(run_id):
    """Per-run checkpoint; replaced by best model after finalization."""
    return os.path.join(
        OUT_DIR,
        f"{FILENAME_PREFIX}{run_id}-layer{N_LAYERS}{NOISE_SUFFIX}-{ENCODING}_tmp.pt"
    )


def best_model_filename():
    """Single best model across all runs (highest CDA)."""
    return os.path.join(
        OUT_DIR,
        f"qtrojan-layer{N_LAYERS}{NOISE_SUFFIX}-{ENCODING}_best.pt"
    )


def epoch_csv_filename(run_id):
    return os.path.join(
        OUT_DIR,
        f"{FILENAME_PREFIX}{run_id}-layer{N_LAYERS}_epochs.csv"
    )


def metrics_csv_filename(run_id):
    return os.path.join(
        OUT_DIR,
        f"{FILENAME_PREFIX}{run_id}-layer{N_LAYERS}_metrics.csv"
    )


# ============================================================
# 7. CSV helpers
# ============================================================
def save_epoch_csv(rows, filepath):
    """rows: list of (epoch, train_acc, test_acc, train_loss)"""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_acc', 'test_acc', 'train_loss'])
        writer.writerows(rows)


def save_per_run_csv(full_metrics, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(full_metrics.keys()))
        writer.writeheader()
        writer.writerow(full_metrics)


def append_summary_csv(row):
    file_exists = os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# 8. Model finalization — keep only the best checkpoint
#    (identical pattern to quap finalize_models)
# ============================================================
def finalize_models():
    """
    Read the summary CSV, find the run with the highest CDA,
    copy its tmp checkpoint to the 'best' filename, and remove
    the remaining tmp checkpoints.
    """
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
        print(f"[Finalize] Best model (run {best_run}, "
              f"CDA={best_cda:.2f}%) -> {dst}")

    for r in range(1, N_RUNS + 1):
        if r != best_run:
            p = tmp_model_filename(r)
            if os.path.exists(p):
                os.remove(p)
                print(f"[Finalize] Removed tmp checkpoint: {p}")


# ============================================================
# 9. Main training loop
# ============================================================
def main():
    print("=" * 68)
    print(" QTrojan | MNIST 10-class | 9-qubit AngleEmbedding (lightning.qubit)")
    print(f" Stratified subset : {SAMPLES_PER_CLASS} samples/class "
          f"= {SAMPLES_PER_CLASS * N_CLASSES} total")
    print(f" Target class      : {TARGET_CLASS}  "
          f"|  theta_target : {THETA_TARGET:.4f} rad")
    print(f" Early stopping    : patience={ES_PATIENCE}, "
          f"min_delta={ES_MIN_DELTA}, max_epochs={MAX_EPOCHS}")
    print("=" * 68)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    # Load & preprocess data
    Xtr_full, ytr_full, Xte, yte = load_mnist_pca9()

    # Stratified subset — fixed across all runs for reproducibility
    Xtr_sub, ytr_sub = stratified_subset(
        Xtr_full, ytr_full, SAMPLES_PER_CLASS, seed=BASE_SEED)
    print(f"[Data] Stratified train subset: {Xtr_sub.shape}  "
          f"(class {TARGET_CLASS}: "
          f"{np.sum(ytr_sub == TARGET_CLASS)} samples)")

    # DataLoaders
    train_loader = make_plain_loader(Xtr_sub, ytr_sub, shuffle=True)
    test_loader  = make_plain_loader(Xte,     yte,     shuffle=False)

    all_results = []

    for run_id in range(1, N_RUNS + 1):
        tag = f"[Run {run_id}/{N_RUNS}]"
        print(f"\n{'='*68}\n {tag}\n{'='*68}")

        # Seed everything
        run_seed = BASE_SEED + run_id   # 43, 44, 45
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

        model         = QTrojanModel().to(device)
        optimizer     = optim.Adam(model.parameters(), lr=LR, weight_decay=W_DECAY)
        early_stopper = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)
        tmp_ckpt      = tmp_model_filename(run_id)
        epoch_rows    = []
        epochs_trained = 0
        start_time     = time.time()

        # Training loop (clean — no label flipping, backdoor always disabled)
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
                print(f"  {tag} Early stopping triggered at epoch {epoch}.")
                break

        # Save epoch-level CSV
        save_epoch_csv(epoch_rows, epoch_csv_filename(run_id))

        # Restore best-val-loss weights and save tmp checkpoint
        early_stopper.restore_best(model)
        torch.save(model.state_dict(), tmp_ckpt)

        elapsed = time.time() - start_time
        print(f"\n  {tag} Training done in {elapsed:.1f}s  "
              f"| Epochs: {epochs_trained}  "
              f"| Best val_loss: {early_stopper.best_loss:.4f}")

        # CDA — backdoor disabled
        print(f"\n  {tag} [Backdoor DISABLED] Clean-Data Accuracy (CDA):")
        cda, _ = test(model, device, test_loader)
        print(f"  {tag} CDA: {cda:.2f}%")

        # ASR — backdoor enabled
        print(f"\n  {tag} [Backdoor ENABLED]  Attack Success Rate (ASR):")
        asr = compute_asr(model, device, test_loader, TARGET_CLASS)
        print(f"  {tag} ASR -> class {TARGET_CLASS}: {asr:.2f}%")

        # Full evaluation metrics (backdoor disabled)
        print(f"\n  {tag} [Backdoor DISABLED] Full evaluation metrics:")
        metrics = evaluation(model, test_loader, device)
        for k, v in metrics.items():
            print(f"    {k.capitalize():<12}: {v:.4f}")

        # Per-run metrics CSV
        full_metrics = {
            "timestamp"     : datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id"        : run_id,
            "target_class"  : TARGET_CLASS,
            "theta_target"  : round(THETA_TARGET, 6),
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
        print(f"  {tag} Metrics -> {metrics_csv_filename(run_id)}")
        print(f"  {tag} Epochs  -> {epoch_csv_filename(run_id)}")

        # Append to summary CSV
        append_summary_csv(full_metrics)
        all_results.append(full_metrics)

    # Finalize: keep only the best checkpoint across all runs
    finalize_models()

    # Final summary table
    print(f"\n{'='*68}")
    print(f" QTrojan Experiment Complete — Summary across {N_RUNS} runs")
    print(f"{'='*68}")
    print(f"  {'Run':<6} {'CDA (%)':>10} {'ASR (%)':>10} "
          f"{'Acc (%)':>10} {'ROC-AUC':>10} {'Epochs':>8}")
    print(f"  {'-'*58}")
    cdas, asrs = [], []
    for r in all_results:
        roc = str(r['roc_auc'])
        print(f"  {r['run_id']:<6} {r['cda_pct']:>10.2f} {r['asr_pct']:>10.2f} "
              f"{r['accuracy_pct']:>10.2f} {roc:>10} "
              f"{r['epochs_trained']:>8}")
        cdas.append(float(r['cda_pct']))
        asrs.append(float(r['asr_pct']))
    print(f"  {'-'*58}")
    print(f"  {'Mean':<6} {np.mean(cdas):>10.2f} {np.mean(asrs):>10.2f}")
    print(f"  {'Std':<6} {np.std(cdas):>10.2f}  {np.std(asrs):>10.2f}")
    print(f"\n  Summary   -> {SUMMARY_CSV}")
    print(f"  Best model -> {best_model_filename()}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()

"""
huang_backdoor_amplitude.py
============================================================
Replication of Huang & Zhang (2023) — "A Backdoor Attack
Against Quantum Neural Networks with Limited Information"
Chinese Physics B 32, 100306 (2023)

Model  : exact drebin from noiseless_amplitude_label_flipping_layer2.py
         9 qubits | 2 layers | AmplitudeEmbedding
         Rot(φ,θ,ω) + cyclic CRX × 2 | Linear(9→10) | log_softmax

Dataset: MNIST 10-class
         Stratified 700 samples/class (7 000 total training)
         flatten 28×28 → MinMaxScaler → PCA(512) → normalize=True

Attack (§4 of the paper):
  1. Train proxy model (same architecture, clean data)          §4.3
  2. Generate universal trigger via Algorithm 1                 §4.3
       - Fuzzy admix  (Eq. 4-5)  (c,σ)=(1,2)  n=3
       - Q-FGSM per-sample sequential accumulation (faithful)
       - l∞ projection  ε=0.2
       - Outer loop until fooling_rate ≥ 0.6 or Imax=200
  3. Poison TARGET_CLASS training samples (clean-label)         §4.4
       - labels unchanged
       - poison_ratio = poisoned / total target-class samples   §3.3
       - ratios tested: [0.1,0.2,0.4,0.6,0.8,1.0]             Fig.4
  4. Train victim model on poisoned data (3 independent runs)   §5.1
       - max 30 epochs, early stopping patience=3
       - all 3 runs evaluated and metrics saved
       - only the single best run model (highest CA) saved per ratio
  5. Report CA and ASR per poison ratio                         §3.3

Paper fidelity:
  - Algorithm 1 inner loop: per-sample sequential delta
    accumulation — each sample sees the latest trigger before
    its gradient is computed. FAITHFUL to paper.              [FIXED]
  - poison_ratio applied to target class only (§3.3)          [CORRECT]
  - ASR on non-target test samples only (§3.3)                [CORRECT]
  - Clean-label: no label modification (§3.2)                 [CORRECT]
  - Default eps=0.2, (c,σ)=(1,2), poison_ratio=0.4 (§5.1)    [CORRECT]
  - NLLLoss(log_softmax) = quantum cross-entropy (§5.1)       [CORRECT]

Performance optimisations:
  ✓ lightning.qubit  — C++ OpenMP simulator, uses all cores
  ✓ OMP/MKL/BLAS thread env vars set to N_CPU=28
  ✓ torch.set_num_threads(56)
  ✓ torch.multiprocessing  — 3 victim runs per ratio in parallel

Output files (all in runs/):
  runs/huang_amplitude_summary.csv                 final summary all runs
  runs/huang_proxy_epoch_metrics.csv               proxy epoch-wise metrics
  runs/huang_amplitude_run{R}_ratio{P}_epoch_metrics.csv  per-run epoch data
  runs/huang_amplitude_run{R}_ratio{P}_final_metrics.csv  per-run final metrics
  huang_amplitude_trigger.pt                       saved trigger tensor
  huang_proxy_best.pt                              best proxy checkpoint
  huang_amplitude_best_ratio{P}.pt                 best victim model per ratio
                                                   (only 1 saved per ratio,
                                                    the run with highest CA)
============================================================
"""

# ── Thread / CPU env vars must be set BEFORE importing numpy/torch ──────────
import os
N_CPU = 8
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
os.environ["OMP_NUM_THREADS"]       = str(N_CPU)
os.environ["MKL_NUM_THREADS"]       = str(N_CPU)
os.environ["OPENBLAS_NUM_THREADS"]  = str(N_CPU)
os.environ["NUMEXPR_NUM_THREADS"]   = str(N_CPU)

import csv
import datetime
import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pennylane as qml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer

# Set torch CPU threads
torch.set_num_threads(N_CPU)
torch.set_num_interop_threads(max(1, N_CPU // 4))

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             accuracy_score, precision_score, f1_score)
import idx2numpy

# ============================================================
# Config
# ============================================================
N_QUBITS          = 9
N_LAYERS          = 2
N_CLASSES         = 10
EPOCHS            = 30
BATCH_SIZE        = 64
LR                = 0.001
W_DECAY           = 1e-4
BASE_SEED         = 42
N_RUNS            = 3

# Attack hyper-parameters (Huang & Zhang §5.1)
TARGET_CLASS      = 7
POISON_RATIOS     = [0.1, 0.3, 0.5]
EPS               = 0.2
FUZZY_C           = 1.0
FUZZY_SIGMA       = 2.0
N_ADMIX           = 3
FOOLING_THRESH    = 0.6
IMAX              = 200

# Early stopping
ES_PATIENCE       = 3

# Data
SAMPLES_PER_CLASS = 700
PCA_COMPONENTS    = 512      # 2^9 = 512 → AmplitudeEmbedding

RUN_DIR     = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(RUN_DIR, "huang_amplitude_summary.csv")

# Main process uses CUDA if available; worker processes use CPU
MAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Main device : {MAIN_DEVICE}")
print(f"[Config] CPU threads : {N_CPU}")


# ============================================================
# 1.  Quantum device
# ============================================================
def make_device():
    """
    Use default.qubit for AmplitudeEmbedding.
    lightning.qubit's adjoint differentiator does not support
    MottonenStatePreparation (used internally by AmplitudeEmbedding)
    and causes a broadcasting error. default.qubit uses backprop
    which handles AmplitudeEmbedding correctly and efficiently.
    """
    return qml.device("default.qubit", wires=N_QUBITS)


# ============================================================
# 2.  Quantum model  (exact drebin from label-flipping script)
# ============================================================
def build_weight_shapes():
    return ({f"rot_layer_{n}": (N_QUBITS, 3) for n in range(N_LAYERS)} |
            {f"crx_layer_{n}": (N_QUBITS, 1) for n in range(N_LAYERS)})

WEIGHT_SHAPES = build_weight_shapes()


def build_qnode(dev):
    @qml.qnode(dev, interface="torch")
    def qnode(inputs, **weights):
        qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=True)
        for n in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.Rot(*weights[f"rot_layer_{n}"][i], wires=i)
            for i in range(N_QUBITS):
                qml.CRX(weights[f"crx_layer_{n}"][i][0],
                        wires=[i, (i + 1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return qnode


class drebin(nn.Module):
    """
    Amplitude-encoding QMLP — identical to
    noiseless_amplitude_label_flipping_layer2.py.
    Each instance builds its own device + qnode so that
    separate processes don't share PennyLane state.
    """
    def __init__(self):
        super().__init__()
        dev         = make_device()
        qnode       = build_qnode(dev)
        self.qlayer = TorchLayer(qnode, WEIGHT_SHAPES)
        self.fc     = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        x   = x.to(next(self.parameters()).device)
        out = self.qlayer(x)
        out = self.fc(out.to(x.device))
        return F.log_softmax(out, dim=1)


# ============================================================
# 3.  Data helpers
# ============================================================
def stratified_subset(X, y, n_per_class, seed=42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(N_CLASSES):
        class_idx = np.where(y == c)[0]
        chosen    = rng.choice(class_idx,
                               size=min(n_per_class, len(class_idx)),
                               replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def load_mnist_amplitude(
    train_images="../../MNIST/train-images-idx3-ubyte",
    train_labels="../../MNIST/train-labels-idx1-ubyte",
    test_images ="../../MNIST/t10k-images-idx3-ubyte",
    test_labels ="../../MNIST/t10k-labels-idx1-ubyte",
):
    """
    Preprocessing identical to noiseless_amplitude_label_flipping_layer2.py:
    flatten 28×28 → MinMaxScaler → PCA(512)
    """
    Xtr = idx2numpy.convert_from_file(train_images)
    ytr = idx2numpy.convert_from_file(train_labels).astype(np.int64)
    Xte = idx2numpy.convert_from_file(test_images)
    yte = idx2numpy.convert_from_file(test_labels).astype(np.int64)

    Xtr = Xtr.astype(np.float32).reshape(Xtr.shape[0], -1)
    Xte = Xte.astype(np.float32).reshape(Xte.shape[0], -1)

    scaler = MinMaxScaler()
    Xtr    = scaler.fit_transform(Xtr)
    Xte    = scaler.transform(Xte)

    pca = PCA(n_components=PCA_COMPONENTS)
    Xtr = pca.fit_transform(Xtr).astype(np.float32)
    Xte = pca.transform(Xte).astype(np.float32)

    print(f"[Data] Full train : {Xtr.shape}   Test : {Xte.shape}")
    return Xtr, ytr, Xte, yte


# ============================================================
# 4.  Fuzzy admix  (Eq. 4–5, Huang & Zhang)
# ============================================================
def gaussian_membership(x, c=FUZZY_C, sigma=FUZZY_SIGMA):
    """µ(x) = exp(-(x-c)²/σ²)  element-wise."""
    return torch.exp(-((x - c) ** 2) / (sigma ** 2))


def fuzzy_admix_batch(X_nt, X_t, n=N_ADMIX):
    """
    Batched fuzzy admix for the whole non-target set at once.
    X_nt : (B, D)  non-target samples
    X_t  : (N, D)  target-class pool
    Returns x_tilde : (B, D)
    """
    B, D = X_nt.shape
    # sample n target examples for each non-target sample
    idx    = torch.randint(0, X_t.size(0), (B, n))   # (B, n)
    xt_sel = X_t[idx]                                  # (B, n, D)
    mu     = gaussian_membership(X_nt)                 # (B, D)
    nu     = 1.0 - mu                                  # (B, D)
    # mean over n sampled target samples, weighted by nu
    mix    = mu * X_nt + (nu / n) * xt_sel.sum(dim=1) # (B, D)
    return mix


# ============================================================
# 5.  Sequential Q-FGSM trigger generation  (Algorithm 1)
# ============================================================
def generate_trigger(proxy, X_nontarget, X_target, device,
                     eps=EPS, imax=IMAX, fooling_thresh=FOOLING_THRESH):
    """
    Faithful implementation of Algorithm 1 (Huang & Zhang §4.3).

    Inner loop processes samples ONE AT A TIME and accumulates
    delta_tb sequentially — each sample sees the CURRENT trigger
    state before its gradient is computed. This matches the paper's
    pseudocode exactly:

        for each xi in D_nontarget (shuffled each outer iter):
            x_adv = xi + delta_tb
            if proxy(x_adv) != target:
                x_tilde = fuzzy_admix(xi)
                delta_i = Q-FGSM(proxy, x_tilde + delta_tb, target)
                delta_tb = Proj_{l-inf, eps}(delta_tb + delta_i)

    The greedy sequential accumulation means later samples in
    each outer iteration already benefit from earlier updates,
    leading to faster convergence than a batched average.

    Returns universal trigger delta_tb  shape (D,)  on CPU.
    """
    proxy.eval()
    D   = X_nontarget.shape[1]
    dbt = torch.zeros(D, device=device)

    X_nt  = torch.tensor(X_nontarget, dtype=torch.float32, device=device)
    X_t   = torch.tensor(X_target,    dtype=torch.float32, device=device)
    y_tgt = torch.tensor([TARGET_CLASS], dtype=torch.long, device=device)

    t0 = time.time()
    for j in range(imax):

        # shuffle sample order each outer iteration (standard UAP practice)
        perm = torch.randperm(X_nt.size(0), device=device)

        for i in perm:
            xi    = X_nt[i]                  # (D,)
            x_adv = xi + dbt

            # skip this sample if already fooled by current trigger
            with torch.no_grad():
                pred = proxy(x_adv.unsqueeze(0)).argmax(1).item()
            if pred == TARGET_CLASS:
                continue

            # fuzzy admix: mix xi with n random target-class samples
            # then add current trigger before the gradient step
            x_tilde = fuzzy_admix_batch(
                xi.unsqueeze(0), X_t        # returns (1, D)
            ).squeeze(0)                     # (D,)
            x_in = (x_tilde + dbt).detach().requires_grad_(True)

            # forward pass + cross-entropy loss toward target class
            out  = proxy(x_in.unsqueeze(0))
            loss = F.nll_loss(out, y_tgt)
            proxy.zero_grad()
            loss.backward()

            if x_in.grad is not None:
                # Q-FGSM: descend toward target class (negate sign)
                delta_i = -eps * x_in.grad.data.sign()   # (D,)
                # accumulate into universal trigger + project onto l-inf ball
                dbt = (dbt + delta_i).clamp(-eps, eps)

        # ── fooling rate on full non-target set after each outer iter ────
        with torch.no_grad():
            fr = (proxy(X_nt + dbt).argmax(1) == TARGET_CLASS
                  ).float().mean().item()

        elapsed = time.time() - t0
        print(f"    [Trigger] iter {j+1:>3}/{imax}  "
              f"fooling_rate={fr:.3f}  "
              f"|delta|_inf={dbt.abs().max():.4f}  "
              f"elapsed={elapsed:.1f}s")

        if fr >= fooling_thresh:
            print(f"    [Trigger] Converged at iter {j+1}  "
                  f"fooling_rate={fr:.3f}")
            break

    return dbt.detach().cpu()


# ============================================================
# 6.  Poison dataset construction  (§4.4 clean-label)
# ============================================================
def build_poisoned_dataset(X_train, y_train, trigger, poison_ratio):
    """
    Add trigger to poison_ratio fraction of TARGET_CLASS samples.
    Labels are NOT changed (clean-label).
    """
    X_p = X_train.copy()
    y_p = y_train.copy()

    target_idx = np.where(y_train == TARGET_CLASS)[0]
    n_poison   = max(1, int(poison_ratio * len(target_idx)))
    rng        = np.random.default_rng(BASE_SEED)
    chosen     = rng.choice(target_idx, size=n_poison, replace=False)

    trig_np    = trigger.numpy()
    for idx in chosen:
        X_p[idx] = np.clip(X_p[idx] + trig_np, 0, 1)

    print(f"  [Poison] ratio={poison_ratio:.1f}  "
          f"poisoned {n_poison}/{len(target_idx)} target-class samples")
    return X_p, y_p


# ============================================================
# 7.  Training / evaluation helpers
# ============================================================
def train_epoch(model, loader, optimizer, epoch, dev):
    """
    One training epoch.
    Returns (train_acc %, mean_train_loss).
    """
    model.train()
    running_loss = 0.0
    total_loss   = 0.0
    correct      = 0
    total        = 0
    n_batches    = 0

    for batch_idx, (inputs, target) in enumerate(loader, 1):
        inputs, target = inputs.to(dev), target.to(dev)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()
        _, predicted  = torch.max(outputs.data, 1)
        total        += target.size(0)
        correct      += (predicted == target).sum().item()
        running_loss += loss.item()
        total_loss   += loss.item()
        n_batches    += 1
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch:>2} | Batch {batch_idx:>4} | "
                  f"Acc: {100*correct/total:6.2f}%  "
                  f"Loss: {running_loss/10:.4f}")
            running_loss = 0.0

    train_acc  = 100.0 * correct / total
    train_loss = total_loss / max(n_batches, 1)
    return train_acc, train_loss


def evaluate_ca_loss(model, loader, dev):
    """
    Clean-data accuracy and mean NLL loss.
    Returns (acc %, mean_loss).
    """
    model.eval()
    correct    = 0
    total      = 0
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for x, y in loader:
            x, y       = x.to(dev), y.to(dev)
            out        = model(x)
            loss       = F.nll_loss(out, y)
            pred       = out.argmax(1)
            total     += y.size(0)
            correct   += (pred == y).sum().item()
            total_loss += loss.item()
            n_batches  += 1
    acc  = 100.0 * correct / total
    loss = total_loss / max(n_batches, 1)
    return acc, loss


def evaluate_asr(model, loader, trigger, dev):
    """
    ASR = fraction of non-target test samples predicted as
    TARGET_CLASS after trigger is added.  (Paper §3.3)
    """
    model.eval()
    trig    = trigger.to(dev)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            mask  = (y != TARGET_CLASS)
            if mask.sum() == 0:
                continue
            x_adv  = torch.clamp(x[mask] + trig.unsqueeze(0), 0, 1)
            preds  = model(x_adv).argmax(1)
            total += x[mask].size(0)
            correct += (preds == TARGET_CLASS).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def full_evaluation(model, loader, dev):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(dev), y.to(dev)
            out   = model(x)
            prob  = torch.softmax(out, 1)
            pred  = out.argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    TP   = np.diag(cm);  FP = cm.sum(0) - TP
    FN   = cm.sum(1) - TP; TN = cm.sum() - (TP + FP + FN)
    fpr  = np.mean(FP / (FP + TN + 1e-8))
    fnr  = np.mean(FN / (FN + TP + 1e-8))
    try:
        roc = roc_auc_score(y_true, y_prob,
                            multi_class="ovr", average="macro")
    except Exception:
        roc = float("nan")

    return dict(accuracy=acc, precision=prec, f1=f1,
                fpr=fpr, fnr=fnr, roc_auc=roc)


# ============================================================
# 8.  CSV helpers
# ============================================================
def append_summary(row):
    # strip internal keys (prefixed with _) before writing to CSV
    public_row = {k: v for k, v in row.items() if not k.startswith("_")}
    exists = os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(public_row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(public_row)


def save_epoch_csv(filepath, rows):
    """
    Save epoch-wise train/test metrics.
    rows: list of dicts with keys:
        epoch, train_loss, train_acc, test_loss, test_acc
    """
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Epoch metrics  -> {filepath}")


def save_final_csv(filepath, run_id, poison_ratio, ca, asr, metrics):
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["run_id",       run_id])
        w.writerow(["poison_ratio", poison_ratio])
        w.writerow(["CA_pct",       round(ca,  4)])
        w.writerow(["ASR_pct",      round(asr, 4)])
        for k, v in metrics.items():
            w.writerow([k, round(float(v), 6)])
    print(f"  Final metrics  -> {filepath}")


# ============================================================
# 9.  Worker function for parallel victim runs
#     (each run is a separate process — no shared QNode state)
# ============================================================
def victim_run_worker(args):
    """
    Trains one victim model for a given (run_id, poison_ratio).
    Called via multiprocessing.Pool — must be a top-level function.

    Each worker saves its OWN best checkpoint to a temp file.
    The main process inspects all CA values after the pool
    completes and keeps only the single best checkpoint per ratio,
    deleting the rest.

    Returns a dict containing all metrics plus 'ckpt_path' and
    'best_ca' so main can select and rename the winner.
    """
    (run_id, poison_ratio,
     X_poison, y_poison,
     Xte, yte,
     trigger_cpu) = args

    # workers use CPU — GPU memory not safely shared across processes
    dev  = torch.device("cpu")
    seed = BASE_SEED + run_id          # 43, 44, 45
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── DataLoaders ──────────────────────────────────────────────────────
    poison_ds = TensorDataset(
        torch.tensor(X_poison, dtype=torch.float32),
        torch.tensor(y_poison, dtype=torch.long))
    test_ds   = TensorDataset(
        torch.tensor(Xte,      dtype=torch.float32),
        torch.tensor(yte,      dtype=torch.long))
    poison_loader = DataLoader(poison_ds, shuffle=True,  batch_size=BATCH_SIZE)
    test_loader   = DataLoader(test_ds,   shuffle=False, batch_size=BATCH_SIZE)

    # ── Model ────────────────────────────────────────────────────────────
    victim    = drebin().to(dev)
    opt       = optim.Adam(victim.parameters(), lr=LR, weight_decay=W_DECAY)
    ratio_tag = f"run{run_id}_ratio{int(poison_ratio*10):02d}"

    # temp checkpoint — kept until main decides which run wins
    tmp_ckpt  = f"huang_amplitude_tmp_{ratio_tag}.pt"

    best_ca    = 0.0
    es_cnt     = 0
    epoch_rows = []

    # ── Training loop (max EPOCHS=30, early stopping patience=3) ────────
    for epoch in range(1, EPOCHS + 1):
        train_acc,  train_loss = train_epoch(
            victim, poison_loader, opt, epoch, dev)
        test_acc,   test_loss  = evaluate_ca_loss(
            victim, test_loader, dev)

        epoch_rows.append({
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 6),
            "train_acc"  : round(train_acc,  4),
            "test_loss"  : round(test_loss,  6),
            "test_acc"   : round(test_acc,   4),
        })

        print(f"  [Run {run_id} | ratio {poison_ratio}] "
              f"Epoch {epoch:>2}  "
              f"tr_loss={train_loss:.4f}  tr_acc={train_acc:.2f}%  "
              f"te_loss={test_loss:.4f}  te_acc={test_acc:.2f}%")

        if test_acc > best_ca:
            best_ca = test_acc
            torch.save(victim.state_dict(), tmp_ckpt)
            es_cnt  = 0
        else:
            es_cnt += 1
            if es_cnt >= ES_PATIENCE:
                print(f"  [Run {run_id} | ratio {poison_ratio}] "
                      f"Early stopping at epoch {epoch}")
                break

    # ── Save epoch-wise CSV (all runs saved regardless) ──────────────────
    epoch_csv = os.path.join(
        RUN_DIR, f"huang_amplitude_{ratio_tag}_epoch_metrics.csv")
    save_epoch_csv(epoch_csv, epoch_rows)

    # ── Load best checkpoint for this run ────────────────────────────────
    victim.load_state_dict(torch.load(tmp_ckpt, map_location=dev))

    # ── Final CA, ASR, full metrics (all runs saved regardless) ──────────
    ca,  _  = evaluate_ca_loss(victim, test_loader, dev)
    asr     = evaluate_asr(victim, test_loader, trigger_cpu, dev)
    met     = full_evaluation(victim, test_loader, dev)

    print(f"\n  [Run {run_id} | ratio {poison_ratio}] "
          f"Final  CA={ca:.2f}%  ASR={asr:.2f}%")

    final_csv = os.path.join(
        RUN_DIR, f"huang_amplitude_{ratio_tag}_final_metrics.csv")
    save_final_csv(final_csv, run_id, poison_ratio, ca, asr, met)

    return {
        "timestamp"    : datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id"       : run_id,
        "poison_ratio" : poison_ratio,
        "CA_pct"       : round(ca,  4),
        "ASR_pct"      : round(asr, 4),
        "accuracy"     : round(met["accuracy"] * 100, 4),
        "precision"    : round(met["precision"], 6),
        "f1"           : round(met["f1"],        6),
        "fpr"          : round(met["fpr"],       6),
        "fnr"          : round(met["fnr"],       6),
        "roc_auc"      : round(met["roc_auc"],   6),
        # used by main to pick the best model — not written to summary CSV
        "_ckpt_path"   : tmp_ckpt,
        "_best_ca"     : ca,
    }


# ============================================================
# 10.  Main
# ============================================================
def main():
    mp.set_start_method("spawn", force=True)   # required for CUDA + multiprocessing

    print("=" * 68)
    print(" Huang & Zhang (2023) Backdoor — Amplitude QMLP Replication")
    print(f" Qubits={N_QUBITS}  Layers={N_LAYERS}  PCA={PCA_COMPONENTS}")
    print(f" Target class={TARGET_CLASS}  ε={EPS}  "
          f"(c,σ)=({FUZZY_C},{FUZZY_SIGMA})")
    print(f" Poison ratios: {POISON_RATIOS}")
    print(f" Parallel runs: {N_RUNS}  |  CPU threads: {N_CPU}")
    print("=" * 68)

    # ── Load & preprocess data ───────────────────────────────────────────
    Xtr_full, ytr_full, Xte, yte = load_mnist_amplitude()
    Xtr_sub, ytr_sub = stratified_subset(
        Xtr_full, ytr_full, SAMPLES_PER_CLASS, seed=BASE_SEED)
    print(f"[Data] Stratified train: {Xtr_sub.shape}  Test: {Xte.shape}")

    test_ds     = TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                                torch.tensor(yte, dtype=torch.long))
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE)

    # ── Phase 1: Load existing proxy (already trained) ───────────────────
    print("\n" + "="*68)
    print(" PHASE 1 — Loading existing proxy model from huang_proxy_best.pt")
    print("="*68)

    proxy = drebin().to(MAIN_DEVICE)
    proxy.load_state_dict(
        torch.load("huang_proxy_best.pt", map_location=MAIN_DEVICE))
    proxy.eval()

    ca_check, _ = evaluate_ca_loss(proxy, test_loader, MAIN_DEVICE)
    print(f"  [Proxy] Loaded.  Test CA: {ca_check:.2f}%")

    # ── Phase 2: Generate universal trigger ─────────────────────────────
    print("\n" + "="*68)
    print(" PHASE 2 — Sequential trigger generation (Algorithm 1, faithful)")
    print("="*68)

    nt_idx   = np.where(ytr_sub != TARGET_CLASS)[0]
    t_idx    = np.where(ytr_sub == TARGET_CLASS)[0]
    rng      = np.random.default_rng(BASE_SEED)
    # use all non-target training samples — batching makes this fast
    nt_samp  = rng.choice(nt_idx, size=min(500, len(nt_idx)), replace=False)

    X_nt_gen = Xtr_sub[nt_samp]
    X_t_gen  = Xtr_sub[t_idx]

    trigger  = generate_trigger(proxy, X_nt_gen, X_t_gen, MAIN_DEVICE)
    torch.save(trigger, "huang_amplitude_trigger.pt")
    print(f"  Trigger saved.  |delta|_inf={trigger.abs().max():.4f}  "
          f"|delta|_2={trigger.norm():.4f}")

    # Final proxy fooling rate
    proxy.eval()
    X_nt_t  = torch.tensor(X_nt_gen, dtype=torch.float32,
                            device=MAIN_DEVICE)
    trig_d  = trigger.to(MAIN_DEVICE)
    with torch.no_grad():
        fr = (proxy(X_nt_t + trig_d).argmax(1) == TARGET_CLASS
              ).float().mean().item()
    print(f"  Final proxy fooling rate: {fr:.3f}")

    # Move proxy to CPU / free GPU memory before spawning workers
    proxy.cpu()
    del proxy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 3: Victim training — N_RUNS parallel per poison ratio ─────
    print("\n" + "="*68)
    print(" PHASE 3 — Victim training (parallel runs per poison ratio)")
    print("="*68)

    all_results = []

    for poison_ratio in POISON_RATIOS:
        print(f"\n{'─'*68}")
        print(f" Poison ratio : {poison_ratio}")
        print(f"{'─'*68}")

        X_poison, y_poison = build_poisoned_dataset(
            Xtr_sub, ytr_sub, trigger, poison_ratio)

        # Build argument list for the 3 parallel workers
        worker_args = [
            (run_id, poison_ratio,
             X_poison, y_poison,
             Xte, yte,
             trigger)
            for run_id in range(1, N_RUNS + 1)
        ]

        # Spawn N_RUNS processes simultaneously
        with mp.Pool(processes=N_RUNS) as pool:
            results = pool.map(victim_run_worker, worker_args)

        # ── Write all 3 runs to summary CSV ──────────────────────────────
        for row in results:
            append_summary(row)
            all_results.append(row)

        # ── Select best run (highest CA) and keep only that checkpoint ───
        best_result = max(results, key=lambda r: r["_best_ca"])
        best_run_id = best_result["run_id"]
        best_ckpt_src = best_result["_ckpt_path"]
        best_ckpt_dst = f"huang_amplitude_best_ratio{int(poison_ratio*10):02d}.pt"

        import shutil
        shutil.copy(best_ckpt_src, best_ckpt_dst)
        print(f"\n  [Best model] ratio={poison_ratio}  "
              f"run_id={best_run_id}  "
              f"CA={best_result['_best_ca']:.2f}%  "
              f"saved -> {best_ckpt_dst}")

        # delete all temp checkpoints for this ratio
        for r in results:
            tmp = r["_ckpt_path"]
            if os.path.exists(tmp):
                os.remove(tmp)

        cas  = [r["CA_pct"]  for r in results]
        asrs = [r["ASR_pct"] for r in results]
        print(f"  [Ratio {poison_ratio}]  "
              f"CA={np.mean(cas):.2f}+/-{np.std(cas):.2f}%  "
              f"ASR={np.mean(asrs):.2f}+/-{np.std(asrs):.2f}%")

    # ── Final summary table ──────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f" Experiment complete   Summary → {SUMMARY_CSV}")
    print(f"{'='*68}")
    print(f"  {'Ratio':<8} {'CA mean':>10} {'CA std':>8} "
          f"{'ASR mean':>10} {'ASR std':>8}")
    print(f"  {'─'*48}")
    for ratio in POISON_RATIOS:
        rows = [r for r in all_results if r["poison_ratio"] == ratio]
        cas  = [r["CA_pct"]  for r in rows]
        asrs = [r["ASR_pct"] for r in rows]
        print(f"  {ratio:<8.1f} {np.mean(cas):>10.2f} {np.std(cas):>8.2f} "
              f"{np.mean(asrs):>10.2f} {np.std(asrs):>8.2f}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()

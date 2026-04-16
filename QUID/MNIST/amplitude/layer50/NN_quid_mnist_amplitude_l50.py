"""
NN_quid_mnist_amplitude_l50.py
============================================================
Combines:
  - QUID attack  (Kundu & Ghosh)
  - Q-Detection defense  (He et al.)

Optimisations over naive implementation:
  1. PROTOTYPE CACHE  (replaces full density-matrix cache)
     Instead of caching all N training density matrices, only the
     10 per-class mean density matrices (prototypes) are cached.
     Prototypes are computed ONCE at startup using all available
     CPU cores (joblib), saved to rho_proto_cache_amplitude.npy,
     and reused across every QUID call.  Re-computation is skipped
     on subsequent runs if the cache file already exists.

  2. MULTI-GPU PARALLEL EXPERIMENTS
     Each (eps, attack, defense) combo is fully independent.
     torch.multiprocessing dispatches one worker per combo,
     each worker pinned to one GPU in round-robin order.
     Falls back to sequential CPU execution if no GPU available.

     Available GPUs are detected automatically. Set
     CUDA_VISIBLE_DEVICES before launching to restrict which
     GPUs are used, e.g.:
       CUDA_VISIBLE_DEVICES=0,1,2,3 python NN_quid_mnist_amplitude_l50.py

Model architecture:
  * Dataset  : MNIST 10-class handwritten digits
  * Features : flatten -> MinMaxScaler -> PCA(n_components=512)
  * Qubits   : 9  |  Layers: 50
  * Encoding : AmplitudeEmbedding (applied ONCE before variational layers)
  * PQC      : Rot(a,b,g) + cyclic CRX  x 50 layers
  * Post     : Linear(9->10) + log_softmax
  * Opt      : Adam lr=0.001, weight_decay=1e-4
  * Epochs   : 10,  batch_size=64,  runs=3  (seeds 43,44,45)

Output folder structure:
  eps0.1/  eps0.3/  eps0.5/
    ├── *.pt   (best-accuracy checkpoint)
    └── *.csv  (per-combo metrics, ['Metric','Value'] format)
  runs/
    └── quid_qdetection_mnist_amplitude_l50_summary.csv
  rho_proto_cache_amplitude.npy  (cached class prototypes, shape [10, 512, 512])
============================================================
"""

import os, math, random, copy, csv, datetime, time, warnings
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES here — workers set their
# own device via the gpu_id argument passed at spawn time.

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

try:
    from joblib import Parallel, delayed
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False
    warnings.warn("joblib not found — density matrix cache will be built single-threaded.")

# ============================================================
# Config
# ============================================================
N_QUBITS   = 9
N_LAYERS   = 50
N_CLASSES  = 10
EPOCHS     = 10
BATCH_SIZE = 64
LR         = 0.001
W_DECAY    = 1e-4
BASE_SEED  = 42
N_RUNS     = 3
EPS_LIST   = [0.1, 0.3, 0.5]

BEST_MODEL_PREFIX    = "qmlp-mnist-quid-run"
FILENAME_PREFIX      = "qmlp-mnist-quid-run"
NOISE_SUFFIX         = "-noiseless"
ENCODING             = "Amplitude"
RHO_PROTO_CACHE_FILE = "rho_proto_cache_amplitude.npy"

RUN_DIR     = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(RUN_DIR, "quid_qdetection_mnist_amplitude_l50_summary.csv")

# Q-Detection subsampling — keeps all 10 classes but limits samples per class.
QDETECT_SUBSET_PER_CLASS = 700   # ~7k total

# CSV write lock — created inside main() after mp context is set,
# then passed explicitly to each worker. Do NOT create at module level
# as it breaks with the 'spawn' start method required for CUDA.


# ============================================================
# 1. Data  — MNIST 10-class
#    flatten -> MinMaxScaler -> PCA(n_components=512) for AmplitudeEmbedding
#    (requires 2^N_QUBITS = 512 features)
#    MNIST files expected at: ./MNIST/
# ============================================================
def load_mnist_pca512(
    train_images="/work/clalarcon/Nowmi/Sok/QUID/MNIST/MNIST/train-images-idx3-ubyte",
    train_labels="/work/clalarcon/Nowmi/Sok/QUID/MNIST/MNIST/train-labels-idx1-ubyte",
    test_images ="/work/clalarcon/Nowmi/Sok/QUID/MNIST/MNIST/t10k-images-idx3-ubyte",
    test_labels ="/work/clalarcon/Nowmi/Sok/QUID/MNIST/MNIST/t10k-labels-idx1-ubyte",
):
    import idx2numpy
    X_train = idx2numpy.convert_from_file(train_images)
    y_train = idx2numpy.convert_from_file(train_labels)
    X_test  = idx2numpy.convert_from_file(test_images)
    y_test  = idx2numpy.convert_from_file(test_labels)

    # flatten 28x28 -> 784 (no downscaling — need >= 512 features for PCA)
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test  = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    pca     = PCA(n_components=512)          # 2^9 = 512 for AmplitudeEmbedding
    X_train = pca.fit_transform(X_train).astype(np.float32)
    X_test  = pca.transform(X_test).astype(np.float32)

    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"[Data] Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ============================================================
# 2. Quantum circuit
#    AmplitudeEmbedding applied ONCE before variational layers.
# ============================================================
def _make_ql_device():
    """Each process creates its own PennyLane device to avoid state sharing."""
    return qml.device("default.qubit", wires=N_QUBITS)

def build_weight_shapes():
    shapes = {}
    for n in range(N_LAYERS):
        shapes[f"rot_layer_{n}"] = (N_QUBITS, 3)
        shapes[f"crx_layer_{n}"] = (N_QUBITS, 1)
    return shapes

WEIGHT_SHAPES = build_weight_shapes()

def make_qnode_torch(dev):
    @qml.qnode(dev, interface="torch")
    def _qnode(inputs, **wkw):
        # AmplitudeEmbedding encodes the full state vector — applied ONCE only.
        # (Unlike AngleEmbedding, re-applying it mid-circuit overwrites the state.)
        qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=True)
        for n in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.Rot(*wkw[f"rot_layer_{n}"][i], wires=i)
            for i in range(N_QUBITS):
                qml.CRX(wkw[f"crx_layer_{n}"][i][0], wires=[i, (i+1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return _qnode


class QMLPAZ(nn.Module):
    """Built fresh in each worker with its own PennyLane device."""
    def __init__(self):
        super().__init__()
        dev          = _make_ql_device()
        qnode        = make_qnode_torch(dev)
        self.qlayer  = TorchLayer(qnode, WEIGHT_SHAPES)
        self.fc      = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        # default.qubit is a CPU simulator — it cannot handle CUDA tensors.
        # Run the quantum layer on CPU, then move the result to the fc device.
        fc_device = next(self.fc.parameters()).device
        out = self.qlayer(x.cpu())          # quantum layer always on CPU
        out = self.fc(out.to(fc_device))    # classical layer on GPU (or CPU)
        return F.log_softmax(out, dim=1)


# ============================================================
# 3. Prototype cache  (replaces full per-sample rho cache)
#    Computes one density matrix per class (N_CLASSES total).
#    Shape: (N_CLASSES, 512, 512) complex128  —  ~96 MB on disk.
#    Built once in the main process, shared read-only by workers.
#    Uses AmplitudeEmbedding — consistent with the circuit above.
# ============================================================
def _compute_one_rho(x512):
    """Single density-matrix computation (called inside joblib worker)."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface=None, diff_method=None)
    def rho_qnode(x):
        qml.AmplitudeEmbedding(x, wires=range(N_QUBITS), normalize=True)
        return qml.density_matrix(wires=list(range(N_QUBITS)))

    return np.array(rho_qnode(pnp.array(x512, dtype=float)))


def _compute_class_prototype(Xtr, indices):
    """Compute the mean density matrix for one class (parallelised over samples)."""
    if _JOBLIB_OK:
        rhos = Parallel(n_jobs=-1)(
            delayed(_compute_one_rho)(Xtr[i]) for i in indices
        )
    else:
        rhos = [_compute_one_rho(Xtr[i]) for i in indices]
    return np.stack(rhos, axis=0).mean(axis=0)


def build_proto_cache(Xtr: np.ndarray,
                      ytr: np.ndarray,
                      cache_file: str = RHO_PROTO_CACHE_FILE,
                      n_jobs: int = -1) -> np.ndarray:
    """
    Compute and cache one mean density matrix (prototype) per class.

    Instead of storing all N density matrices, only N_CLASSES=10
    prototypes are stored.

    Prototypes are built from the FULL training set.

    Returns: array of shape (N_CLASSES, 512, 512) complex128.
    """
    if os.path.exists(cache_file):
        print(f"[ProtoCache] Loading prototypes from {cache_file} ...")
        t0     = time.time()
        protos = np.load(cache_file)
        print(f"[ProtoCache] Loaded {protos.shape[0]} prototypes "
              f"in {time.time()-t0:.1f}s  "
              f"(size: {protos.nbytes / 1e6:.1f} MB)")
        return protos

    dim = 2 ** N_QUBITS
    print(f"[ProtoCache] Computing {N_CLASSES} class prototypes ...")
    print(f"[ProtoCache] Each prototype averages all per-class density matrices.")
    print(f"[ProtoCache] This runs ONCE and is cached to {cache_file}.")
    t0     = time.time()
    protos = np.zeros((N_CLASSES, dim, dim), dtype=np.complex128)

    for c in range(N_CLASSES):
        idx_c = np.where(ytr == c)[0]
        if len(idx_c) == 0:
            protos[c] = np.eye(dim, dtype=np.complex128) / dim
            print(f"  Class {c:>2}: no samples — using maximally mixed state")
            continue
        print(f"  Class {c:>2}: averaging {len(idx_c):>6} density matrices ...")
        tc = time.time()
        protos[c] = _compute_class_prototype(Xtr, idx_c)
        print(f"  Class {c:>2}: done in {time.time()-tc:.1f}s")

    np.save(cache_file, protos)
    print(f"[ProtoCache] All done in {time.time()-t0:.1f}s  "
          f"→  saved to {cache_file}  "
          f"({protos.nbytes / 1e6:.1f} MB)")
    return protos


# ============================================================
# 4. QUID attack — uses pre-built prototype cache
# ============================================================
def frobenius_dist(A, B):
    D = A - B
    return float(np.sqrt(np.sum(np.real(D * np.conj(D)))))


def quid_label_flip_with_data(Xtr, proto_cache, y, epsilon, seed=BASE_SEED):
    """
    QUID Algorithm 1 — prototype-cache edition.

    Poisoned samples have their density matrix computed on-the-fly
    (one _compute_one_rho call per poisoned sample, parallelised).
    Class prototypes come from proto_cache (pre-built, ~96 MB).

    This avoids storing N~184k density matrices on disk.
    """
    if epsilon <= 0.0:
        return y.copy(), np.array([], dtype=int)

    rng          = np.random.default_rng(seed)
    n            = len(y)
    k            = int(round(epsilon * n))
    poisoned_idx = rng.choice(n, size=k, replace=False)

    protos = [proto_cache[c] for c in range(N_CLASSES)]

    print(f"  [QUID] Computing density matrices for {k} poisoned samples ...")
    t0 = time.time()
    if _JOBLIB_OK:
        rhos_poisoned = Parallel(n_jobs=-1, verbose=0, prefer="threads")(
            delayed(_compute_one_rho)(Xtr[i]) for i in poisoned_idx
        )
    else:
        rhos_poisoned = [_compute_one_rho(Xtr[i]) for i in poisoned_idx]

    print(f"  [QUID] Density matrices done in {time.time()-t0:.1f}s")

    print(f"  [QUID] Assigning adversarial labels ...")
    t0       = time.time()
    y_poison = y.copy()
    for j, i in enumerate(poisoned_idx):
        dists       = [frobenius_dist(rhos_poisoned[j], protos[c])
                       for c in range(N_CLASSES)]
        y_poison[i] = int(np.argmax(dists))
    print(f"  [QUID] Label assignment done in {time.time()-t0:.1f}s")

    changed = int((y_poison[poisoned_idx] != y[poisoned_idx]).sum())
    print(f"  [QUID] Labels changed: {changed}/{k}")
    return y_poison, poisoned_idx


def random_label_flip(y, epsilon, seed=BASE_SEED, n_classes=N_CLASSES):
    if epsilon <= 0.0:
        return y.copy(), np.array([], dtype=int)
    rng          = np.random.default_rng(seed + 999)
    k            = int(round(epsilon * len(y)))
    poisoned_idx = rng.choice(len(y), size=k, replace=False)
    y_poison     = y.copy()
    for i in poisoned_idx:
        choices     = [c for c in range(n_classes) if c != int(y[i])]
        y_poison[i] = int(rng.choice(choices))
    return y_poison, poisoned_idx


# ============================================================
# 5. Q-Detection — Q-WAN Ising Bank
# ============================================================
class QWANIsingBank:
    def __init__(self, n_total, eta=0.05, alpha=1.0,
                 beta_start=0.1, beta_end=2.0, sa_sweeps=50, seed=BASE_SEED):
        self.N=n_total; self.eta=eta; self.alpha=alpha
        self.beta_start=beta_start; self.beta_end=beta_end
        self.sa_sweeps=sa_sweeps
        self.rng = np.random.default_rng(seed)
        self.h   = np.zeros(n_total, dtype=np.float64)
        self.J   = np.zeros((n_total, n_total), dtype=np.float64)

    def _betas(self, steps=30):
        return np.linspace(self.beta_start, self.beta_end, steps)

    def _sa(self, h_sub, J_sub, betas, init=None):
        B = len(h_sub)
        s = init.copy() if init is not None else self.rng.choice([-1,1], size=B)
        samples = []
        for k, beta in enumerate(betas):
            for _ in range(self.sa_sweeps):
                i = self.rng.integers(0, B)
                ns = -s[i]; delta = ns - s[i]
                dE = -(h_sub[i]*delta + np.dot(J_sub[i], s)*delta)
                if dE <= 0 or self.rng.random() < math.exp(-beta*dE):
                    s[i] = ns
            if k >= int(0.7*len(betas)):
                samples.append(s.copy())
        return s, samples

    def _exp(self, samples):
        if not samples: return None, None
        X = np.stack(samples, axis=0)
        return X.mean(axis=0), (X.T @ X) / X.shape[0]

    def train_step(self, idx_batch, losses_batch):
        idx = np.asarray(idx_batch, dtype=int)
        Ln  = (losses_batch - losses_batch.mean()) / (losses_batch.std() + 1e-8)
        h_s = self.h[idx].copy()
        J_s = self.J[np.ix_(idx,idx)].copy(); np.fill_diagonal(J_s, 0.0)
        betas = self._betas()
        _, fs = self._sa(h_s, J_s, betas)
        mf, Cf = self._exp(fs)
        if mf is None: mf=np.zeros_like(h_s); Cf=np.zeros_like(J_s)
        _, gs = self._sa(h_s - 0.5*self.alpha*Ln, J_s, betas)
        mg, Cg = self._exp(gs)
        if mg is None: mg=np.zeros_like(h_s); Cg=np.zeros_like(J_s)
        dC = 0.5*((Cg-Cf)+(Cg-Cf).T); np.fill_diagonal(dC, 0.0)
        self.J[np.ix_(idx,idx)] -= self.eta * dC
        self.h[idx]             -= self.eta * (mg - mf)
        self.J = 0.5*(self.J+self.J.T); np.fill_diagonal(self.J, 0.0)

    def weights(self, idx_batch, losses_batch):
        idx = np.asarray(idx_batch, dtype=int)
        Ln  = (losses_batch - losses_batch.mean()) / (losses_batch.std() + 1e-8)
        h_s = self.h[idx].copy()
        J_s = self.J[np.ix_(idx,idx)].copy(); np.fill_diagonal(J_s, 0.0)
        _, gs = self._sa(h_s - 0.5*self.alpha*Ln, J_s, self._betas())
        if not gs:
            s = np.sign(h_s - 0.5*self.alpha*Ln + 1e-6)
            return np.clip((s+1.0)/2.0, 0.0, 1.0)
        mg, _ = self._exp(gs)
        return np.clip((mg+1.0)/2.0, 0.0, 1.0)


# ============================================================
# 6. Dataset / loader helpers
# ============================================================
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, X_t, y_t):
        self.X=X_t; self.y=y_t
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], idx

def make_indexed_loader(X, y):
    ds = IndexedDataset(torch.tensor(X, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

def make_plain_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ============================================================
# 6b. Stratified subset for Q-Detection
#     Keeps all 10 classes, samples up to QDETECT_SUBSET_PER_CLASS
#     samples per class so the J matrix stays manageable.
# ============================================================
def stratified_subset(X, y, n_per_class, seed=BASE_SEED):
    """Return a stratified subset of (X, y) with at most n_per_class per class."""
    rng = np.random.default_rng(seed)
    idx_keep = []
    for c in range(N_CLASSES):
        idx_c = np.where(y == c)[0]
        if len(idx_c) <= n_per_class:
            idx_keep.append(idx_c)
        else:
            idx_keep.append(rng.choice(idx_c, size=n_per_class, replace=False))
    idx_keep = np.concatenate(idx_keep)
    idx_keep = np.sort(idx_keep)
    return X[idx_keep], y[idx_keep], idx_keep


# ============================================================
# 7. train() and test()
# ============================================================
def train(model, device, train_loader, optimizer, epoch,
          qwan=None, use_qdetect=False):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total   = 0.0

    for batch_idx, batch in enumerate(train_loader, 0):
        if use_qdetect:
            inputs, target, idx_b = batch
            idx_np = idx_b.numpy() if isinstance(idx_b, torch.Tensor) else np.array(idx_b)
        else:
            inputs, target = batch
            idx_np = None

        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        if use_qdetect and qwan is not None:
            model.eval()
            with torch.no_grad():
                per_losses = F.nll_loss(model(inputs), target,
                                        reduction="none").cpu().numpy()
            model.train()
            for _ in range(2):
                qwan.train_step(idx_np, per_losses)
            w_np = qwan.weights(idx_np, per_losses)
            per  = F.nll_loss(model(inputs), target, reduction="none")
            w    = torch.tensor(w_np, dtype=torch.float32,
                                device=per.device).clamp(min=1e-12)
            loss = (w * per).sum() / w.sum()
        else:
            outputs = model(inputs)
            loss    = F.nll_loss(outputs, target)

        loss.backward()
        optimizer.step()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        total        += target.size(0)
        correct      += (predicted == target).sum().item()
        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx+1}, "
                  f"Acc: {100 * correct / total:.2f}%, "
                  f"Loss: {running_loss / 10:.4f}")
            running_loss = 0.0


def test(model, device, test_loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted   = torch.max(model(inputs).data, dim=1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy on test set: {acc:.2f}%")
    return acc


# ============================================================
# 8. evaluation()
# ============================================================
def evaluation(model, test_loader, device, num_classes=N_CLASSES):
    model.eval()
    outputs_list, y_true_list = [], []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            out = model(x_batch)
            outputs_list.append(out)
            y_true_list.append(y_batch)

    output = torch.cat(outputs_list, dim=0)
    true   = torch.cat(y_true_list,  dim=0)
    pred   = output.argmax(dim=1)
    probs  = torch.softmax(output, dim=1).detach().cpu().numpy()
    acc    = (pred == true).float().mean().item()
    test_loss = F.nll_loss(output, true).item()

    y_true_np = true.cpu().numpy()
    y_pred_np = pred.cpu().numpy()

    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes)))
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    precision_macro = np.mean(TP / (TP + FP + 1e-8))
    recall_macro    = np.mean(TP / (TP + FN + 1e-8))
    f1_macro        = 2*(precision_macro*recall_macro) / (precision_macro+recall_macro+1e-8)
    fpr_macro       = np.mean(FP / (FP + TN + 1e-8))
    fnr_macro       = np.mean(FN / (FN + TP + 1e-8))

    try:
        roc_auc = roc_auc_score(y_true_np, probs, multi_class='ovr', average='macro')
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
# 9. File naming helpers
# ============================================================
def eps_folder(eps):
    folder = f"eps{eps}"
    os.makedirs(folder, exist_ok=True)
    return folder

def model_filename(run_id, eps, attack, defense):
    def_str = defense.replace("-", "")
    fname   = (f"{BEST_MODEL_PREFIX}{run_id}"
               f"-layer{N_LAYERS}{NOISE_SUFFIX}-{ENCODING}"
               f"_eps{eps}_{attack}_{def_str}.pt")
    return os.path.join(eps_folder(eps), fname)

def metrics_csv_filename(run_id, eps, attack, defense):
    def_str = defense.replace("-", "")
    fname   = (f"{FILENAME_PREFIX}{run_id}"
               f"-layer{N_LAYERS}_quid"
               f"_eps{eps}_{attack}_{def_str}.csv")
    return os.path.join(eps_folder(eps), fname)


# ============================================================
# 10. CSV helpers
# ============================================================
def save_per_run_csv(metrics, filepath):
    """['Metric','Value'] format."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(f"Metrics saved to {filepath}")


_SUMMARY_FIELDS = [
    "timestamp", "run_id", "attack", "defense", "epsilon",
    "n_poisoned", "best_acc_pct", "asr",
    "accuracy", "loss", "precision", "recall",
    "f1", "fpr", "fnr", "roc_auc", "pr_auc",
    "train_time_s", "gpu_id"
]

def append_summary_csv(row, lock):
    """Thread/process-safe write using the shared lock."""
    with lock:
        exists = os.path.exists(SUMMARY_CSV) and os.path.getsize(SUMMARY_CSV) > 0
        with open(SUMMARY_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
            if not exists:
                w.writeheader()
            w.writerow(row)
            f.flush()


# ============================================================
# 11. Worker function — runs one (run_id, eps, attack, defense)
#     combo on a specific GPU.
#     Called by mp.Process — must be picklable (no lambdas).
# ============================================================
def experiment_worker(task, Xtr, ytr, Xte, yte,
                      proto_cache_file, csv_lock):
    """
    task = (run_id, eps, attack, use_qdetect, run_seed, gpu_id)

    Each worker:
      1. Sets its own CUDA device
      2. Loads prototype cache (~96 MB, no RAM duplication issue)
      3. Poisons labels (QUID computes rhos only for poisoned samples)
      4. Trains model
      5. Evaluates and saves CSV + checkpoint
      6. Appends row to shared summary CSV (lock-protected)
    """
    run_id, eps, attack, use_qdetect, run_seed, gpu_id = task

    # ---- device setup ----
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")

    defense_name = "q-detection" if use_qdetect else "none"
    tag = f"[Run {run_id} | eps={eps} | {attack} | {defense_name} | GPU {gpu_id}]"
    print(f"\n{'='*68}\n {tag}\n{'='*68}")

    # ---- seeds ----
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    # ---- load prototype cache (~96 MB — trivial RAM cost) ----
    proto_cache = np.load(proto_cache_file)

    # ---- stratified subset FIRST (subset, then poison) ----
    Xtr_sub, ytr_sub, sub_idx = stratified_subset(
        Xtr, ytr, QDETECT_SUBSET_PER_CLASS, seed=run_seed)
    n_sub = len(ytr_sub)
    print(f"  {tag} Stratified subset: {n_sub} samples "
          f"({QDETECT_SUBSET_PER_CLASS} per class max, from {len(ytr)} total)")

    # ---- poison labels on the subset ----
    if attack == "quid":
        y_poisoned, p_idx = quid_label_flip_with_data(Xtr_sub, proto_cache, ytr_sub,
                                                       epsilon=eps, seed=run_seed)
    else:
        y_poisoned, p_idx = random_label_flip(ytr_sub, epsilon=eps, seed=run_seed)

    n_poisoned = len(p_idx)
    print(f"  {tag} Poisoned: {n_poisoned}/{n_sub}")

    # ---- loaders ----
    test_loader = make_plain_loader(Xte, yte, shuffle=False)
    if use_qdetect:
        print(f"  {tag} J matrix: {n_sub}x{n_sub} "
              f"({n_sub**2*8/1e6:.0f} MB)")
        train_loader = make_indexed_loader(Xtr_sub, y_poisoned)
        qwan = QWANIsingBank(n_total=n_sub, eta=0.05,
                             alpha=1.0, seed=run_seed)
    else:
        train_loader = make_plain_loader(Xtr_sub, y_poisoned, shuffle=True)
        qwan = None

    # ---- model & optimizer ----
    model = QMLPAZ().to(device)
    # default.qubit is a CPU simulator — keep qlayer weights on CPU so
    # PennyLane can build gate matrices without hitting CUDA driver errors.
    model.qlayer.to('cpu')
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=W_DECAY)
    best_model_path = model_filename(run_id, eps, attack, defense_name)
    best_acc   = 0.0
    start_time = time.time()

    # ---- training loop ----
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch,
              qwan=qwan, use_qdetect=use_qdetect)
        acc = test(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)

    elapsed = time.time() - start_time
    print(f"  {tag} Training Time: {elapsed:.2f}s  Best Acc: {best_acc:.2f}%")

    # ---- evaluate best checkpoint ----
    print(f"\n  {tag} Evaluating final model on clean test data...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    metrics = evaluation(model, test_loader, device)

    # ---- ASR — subset first then poisoned, so p_idx indexes into Xtr_sub ----
    asr = float("nan")
    if n_poisoned > 0:
        model.eval()
        Xp = torch.tensor(Xtr_sub[p_idx], dtype=torch.float32).to(device)
        yp = torch.tensor(y_poisoned[p_idx], dtype=torch.long).to(device)
        with torch.no_grad():
            asr = (model(Xp).argmax(dim=1) == yp).float().mean().item() * 100.0

    # ---- print all metrics including ASR ----
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12}: {v:.4f}")
    print(f"  {'Asr':<12}: {asr:.4f}" if not math.isnan(asr) else f"  {'Asr':<12}: nan")

    # ---- per-combo CSV ----
    csv_path = metrics_csv_filename(run_id, eps, attack, defense_name)
    save_per_run_csv(metrics, csv_path)

    # ---- append to shared summary CSV (lock-protected) ----
    row = {
        "timestamp"    : datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id"       : run_id,
        "attack"       : attack,
        "defense"      : defense_name,
        "epsilon"      : eps,
        "n_poisoned"   : int(n_poisoned),
        "best_acc_pct" : round(best_acc, 4),
        "asr"          : round(asr, 4) if not math.isnan(asr) else "nan",
        "accuracy"     : round(metrics['accuracy'] * 100, 4),
        "loss"         : round(metrics['loss'], 6),
        "precision"    : round(metrics['precision'], 6),
        "recall"       : round(metrics['recall'], 6),
        "f1"           : round(metrics['f1'], 6),
        "fpr"          : round(metrics['fpr'], 6),
        "fnr"          : round(metrics['fnr'], 6),
        "roc_auc"      : round(metrics['roc_auc'], 6),
        "pr_auc"       : metrics['pr_auc'],
        "train_time_s" : round(elapsed, 1),
        "gpu_id"       : gpu_id,
    }
    append_summary_csv(row, csv_lock)
    print(f"  {tag} Done. Saved → {best_model_path}")


# ============================================================
# 12. Main — builds cache then dispatches parallel workers
# ============================================================
def main():
    print("=" * 68)
    print(" QUID + Q-Detection | MNIST 10-class | 9-qubit Amplitude model (PCA->512)")
    print(" Optimisations: prototype cache + multi-GPU parallelism")
    print("=" * 68)

    # ---- detect available GPUs ----
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("[GPU] No CUDA GPUs found — running sequentially on CPU.")
    else:
        print(f"[GPU] {n_gpus} GPU(s) detected: "
              f"{[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")

    # ---- load & preprocess data (main process only) ----
    Xtr, ytr, Xte, yte = load_mnist_pca512()

    # ---- build/load prototype cache ----
    proto_cache = build_proto_cache(Xtr, ytr,
                                    cache_file=RHO_PROTO_CACHE_FILE, n_jobs=-1)
    print(f"[ProtoCache] shape: {proto_cache.shape}  "
          f"dtype: {proto_cache.dtype}  "
          f"size: {proto_cache.nbytes / 1e6:.1f} MB")

    # ---- build task list ----
    # All 3 runs × 3 eps × 2 attacks × 2 defenses = 36 tasks
    tasks = []
    for run_id in range(1, N_RUNS + 1):
        run_seed = BASE_SEED + run_id
        for eps in EPS_LIST:
            for attack in ["quid", "random"]:
                for use_qdetect in [False, True]:
                    if use_qdetect:
                        n_sub = min(len(ytr), N_CLASSES * QDETECT_SUBSET_PER_CLASS)
                        print(f"  [INFO] Q-Detection will use stratified subset: "
                              f"~{n_sub} samples ({QDETECT_SUBSET_PER_CLASS}/class) "
                              f"-> J matrix {n_sub}x{n_sub} "
                              f"({n_sub**2*8/1e6:.0f} MB).")
                    tasks.append((run_id, eps, attack, use_qdetect,
                                  run_seed, -1))   # gpu_id filled below

    # ---- assign GPUs round-robin ----
    if n_gpus > 0:
        tasks = [(*t[:-1], i % n_gpus) for i, t in enumerate(tasks)]

    print(f"\n[Main] Total experiments: {len(tasks)}")
    print(f"[Main] Parallelism: "
          f"{'up to ' + str(n_gpus) + ' concurrent workers (1 per GPU)' if n_gpus > 0 else 'sequential (CPU)'}")

    # ---- run experiments ----
    if n_gpus <= 1:
        # Sequential: single GPU or CPU — create a dummy lock
        csv_lock = mp.Manager().Lock()
        for task in tasks:
            experiment_worker(task, Xtr, ytr, Xte, yte,
                              RHO_PROTO_CACHE_FILE, csv_lock)
    else:
        # Parallel: one process per GPU, tasks distributed round-robin.
        # Lock must be created from a Manager so it is shareable across
        # spawned processes (plain mp.Lock() is not picklable after spawn).
        mp.set_start_method("spawn", force=True)
        manager  = mp.Manager()
        csv_lock = manager.Lock()

        for batch_start in range(0, len(tasks), n_gpus):
            batch = tasks[batch_start : batch_start + n_gpus]
            procs = []
            for task in batch:
                p = mp.Process(
                    target=experiment_worker,
                    args=(task, Xtr, ytr, Xte, yte,
                          RHO_PROTO_CACHE_FILE, csv_lock)
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            # Check for failures
            failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
            if failed:
                print(f"  [WARN] {len(failed)} worker(s) in this batch "
                      f"exited with non-zero code: "
                      f"{[batch[i] for i in failed]}")

    print(f"\n{'='*68}")
    print(f" All experiments complete.")
    print(f" Summary → {SUMMARY_CSV}")
    print(f" Cache   → {RHO_PROTO_CACHE_FILE}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()

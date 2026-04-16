"""
depo_quid_az_amplitude_l10.py
============================================================
Depolarizing-noise evaluation of models saved by NN_quid_az_amplitude_l2.py.

For every saved best-model checkpoint (3 runs × 3 eps × 2 attacks × 2 defenses
= 36 models total) this script:
  1. Rebuilds the QMLPAZ model with a depolarizing-noise device
  2. Loads the saved weights (trained noiseless, evaluated noisy)
  3. Reconstructs poisoned subset deterministically (same seeds as QUID script)
  4. Runs noisy inference on clean test set → full metrics
  5. Runs noisy inference on poisoned samples → ASR under noise
  6. Saves per-model metrics CSV + confusion matrix PNG + summary CSV

Speed optimisations (all three applied):
  Option 1 — GPU: qiskit.aer uses CUDA GPU if available
  Option 2 — shots=256: reduces qiskit.aer shots from 1024 → 256 (~4x faster)
  Option 3 — default.mixed: PennyLane density-matrix simulator with inline
              DepolarizingChannel — exact simulation, no shots needed,
              faster than qiskit.aer for 9 qubits.

  USE_DEFAULT_MIXED = True  → uses default.mixed (option 3, fastest)
  USE_DEFAULT_MIXED = False → uses qiskit.aer + GPU + shots=256 (options 1+2)

Output folder structure mirrors the QUID script but with "depo_" prefix:
  depo_eps0.1/  depo_eps0.3/  depo_eps0.5/
    ├── depo-qmlp-az-quid-run{r}-layer10_quid_eps{e}_{attack}_{def}.csv
    └── depo-qmlp-az-quid-run{r}-layer10_quid_eps{e}_{attack}_{def}_cm.png
  depo_runs/
    └── depo_quid_qdetection_AZ23_amplitude_l10_summary.csv

Noise model: depolarizing p=0.01 on all 1-qubit and 2-qubit gates.
============================================================
"""

import os, csv, datetime, math, random, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from joblib import Parallel, delayed
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False

# ============================================================
# Config — must match NN_quid_az_amplitude_l2.py exactly
# ============================================================
N_QUBITS   = 9
N_LAYERS   = 10
N_CLASSES  = 23
BATCH_SIZE = 256          # increased from 64 — reduces Python loop overhead
BASE_SEED  = 42

BEST_MODEL_PREFIX    = "qmlp-az-quid-run"
FILENAME_PREFIX      = "qmlp-az-quid-run"
NOISE_SUFFIX         = "-noiseless"
ENCODING             = "Amplitude"
RHO_PROTO_CACHE_FILE = "rho_proto_cache_amplitude.npy"

N_RUNS   = 3
EPS_LIST = [0.1, 0.3, 0.5]
QDETECT_SUBSET_PER_CLASS = 700

DEPO_P           = 0.01
DEPO_PREFIX      = "depo_"
DEPO_RUN_DIR     = "depo_runs"
DEPO_SUMMARY_CSV = os.path.join(DEPO_RUN_DIR,
                                "depo_quid_qdetection_AZ23_amplitude_l10_summary.csv")

# ---- Speed optimisation selector ----
# True  → default.mixed + DepolarizingChannel (option 3, fastest, recommended)
# False → qiskit.aer + GPU + shots=256 (options 1+2)
USE_DEFAULT_MIXED = True
SHOTS = 256               # used only when USE_DEFAULT_MIXED = False

os.makedirs(DEPO_RUN_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Torch device: {device}")

# ============================================================
# 1. Noise device factory
#    Option 3: default.mixed — exact density matrix, no shots needed
#    Options 1+2: qiskit.aer with GPU + shots=256
# ============================================================
def _make_noisy_device():
    if USE_DEFAULT_MIXED:
        # Option 3: fastest — exact density matrix simulation
        # Noise is injected via DepolarizingChannel inside the circuit
        return qml.device("default.mixed", wires=N_QUBITS)
    else:
        # Options 1+2: qiskit.aer with GPU acceleration + reduced shots
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        noise_model = NoiseModel(basis_gates=["id", "rz", "sx", "cx", "x"])
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(DEPO_P, 1), ["id", "rz", "sx", "x"])
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(DEPO_P, 2), ["cx"])

        # Option 1: try GPU first, fall back to CPU
        try:
            dev = qml.device("qiskit.aer", wires=N_QUBITS,
                             backend="aer_simulator",
                             noise_model=noise_model,
                             shots=SHOTS,
                             device="GPU")
            print(f"[Device] qiskit.aer GPU backend active (shots={SHOTS})")
            return dev
        except Exception as e:
            warnings.warn(f"GPU backend failed ({e}), falling back to CPU.")
            dev = qml.device("qiskit.aer", wires=N_QUBITS,
                             backend="aer_simulator",
                             noise_model=noise_model,
                             shots=SHOTS)
            print(f"[Device] qiskit.aer CPU backend (shots={SHOTS})")
            return dev


# ============================================================
# 2. Quantum circuit + model
#    Option 3 circuit: inline DepolarizingChannel after every gate
#    Options 1+2 circuit: noise injected by qiskit.aer noise model
#    AmplitudeEmbedding requires broadcast_expand for batched inputs.
# ============================================================
def build_weight_shapes():
    shapes = {}
    for n in range(N_LAYERS):
        shapes[f"rot_layer_{n}"] = (N_QUBITS, 3)
        shapes[f"crx_layer_{n}"] = (N_QUBITS, 1)
    return shapes

WEIGHT_SHAPES = build_weight_shapes()

def make_noisy_qnode(dev):
    @qml.transforms.broadcast_expand
    @qml.qnode(dev, interface="torch")
    def _qnode(inputs, **wkw):
        # AmplitudeEmbedding applied ONCE before variational layers
        qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=True)
        for n in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.Rot(*wkw[f"rot_layer_{n}"][i], wires=i)
                if USE_DEFAULT_MIXED:
                    # Option 3: inject 1-qubit depolarizing noise after Rot
                    qml.DepolarizingChannel(DEPO_P, wires=i)
            for i in range(N_QUBITS):
                qml.CRX(wkw[f"crx_layer_{n}"][i][0], wires=[i, (i+1) % N_QUBITS])
                if USE_DEFAULT_MIXED:
                    # Option 3: inject 1-qubit depolarizing noise after CRX
                    # on both target qubits (approximates 2-qubit depolarizing)
                    qml.DepolarizingChannel(DEPO_P, wires=i)
                    qml.DepolarizingChannel(DEPO_P, wires=(i+1) % N_QUBITS)
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return _qnode


class QMLPAZ(nn.Module):
    """Same architecture as NN_quid_az_amplitude_l2.py but with noisy device."""
    def __init__(self):
        super().__init__()
        dev          = _make_noisy_device()
        qnode        = make_noisy_qnode(dev)
        self.qlayer  = TorchLayer(qnode, WEIGHT_SHAPES)
        self.fc      = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        x   = x.to(next(self.parameters()).device)
        out = self.qlayer(x)
        out = self.fc(out.to(x.device))
        return F.log_softmax(out, dim=1)


# ============================================================
# 3. Data — must match NN_quid_az_amplitude_l2.py exactly
#    Returns BOTH train and test (train needed for poisoning replay)
# ============================================================
def load_az23_pca512(
    train_npz="./AZ_23/AZ-Class-Task_23_families_train.npz",
    test_npz ="./AZ_23/AZ-Class-Task_23_families_test.npz"
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

    pca = PCA(n_components=512)
    Xtr = pca.fit_transform(Xtr).astype(np.float32)
    Xte = pca.transform(Xte).astype(np.float32)

    print(f"[Data] Train: {Xtr.shape}  Test: {Xte.shape}")
    return Xtr, ytr, Xte, yte


def make_test_loader(Xte, yte):
    ds = TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                       torch.tensor(yte, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 4. Poisoning replay — identical to NN_quid_az_amplitude_l2.py
#    Deterministic: same seeds → same poisoned_idx and y_poisoned
# ============================================================
def _compute_one_rho_amplitude(x512):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface=None, diff_method=None)
    def rho_qnode(x):
        qml.AmplitudeEmbedding(x, wires=range(N_QUBITS), normalize=True)
        return qml.density_matrix(wires=list(range(N_QUBITS)))

    return np.array(rho_qnode(pnp.array(x512, dtype=float)))


def frobenius_dist(A, B):
    D = A - B
    return float(np.sqrt(np.sum(np.real(D * np.conj(D)))))


def stratified_subset(X, y, n_per_class, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    idx_keep = []
    for c in range(N_CLASSES):
        idx_c = np.where(y == c)[0]
        if len(idx_c) <= n_per_class:
            idx_keep.append(idx_c)
        else:
            idx_keep.append(rng.choice(idx_c, size=n_per_class, replace=False))
    idx_keep = np.concatenate(idx_keep)
    return X[np.sort(idx_keep)], y[np.sort(idx_keep)], np.sort(idx_keep)


def quid_label_flip_with_data(Xtr, proto_cache, y, epsilon, seed=BASE_SEED):
    if epsilon <= 0.0:
        return y.copy(), np.array([], dtype=int)
    rng          = np.random.default_rng(seed)
    n            = len(y)
    k            = int(round(epsilon * n))
    poisoned_idx = rng.choice(n, size=k, replace=False)
    protos = [proto_cache[c] for c in range(N_CLASSES)]
    if _JOBLIB_OK:
        rhos_poisoned = Parallel(n_jobs=-1, verbose=0, prefer="threads")(
            delayed(_compute_one_rho_amplitude)(Xtr[i]) for i in poisoned_idx
        )
    else:
        rhos_poisoned = [_compute_one_rho_amplitude(Xtr[i]) for i in poisoned_idx]
    y_poison = y.copy()
    for j, i in enumerate(poisoned_idx):
        dists       = [frobenius_dist(rhos_poisoned[j], protos[c])
                       for c in range(N_CLASSES)]
        y_poison[i] = int(np.argmax(dists))
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
# 5. Evaluation — matches NN_quid_az_amplitude_l2.py exactly
# ============================================================
def evaluation(model, test_loader, num_classes=N_CLASSES):
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
    }, cm


def compute_asr(model, Xtr_sub, y_poisoned, p_idx):
    """ASR under depolarizing noise — same formula as QUID script."""
    if len(p_idx) == 0:
        return float("nan")
    model.eval()
    Xp = torch.tensor(Xtr_sub[p_idx], dtype=torch.float32).to(device)
    yp = torch.tensor(y_poisoned[p_idx], dtype=torch.long).to(device)
    with torch.no_grad():
        asr = (model(Xp).argmax(dim=1) == yp).float().mean().item() * 100.0
    return asr


# ============================================================
# 6. File naming helpers (mirror QUID script, with depo_ prefix)
# ============================================================
def depo_eps_folder(eps):
    folder = f"{DEPO_PREFIX}eps{eps}"
    os.makedirs(folder, exist_ok=True)
    return folder

def source_model_path(run_id, eps, attack, defense):
    def_str = defense.replace("-", "")
    fname   = (f"{BEST_MODEL_PREFIX}{run_id}"
               f"-layer{N_LAYERS}{NOISE_SUFFIX}-{ENCODING}"
               f"_eps{eps}_{attack}_{def_str}.pt")
    return os.path.join(f"eps{eps}", fname)

def depo_csv_path(run_id, eps, attack, defense):
    def_str = defense.replace("-", "")
    fname   = (f"depo-{FILENAME_PREFIX}{run_id}"
               f"-layer{N_LAYERS}_quid"
               f"_eps{eps}_{attack}_{def_str}.csv")
    return os.path.join(depo_eps_folder(eps), fname)

def depo_cm_path(run_id, eps, attack, defense):
    def_str = defense.replace("-", "")
    fname   = (f"depo-{FILENAME_PREFIX}{run_id}"
               f"-layer{N_LAYERS}_quid"
               f"_eps{eps}_{attack}_{def_str}_cm.png")
    return os.path.join(depo_eps_folder(eps), fname)


# ============================================================
# 7. CSV helpers
# ============================================================
def save_per_run_csv(metrics, asr, filepath):
    """['Metric','Value'] format — matches QUID script, adds asr."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
        writer.writerow(['asr', asr])
    print(f"  Metrics saved → {filepath}")


_SUMMARY_FIELDS = [
    "timestamp", "run_id", "attack", "defense", "epsilon",
    "n_poisoned", "asr",
    "accuracy", "loss", "precision", "recall",
    "f1", "fpr", "fnr", "roc_auc", "pr_auc",
    "depo_p", "noise_backend"
]

def append_summary_csv(row):
    exists = os.path.exists(DEPO_SUMMARY_CSV) and os.path.getsize(DEPO_SUMMARY_CSV) > 0
    with open(DEPO_SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()


# ============================================================
# 8. Confusion matrix plot
# ============================================================
def save_confusion_matrix(cm, run_id, eps, attack, defense, filepath):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    backend_str = "default.mixed" if USE_DEFAULT_MIXED else f"qiskit.aer (shots={SHOTS})"
    ax.set_title(
        f"Confusion Matrix — Depolarizing Noise (p={DEPO_P}, {backend_str})\n"
        f"Run {run_id} | eps={eps} | {attack} | {defense}",
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
    noise_backend = "default.mixed" if USE_DEFAULT_MIXED else f"qiskit.aer (GPU+shots={SHOTS})"
    print("=" * 68)
    print(f" Depolarizing Noise Evaluation | QUID AZ-Amplitude models | p={DEPO_P}")
    print(f" Noise backend : {noise_backend}")
    print(f" Batch size    : {BATCH_SIZE}")
    print(f" Models        : {N_RUNS} runs × {len(EPS_LIST)} eps × 2 attacks × 2 defenses"
          f" = {N_RUNS * len(EPS_LIST) * 4} total")
    print("=" * 68)

    # Load full train+test data (train needed to replay poisoning)
    Xtr, ytr, Xte, yte = load_az23_pca512()
    test_loader = make_test_loader(Xte, yte)

    # Load prototype cache (needed for QUID replay)
    if not os.path.exists(RHO_PROTO_CACHE_FILE):
        raise FileNotFoundError(
            f"Proto cache not found: {RHO_PROTO_CACHE_FILE}\n"
            f"Run NN_quid_az_amplitude_l2.py first to build it.")
    proto_cache = np.load(RHO_PROTO_CACHE_FILE)
    print(f"[ProtoCache] Loaded {proto_cache.shape[0]} prototypes "
          f"from {RHO_PROTO_CACHE_FILE}")

    total   = 0
    skipped = 0

    for run_id in range(1, N_RUNS + 1):
        run_seed = BASE_SEED + run_id

        for eps in EPS_LIST:
            for attack in ["quid", "random"]:
                for defense in ["none", "q-detection"]:

                    model_path = source_model_path(run_id, eps, attack, defense)
                    tag = (f"[Run {run_id} | eps={eps} | "
                           f"{attack} | {defense}]")

                    print(f"\n{'-'*68}")
                    print(f" {tag}")
                    print(f"  Model: {model_path}")

                    if not os.path.exists(model_path):
                        print(f"  [SKIP] Model not found: {model_path}")
                        skipped += 1
                        continue

                    # ---- replay poisoning (deterministic, same seeds) ----
                    Xtr_sub, ytr_sub, _ = stratified_subset(
                        Xtr, ytr, QDETECT_SUBSET_PER_CLASS, seed=run_seed)

                    if attack == "quid":
                        y_poisoned, p_idx = quid_label_flip_with_data(
                            Xtr_sub, proto_cache, ytr_sub,
                            epsilon=eps, seed=run_seed)
                    else:
                        y_poisoned, p_idx = random_label_flip(
                            ytr_sub, epsilon=eps, seed=run_seed)

                    n_poisoned = len(p_idx)
                    print(f"  Poisoned replay: {n_poisoned} samples")

                    # ---- build fresh noisy model and load weights ----
                    model = QMLPAZ().to(device)
                    model.load_state_dict(
                        torch.load(model_path, map_location=device))
                    model.eval()
                    print(f"  Weights loaded. Running noisy inference ...")

                    # ---- evaluate on clean test set ----
                    metrics, cm = evaluation(model, test_loader)

                    # ---- compute ASR under noise ----
                    asr = compute_asr(model, Xtr_sub, y_poisoned, p_idx)
                    asr_str = f"{asr:.4f}" if not math.isnan(asr) else "nan"

                    for k, v in metrics.items():
                        print(f"  {k.capitalize():<12}: {v:.4f}")
                    print(f"  {'Asr':<12}: {asr_str}")

                    # ---- save per-model CSV (includes asr) ----
                    save_per_run_csv(
                        metrics, asr_str,
                        depo_csv_path(run_id, eps, attack, defense))

                    # ---- save confusion matrix ----
                    save_confusion_matrix(
                        cm, run_id, eps, attack, defense,
                        depo_cm_path(run_id, eps, attack, defense))

                    # ---- append to summary CSV ----
                    row = {
                        "timestamp"    : datetime.datetime.now().isoformat(timespec="seconds"),
                        "run_id"       : run_id,
                        "attack"       : attack,
                        "defense"      : defense,
                        "epsilon"      : eps,
                        "n_poisoned"   : int(n_poisoned),
                        "asr"          : round(asr, 4) if not math.isnan(asr) else "nan",
                        "accuracy"     : round(metrics['accuracy'] * 100, 4),
                        "loss"         : round(metrics['loss'], 6),
                        "precision"    : round(metrics['precision'], 6),
                        "recall"       : round(metrics['recall'], 6),
                        "f1"           : round(metrics['f1'], 6),
                        "fpr"          : round(metrics['fpr'], 6),
                        "fnr"          : round(metrics['fnr'], 6),
                        "roc_auc"      : round(metrics['roc_auc'], 6) if not math.isnan(metrics['roc_auc']) else "nan",
                        "pr_auc"       : metrics['pr_auc'],
                        "depo_p"       : DEPO_P,
                        "noise_backend": noise_backend,
                    }
                    append_summary_csv(row)
                    total += 1

    print(f"\n{'='*68}")
    print(f" Done. Evaluated {total} model(s), skipped {skipped}.")
    print(f" Summary → {DEPO_SUMMARY_CSV}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()

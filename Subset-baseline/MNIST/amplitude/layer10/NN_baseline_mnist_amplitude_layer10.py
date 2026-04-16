"""
NN_baseline_mnist_amplitude_layer10.py
============================================================
Baseline — NO attack, NO defense.
Dataset      : MNIST 10-class handwritten digits
Encoding     : AmplitudeEmbedding (applied ONCE before variational layers)
Layer config : 10 layers
Runs         : 1  (seed 43)
Epochs       : 30 (early stopping patience=5)
GPUs         : 1 MIG slice

Model architecture (identical to NN_quid_mnist_amplitude_l10.py):
  * Features  : flatten(784) -> MinMaxScaler -> PCA(512)
  * Qubits    : 9
  * Encoding  : AmplitudeEmbedding normalize=True (applied ONCE)
  * PQC       : Rot(a,b,g) + cyclic CRX  x 10 layers
  * Post      : Linear(9->10) + log_softmax
  * Opt       : Adam lr=0.001, weight_decay=1e-4
  * Subset    : stratified 700/class (~7000 total)

Output structure:
  baseline_mnist_amplitude_layer10/
    baseline_mnist_amplitude_layer10_run1_epoch_metrics.csv
    baseline_mnist_amplitude_layer10_run1_metrics.csv
    baseline_mnist_amplitude_layer10_summary.csv
    best_model_run1.pt
============================================================
"""

import os, csv, random, time
import numpy as np
import pennylane as qml

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

import idx2numpy

# ============================================================
# Config
# ============================================================
N_QUBITS         = 9
N_LAYERS         = 10
N_CLASSES        = 10
PCA_COMPONENTS   = 512
EPOCHS           = 30
PATIENCE         = 5
BATCH_SIZE       = 64
LR               = 0.001
W_DECAY          = 1e-4
BASE_SEED        = 42
N_RUNS           = 1
SUBSET_PER_CLASS = 700
GPUS             = [0, 1]

MNIST_DIR = "/work/clalarcon/Nowmi/Sok/Baseline/MNIST/MNIST"

OUT_DIR     = "baseline_mnist_amplitude_layer10"
SUMMARY_CSV = os.path.join(OUT_DIR, "baseline_mnist_amplitude_layer10_summary.csv")

SUMMARY_FIELDS = [
    "n_layers", "run_id", "seed",
    "test_accuracy_pct", "loss",
    "fnr", "fpr",
    "roc_auc", "pr_auc",
    "precision", "recall", "f1",
    "train_time_s", "gpu_id",
]
MV_FIELDS    = ["Metric", "Value"]
EPOCH_FIELDS = ["epoch", "train_loss", "train_acc_pct", "test_acc_pct"]
SEP          = "=" * 60


# ============================================================
# 1. Data loading — flatten -> MinMaxScaler -> PCA(512)
# ============================================================
def load_mnist_pca512():
    train_images = os.path.join(MNIST_DIR, "train-images-idx3-ubyte")
    train_labels = os.path.join(MNIST_DIR, "train-labels-idx1-ubyte")
    test_images  = os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte")
    test_labels  = os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte")

    Xtr = idx2numpy.convert_from_file(train_images)
    ytr = idx2numpy.convert_from_file(train_labels).astype(np.int64)
    Xte = idx2numpy.convert_from_file(test_images)
    yte = idx2numpy.convert_from_file(test_labels).astype(np.int64)

    # flatten 28x28 -> 784 (no downscaling — need >= 512 features for PCA)
    Xtr = Xtr.reshape(Xtr.shape[0], -1).astype(np.float32)
    Xte = Xte.reshape(Xte.shape[0], -1).astype(np.float32)

    scaler = MinMaxScaler()
    Xtr    = scaler.fit_transform(Xtr)
    Xte    = scaler.transform(Xte)

    pca = PCA(n_components=PCA_COMPONENTS)
    Xtr = pca.fit_transform(Xtr).astype(np.float32)
    Xte = pca.transform(Xte).astype(np.float32)

    print(f"[Data] Classes : {len(np.unique(ytr))}  "
          f"Train : {Xtr.shape}  Test : {Xte.shape}")
    return Xtr, ytr, Xte, yte


# ============================================================
# 2. Stratified subset
# ============================================================
def stratified_subset(X, y, n_per_class, seed):
    rng      = np.random.default_rng(seed)
    idx_keep = []
    for c in range(N_CLASSES):
        idx_c = np.where(y == c)[0]
        if len(idx_c) <= n_per_class:
            idx_keep.append(idx_c)
        else:
            idx_keep.append(rng.choice(idx_c, size=n_per_class, replace=False))
    idx_keep = np.sort(np.concatenate(idx_keep))
    return X[idx_keep], y[idx_keep]


# ============================================================
# 3. Quantum model — AmplitudeEmbedding applied ONCE
# ============================================================
def build_weight_shapes():
    shapes = {}
    for n in range(N_LAYERS):
        shapes[f"rot_layer_{n}"] = (N_QUBITS, 3)
        shapes[f"crx_layer_{n}"] = (N_QUBITS, 1)
    return shapes


def make_qnode_torch(dev):
    @qml.transforms.broadcast_expand
    @qml.qnode(dev, interface="torch")
    def _qnode(inputs, **wkw):
        # AmplitudeEmbedding encodes the full state vector — applied ONCE only.
        # (Unlike AngleEmbedding, re-applying it mid-circuit overwrites the state.)
        qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=True)
        for n in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.Rot(*wkw[f"rot_layer_{n}"][i], wires=i)
            for i in range(N_QUBITS):
                qml.CRX(wkw[f"crx_layer_{n}"][i][0],
                        wires=[i, (i + 1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return _qnode


class QMLPMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        dev           = qml.device("default.qubit", wires=N_QUBITS)
        weight_shapes = build_weight_shapes()
        qnode         = make_qnode_torch(dev)
        self.qlayer   = TorchLayer(qnode, weight_shapes)
        self.fc       = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        # default.qubit is a CPU simulator — qlayer always on CPU
        fc_device = next(self.fc.parameters()).device
        out = self.qlayer(x.cpu())
        out = self.fc(out.to(fc_device))
        return F.log_softmax(out, dim=1)


# ============================================================
# 4. Loaders
# ============================================================
def make_loader(X, y, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ============================================================
# 5. Train one epoch
# ============================================================
def train_one_epoch(model, device, train_loader, optimizer):
    model.train()
    correct = 0; total = 0; running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = F.nll_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted  = torch.max(outputs.data, dim=1)
        total        += targets.size(0)
        correct      += (predicted == targets).sum().item()
        running_loss += loss.item() * targets.size(0)
    return 100.0 * correct / total, running_loss / total


# ============================================================
# 6. Test accuracy
# ============================================================
def test_accuracy(model, device, test_loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted   = torch.max(model(inputs).data, dim=1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()
    return 100.0 * correct / total


# ============================================================
# 7. Full evaluation
# ============================================================
def evaluation(model, test_loader, device):
    model.eval()
    outputs_list, y_true_list = [], []
    for x_batch, y_batch in test_loader:
        with torch.no_grad():
            out = model(x_batch.to(device))
        outputs_list.append(out)
        y_true_list.append(y_batch.to(device))

    output    = torch.cat(outputs_list, dim=0)
    true      = torch.cat(y_true_list,  dim=0)
    pred      = output.argmax(dim=1)
    probs     = torch.softmax(output, dim=1).detach().cpu().numpy()
    acc       = (pred == true).float().mean().item()
    loss      = F.nll_loss(output, true).item()

    y_true_np = true.cpu().numpy()
    y_pred_np = pred.cpu().numpy()

    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(N_CLASSES)))
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    precision = np.mean(TP / (TP + FP + 1e-8))
    recall    = np.mean(TP / (TP + FN + 1e-8))
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fpr       = np.mean(FP / (FP + TN + 1e-8))
    fnr       = np.mean(FN / (FN + TP + 1e-8))

    try:
        roc_auc = roc_auc_score(y_true_np, probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(
            np.eye(N_CLASSES)[y_true_np], probs, average="macro")
    except Exception:
        pr_auc = float("nan")

    return dict(accuracy=acc, loss=loss, precision=precision,
                recall=recall, f1=f1, fpr=fpr, fnr=fnr,
                roc_auc=roc_auc, pr_auc=pr_auc)


# ============================================================
# 8. CSV helpers
# ============================================================
def save_epoch_csv(epoch_recs, run_id):
    path = os.path.join(OUT_DIR,
                        f"baseline_mnist_amplitude_layer10_run{run_id}_epoch_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPOCH_FIELDS)
        w.writeheader()
        for rec in epoch_recs:
            w.writerow({
                "epoch"        : rec["epoch"],
                "train_loss"   : rec["train_loss"],
                "train_acc_pct": rec["train_acc_pct"],
                "test_acc_pct" : rec["test_acc_pct"],
            })
    print(f"  [R{run_id}] Epoch CSV    -> {path}")


def save_per_run_csv(metrics, run_id):
    path = os.path.join(OUT_DIR,
                        f"baseline_mnist_amplitude_layer10_run{run_id}_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MV_FIELDS)
        w.writeheader()
        for key, value in metrics.items():
            w.writerow({"Metric": key, "Value": value})
    print(f"  [R{run_id}] Per-run CSV  -> {path}")


def append_summary_csv(row, csv_lock):
    with csv_lock:
        exists = os.path.exists(SUMMARY_CSV) and os.path.getsize(SUMMARY_CSV) > 0
        with open(SUMMARY_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if not exists:
                w.writeheader()
            w.writerow(row)


# ============================================================
# 9. Worker
# ============================================================
def experiment_worker(task, Xtr, ytr, Xte, yte, csv_lock):
    run_id, gpu_id = task
    run_seed = BASE_SEED + run_id
    tag      = f"[MNIST|Amp|L10|R{run_id}|GPU{gpu_id}]"

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    torch.manual_seed(run_seed)
    torch.cuda.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    print(SEP)
    print(f"  {tag} Starting")
    print(SEP)

    Xtr_sub, ytr_sub = stratified_subset(Xtr, ytr, SUBSET_PER_CLASS, seed=run_seed)
    print(f"  {tag} Subset: {len(ytr_sub)} samples "
          f"({SUBSET_PER_CLASS}/class from {len(ytr)} total)")

    train_loader = make_loader(Xtr_sub, ytr_sub, shuffle=True)
    test_loader  = make_loader(Xte,     yte,     shuffle=False)

    model = QMLPMNIST().to(device)
    # default.qubit is a CPU simulator — keep qlayer weights on CPU
    model.qlayer.to('cpu')
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=W_DECAY)

    best_acc     = 0.0
    best_state   = None
    epoch_recs   = []
    patience_ctr = 0
    start_time   = time.time()

    print(f"  {tag} {'Epoch':>5}  {'TrainAcc%':>10}  {'TrainLoss':>10}  {'TestAcc%':>9}  {'Patience':>8}")
    print(f"  {tag} {'-----':>5}  {'----------':>10}  {'----------':>10}  {'--------':>9}  {'--------':>8}")

    for epoch in range(1, EPOCHS + 1):
        tr_acc, tr_loss = train_one_epoch(model, device, train_loader, optimizer)
        te_acc          = test_accuracy(model, device, test_loader)

        if te_acc > best_acc:
            best_acc     = te_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        print(f"  {tag} {epoch:>5}  {tr_acc:>10.4f}  {tr_loss:>10.6f}  {te_acc:>9.4f}  {patience_ctr:>8}/{PATIENCE}")

        epoch_recs.append({
            "epoch"        : epoch,
            "train_acc_pct": round(tr_acc,  6),
            "train_loss"   : round(tr_loss, 8),
            "test_acc_pct" : round(te_acc,  6),
        })

        if patience_ctr >= PATIENCE:
            print(f"  {tag} Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    elapsed = time.time() - start_time
    print(f"\n  {tag} Training time: {elapsed:.2f}s  Best test acc: {best_acc:.4f}%")

    save_epoch_csv(epoch_recs, run_id)

    model_path = os.path.join(OUT_DIR, f"best_model_run{run_id}.pt")
    torch.save(best_state, model_path)
    print(f"  [R{run_id}] Model saved  -> {model_path}")

    model.load_state_dict(best_state)
    model.qlayer.to('cpu')
    m = evaluation(model, test_loader, device)
    save_per_run_csv(m, run_id)

    print(f"\n  {tag} Final metrics:")
    print(f"  {tag}   Test Accuracy % : {m['accuracy']*100:.6f}")
    print(f"  {tag}   Loss            : {m['loss']:.6f}")
    print(f"  {tag}   FNR             : {m['fnr']:.6f}")
    print(f"  {tag}   FPR             : {m['fpr']:.6f}")
    print(f"  {tag}   ROC-AUC         : {m['roc_auc']:.6f}")
    print(f"  {tag}   PR-AUC          : {m['pr_auc']:.6f}")
    print(f"  {tag}   Precision       : {m['precision']:.6f}")
    print(f"  {tag}   Recall          : {m['recall']:.6f}")
    print(f"  {tag}   F1              : {m['f1']:.6f}")

    row = {
        "n_layers"         : N_LAYERS,
        "run_id"           : run_id,
        "seed"             : run_seed,
        "test_accuracy_pct": round(m["accuracy"] * 100, 6),
        "loss"             : round(m["loss"],      8),
        "fnr"              : round(m["fnr"],       6),
        "fpr"              : round(m["fpr"],       6),
        "roc_auc"          : round(m["roc_auc"],   6),
        "pr_auc"           : round(m["pr_auc"],    6),
        "precision"        : round(m["precision"], 6),
        "recall"           : round(m["recall"],    6),
        "f1"               : round(m["f1"],        6),
        "train_time_s"     : round(elapsed, 2),
        "gpu_id"           : gpu_id,
    }
    append_summary_csv(row, csv_lock)
    print(f"  {tag} Done.")


# ============================================================
# 10. Main
# ============================================================
def main():
    global GPUS
    print(SEP)
    print(f"  Baseline | MNIST 10-class | 9-qubit | AmplitudeEmbedding | 10 layers")
    print(f"  Runs: {N_RUNS}  Epochs: {EPOCHS}  Patience: {PATIENCE}")
    print("  No attack — No defense")
    print(SEP)

    n_avail = torch.cuda.device_count()
    print(f"[GPU] {n_avail} CUDA device(s) visible to PyTorch")
    for i in range(n_avail):
        print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
    if n_avail == 0:
        raise RuntimeError("No CUDA devices visible — check SLURM GPU allocation.")
    if n_avail == 1:
        print("[WARN] Only 1 CUDA device visible — workers will use cuda:0.")
        GPUS = [0, 0]
    else:
        GPUS = list(range(min(n_avail, len(GPUS))))
        print(f"[GPU] Using devices: {GPUS}")

    os.makedirs(OUT_DIR, exist_ok=True)

    Xtr, ytr, Xte, yte = load_mnist_pca512()

    tasks = [(run_id, GPUS[(run_id - 1) % len(GPUS)])
             for run_id in range(1, N_RUNS + 1)]

    print("\n[Main] Task schedule:")
    print("  Run  cuda-device")
    print("  ---  -----------")
    for run_id, gpu_id in tasks:
        print(f"  {run_id:<3}  cuda:{gpu_id}")

    mp.set_start_method("spawn", force=True)
    manager  = mp.Manager()
    csv_lock = manager.Lock()

    total_start = time.time()
    for batch_start in range(0, len(tasks), len(GPUS)):
        batch = tasks[batch_start : batch_start + len(GPUS)]
        procs = []
        for task in batch:
            p = mp.Process(
                target=experiment_worker,
                args=(task, Xtr, ytr, Xte, yte, csv_lock),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        failed = [batch[i] for i, p in enumerate(procs) if p.exitcode != 0]
        if failed:
            print(f"\n[WARN] Failed tasks in batch: {failed}")

    total_elapsed = time.time() - total_start

    print(SEP)
    print(f"  All runs complete in {total_elapsed/60:.1f} min")
    print(SEP)

    try:
        with open(SUMMARY_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print("[WARN] Summary CSV not found.")
        return

    accs = [float(r["test_accuracy_pct"]) for r in rows]
    rocs = [float(r["roc_auc"])           for r in rows]
    prs  = [float(r["pr_auc"])            for r in rows]
    fnrs = [float(r["fnr"])               for r in rows]
    fprs = [float(r["fpr"])               for r in rows]

    print(f"  MNIST Amplitude | Layer 10 — mean ± std across {N_RUNS} runs:")
    print(f"  Test Accuracy : {np.mean(accs):.4f}% ± {np.std(accs):.4f}%")
    print(f"  ROC-AUC       : {np.mean(rocs):.6f} ± {np.std(rocs):.6f}")
    print(f"  PR-AUC        : {np.mean(prs):.6f} ± {np.std(prs):.6f}")
    print(f"  FNR           : {np.mean(fnrs):.6f} ± {np.std(fnrs):.6f}")
    print(f"  FPR           : {np.mean(fprs):.6f} ± {np.std(fprs):.6f}")
    print(f"\n  Summary CSV  -> {SUMMARY_CSV}")
    print(f"  Epoch CSVs   -> {OUT_DIR}/baseline_mnist_amplitude_layer10_run*_epoch_metrics.csv")
    print(f"  Per-run CSVs -> {OUT_DIR}/baseline_mnist_amplitude_layer10_run*_metrics.csv")
    print(SEP)


if __name__ == "__main__":
    main()

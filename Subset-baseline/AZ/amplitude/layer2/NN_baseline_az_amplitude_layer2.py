"""
NN_baseline_az_amplitude_layer2.py
============================================================
Baseline — NO attack, NO defense.  (CPU-only version)
Encoding     : AmplitudeEmbedding
Layer config : 2 layers
Runs         : 1  (seed 43)
Epochs       : 30 (early stopping patience=5)

Model architecture:
  * Dataset   : AZ-Class FULL 23-family malware dataset
  * Subset    : stratified 700/class  (~16 100 total)
  * Features  : MinMaxScaler -> PCA(n_components=512)
  * Qubits    : 9
  * Encoding  : AmplitudeEmbedding (normalize=True, applied ONCE)
  * PQC       : Rot(a,b,g) + cyclic CRX  x 2 layers
  * Post      : Linear(9->23) + log_softmax
  * Opt       : Adam lr=0.001, weight_decay=1e-4

Output structure:
  baseline_amplitude_layer2/
    baseline_amplitude_layer2_run1_epoch_metrics.csv
    baseline_amplitude_layer2_run1_metrics.csv
    baseline_amplitude_layer2_summary.csv
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
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# ============================================================
# Config
# ============================================================
N_QUBITS         = 9
N_LAYERS         = 2
N_CLASSES        = 23
PCA_COMPONENTS   = 512          # AmplitudeEmbedding needs 2^N_QUBITS = 512 features
EPOCHS           = 30
PATIENCE         = 5
BATCH_SIZE       = 64
LR               = 0.001
W_DECAY          = 1e-4
BASE_SEED        = 42
SUBSET_PER_CLASS = 700

DEVICE = torch.device("cpu")

OUT_DIR     = "baseline_amplitude_layer2"
SUMMARY_CSV = os.path.join(OUT_DIR, "baseline_amplitude_layer2_summary.csv")

SUMMARY_FIELDS = [
    "n_layers", "run_id", "seed",
    "test_accuracy_pct", "loss",
    "fnr", "fpr",
    "roc_auc", "pr_auc",
    "precision", "recall", "f1",
    "train_time_s",
]
MV_FIELDS    = ["Metric", "Value"]
EPOCH_FIELDS = ["epoch", "train_loss", "train_acc_pct", "test_acc_pct"]
SEP          = "=" * 60


# ============================================================
# 1. Data loading  — PCA(512) for AmplitudeEmbedding
# ============================================================
def load_az23_pca512(
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
# 3. Quantum model — AmplitudeEmbedding
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
        qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=True)
        for n in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.Rot(*wkw[f"rot_layer_{n}"][i], wires=i)
            for i in range(N_QUBITS):
                qml.CRX(wkw[f"crx_layer_{n}"][i][0],
                        wires=[i, (i + 1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return _qnode


class QMLPAZ(nn.Module):
    def __init__(self):
        super().__init__()
        dev           = qml.device("default.qubit", wires=N_QUBITS)
        weight_shapes = build_weight_shapes()
        qnode         = make_qnode_torch(dev)
        self.qlayer   = TorchLayer(qnode, weight_shapes)
        self.fc       = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        out = self.qlayer(x)
        out = self.fc(out)
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
def train_one_epoch(model, train_loader, optimizer):
    model.train()
    correct = 0; total = 0; running_loss = 0.0
    for inputs, targets in train_loader:
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
def test_accuracy(model, test_loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            _, predicted   = torch.max(model(inputs).data, dim=1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()
    return 100.0 * correct / total


# ============================================================
# 7. Full evaluation
# ============================================================
def evaluation(model, test_loader):
    model.eval()
    outputs_list, y_true_list = [], []
    for x_batch, y_batch in test_loader:
        with torch.no_grad():
            out = model(x_batch)
        outputs_list.append(out)
        y_true_list.append(y_batch)

    output    = torch.cat(outputs_list, dim=0)
    true      = torch.cat(y_true_list,  dim=0)
    pred      = output.argmax(dim=1)
    probs     = torch.softmax(output, dim=1).detach().numpy()
    acc       = (pred == true).float().mean().item()
    loss      = F.nll_loss(output, true).item()

    y_true_np = true.numpy()
    y_pred_np = pred.numpy()

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
                        f"baseline_amplitude_layer2_run{run_id}_epoch_metrics.csv")
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
                        f"baseline_amplitude_layer2_run{run_id}_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MV_FIELDS)
        w.writeheader()
        for key, value in metrics.items():
            w.writerow({"Metric": key, "Value": value})
    print(f"  [R{run_id}] Per-run CSV  -> {path}")


def save_summary_csv(row):
    exists = os.path.exists(SUMMARY_CSV) and os.path.getsize(SUMMARY_CSV) > 0
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ============================================================
# 9. Main
# ============================================================
def main():
    run_id   = 1
    run_seed = BASE_SEED + run_id   # 43
    tag      = f"[Amp|L2|R{run_id}|CPU]"

    print(SEP)
    print(f"  Baseline | AZ-Class 23-family | 9-qubit | AmplitudeEmbedding | 2 layers")
    print(f"  Run: {run_id}  Seed: {run_seed}  Epochs: {EPOCHS}  Patience: {PATIENCE}")
    print(f"  Device: CPU")
    print("  No attack — No defense")
    print(SEP)

    # ---- seeds ----
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- data ----
    Xtr, ytr, Xte, yte = load_az23_pca512()
    Xtr_sub, ytr_sub = stratified_subset(Xtr, ytr, SUBSET_PER_CLASS, seed=run_seed)
    print(f"  {tag} Subset: {len(ytr_sub)} samples "
          f"({SUBSET_PER_CLASS}/class from {len(ytr)} total)")

    train_loader = make_loader(Xtr_sub, ytr_sub, shuffle=True)
    test_loader  = make_loader(Xte,     yte,     shuffle=False)

    # ---- model & optimiser ----
    model     = QMLPAZ()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=W_DECAY)

    best_acc     = 0.0
    best_state   = None
    epoch_recs   = []
    patience_ctr = 0
    start_time   = time.time()

    # ---- epoch loop ----
    print(f"  {tag} {'Epoch':>5}  {'TrainAcc%':>10}  {'TrainLoss':>10}  {'TestAcc%':>9}  {'Patience':>8}")
    print(f"  {tag} {'-----':>5}  {'----------':>10}  {'----------':>10}  {'--------':>9}  {'--------':>8}")

    for epoch in range(1, EPOCHS + 1):
        tr_acc, tr_loss = train_one_epoch(model, train_loader, optimizer)
        te_acc          = test_accuracy(model, test_loader)

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

    # ---- save epoch CSV ----
    save_epoch_csv(epoch_recs, run_id)

    # ---- save best model ----
    model_path = os.path.join(OUT_DIR, f"best_model_run{run_id}.pt")
    torch.save(best_state, model_path)
    print(f"  [R{run_id}] Model saved  -> {model_path}")

    # ---- final evaluation ----
    model.load_state_dict(best_state)
    m = evaluation(model, test_loader)
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
    }
    save_summary_csv(row)

    print(SEP)
    print(f"  Run complete in {elapsed/60:.1f} min")
    print(f"  Summary CSV  -> {SUMMARY_CSV}")
    print(f"  Epoch CSV    -> {OUT_DIR}/baseline_amplitude_layer2_run1_epoch_metrics.csv")
    print(f"  Per-run CSV  -> {OUT_DIR}/baseline_amplitude_layer2_run1_metrics.csv")
    print(SEP)


if __name__ == "__main__":
    main()

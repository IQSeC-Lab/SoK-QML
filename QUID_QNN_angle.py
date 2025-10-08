# faithful_quid_ess_qnn_4class_mnist_cifar_paperfaithful_with_noiseless.py
# Paper-faithful replication (MNIST & CIFAR-100), including BOTH regimes:
#   - Noiseless (p=0, lightning.qubit, no noise channels)
#   - Noisy (p=0.05, default.mixed, AmplitudeDamping + Depolarizing AFTER EACH GATE)
#
# Features:
#   * CAE latent d=8, 4-class subsets (175 train / 75 test per class)
#   * ESS (Frobenius) for angle & amplitude
#   * QUID label-flip (density space), seeded per run
#   * QNNs: PQC-1 / PQC-6 / PQC-8, angle encoding for training
#   * Hybrid head: Z-expectations -> Linear layer (Adam); quantum params via SPSA
#   * Shots = 1000; Epochs = 30; ε ∈ {0.1, 0.3, 0.5, 0.7}
#
# CLI: --gpu <id>

import os, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0", help="GPU id to use (sets CUDA_VISIBLE_DEVICES)")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import math, json, random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

import pennylane as qml

# ---------------- constants ----------------
RESULTS_DIR = Path("results_faithful_4cls_paper_noiseless_noisy")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
EPS_LIST = [0.1, 0.3, 0.5, 0.7]     # per paper grid

# QNN/encoding
N_QUBITS = 4                         # 4-qubit classifiers (PQC variants below)
LATENT = 8                           # CAE latent dimension
ANGLES_PER_QUBIT = 2                 # 4*2 = 8 angles
AMPLITUDE_QUBITS = 4                 # for ESS with amplitude encoding
SHOTS = 1000
EPOCHS = 30
BATCH = 32
SPSA_LR = 1e-2                       # SPSA step
HEAD_LR = 1e-2                       # Adam on head
CAE_EPOCHS = 10
CAE_BS = 64

# dataset 4-class subset sizes (175 train, 75 test per class → 700/300 total)
TRAIN_PER_CLASS = 175
TEST_PER_CLASS  = 75

MNIST_KEEP = [0,1,2,3]
CIFAR_KEEP = [0,1,2,3]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
torch.backends.cudnn.benchmark = True

# ---------------- data ----------------
def tfm_for(name):
    return transforms.Compose([transforms.ToTensor()])

def load_raw_mnist():
    tfm = tfm_for("mnist")
    tr = datasets.MNIST("./data", train=True,  download=True, transform=tfm)
    te = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    return tr, te, 1, 28  # channels, size

def load_raw_cifar100():
    tfm = tfm_for("cifar100")
    tr = datasets.CIFAR100("./data", train=True,  download=True, transform=tfm)
    te = datasets.CIFAR100("./data", train=False, download=True, transform=tfm)
    return tr, te, 3, 32  # channels, size

def index_subset(ds, keep_classes, per_class):
    counts = {c:0 for c in keep_classes}; idx=[]
    for i in range(len(ds)):
        y = int(ds[i][1])
        if y in keep_classes and counts[y] < per_class:
            idx.append(i); counts[y]+=1
        if all(counts[c] >= per_class for c in keep_classes): break
    return Subset(ds, idx)

def to_xy(sub, keep_classes):
    label_map = {c:i for i,c in enumerate(keep_classes)}
    X, Y = [], []
    for i in range(len(sub)):
        xi, yi = sub[i]
        X.append(xi.numpy()); Y.append(label_map[int(yi)])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.long)
    return X, Y

class SmallCAE(nn.Module):
    """Parameterizes output size: 28 (MNIST) or 32 (CIFAR)."""
    def __init__(self, in_ch=1, latent=8, out_size=28):
        super().__init__()
        assert out_size in (28, 32)
        enc_side = out_size // 4  # after two MaxPools
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,32,3,padding=1),    nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((enc_side, enc_side)), nn.Flatten()
        )
        self.fc_enc = nn.Linear(32*enc_side*enc_side, latent)
        self.fc_dec = nn.Linear(latent, 32*enc_side*enc_side)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (32, enc_side, enc_side)),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(32,32,3,padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(32,in_ch,3,padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.enc(x); h = self.fc_enc(z); z2 = self.fc_dec(h); xhat = self.dec(z2)
        return xhat, h

def get_4class_latents(dataset_name, keep_classes):
    if dataset_name=="mnist":
        tr_raw, te_raw, in_ch, out_size = load_raw_mnist()
    else:
        tr_raw, te_raw, in_ch, out_size = load_raw_cifar100()

    tr_sub = index_subset(tr_raw, keep_classes, TRAIN_PER_CLASS)
    te_sub = index_subset(te_raw, keep_classes, TEST_PER_CLASS)
    Xtr, Ytr = to_xy(tr_sub, keep_classes)
    Xte, Yte = to_xy(te_sub, keep_classes)

    cae = SmallCAE(in_ch=in_ch, latent=LATENT, out_size=out_size).to(DEVICE)
    opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(Xtr, Xtr), batch_size=CAE_BS, shuffle=True)
    cae.train()
    for _ in range(CAE_EPOCHS):
        for xb, x_target in dl:
            xb, x_target = xb.to(DEVICE), x_target.to(DEVICE)
            opt.zero_grad(); xhat, _ = cae(xb); F.mse_loss(xhat, x_target).backward(); opt.step()
    cae.eval()
    with torch.no_grad():
        _, Htr = cae(Xtr.to(DEVICE))
        _, Hte = cae(Xte.to(DEVICE))
    return (Htr.cpu().numpy(), Ytr.numpy()), (Hte.cpu().numpy(), Yte.numpy())

# ---------------- noise & encodings ----------------
def per_gate_noise(p, wires):
    if p > 0:
        for w in wires:
            qml.AmplitudeDamping(p, wires=w)
            qml.DepolarizingChannel(p, wires=w)

def angle_embed_with_noise(x, wires, p, angles_per_qubit=ANGLES_PER_QUBIT):
    d = len(x); it = 0
    for w in wires:
        for _ in range(angles_per_qubit):
            if it >= d: break
            qml.RZ(float(x[it]), w);        per_gate_noise(p, [w]); it += 1
            if it < d:
                qml.RX(float(x[it]), w);    per_gate_noise(p, [w]); it += 1

def amplitude_embed_with_noise(x, wires, p):
    q = len(wires); dim = 2**q
    v = np.zeros(dim, dtype=np.float64)
    vv = np.asarray(x, dtype=np.float64)
    v[:len(vv)] = vv / (np.linalg.norm(vv) + 1e-12)
    qml.AmplitudeEmbedding(v, wires=wires, normalize=False)
    per_gate_noise(p, wires)

# ---------------- PQC variants (PQC-1/6/8) ----------------
def pqc_layer_rot_ent(wires, theta_block, p):
    # single-qubit RY/RZ on all qubits + linear entangling chain (per-gate noise)
    for i, w in enumerate(wires):
        qml.RY(theta_block[i,0], w); per_gate_noise(p, [w])
        qml.RZ(theta_block[i,1], w); per_gate_noise(p, [w])
    for i in range(len(wires)-1):
        qml.CNOT([wires[i], wires[i+1]]); per_gate_noise(p, [wires[i], wires[i+1]])

def build_pqc(shape_name, wires, p):
    """
    PQC-1: 1 layer
    PQC-6: 3 layers + mirrored entanglers (effective depth ~6)
    PQC-8: 4 layers + mirrored entanglers (effective depth ~8)
    """
    n = len(wires)
    if shape_name == "PQC-1":
        L = 1
    elif shape_name == "PQC-6":
        L = 3
    elif shape_name == "PQC-8":
        L = 4
    else:
        raise ValueError("shape_name must be one of PQC-1, PQC-6, PQC-8")
    def apply(theta):
        for l in range(L):
            pqc_layer_rot_ent(wires, theta[l], p)
    return apply, (L, n, 2)

# ---------------- devices (parametrized by noise p) ----------------
def build_devices(n_qubits, p, shots_run=SHOTS, seed=0):
    if p > 0:
        dev_density = qml.device("default.mixed", wires=n_qubits, shots=None, seed=seed)
        dev_run     = qml.device("default.mixed", wires=n_qubits, shots=shots_run, seed=seed)
    else:
        dev_density = qml.device("lightning.qubit", wires=n_qubits, seed=seed)
        dev_run     = qml.device("lightning.qubit", wires=n_qubits, shots=shots_run, seed=seed)
    return dev_density, dev_run

# ---------------- density QNodes (ESS + QUID) ----------------
def make_density_qnode(encoding, seed, p):
    q_for_enc = (AMPLITUDE_QUBITS if encoding=="amplitude" else N_QUBITS)
    dev_density, _ = build_devices(q_for_enc, p, seed=seed)
    wires = list(range(q_for_enc))
    @qml.qnode(dev_density)
    def rho(x):
        if encoding=="angle":
            angle_embed_with_noise(x, wires, p)
        else:
            amplitude_embed_with_noise(x, wires, p)
        return qml.density_matrix(wires=wires)
    return rho

def frob(a, b):
    d = a - b
    return float(np.linalg.norm(d, 'fro'))

def compute_ess_tables(X, Y, encoding, seed=0, p=0.05):
    rho_fn = make_density_qnode(encoding, seed=seed, p=p)
    classes = sorted(np.unique(Y))
    dens_by_cls = {c:[] for c in classes}
    for i in range(len(X)):
        dens_by_cls[int(Y[i])].append(rho_fn(X[i]))
    intr = {}
    for c in classes:
        mats = dens_by_cls[c]
        if len(mats) <= 1:
            intr[c] = 0.0
        else:
            s=0.0; cnt=0
            for i in range(len(mats)):
                for j in range(i+1, len(mats)):
                    s += frob(mats[i], mats[j]); cnt += 1
            intr[c] = s / max(cnt,1)
    cent = {c: np.mean(dens_by_cls[c], axis=0) for c in classes}
    inter = {}
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if j<=i: continue
            inter[(ci,cj)] = frob(cent[ci], cent[cj])
    return intr, inter

# ---------------- QUID (label flip) ----------------
def quid_poison(Xtr, Ytr, eps, encoding, seed, p):
    rng = np.random.default_rng(seed)
    n = len(Ytr); n_poison = int(round(eps*n))
    idx = rng.permutation(n)
    ip, ic = idx[:n_poison], idx[n_poison:]
    rho_fn = make_density_qnode(encoding, seed=seed, p=p)
    classes = sorted(np.unique(Ytr))
    clean_by_cls = {c: [] for c in classes}
    for j in ic:
        clean_by_cls[int(Ytr[j])].append(rho_fn(Xtr[j]))
    flips = {}; Ynew = Ytr.copy()
    for i in ip:
        ri = rho_fn(Xtr[i])
        Dcls = {}
        for c in classes:
            mats = clean_by_cls[c]
            Dcls[c] = float(np.mean([frob(ri, rj) for rj in mats])) if len(mats)>0 else -1e9
        tgt = max(Dcls.items(), key=lambda kv: kv[1])[0]
        if tgt != Ynew[i]:
            flips[int(i)] = {"old": int(Ynew[i]), "new": int(tgt)}
            Ynew[i] = tgt
    return Ynew, flips

# ---------------- QNN + linear head (hybrid) ----------------
class LinearHead(nn.Module):
    def __init__(self, in_feat, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_feat, n_classes)
    def forward(self, x):  # x: [B, in_feat]
        return self.fc(x)

def make_qnode_features(encoding, pqc_name, n_classes, seed, p):
    _, dev_run = build_devices(N_QUBITS, p, SHOTS, seed=seed)
    wires = list(range(N_QUBITS))
    readout_wires = list(range(int(math.ceil(math.log2(n_classes)))))  # 2 for 4 classes
    pqc_apply, shape = build_pqc(pqc_name, wires, p)

    @qml.qnode(dev_run, diff_method="parameter-shift")
    def qnode(x, theta):
        if encoding=="angle":
            angle_embed_with_noise(x, wires, p)
        else:
            amplitude_embed_with_noise(x, wires, p)
        pqc_apply(theta)
        return [qml.expval(qml.PauliZ(w)) for w in readout_wires]

    def init_theta(seed_):
        rng = np.random.default_rng(seed_)
        return 0.01 * rng.standard_normal(size=shape, dtype=np.float64)

    def features(theta, Xb):
        feats = []
        for i in range(len(Xb)):
            feats.append(np.array(qnode(Xb[i], theta), dtype=np.float64))
        return np.stack(feats, axis=0)  # [B, F]
    return qnode, init_theta, features, len(readout_wires)

def spsa_step(cost_fn, theta, lr, rng):
    a = lr
    c = 0.01
    delta = rng.choice([-1.0, 1.0], size=theta.shape)
    theta_plus  = theta + c * delta
    theta_minus = theta - c * delta
    f_plus  = cost_fn(theta_plus)
    f_minus = cost_fn(theta_minus)
    ghat = (f_plus - f_minus) / (2.0 * c * delta)
    return theta - a * ghat

def train_hybrid_qnn(
    Xtr, Ytr_pois, Xte, Yte, n_classes, encoding, pqc_name,
    out_dir, tag, seed, p
):
    # Seeds
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # QNode + features + θ
    qnode, init_theta, get_features, in_feat = make_qnode_features(encoding, pqc_name, n_classes, seed=seed, p=p)
    theta = init_theta(seed)  # seeded init

    # Linear head
    head = LinearHead(in_feat, n_classes).to(DEVICE)
    opt_head = torch.optim.Adam(head.parameters(), lr=HEAD_LR)

    idx = np.arange(len(Ytr_pois))
    n_batches = max(1, len(idx)//BATCH)

    epoch_rows = []
    prev_theta = theta.copy()

    def batch_loss_numpy(theta_used, xb_np, yb_np):
        feats_np = get_features(theta_used, xb_np)
        with torch.no_grad():
            logits = head(torch.tensor(feats_np, dtype=torch.float32, device=DEVICE))
            loss = F.cross_entropy(logits, torch.tensor(yb_np, dtype=torch.long, device=DEVICE))
        return float(loss.item())

    def eval_accuracy_numpy(theta_used, X_np, Y_np):
        feats_np = get_features(theta_used, X_np)
        with torch.no_grad():
            logits = head(torch.tensor(feats_np, dtype=torch.float32, device=DEVICE))
            ypred = logits.argmax(dim=1).cpu().numpy()
        return float((ypred == Y_np).mean()), ypred

    # training
    for ep in range(1, EPOCHS+1):
        rng.shuffle(idx)
        for bi in range(n_batches):
            sel = idx[bi*BATCH:(bi+1)*BATCH]
            xb = Xtr[sel]
            yb = Ytr_pois[sel]

            # 1) SPSA step on quantum params (head frozen)
            def cost_fn_th(thet):
                return batch_loss_numpy(thet, xb, yb)
            theta = spsa_step(cost_fn_th, theta, SPSA_LR, rng)

            # 2) Head step (theta frozen)
            feats_np = get_features(theta, xb)
            feats = torch.tensor(feats_np, dtype=torch.float32, device=DEVICE)
            yb_t   = torch.tensor(yb, dtype=torch.long, device=DEVICE)
            opt_head.zero_grad()
            logits = head(feats)
            loss = F.cross_entropy(logits, yb_t)
            loss.backward()
            opt_head.step()

        tr_acc, _ = eval_accuracy_numpy(theta, Xtr, Ytr_pois)
        delta_theta = float(np.linalg.norm(theta - prev_theta))
        prev_theta = theta.copy()
        epoch_rows.append({"epoch": ep, "train_accuracy": tr_acc, "delta_theta": delta_theta})

    # save epoch log
    pd.DataFrame(epoch_rows).to_csv(out_dir / f"{tag}_epoch_metrics.csv", index=False)

    # final TEST eval
    te_acc, ypred_te = eval_accuracy_numpy(theta, Xte, Yte)
    rep = classification_report(Yte, ypred_te, output_dict=True, zero_division=0)
    acc = float(rep["accuracy"])
    prec = float(rep["macro avg"]["precision"])
    rec  = float(rep["macro avg"]["recall"])
    f1   = float(rep["macro avg"]["f1-score"])
    cm = confusion_matrix(Yte, ypred_te)
    fnr_list, fpr_list = [], []
    for k in range(n_classes):
        tp = cm[k,k]; fn = cm[k,:].sum()-tp; fp = cm[:,k].sum()-tp; tn = cm.sum()-tp-fn-fp
        fnr_list.append(fn/(tp+fn+1e-12)); fpr_list.append(fp/(fp+tn+1e-12))
    fnr, fpr = float(np.mean(fnr_list)), float(np.mean(fpr_list))

    # save artifacts
    pd.DataFrame({"index": np.arange(len(Yte)), "y_true": Yte, "y_pred": ypred_te}).to_csv(
        out_dir / f"preds_{tag}.csv", index=False
    )
    np.save(out_dir / f"qnn_{tag}_theta.npy", theta)
    torch.save(head.state_dict(), out_dir / f"head_{tag}.pt")

    return acc, prec, rec, f1, fnr, fpr

def compute_asr(y_true, y_pred, flips, n_classes):
    flip_counts = defaultdict(lambda: Counter())
    for _,chg in flips.items():
        flip_counts[chg["old"]][chg["new"]] += 1
    if len(flip_counts)==0:
        return {f"{k}->?":0.0 for k in range(n_classes)}, 0.0
    asr_map = {}
    for s in range(n_classes):
        if s not in flip_counts:
            asr_map[f"{s}->?"] = 0.0; continue
        tgt = flip_counts[s].most_common(1)[0][0]
        sel = (y_true==s)
        asr_map[f"{s}->{tgt}"] = float((y_pred[sel]==tgt).mean()) if sel.sum()>0 else 0.0
    macro_asr = float(np.mean(list(asr_map.values()))) if len(asr_map)>0 else 0.0
    return asr_map, macro_asr

# ---------------- runner (param p) ----------------
def run_dataset(dataset_name, keep_classes, encoding_name_for_training, p, regime_name):
    # Seed CAE/ESS once at 0 for determinism across runs
    np.random.seed(0); random.seed(0); torch.manual_seed(0)
    out_root = RESULTS_DIR / f"{regime_name}_p{p}" / f"{dataset_name}_4class_d{LATENT}" / f"trainenc-{encoding_name_for_training}_q{N_QUBITS}"
    out_root.mkdir(parents=True, exist_ok=True)

    (Xtr, Ytr), (Xte, Yte) = get_4class_latents(dataset_name, keep_classes)
    n_classes = 4

    # ESS tables (angle & amplitude), computed once (seed=0) under THIS p
    for enc in ["angle","amplitude"]:
        intr, inter = compute_ess_tables(Xtr, Ytr, enc, seed=0, p=p)
        pd.DataFrame([intr]).to_csv(out_root/f"ESS_intra_{enc}.csv", index=False)
        pd.DataFrame([{"%d-%d"%(a,b):v for (a,b),v in inter.items()}]).to_csv(out_root/f"ESS_inter_{enc}.csv", index=False)

    # Train/eval across seeds, eps, and PQC variants
    all_rows = []
    pqc_list = ["PQC-1","PQC-6","PQC-8"]

    for seed in SEEDS:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        for pqc_name in pqc_list:
            per_run = []
            for eps in EPS_LIST:
                tag = f"{pqc_name}_seed{seed}_eps{eps}"
                # QUID poison in density space (use THIS p)
                Ypoison, flips = quid_poison(Xtr, Ytr.copy(), eps, encoding_name_for_training, seed=seed, p=p)
                (out_root / f"flips_{pqc_name}_e{eps}_seed{seed}.json").write_text(json.dumps(flips, indent=2))

                # hybrid training: angle-encoded QNN + linear head
                acc, prec, rec, f1, fnr, fpr = train_hybrid_qnn(
                    Xtr, Ypoison, Xte, Yte, n_classes,
                    encoding_name_for_training, pqc_name,
                    out_root, tag, seed, p
                )

                # compute ASR on test using flips map
                ytrue_pred_df = pd.read_csv(out_root / f"preds_{tag}.csv")
                y_true = ytrue_pred_df["y_true"].to_numpy()
                y_pred = ytrue_pred_df["y_pred"].to_numpy()
                _, asr_macro = compute_asr(y_true, y_pred, flips, n_classes)

                row = {"regime":regime_name, "p":p, "pqc":pqc_name, "seed":seed,"eps":eps,
                       "accuracy":acc,"precision":prec,"recall":rec,"f1":f1,
                       "fnr":fnr,"fpr":fpr,"asr_macro":asr_macro}
                per_run.append(row); all_rows.append(row)

            pd.DataFrame(per_run).to_csv(out_root / f"run_{pqc_name}_seed{seed}.csv", index=False)

    # summary across seeds & pqc (per eps)
    df = pd.DataFrame(all_rows); summary=[]
    for pqc_name in ["PQC-1","PQC-6","PQC-8"]:
        for eps in EPS_LIST:
            d = df[(df["eps"]==eps) & (df["pqc"]==pqc_name)]
            def ms(k):
                m = float(d[k].mean()) if len(d)>0 else float("nan")
                s = float(d[k].std(ddof=1)) if len(d)>1 else 0.0
                return m, s
            acc_m, acc_s = ms("accuracy")
            pre_m, pre_s = ms("precision")
            rec_m, rec_s = ms("recall")
            f1_m,  f1_s  = ms("f1")
            fnr_m, fnr_s = ms("fnr")
            fpr_m, fpr_s = ms("fpr")
            asr_m, asr_s = ms("asr_macro")
            summary.append({"regime":regime_name, "p":p, "pqc":pqc_name, "eps":eps,
                            "accuracy_mean":acc_m, "accuracy_std":acc_s,
                            "precision_mean":pre_m, "precision_std":pre_s,
                            "recall_mean":rec_m, "recall_std":rec_s,
                            "f1_mean":f1_m, "f1_std":f1_s,
                            "fnr_mean":fnr_m, "fnr_std":fnr_s,
                            "fpr_mean":fpr_m, "fpr_std":fpr_s,
                            "asr_macro_mean":asr_m, "asr_macro_std":asr_s})
    pd.DataFrame(summary).to_csv(out_root / "summary.csv", index=False)
    print(f"[{regime_name}|{dataset_name}] done -> {out_root}")

if __name__ == "__main__":
    # Run BOTH regimes, per the paper
    regimes = [("noiseless", 0.0), ("noisy", 0.05)]
    for regime_name, p in regimes:
        # MNIST (angle-encoded training)
        run_dataset("mnist", MNIST_KEEP, encoding_name_for_training="angle", p=p, regime_name=regime_name)
        # CIFAR-100 (angle-encoded training)
        run_dataset("cifar100", CIFAR_KEEP, encoding_name_for_training="angle", p=p, regime_name=regime_name)

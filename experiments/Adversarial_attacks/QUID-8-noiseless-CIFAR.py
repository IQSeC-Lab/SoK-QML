# qml_quid_repro_pqc6.py
# Reproduce "Adversarial Data Poisoning Attacks on Quantum Machine Learning"
# with Sim-2019 Circuit-6 (CRX) ONLY.
# (Added) Random label-flip baseline; logs attack_type in CSV.
# (Added) Q-Detection with paper-style Q-WAN (Ising/QUBO) that PERSISTS across minibatches/epochs.
#
# NOTE: All prior settings (PQC-6, encodings, noisy/noiseless, hyperparams, QUID) are preserved.
#       Only the defense path is added and toggled via `use_qdetect`.

import os, math, random, sys, csv, datetime, copy as _copy
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

import pennylane as qml
from pennylane import numpy as pnp

import torch
import torch.nn as nn
import torch.optim as torchopt
from torchvision import datasets, transforms

# ---------------------------
# Paths & logging
# ---------------------------
RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)
LOG_CSV = os.path.join(RUN_DIR, "quid_results_pqc6.csv")

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
pnp.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Config
# ---------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    spsa_stepsize: float = 0.01
    spsa_perturb: float = 0.02
    spsa_avg: int = 1
    shots: int = 1000

@dataclass
class EncoderConfig:
    n_qubits: int = 4
    noise_p: float = 0.05
    noisy: bool = False  # toggled in the sweep

# eps sweep
EPS_LIST = [0.0, 0.1, 0.3, 0.5, 0.7]
REPORT_ASR = True




# ---------------------------
# Data: CIFAR-100(0,1,2,3) -> Conv AE -> 8D latent
# ---------------------------
class ConvAE8(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (B, 3, 32, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(True),   # -> (16, 16, 16)
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(True),  # -> (32, 8, 8)
            nn.Flatten(),                                              # -> 32*8*8 = 2048
            nn.Linear(32*8*8, 128), nn.ReLU(True),
            nn.Linear(128, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(True),
            nn.Linear(128, 32*8*8), nn.ReLU(True),
            nn.Unflatten(1, (32,8,8)),
            nn.ConvTranspose2d(32,16,3,stride=2, padding=1, output_padding=1), nn.ReLU(True),  # -> (16,16,16)
            nn.ConvTranspose2d(16,3,3,stride=2, padding=1, output_padding=1),                  # -> (3,32,32)
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, z

def load_cifar100_ae_latent(n_train=700, n_test=300) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tfm = transforms.Compose([transforms.ToTensor()])  # scales to [0,1]
    root = "./data"
    ds  = datasets.CIFAR100(root=root, train=True,  download=True, transform=tfm)
    ds2 = datasets.CIFAR100(root=root, train=False, download=True, transform=tfm)

    def filter_4(d):
        X, Y = [], []
        for x,y in d:
            if y in [0,1,2,3]:
                X.append(x.numpy())
                Y.append(y)
        return np.stack(X, axis=0), np.array(Y)

    X1, y1 = filter_4(ds)
    X2, y2 = filter_4(ds2)

    def stratified_pick(X, y, n_per_class):
        idxs = []
        for c in [0,1,2,3]:
            cand = np.where(y==c)[0]
            rng = np.random.default_rng(SEED + c)
            pick = rng.choice(cand, size=n_per_class, replace=False)
            idxs.extend(pick.tolist())
        idxs = np.array(idxs)
        return X[idxs], y[idxs]

    Xall = np.concatenate([X1, X2], axis=0)
    yall = np.concatenate([y1, y2], axis=0)
    X_all_1000, y_all_1000 = stratified_pick(Xall, yall, n_per_class=250)

    def stratified_split(X, y, n_train_per_class):
        tr_idx, te_idx = [], []
        for c in [0,1,2,3]:
            idx = np.where(y==c)[0]
            rng = np.random.default_rng(SEED+100+c)
            rng.shuffle(idx)
            tr_idx.extend(idx[:n_train_per_class].tolist())
            te_idx.extend(idx[n_train_per_class:].tolist())
        return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]

    Xtr_img, ytr, Xte_img, yte = stratified_split(X_all_1000, y_all_1000, n_train_per_class=175)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = ConvAE8().to(device)
    opt = torchopt.Adam(ae.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    Xtr_t = torch.tensor(Xtr_img, dtype=torch.float32).to(device)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr_t, Xtr_t),
                                         batch_size=128, shuffle=True, drop_last=False)
    ae.train()
    for _ in range(10):
        for xb, yb in loader:
            xr, z = ae(xb)
            loss = bce(xr, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    ae.eval()
    with torch.no_grad():
        def to_lat8(Ximg):
            Xt = torch.tensor(Ximg, dtype=torch.float32).to(device)
            _, z = ae(Xt)
            return z.cpu().numpy()
        Xtr_lat8 = to_lat8(Xtr_img)
        Xte_lat8 = to_lat8(Xte_img)

    # Scale features for angle encoding (to roughly [-pi, pi])
    mu = Xtr_lat8.mean(axis=0)
    sigma = Xtr_lat8.std(axis=0) + 1e-8
    Xtr_lat8 = (Xtr_lat8 - mu) / sigma
    Xte_lat8 = (Xte_lat8 - mu) / sigma

    return Xtr_lat8, ytr, Xte_lat8, yte



# ---------------------------
# Devices
# ---------------------------
def make_train_device(n_qubits: int, noisy: bool, shots: int):
    if noisy:
        return qml.device("default.mixed", wires=n_qubits, shots=shots)
    else:
        return qml.device("lightning.qubit", wires=n_qubits, shots=None)

def make_ess_device(n_qubits: int):
    return qml.device("lightning.qubit", wires=n_qubits, shots=None)

# ---------------------------
# Noise after each gate (noisy branch)
# ---------------------------
def amp_depol(noisy: bool, p: float, wires):
    if not noisy:
        return
    ws = wires if isinstance(wires, (list, tuple)) else [wires]
    for w in ws:
        qml.AmplitudeDamping(p, wires=w)
        qml.DepolarizingChannel(p, wires=w)

# ---------------------------
# Angle encoding block
# ---------------------------
def _scale_to_pi(x):
    return pnp.pi * pnp.tanh(x)

def angle_encode_once(x8: pnp.ndarray, cfg: EncoderConfig):
    assert len(x8) == 2 * cfg.n_qubits
    x = _scale_to_pi(x8)
    for q in range(cfg.n_qubits):
        a, b = x[2*q], x[2*q+1]
        qml.RZ(a,   wires=q); amp_depol(cfg.noisy, cfg.noise_p, q)
        qml.RX(b,   wires=q); amp_depol(cfg.noisy, cfg.noise_p, q)
        qml.RZ(a/2, wires=q); amp_depol(cfg.noisy, cfg.noise_p, q)
        qml.RX(b/2, wires=q); amp_depol(cfg.noisy, cfg.noise_p, q)

# ---------------------------
# Sim-2019 Circuit-6 variational layer (CRX, all-to-all)
# ---------------------------
def pqc6_layer(theta_L, wires, cfg):
    n = len(wires)
    loc = theta_L["locals"]
    A   = theta_L["crx_matrix"]
    for q in range(n):
        qml.RY(loc[q,0], wires=wires[q]); amp_depol(cfg.noisy, cfg.noise_p, wires[q])
        qml.RZ(loc[q,1], wires=wires[q]); amp_depol(cfg.noisy, cfg.noise_p, wires[q])
        qml.RX(loc[q,2], wires=wires[q]); amp_depol(cfg.noisy, cfg.noise_p, wires[q])
        qml.RZ(loc[q,3], wires=wires[q]); amp_depol(cfg.noisy, cfg.noise_p, wires[q])
    for i in range(n):
        for j in range(i+1, n):
            qml.CRX(A[i,j], wires=[wires[i], wires[j]])
            amp_depol(cfg.noisy, cfg.noise_p, [wires[i], wires[j]])

def build_circuit_pqc6(theta, x8, cfg, n_layers: int = 6):
    wires = list(range(cfg.n_qubits))
    angle_encode_once(x8, cfg)
    for l in range(n_layers):
        pqc6_layer(theta[l], wires, cfg)

# ---------------------------
# QNodes
# ---------------------------
def make_variational_qnode_noiseless(train_dev, cfg: EncoderConfig):
    @qml.qnode(train_dev, interface="autograd", diff_method="adjoint")
    def qnode(theta, x8):
        build_circuit_pqc6(theta, x8, cfg, n_layers=6)
        return [qml.expval(qml.PauliZ(w)) for w in range(cfg.n_qubits)]
    return qnode

def make_variational_qnode_noisy(train_dev, cfg: EncoderConfig):
    @qml.qnode(train_dev, interface="auto", diff_method=None)
    def qnode(theta, x8):
        build_circuit_pqc6(theta, x8, cfg, n_layers=6)
        return [qml.expval(qml.PauliZ(w)) for w in range(cfg.n_qubits)]
    return qnode

def make_rho_qnode(ess_dev, cfg: EncoderConfig):
    @qml.qnode(ess_dev, interface=None, diff_method=None)
    def rho_x(x8):
        saved = cfg.noisy
        cfg.noisy = False
        angle_encode_once(x8, cfg)
        cfg.noisy = saved
        return qml.density_matrix(wires=list(range(cfg.n_qubits)))
    return rho_x

# ---------------------------
# Pytree helpers
# ---------------------------
def tree_map(fn, tree):
    if isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(fn, t) for t in tree)
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    return fn(tree)

def tree_zip_map(fn, a, b):
    if isinstance(a, (list, tuple)):
        return type(a)(tree_zip_map(fn, x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return {k: tree_zip_map(fn, a[k], b[k]) for k in a}
    return fn(a, b)

def tree_zeros_like(tree): return tree_map(lambda x: pnp.zeros_like(x), tree)
def tree_random_sign_like(tree): return tree_map(lambda x: pnp.sign(pnp.random.uniform(-1, 1, size=x.shape)), tree)
def tree_add_scaled(a, b, scale): return tree_zip_map(lambda x, y: x + scale * y, a, b)

# ---------------------------
# Noiseless optimizer (Adam)
# ---------------------------
class AdamNoiseless:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = None
        self.v = None
        self.t = 0
    def step(self, params, grads):
        if self.m is None:
            self.m, self.v = tree_zeros_like(params), tree_zeros_like(params)
        self.t += 1
        self.m = tree_zip_map(lambda m,g: self.b1*m + (1-self.b1)*g, self.m, grads)
        self.v = tree_zip_map(lambda v,g: self.b2*v + (1-self.b2)*(g*g), self.v, grads)
        mhat = tree_map(lambda m: m/(1-self.b1**self.t), self.m)
        vhat = tree_map(lambda v: v/(1-self.b2**self.t), self.v)
        upd  = tree_zip_map(lambda m,v: self.lr * m / (pnp.sqrt(v) + self.eps), mhat, vhat)
        return tree_zip_map(lambda p,u: p - u, params, upd)

def logits_from_embed(embed: pnp.ndarray, W: pnp.ndarray, b: pnp.ndarray):
    return embed @ W + b

def xent_loss(logits: pnp.ndarray, y: np.ndarray):
    m = logits - pnp.max(logits, axis=1, keepdims=True)
    ex = pnp.exp(m)
    logp = m - pnp.log(pnp.sum(ex, axis=1, keepdims=True))
    n = logits.shape[0]
    return -pnp.mean(logp[pnp.arange(n), y])

def predict(logits):
    return pnp.argmax(logits, axis=1)

def make_noiseless_loss(qnode):
    def loss_fn(theta, W, b, Xb, yb):
        E = pnp.stack([qnode(theta, Xb[i]) for i in range(len(yb))], axis=0)
        lg = logits_from_embed(E, W, b)
        return xent_loss(lg, yb)
    return loss_fn

def train_epoch_noiseless(params, Xtr8, ytr, qnode, tr: TrainConfig, opt: AdamNoiseless):
    theta, W, b = params
    loss_fn = make_noiseless_loss(qnode)
    grad_fn = qml.grad(lambda th, Wh, bh, Xb, yb: loss_fn(th, Wh, bh, Xb, yb), argnum=[0,1,2])
    losses = []
    for Xb, yb, _ in make_minibatches(Xtr8, ytr, tr.batch_size, shuffle=True):
        g_theta, g_W, g_b = grad_fn(theta, W, b, Xb, yb)
        theta, W, b = opt.step([theta, W, b], [g_theta, g_W, g_b])
        losses.append(float(loss_fn(theta, W, b, Xb, yb)))
    return [theta, W, b], float(np.mean(losses))

# ---------------------------
# Noisy optimizer (SPSA)
# ---------------------------
class SPSAWrapper:
    def __init__(self, qnode, cfg: EncoderConfig, tr: TrainConfig):
        self.qnode, self.cfg, self.tr = qnode, cfg, tr
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.lr = tr.spsa_stepsize
        self.m = None
        self.v = None
        self.t = 0

    def _adam_update(self, params, grads):
        if self.m is None:
            self.m = tree_zeros_like(params)
            self.v = tree_zeros_like(params)
            self.t = 0
        self.t += 1
        self.m = tree_zip_map(lambda m, g: self.beta1 * m + (1 - self.beta1) * g, self.m, grads)
        self.v = tree_zip_map(lambda v, g: self.beta2 * v + (1 - self.beta2) * (g * g), self.v, grads)
        mhat = tree_map(lambda m: m / (1 - self.beta1 ** self.t), self.m)
        vhat = tree_map(lambda v: v / (1 - self.beta2 ** self.t), self.v)
        step = tree_zip_map(lambda m, v: self.lr * m / (pnp.sqrt(v) + self.eps), mhat, vhat)
        return tree_zip_map(lambda p, s: p - s, params, step)

    def _eval_loss(self, th, Wh, bh, Xb, yb):
        E = pnp.stack([self.qnode(th, Xb[i]) for i in range(len(yb))], axis=0)
        return xent_loss(logits_from_embed(E, Wh, bh), yb)

    def _eval_loss_weighted(self, th, Wh, bh, Xb, yb, weights_np):
        E = pnp.stack([self.qnode(th, Xb[i]) for i in range(len(yb))], axis=0)
        lg = logits_from_embed(E, Wh, bh)
        m = lg - pnp.max(lg, axis=1, keepdims=True)
        ex = pnp.exp(m); logp = m - pnp.log(pnp.sum(ex, axis=1, keepdims=True))
        per = -logp[pnp.arange(len(yb)), yb]
        w = pnp.array(weights_np).flatten() + 1e-12
        return pnp.sum(w * per) / pnp.sum(w)

    def step(self, params, Xb, yb):
        theta, W, b = params
        c = self.tr.spsa_perturb
        grads = []
        obj = 0.0
        for _ in range(self.tr.spsa_avg):
            S_theta = tree_random_sign_like(theta)
            S_W     = pnp.sign(pnp.random.uniform(-1, 1, size=W.shape))
            S_b     = pnp.sign(pnp.random.uniform(-1, 1, size=b.shape))
            theta_p = tree_add_scaled(theta, S_theta, +c)
            theta_m = tree_add_scaled(theta, S_theta, -c)
            W_p, W_m = W + c * S_W, W - c * S_W
            b_p, b_m = b + c * S_b, b - c * S_b
            lp = self._eval_loss(theta_p, W_p, b_p, Xb, yb)
            lm = self._eval_loss(theta_m, W_m, b_m, Xb, yb)
            scale = (lp - lm) / (2 * c)
            g_theta = tree_map(lambda s: scale * s, S_theta)
            g_W     = scale * S_W
            g_b     = scale * S_b
            if not grads:
                grads = [g_theta, g_W, g_b]
            else:
                grads[0] = tree_zip_map(lambda a, b_: a + b_, grads[0], g_theta)
                grads[1] = grads[1] + g_W
                grads[2] = grads[2] + g_b
            obj += 0.5 * (lp + lm)
        grads[0] = tree_map(lambda x: x / self.tr.spsa_avg, grads[0])
        grads[1] = grads[1] / self.tr.spsa_avg
        grads[2] = grads[2] / self.tr.spsa_avg
        new_params = self._adam_update([theta, W, b], grads)
        return new_params, float(obj / self.tr.spsa_avg)

    def step_weighted(self, params, Xb, yb, weights_np):
        theta, W, b = params
        c = self.tr.spsa_perturb
        grads = []
        obj = 0.0
        for _ in range(self.tr.spsa_avg):
            S_theta = tree_random_sign_like(theta)
            S_W     = pnp.sign(pnp.random.uniform(-1, 1, size=W.shape))
            S_b     = pnp.sign(pnp.random.uniform(-1, 1, size=b.shape))
            theta_p = tree_add_scaled(theta, S_theta, +c)
            theta_m = tree_add_scaled(theta, S_theta, -c)
            W_p, W_m = W + c * S_W, W - c * S_W
            b_p, b_m = b + c * S_b, b - c * S_b
            lp = self._eval_loss_weighted(theta_p, W_p, b_p, Xb, yb, weights_np)
            lm = self._eval_loss_weighted(theta_m, W_m, b_m, Xb, yb, weights_np)
            scale = (lp - lm) / (2 * c)
            g_theta = tree_map(lambda s: scale * s, S_theta)
            g_W     = scale * S_W
            g_b     = scale * S_b
            if not grads:
                grads = [g_theta, g_W, g_b]
            else:
                grads[0] = tree_zip_map(lambda a, b_: a + b_, grads[0], g_theta)
                grads[1] = grads[1] + g_W
                grads[2] = grads[2] + g_b
            obj += 0.5 * (lp + lm)
        grads[0] = tree_map(lambda x: x / self.tr.spsa_avg, grads[0])
        grads[1] = grads[1] / self.tr.spsa_avg
        grads[2] = grads[2] / self.tr.spsa_avg
        new_params = self._adam_update([theta, W, b], grads)
        return new_params, float(obj / self.tr.spsa_avg)

# ---------------------------
# Minibatches
# ---------------------------
def make_minibatches(X, y, bs, shuffle=True):
    idx = np.arange(len(y))
    if shuffle:
        np.random.default_rng(SEED+123).shuffle(idx)
    for s in range(0, len(y), bs):
        j = idx[s:s+bs]
        yield X[j], y[j], j  # also yield batch indices into the global training set

# ---------------------------
# QUID: ESS (Frobenius) label flipping
# ---------------------------
def frob(A, B):
    D = A - B
    return pnp.sqrt(pnp.sum(pnp.real(D * pnp.conj(D))))

def class_prototypes(rhos: List[pnp.ndarray], ys: np.ndarray, n_classes=4):
    protos = []
    for c in range(n_classes):
        idx = np.where(ys==c)[0]
        mats = [rhos[i] for i in idx]
        protos.append(pnp.mean(pnp.stack(mats, axis=0), axis=0))
    return protos

def make_rho_encoder(ess_dev, cfg: EncoderConfig):
    @qml.qnode(ess_dev, interface=None, diff_method=None)
    def rho_x(x8):
        saved = cfg.noisy
        cfg.noisy = False
        angle_encode_once(x8, cfg)
        cfg.noisy = saved
        return qml.density_matrix(wires=list(range(cfg.n_qubits)))
    return rho_x

def quid_label_flip(Xtr8, ytr, rho_x_fun, epsilon: float, seed=SEED):
    if epsilon <= 0.0:
        return ytr.copy(), np.array([], dtype=int), None
    rng = np.random.default_rng(seed)
    n = len(ytr)
    k = int(round(epsilon * n))
    all_idx = np.arange(n)
    poisoned_idx = rng.choice(all_idx, size=k, replace=False)
    clean_idx = np.setdiff1d(all_idx, poisoned_idx)
    rhos_clean = [pnp.array(rho_x_fun(Xtr8[i])) for i in clean_idx]
    protos = class_prototypes(rhos_clean, ytr[clean_idx], n_classes=4)
    y_poison = ytr.copy()
    for i in poisoned_idx:
        rho_i = pnp.array(rho_x_fun(Xtr8[i]))
        dists = [frob(rho_i, protos[c]) for c in range(4)]
        tgt = int(pnp.argmax(pnp.stack(dists)))
        y_poison[i] = tgt
    return y_poison, poisoned_idx, protos

# ---------------------------
# RANDOM label flip baseline
# ---------------------------
def random_label_flip(Xtr8, ytr, epsilon: float, seed=SEED, n_classes=4):
    if epsilon <= 0.0:
        return ytr.copy(), np.array([], dtype=int), None
    rng = np.random.default_rng(seed + 999)
    n = len(ytr)
    k = int(round(epsilon * n))
    all_idx = np.arange(n)
    poisoned_idx = rng.choice(all_idx, size=k, replace=False)
    y_poison = ytr.copy()
    for i in poisoned_idx:
        choices = [c for c in range(n_classes) if c != int(ytr[i])]
        y_poison[i] = int(rng.choice(choices))
    return y_poison, poisoned_idx, None

# ---------------------------
# Inference & metrics
# ---------------------------
def infer(params, qnode, X):
    theta, W, b = params
    E = pnp.stack([qnode(theta, X[i]) for i in range(len(X))], axis=0)
    logits = logits_from_embed(E, W, b)
    preds  = predict(logits)
    return np.array(preds), np.array(logits, dtype=float)

def accuracy(yhat, y):
    return 100.0 * (yhat == y).mean()

def _append_csv(row_dict: dict, path: str = LOG_CSV):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()

# ============================================================
# Q-DETECTION — Paper-style Q-WAN BANK (persistent across minibatches)
# ============================================================

class QWANIsingBank:
    """
    Persistent Q-WAN over the entire training set (N spins).
      - Ising energy:  E(s) = -Σ_i h_i s_i - 1/2 Σ_{i!=j} J_ij s_i s_j
      - Guided:        E'(s)= E(s) + α/2 Σ_{i in batch} L_i (1 + s_i)
      - Update rule on the *batch subgraph* (indices idx):
           ΔJ_ij = -η ( <s_i s_j>_guided - <s_i s_j>_free ),  for i,j in idx
           Δh_i  = -η ( <s_i>_guided      - <s_i>_free ),      for i in idx
      - We simulate sampling with simple Metropolis simulated annealing (SA).
    """
    def __init__(self,
                 n_spins_total: int,
                 eta: float = 0.05,
                 alpha: float = 1.0,
                 beta_start: float = 0.1,
                 beta_end: float = 2.0,
                 sa_sweeps: int = 50,
                 seed: int = SEED):
        self.N = n_spins_total
        self.eta = eta
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.sa_sweeps = sa_sweeps
        self.rng = np.random.default_rng(seed)
        # persistent parameters
        self.h = np.zeros((self.N,), dtype=np.float64)
        self.J = np.zeros((self.N, self.N), dtype=np.float64)  # symmetric, zero diag

    def _beta_schedule(self, steps=30):
        return np.linspace(self.beta_start, self.beta_end, steps)

    def _energy_delta(self, s, i, new_si, h, J):
        delta = new_si - s[i]
        return -(h[i] * delta + np.dot(J[i], s) * delta)

    def _sa_sample_subgraph(self, idx, h_sub, J_sub, beta_schedule, init=None):
        """
        SA restricted to the subgraph defined by idx (batch indices).
        h_sub, J_sub correspond to idx order.
        """
        B = len(idx)
        s = init.copy() if init is not None else self.rng.choice([-1, 1], size=B)
        samples = []
        for k, beta in enumerate(beta_schedule):
            for _ in range(self.sa_sweeps):
                i = self.rng.integers(0, B)
                new_si = -s[i]
                # neighbor contributions via subgraph J_sub and full spin vector s (batch only)
                delta = new_si - s[i]
                dE = -(h_sub[i] * delta + np.dot(J_sub[i], s) * delta)
                if dE <= 0 or self.rng.random() < math.exp(-beta * dE):
                    s[i] = new_si
            if k >= int(0.7 * len(beta_schedule)):
                samples.append(s.copy())
        return s, samples

    def _expectations(self, samples):
        if len(samples) == 0:
            return None, None
        X = np.stack(samples, axis=0)   # (S, B)
        m = X.mean(axis=0)              # (B,)
        C = (X.T @ X) / X.shape[0]      # (B,B)
        return m, C

    def train_step(self, idx_batch: np.ndarray, per_sample_losses_batch: np.ndarray):
        """
        Adversarial Filtering update on the batch subgraph.
        Applies correlation-difference update to J,h for those indices.
        """
        idx = np.asarray(idx_batch, dtype=int)
        L = np.asarray(per_sample_losses_batch, dtype=np.float64)
        # normalize losses to stabilize guidance
        Ln = (L - L.mean()) / (L.std() + 1e-8)

        # extract sub-parameters
        h_sub = self.h[idx].copy()
        J_sub = self.J[np.ix_(idx, idx)].copy()
        np.fill_diagonal(J_sub, 0.0)

        betas = self._beta_schedule(steps=30)

        # FREE phase on subgraph
        _, free_samples = self._sa_sample_subgraph(idx, h_sub, J_sub, betas, init=None)
        m_free, C_free = self._expectations(free_samples)
        if m_free is None:
            # initialize a tiny jitter to avoid degenerate start
            m_free = np.zeros_like(h_sub)
            C_free = np.zeros_like(J_sub)

        # GUIDED phase: effective field shift for batch indices only
        h_eff = h_sub - 0.5 * self.alpha * Ln
        _, guided_samples = self._sa_sample_subgraph(idx, h_eff, J_sub, betas, init=None)
        m_guided, C_guided = self._expectations(guided_samples)
        if m_guided is None:
            m_guided = np.zeros_like(h_sub)
            C_guided = np.zeros_like(J_sub)

        # parameter updates on subgraph
        dC = C_guided - C_free
        dm = m_guided - m_free
        dC = 0.5 * (dC + dC.T)
        np.fill_diagonal(dC, 0.0)

        J_sub_new = J_sub - self.eta * dC
        h_sub_new = h_sub - self.eta * dm

        # write back to global J,h
        self.h[idx] = h_sub_new
        self.J[np.ix_(idx, idx)] = J_sub_new
        # keep symmetry and zero diag
        self.J = 0.5 * (self.J + self.J.T)
        np.fill_diagonal(self.J, 0.0)

    def weights(self, idx_batch: np.ndarray, per_sample_losses_batch: np.ndarray):
        """
        Produce v in [0,1] for the current batch via guided sampling on subgraph.
        """
        idx = np.asarray(idx_batch, dtype=int)
        L = np.asarray(per_sample_losses_batch, dtype=np.float64)
        Ln = (L - L.mean()) / (L.std() + 1e-8)

        h_sub = self.h[idx].copy()
        J_sub = self.J[np.ix_(idx, idx)].copy()
        np.fill_diagonal(J_sub, 0.0)

        betas = self._beta_schedule(steps=30)
        h_eff = h_sub - 0.5 * self.alpha * Ln
        _, guided_samples = self._sa_sample_subgraph(idx, h_eff, J_sub, betas, init=None)

        if len(guided_samples) == 0:
            s = np.sign(h_eff + 1e-6)
            v = (s + 1.0) / 2.0
            return np.clip(v, 0.0, 1.0)

        m_guided, _ = self._expectations(guided_samples)
        v = (m_guided + 1.0) / 2.0
        return np.clip(v, 0.0, 1.0)

# ---------------------------
# Q-Detection helpers
# ---------------------------
def compute_per_sample_losses(qnode, params, Xb, yb):
    theta, W, b = params
    E = pnp.stack([qnode(theta, Xb[i]) for i in range(len(yb))], axis=0)
    logits = logits_from_embed(E, W, b)
    m = logits - pnp.max(logits, axis=1, keepdims=True)
    ex = pnp.exp(m); logp = m - pnp.log(pnp.sum(ex, axis=1, keepdims=True))
    per = -logp[pnp.arange(len(yb)), yb]
    return np.asarray(per)

def weighted_loss_mean(qnode, params, Xb, yb, weights_np):
    theta, W, b = params
    E = pnp.stack([qnode(theta, Xb[i]) for i in range(len(yb))], axis=0)
    logits = logits_from_embed(E, W, b)
    m = logits - pnp.max(logits, axis=1, keepdims=True)
    ex = pnp.exp(m); logp = m - pnp.log(pnp.sum(ex, axis=1, keepdims=True))
    per = -logp[pnp.arange(len(yb)), yb]
    w = pnp.array(weights_np).flatten() + 1e-12
    return pnp.sum(w * per) / pnp.sum(w)


def train_epoch_noiseless_qd(params, Xtr8, ytr, qnode, tr, opt, qwan_bank):
    theta, W, b = params
    losses_epoch = []
    # was: for Xb, yb, _ in make_minibatches(...):
    for Xb, yb, idxb in make_minibatches(Xtr8, ytr, tr.batch_size, shuffle=True):
        per_losses = compute_per_sample_losses(qnode, [theta, W, b], Xb, yb)
        for _ in range(2):
            qwan_bank.train_step(idxb, per_losses)
        v = qwan_bank.weights(idxb, per_losses)

        wl_fn = lambda th, Wh, bh: weighted_loss_mean(qnode, [th, Wh, bh], Xb, yb, v)
        grad_fn = qml.grad(wl_fn, argnum=[0,1,2])
        g_theta, g_W, g_b = grad_fn(theta, W, b)
        theta, W, b = opt.step([theta, W, b], [g_theta, g_W, g_b])

        losses_epoch.append(float(xent_loss(
            logits_from_embed(pnp.stack([qnode(theta, Xb[i]) for i in range(len(yb))], axis=0), W, b), yb
        )))
    return [theta, W, b], float(np.mean(losses_epoch))



# ---------------------------
# Runner (PQC-6) with/without Q-Detection
# ---------------------------
def init_params_pqc6(n_qubits: int, n_layers: int = 6, n_classes: int = 4):
    theta = []
    for _ in range(n_layers):
        locals_ = pnp.random.uniform(-0.1, 0.1, size=(n_qubits, 4))
        M = pnp.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                M[i, j] = np.random.uniform(-0.1, 0.1)
        theta.append({"locals": locals_, "crx_matrix": M})
    W = pnp.random.uniform(-0.1, 0.1, size=(n_qubits, n_classes))
    b = pnp.zeros((n_classes,))
    return theta, W, b

def run_one_setting_pqc6(noisy: bool, eps_list=EPS_LIST):
    print("\n==============================================")
    print(f"PQC6 | n=4 | encoding=RZ-RX-RZ-RX | noise={'(damp=0.05,depol=0.05)' if noisy else '(none)'}")
    print("==============================================")

    Xtr8, ytr, Xte8, yte = load_cifar100_ae_latent(n_train=700, n_test=300)

    cfg = EncoderConfig(n_qubits=4, noise_p=0.05, noisy=noisy)
    tr  = TrainConfig(epochs=30, batch_size=32, spsa_stepsize=0.01, spsa_perturb=0.02, spsa_avg=1, shots=1000)

    train_dev = make_train_device(cfg.n_qubits, noisy=cfg.noisy, shots=tr.shots)
    qnode = make_variational_qnode_noisy(train_dev, cfg) if noisy else make_variational_qnode_noiseless(train_dev, cfg)

    ess_dev = make_ess_device(cfg.n_qubits)
    rho_x   = make_rho_qnode(ess_dev, cfg)

    theta0, W0, b0 = init_params_pqc6(cfg.n_qubits, n_layers=6, n_classes=4)

    for eps in eps_list:
        for attack_type in ["quid", "random"]:
            if attack_type == "quid":
                ytr_poison, poisoned_idx, _ = quid_label_flip(Xtr8, ytr, rho_x, epsilon=eps, seed=SEED)
            else:
                ytr_poison, poisoned_idx, _ = random_label_flip(Xtr8, ytr, epsilon=eps, seed=SEED)

            for use_qdetect in [False, True]:
                theta = _copy.deepcopy(theta0); W = pnp.array(W0); b = pnp.array(b0)
                params = [theta, W, b]
                defense_name = "q-detection" if use_qdetect else "none"
                print(f"\n[attack={attack_type}] eps={eps} | defense={defense_name} | poisoned={len(poisoned_idx)}")

                if noisy:
                    spsa = SPSAWrapper(qnode, cfg, tr)
                    if use_qdetect:
                        # persistent Q-WAN over entire training set
                        qwan_bank = QWANIsingBank(n_spins_total=len(ytr_poison), eta=0.05, alpha=1.0)
                        for _ in range(tr.epochs):
                            for Xb, yb, idxb in make_minibatches(Xtr8, ytr_poison, tr.batch_size, shuffle=True):
                                # AF: update Q-WAN on this batch
                                per_losses = compute_per_sample_losses(qnode, params, Xb, yb)
                                for _inner in range(2):
                                    qwan_bank.train_step(idxb, per_losses)
                                v = qwan_bank.weights(idxb, per_losses)
                                # AU: weighted SPSA on real params
                                params, _ = spsa.step_weighted(params, Xb, yb, v)
                    else:
                        for _ in range(tr.epochs):
                            params, _ = spsa.step(params, Xtr8, ytr_poison)

                else:
                    opt = AdamNoiseless(lr=0.01)
                    if use_qdetect:
                        qwan_bank = QWANIsingBank(n_spins_total=len(ytr_poison), eta=0.05, alpha=1.0)
                        for _ in range(tr.epochs):
                            params, _ = train_epoch_noiseless_qd(params, Xtr8, ytr_poison, qnode, tr, opt, qwan_bank)
                    else:
                        for _ in range(tr.epochs):
                            params, _ = train_epoch_noiseless(params, Xtr8, ytr_poison, qnode, tr, opt)

                # Evaluate on test
                yhat, _ = infer(params, qnode, Xte8)
                acc = accuracy(yhat, yte)

                # ASR on poisoned training points
                if REPORT_ASR and len(poisoned_idx) > 0:
                    yhat_tr, _ = infer(params, qnode, Xtr8[poisoned_idx])
                    asr = 100.0 * np.mean(yhat_tr == ytr_poison[poisoned_idx])
                else:
                    asr = 0.0 if eps == 0.0 else float("nan")

                print(f"[{attack_type} eps={eps} | {defense_name}] Test Acc = {acc:6.2f}% | ASR = {asr:6.2f}%")

                _append_csv({
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "which_pqc": "pqc6",
                    "attack_type": attack_type,
                    "defense": defense_name,
                    "noisy": noisy,
                    "n_qubits": cfg.n_qubits,
                    "shots": tr.shots if noisy else 0,
                    "epochs": tr.epochs,
                    "batch_size": tr.batch_size,
                    "epsilon": eps,
                    "poisoned_count": int(len(poisoned_idx)),
                    "test_acc_pct": float(f"{acc:.4f}"),
                    "asr_pct": float(f"{asr:.4f}") if REPORT_ASR else float("nan"),
                    "seed": int(SEED),
                })

def main():
    # NOISELESS-ONLY split of QUID-8.py
    run_one_setting_pqc6(noisy=False, eps_list=EPS_LIST)

if __name__ == "__main__":
    main()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pennylane as qml
import random
from pennylane import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer
import seaborn as sns
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Parameters
bsz = 64
n_qubits = 9
n_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filename_prefix = "qmlp-mnist-labelflip"  # adjust base name if needed
noise_afterfix = f"depo-LF{n_qubits}-{n_layers}"
p = 0.01  # depolarizing noise level

# Depolarizing noise model
noise_model = NoiseModel(basis_gates=['id', 'rz', 'sx', 'cx', 'x'])
depol_1q = depolarizing_error(p, 1)
depol_2q = depolarizing_error(p, 2)
noise_model.add_all_qubit_quantum_error(depol_1q, ['id', 'rz', 'sx', 'x'])
noise_model.add_all_qubit_quantum_error(depol_2q, ['cx'])

# Noisy quantum device
dev = qml.device('qiskit.aer', wires=n_qubits, backend="aer_simulator", noise_model=noise_model)

# Weight shapes
def modular_w(n_qubits, n_layers):
    return {f"rot_layer_{n}": (n_qubits, 3) for n in range(n_layers)} | \
           {f"crx_layer_{n}": (n_qubits, 1) for n in range(n_layers)}

weight_shapes = modular_w(n_qubits, n_layers)

@qml.qnode(dev, interface="torch")
def qnode(inputs, **weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    for n in range(n_layers):
        for i in range(n_qubits):
            qml.Rot(*weights[f"rot_layer_{n}"][i], wires=i)
        for i in range(n_qubits):
            qml.CRX(weights[f"crx_layer_{n}"][i][0], wires=[i, (i+1)%n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qlayer = TorchLayer(qnode, weight_shapes)

class QMLP_AZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        self.fc = nn.Linear(n_qubits, 23)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        out = self.qlayer(x)
        out = self.fc(out.to(x.device))
        return F.log_softmax(out, dim=1)

def evaluate(model, device, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    fnr_list, fpr_list = [], []
    for i in range(23):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        fnr_list.append(fn / (fn + tp) if (fn + tp) else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) else 0)
    return acc, prec, f1, cm, fnr_list, fpr_list, np.mean(fnr_list), np.mean(fpr_list)

# Load and preprocess AZ test data
az_test = np.load('./AZ_23/AZ-Class-Task_23_families_test.npz')
X_test = az_test['X_test'].astype(np.float32)
y_test = az_test['Y_test'].astype(np.int64)

scaler = MinMaxScaler()
X_test = scaler.fit_transform(X_test)
pca = PCA(n_components=512)
X_test = pca.fit_transform(X_test)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz)


# Run 3 evaluations under depolarizing noise
for run_id in range(1, 4):
    print(f"\n========= Noise Evaluation RUN {run_id} =========")

    # Set different seeds
    torch.manual_seed(42 + run_id)
    np.random.seed(42 + run_id)
    random.seed(42 + run_id)

    # Reload model (to ensure same starting point for each noise run)
    model = QMLP_AZ().to(device)
    model.load_state_dict(torch.load("qmlp-mnist-labelflip_best_run.pt", map_location=device))
    model.eval()

    # Evaluate under noise
    acc, prec, f1, cm, fnr_list, fpr_list, avg_fnr, avg_fpr = evaluate(model, device, test_loader)

    # Save results
    run_filename_prefix = f"{filename_prefix}_{noise_afterfix}_run{run_id}"
    with open(f"{run_filename_prefix}_eval_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Accuracy", "Precision", "F1", "Avg_FNR", "Avg_FPR"])
        writer.writerow([acc, prec, f1, avg_fnr, avg_fpr])
        writer.writerow([])
        writer.writerow(["Class", "FNR", "FPR"])
        for i in range(23):
            writer.writerow([i, fnr_list[i], fpr_list[i]])

    # Save confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix with Depolarizing Noise (p={p}) - Run {run_id}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{run_filename_prefix}_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"Run {run_id} | Acc: {acc:.4f}, F1: {f1:.4f}, Avg FNR: {avg_fnr:.4f}, Avg FPR: {avg_fpr:.4f}")


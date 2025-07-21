import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pennylane as qml
import random
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer
import torch.optim as optim
import seaborn as sns
import idx2numpy

# Parameters
bsz = 64
epochs = 30
lr = 0.001
w_decay = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filename_prefix = "qmlp-mnist-labelflip-random"
best_model_prefix = filename_prefix
n_qubits = 9
n_layers = 10
n_random_encodings = 5

# Quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Label flipping function
def flip_labels(y, num_classes, flip_fraction=0.5, seed=42):
    np.random.seed(seed)
    y_flipped = y.copy()
    n_samples = len(y)
    n_flip = int(flip_fraction * n_samples)
    flip_indices = np.random.choice(n_samples, n_flip, replace=False)
    for idx in flip_indices:
        original = y_flipped[idx]
        y_flipped[idx] = np.random.choice([l for l in range(num_classes) if l != original])
    return y_flipped

# Load MNIST
X_train = idx2numpy.convert_from_file('../../MNIST/train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file('../../MNIST/train-labels-idx1-ubyte')
X_test = idx2numpy.convert_from_file('../../MNIST/t10k-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file('../../MNIST/t10k-labels-idx1-ubyte')

X_train = X_train.astype(np.float32).reshape(X_train.shape[0], -1)
X_test = X_test.astype(np.float32).reshape(X_test.shape[0], -1)

# Normalize and reduce
num_classes = len(np.unique(y_train))
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components=n_qubits)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Label flip
y_train = flip_labels(y_train, num_classes=10, flip_fraction=0.5)

# Dataloaders
def get_dataloaders(X_train, y_train, X_test, y_test):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz)
    return train_loader, test_loader

# Quantum circuit with random encoding
def random_encoding(x, weights):
    for i in range(n_qubits):
        qml.RX(weights[i][0] * x[i], wires=i)
        qml.RY(weights[i][1] * x[i], wires=i)
        qml.RZ(weights[i][2] * x[i], wires=i)

def modular_w(n_qubits, n_layers):
    return {f"rot_layer_{n}": (n_qubits, 3) for n in range(n_layers)} | \
           {f"crx_layer_{n}": (n_qubits, 1) for n in range(n_layers)}

weight_shapes = modular_w(n_qubits, n_layers)

# QNode factory with external weights passed
def create_qnode():
    @qml.qnode(dev, interface="torch")
    def qnode(inputs, rand_weights, **weights):
        random_encoding(inputs, rand_weights)
        for n in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(*weights[f"rot_layer_{n}"][i], wires=i)
            for i in range(n_qubits):
                qml.CRX(weights[f"crx_layer_{n}"][i][0], wires=[i, (i+1)%n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return qnode

class QMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.qparams = nn.ParameterDict({name: nn.Parameter(torch.randn(*shape)) for name, shape in weight_shapes.items()})
        self.fc = nn.Linear(n_qubits, num_classes)
        self.qnode_fn = create_qnode()

    def forward(self, x):
        outputs = []
        for i in range(len(x)):
            x_i = x[i]
            preds = []
            for _ in range(n_random_encodings):
                rand_weights = torch.tensor(np.random.uniform(0, np.pi, (n_qubits, 3)), dtype=torch.float32).to(x.device)
                preds.append(torch.tensor(self.qnode_fn(x_i, rand_weights, **{k: v for k, v in self.qparams.items()}), dtype=torch.float32))

            q_out = torch.stack(preds).mean(dim=0)
            outputs.append(q_out)
        q_output = torch.stack(outputs)
        return F.log_softmax(self.fc(q_output.to(self.fc.weight.device)), dim=1)

# Train function
def train(model, device, loader, optimizer):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

# Evaluation function remains unchanged
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
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        fnr_list.append(fn / (fn + tp) if (fn + tp) else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) else 0)
    avg_fnr = np.mean(fnr_list)
    avg_fpr = np.mean(fpr_list)
    return acc, prec, f1, cm, fnr_list, fpr_list, avg_fnr, avg_fpr

# Run 3 times and save best model
best_acc = 0
for run_id in range(1, 4):
    print(f"\n========= RUN {run_id} =========")
    torch.manual_seed(42 + run_id)
    np.random.seed(42 + run_id)
    random.seed(42 + run_id)

    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)
    model = QMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)

    test_accuracies = []
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        acc, prec, f1, _, _, _, _, _ = evaluate(model, device, test_loader)
        test_accuracies.append(acc)
        with open(f"{filename_prefix}_epoch_metrics_run{run_id}.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, acc, prec, f1])

    acc, prec, f1, cm, fnr_list, fpr_list, avg_fnr, avg_fpr = evaluate(model, device, test_loader)

    if acc > best_acc:
        torch.save(model.state_dict(), f"{best_model_prefix}_best_run.pt")
        best_acc = acc
        best_run_id = run_id
        best_cm = cm
        best_fnr_list = fnr_list
        best_fpr_list = fpr_list
        best_test_accuracies = test_accuracies

# Final outputs
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Best Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{filename_prefix}_best_confusion_matrix.png", dpi=300)
plt.close()

with open(f"{filename_prefix}_final_fnr_fpr.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "FNR", "FPR"])
    for i in range(num_classes):
        writer.writerow([i, best_fnr_list[i], best_fpr_list[i]])
    writer.writerow(["Average", avg_fnr, avg_fpr])

plt.plot(range(1, epochs + 1), best_test_accuracies, marker='o')
plt.title("Best Run: QMLP Test Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{filename_prefix}_best_accuracy_plot.png", dpi=300)
print(f"Best Run ID: {best_run_id}, Accuracy: {best_acc:.4f}")

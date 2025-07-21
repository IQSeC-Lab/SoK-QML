import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pennylane as qml
import random
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer
import torch.optim as optim
import seaborn as sns

# Parameters
bsz = 64
epochs = 30
lr = 0.001
w_decay = 1e-4
label_smoothing = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model_prefix = "qmlp-az-amplitude-label-flip-run"
filename_prefix = "qmlp-az-amplitude-label-flip-run"
encoding = "Amplitude"
n_qubits = 9
n_layers = 10

# Quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum circuit weights
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

# AZ classifier model
class drebin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.qlayer = qlayer
        self.fc = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        out = self.qlayer(x)
        out = self.fc(out.to(x.device))
        return F.log_softmax(out, dim=1)

# Training loop with label smoothing
def train(model, DEVICE, train_loader, optimizer, epoch, num_classes):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, target) in enumerate(train_loader, 0):
        inputs, target = inputs.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        if label_smoothing > 0.0:
            true_dist = torch.zeros_like(outputs)
            true_dist.fill_(label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - label_smoothing)
            loss = torch.mean(torch.sum(-true_dist * outputs, dim=1))
        else:
            loss = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx+1}, Acc: {100 * correct / total:.2f}%, Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

# Evaluation
def evaluate(model, device, loader, num_classes):
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
    return acc, prec, f1, cm, fnr_list, fpr_list, np.mean(fnr_list), np.mean(fpr_list)

# Label flipping
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

# Load AZ-Class dataset
az_train = np.load('./AZ_23/AZ-Class-Task_23_families_train.npz')
az_test = np.load('./AZ_23/AZ-Class-Task_23_families_test.npz')
X_train = az_train['X_train'].astype(np.float32)
y_train = az_train['Y_train'].astype(np.int64)
X_test = az_test['X_test'].astype(np.float32)
y_test = az_test['Y_test'].astype(np.int64)

num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

# Normalize and PCA
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components=512)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Apply label flipping
y_train = flip_labels(y_train, num_classes=num_classes)

# Dataloaders
def get_dataloaders(X_train, y_train, X_test, y_test):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz)
    return train_loader, test_loader

# Run 3 experiments
best_acc = 0
for run_id in range(1, 4):
    print(f"\n========= RUN {run_id} =========")
    torch.manual_seed(42 + run_id)
    np.random.seed(42 + run_id)
    random.seed(42 + run_id)

    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)
    model = drebin(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)

    test_accuracies = []
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, num_classes)
        acc, prec, f1, _, _, _, _, _ = evaluate(model, device, test_loader, num_classes)
        test_accuracies.append(acc)
        with open(f"{filename_prefix}_epoch_metrics_run{run_id}.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, acc, prec, f1])

    acc, prec, f1, cm, fnr_list, fpr_list, avg_fnr, avg_fpr = evaluate(model, device, test_loader, num_classes)

    if acc > best_acc:
        torch.save(model.state_dict(), f"{best_model_prefix}_best_run.pt")
        best_acc = acc
        best_run_id = run_id
        best_cm = cm
        best_fnr_list = fnr_list
        best_fpr_list = fpr_list
        best_test_accuracies = test_accuracies

# Save confusion matrix
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Best Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{filename_prefix}_best_confusion_matrix.png", dpi=300)
plt.close()

# Save FNR/FPR CSV
with open(f"{filename_prefix}_final_fnr_fpr.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "FNR", "FPR"])
    for i in range(num_classes):
        writer.writerow([i, best_fnr_list[i], best_fpr_list[i]])
    writer.writerow(["Average", np.mean(best_fnr_list), np.mean(best_fpr_list)])

# Plot test accuracy
plt.plot(range(1, epochs + 1), best_test_accuracies, marker='o')
plt.title("Best Run: QMLP Test Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{filename_prefix}_best_accuracy_plot.png", dpi=300)

print(f"Best Run ID: {best_run_id}, Accuracy: {best_acc:.4f}")

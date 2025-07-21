import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pennylane as qml
import random
from pennylane import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import time
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from pennylane.qnn import TorchLayer

# Parameters
bsz = 64
epochs = 30
lr = 0.001
w_decay = 1e-4
label_smoothing = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_model_prefix = "qmlp-az-angle-label-flip-run"
filename_prefix = "qmlp-az-angle-label-flip-run"
noise_afterfix = "-noiseless"
encoding = "Angle"
n_qubits = 9
n_layers = 5

dev = qml.device("default.qubit", wires=n_qubits)

# Load AZ-Class data
train_data = np.load('./AZ_23/AZ-Class-Task_23_families_train.npz')
test_data = np.load('./AZ_23/AZ-Class-Task_23_families_test.npz')

X_train = train_data['X_train'].astype(np.float32)
y_train = train_data['Y_train'].astype(np.int64)
X_test = test_data['X_test'].astype(np.float32)
y_test = test_data['Y_test'].astype(np.int64)

num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

# Normalize & PCA
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components=n_qubits)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Untargeted Label Flipping
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

y_train = flip_labels(y_train, num_classes=num_classes, flip_fraction=0.5)

# Label Smoothing
def smooth_labels(y, num_classes, smoothing=0.2):
    with torch.no_grad():
        y_smoothed = torch.full((y.size(0), num_classes), smoothing / (num_classes - 1)).to(y.device)
        y_smoothed.scatter_(1, y.unsqueeze(1), 1.0 - smoothing)
    return y_smoothed

# Dataloaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz)

# Quantum Circuit
def modular_w(n_qubits, n_layers):
    shapes = {}
    for n in range(n_layers):
        shapes[f"rot_layer_{n}"] = (n_qubits, 3)
        shapes[f"crx_layer_{n}"] = (n_qubits, 1)
    return shapes

weight_shapes = modular_w(n_qubits, n_layers)

@qml.qnode(dev, interface="torch")
def qnode(inputs, **weights_kwargs):
    for n in range(n_layers):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.Rot(*weights_kwargs[f"rot_layer_{n}"][i], wires=i)
        for i in range(n_qubits):
            qml.CRX(weights_kwargs[f"crx_layer_{n}"][i][0], wires=[i, (i + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qlayer = TorchLayer(qnode, weight_shapes)

# Hybrid Model
class QMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        self.fc = nn.Linear(n_qubits, num_classes)
    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        out = self.qlayer(x)
        out = self.fc(out.to(x.device))
        return F.log_softmax(out, dim=1)

# ORIGINAL Training Function
def train(model, DEVICE, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader, 0):
        inputs, target = inputs.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        smoothed_targets = smooth_labels(target, num_classes=num_classes, smoothing=label_smoothing)
        loss = F.kl_div(outputs, smoothed_targets, reduction='batchmean')
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx+1}, Acc: {100 * correct / total:.2f}%, Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

# Testing
def test(model, DEVICE, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy on test set: {acc:.2f}%")
    return acc

# Evaluation
def evaluation(model, test_loader, num_classes):
    model.eval()
    outputs, y_true = [], []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            out = model(x_batch)
            outputs.append(out)
            y_true.append(y_batch)
    output = torch.cat(outputs, dim=0)
    true = torch.cat(y_true, dim=0)
    pred = output.argmax(dim=1)
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()
    acc = (pred == true).float().mean().item()
    test_loss = F.nll_loss(output, true).item()

    y_true_np = true.cpu().numpy()
    y_pred_np = pred.cpu().numpy()
    cm = confusion_matrix(y_true_np, y_pred_np)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)
    precision_macro = np.mean(TP / (TP + FP + 1e-8))
    recall_macro = np.mean(TP / (TP + FN + 1e-8))
    f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro + 1e-8)
    fpr_macro = np.mean(FP / (FP + TN + 1e-8))
    fnr_macro = np.mean(FN / (FN + TP + 1e-8))
    roc_auc = roc_auc_score(y_true_np, probs, multi_class='ovr', average='macro')
    pr_auc = float('nan')

    metrics = {
        'accuracy': acc,
        'loss': test_loss,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'fpr': fpr_macro,
        'fnr': fnr_macro,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    return metrics

# Training Loop
for run_id in range(1, 4):
    print(f'\n========== Training Run {run_id} ==========')
    manual_seed = 42 + run_id
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)

    model = QMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    best_model = f'{best_model_prefix}{run_id}-layer{n_layers}{noise_afterfix}-{encoding}.pt'

    best_acc = 0.0
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model)
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    print(f"Best Accuracy for Run {run_id}: {best_acc:.2f}%")

    print("\nEvaluating final model on clean test data...")
    model.load_state_dict(torch.load(best_model))
    metrics = evaluation(model, test_loader, num_classes)
    for k, v in metrics.items():
        print(f"{k.capitalize():<12}: {v:.4f}")

    final_csv_filename = f'{filename_prefix}_run{run_id}-layer{n_layers}_label_flip.csv'
    write_header = not os.path.exists(final_csv_filename)
    with open(final_csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(f"Metrics saved to {final_csv_filename}")

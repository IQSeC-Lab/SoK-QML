import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # setting
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import matplotlib.pyplot as plt
import math
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix
from pennylane.qnn import TorchLayer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import qiskit_aer.noise as noise
from qiskit_aer.noise import NoiseModel
import idx2numpy
#################################################################################################
#################################################################################################
#################################################################################################

def downscale_images(X, new_size=(4, 4)):
    downscaled = []
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(new_size)

    for img_array in X:
        # Convert numpy array (28x28) to PIL Image
        img = Image.fromarray(img_array.astype(np.uint8))
        # Resize and convert back to numpy
        img_resized = resize(img)
        img_tensor = to_tensor(img_resized).squeeze(0).numpy()
        downscaled.append(img_tensor)

    return np.array(downscaled)

#################################################################################################
# Evaluation
#################################################################################################
def evaluation(model, test_loader, epsilon, num_classes):
    model.eval()
    outputs, y_true = [], []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if epsilon > 0:
            x_batch = fgsm_attack(model, x_batch, y_batch, epsilon)

        with torch.no_grad():
            out = model(x_batch)
            outputs.append(out)
            y_true.append(y_batch)

    output = torch.cat(outputs, dim=0)
    true = torch.cat(y_true, dim=0)
    pred = output.argmax(dim=1)
    probs = torch.softmax(output, dim=1).cpu().numpy()
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

    if num_classes == 2:
        roc_auc = roc_auc_score(y_true_np, probs[:, 1])
        pr_curve, rc_curve, _ = precision_recall_curve(y_true_np, probs[:, 1])
        pr_auc = auc(rc_curve, pr_curve)
    else:
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
#################################################################################################

#################################################################################################
# Attacks
#################################################################################################

def fgsm_attack(model, X, y, epsilon):
    # Clone and enable gradient tracking on input
    X_adv = X.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(X_adv)
    loss = torch.nn.functional.cross_entropy(output, y)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Apply FGSM perturbation
    X_adv = X_adv + epsilon * X_adv.grad.sign()

    # Clamp to valid data range [0, 1]
    X_adv = torch.clamp(X_adv, 0, 1)

    return X_adv.detach() 
#################################################################################################

bsz = 64
epochs = 30
lr = 0.001
w_decay = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#################################################################################################


best_model_prefix = "qmlp-mnist-run"
cm_name_prefix= "cm_qmlp-mnist-run"
filename_prefix = "qmlp-mnist-run"
############################################################################################################################################################################

# data_train = np.load("../dataset/API_graph/APIgraph_train23-fam.npz") 
# data_test = np.load("../dataset/API_graph/APIgraph_test23-fam.npz") 
# Ember and AZ
# X_train = data_train["X_train"]
# y_train_raw = data_train["Y_train"]
# y_train = y_train_raw
# X_test = data_test["X_test"]
# y_test_raw = data_test["Y_test"]
# y_test = y_test_raw

X_train = idx2numpy.convert_from_file('../datasets/mnist/train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file('../datasets/mnist/train-labels-idx1-ubyte')
X_test = idx2numpy.convert_from_file('../datasets/mnist/t10k-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file('../datasets/mnist/t10k-labels-idx1-ubyte')


X_train = downscale_images(X_train)  # Shape: (N, 14, 14)
X_test = downscale_images(X_test)

# Flatten
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)  # (N, 196)
X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

num_classes = len(np.unique(y_train)) 
print(num_classes)
# Normalize to [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=16)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Convert to tensors
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)
# DataLoaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz)

###########################################################################################################################################################################
def modular_w(n_qubits,n_layers):
    shapes = {}
    for n in range(n_layers):
        shapes[f"rot_layer_{n}"] = (n_qubits, 3) 
        shapes[f"crx_layer_{n}"] = (n_qubits, 1) 
    return shapes

###########################################################################################################################################################################
n_qubits = 16
n_layers = 5
# HERE CHANGE THE SIMULATOR

dev = qml.device("default.qubit", wires=n_qubits) # w/o noise


@qml.qnode(dev, interface="torch")
def qnode(inputs,**weights_kwargs):
    # Layer 1 — Data reuploading + trainable Rot + entanglement

    for n in range(n_layers):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        # qml.AmplitudeEmbedding(inputs, wires=range(n_qubits))
        for i in range(16):
            qml.Rot(*weights_kwargs[f"rot_layer_{n}"][i], wires=i)
        for i in range(16):
            qml.CRX(weights_kwargs[f"crx_layer_{n}"][i][0], wires=[i, (i + 1) % n_qubits])
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(16)]


weight_shapes = modular_w(n_qubits, n_layers)

#################################################################################################################################################################################
qlayer = TorchLayer(qnode, weight_shapes)
class drebin(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        self.fc = nn.Linear(16, num_classes)
    def forward(self,x):
        x = x.to(next(self.parameters()).device)
        out = self.qlayer(x)  # Batched input directly
        out = self.fc(out)
        return F.log_softmax(out, dim=1)




###########################################################################################################################################################################
def train(model, DEVICE, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0.0

    for batch_idx, (inputs, target) in enumerate(train_loader, 0):
        inputs, target = inputs.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
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

###########################################################################################################################################################################
best_acc = 0.0

for run_id  in range (1,4):
    print(f' Training Run {run_id}')
    best_model = f'{best_model_prefix}{run_id}-layer{n_layers}.pt'
    # cm_name = f'{cm_name}{run_id}.png'
    # filename = f'{filename_prefix}{run_id}.txt' 
    best_acc = 0.0
    model = drebin().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)


    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model)
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")
########################################################################




######################################################################
# EVALUATION
#######################################################################


# Evaluation
for run_id in range (1,4):
    best_model = f'{best_model_prefix}{run_id}-layer{n_layers}.pt'
    cm_name = f'{cm_name_prefix}{run_id}-layer{n_layers}.png'
    filename = f'{filename_prefix}_runs-layer{n_layers}.csv' 
    if os.path.exists(best_model):
        print("Loading previous model...")
        model = drebin().to(device)
        model.load_state_dict(torch.load(best_model))
        epsilons = [0, 0.01, 0.1, 0.15]
        results = {}
        for eps in epsilons:
            print(f"\n>>> Evaluating for ε = {eps}")
            metrics = evaluation(model, test_loader, epsilon=eps, num_classes=num_classes)
            results[eps] = metrics

            for k, v in metrics.items():
                print(f"{k.capitalize():<10}: {v:.4f}")
        plt.figure(figsize=(8, 5))
        plt.plot(epsilons, [results[eps]['accuracy'] for eps in epsilons], marker='o')
        plt.title(f"Adversarial Robustness of QCNN (FGSM) -- run {run_id}")
        plt.xlabel("Epsilon (FGSM strength)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"qcnn_fgsm_accuracy_plot_{run_id}_{n_layers}.png", dpi=300)
        
        if os.path.exists(filename): 
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                # Get metric names from the results (assuming consistent metrics across epsilons)
                if epsilons: 
                    metric_names = list(results[epsilons[0]].keys())
                else:
                    metric_names = [] 
                # writer.writerow(header) # Write header for this specific CSV file
                
                # Write data rows for all epsilons of the current run_id
                for eps in epsilons:
                    row = [run_id, eps] + [results[eps][k] for k in metric_names]
                    writer.writerow(row)
            print(f"Results saved to {filename}")
        else:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Get metric names from the results (assuming consistent metrics across epsilons)
                if epsilons: 
                    metric_names = list(results[epsilons[0]].keys())
                else:
                    metric_names = [] 

                header = ['Run ID', 'Epsilon'] + metric_names
                writer.writerow(header) # Write header for this specific CSV file
                
                # Write data rows for all epsilons of the current run_id
                for eps in epsilons:
                    row = [run_id, eps] + [results[eps][k] for k in metric_names]
                    writer.writerow(row)
            print(f"Results saved to {filename}")
    else: 
        print(f'{best_model} not found')


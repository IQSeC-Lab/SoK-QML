import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Change for the GPU to use
import pennylane as qml

import random
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import matplotlib.pyplot as plt
import math
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_recall_curve, auc)
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
import argparse


parser = argparse.ArgumentParser(description="Configure a Quantum Machine Learning model.")

parser.add_argument(
    "-l", "--layers",
    type=int,
    default=1,
    help="Number of layers in the QML model (default: 1)"
)

args = parser.parse_args()




####################################################################################################
# Global Parameters
bsz = 64
epochs = 30
lr = 0.001
w_decay = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prefix for the names of the models
best_model_prefix = "qmlp-az-run"
cm_name_prefix= "cm_qmlp-az-run"
filename_prefix = "qmlp-az-run"
noise_afterfix = ""

#Change accordingly to the type of encoding
encoding = "Amplitude"

# Circuit 
n_qubits = 9 # qubits of the the circuit
# n_layers = 10 # modify this for the number of layers 

n_layers = args.layers
print(f"using layers {n_layers}")
# noiseless
dev = qml.device("default.qubit", wires=n_qubits)
####################################################################################################

# Train and test functions for the hybrid models
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
####################################################################################################

"""
Image downscaler for processing MNIST Dataset
"""
def downscale_images(X, new_size=(3,3)):
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


"""
Evaluation Function for the QML Models
"""
def evaluation(model, test_loader, epsilon, num_classes, attack_type, pgd_alpha, pgd_num_iter, pgd_random_start):
    model.eval()
    outputs, y_true = [], []
    
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if epsilon > 0:
            if attack_type == 'fgsm':
                x_batch = fgsm_attack(model, x_batch, y_batch, epsilon)
            elif attack_type == 'pgd':
                x_batch = pgd_attack(model, x_batch, y_batch, epsilon, pgd_alpha, pgd_num_iter, pgd_random_start)
            elif attack_type == 'none':
                pass 
            else:
                raise ValueError(f"Unknown attack type: {attack_type}. Choose 'none', 'fgsm', or 'pgd'.")

        with torch.no_grad():
            out = model(x_batch)
            outputs.append(out)
            y_true.append(y_batch)

    output = torch.cat(outputs, dim=0)
    true = torch.cat(y_true, dim=0)
    pred = output.argmax(dim=1)
    # probs = torch.softmax(output, dim=1).cpu().numpy()
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
# Attacks
#################################################################################################
"""
FGSM Attack
"""
def fgsm_attack(model, X, y, epsilon):
    X_adv = X.clone().detach().requires_grad_(True)
    output = model(X_adv)
    loss = torch.nn.functional.cross_entropy(output, y)
    model.zero_grad()
    loss.backward()
    X_adv = X_adv + epsilon * X_adv.grad.sign()
    X_adv = torch.clamp(X_adv, 0, 1)
    return X_adv.detach() 

"""
PGD Attack
"""
def pgd_attack(model, X, y, epsilon, alpha, num_iter, random_start):
    model.eval() # Ensure model is in evaluation mode
    X_original = X.clone().detach() # Keep a copy of the original input
    # Initialize adversarial example
    if random_start:
        random_noise = (torch.rand_like(X) * 2 * epsilon - epsilon) # Noise within [-epsilon, epsilon]
        X_adv = X_original + random_noise
        X_adv = torch.clamp(X_adv, 0, 1).detach() # Clamp and detach
    else:
        X_adv = X_original.clone().detach()

    for i in range(num_iter):
        X_adv.requires_grad_(True) # Enable gradient tracking for the adversarial example
        
        output = model(X_adv)
        loss = F.nll_loss(output, y) # Or F.cross_entropy if model outputs logits

        model.zero_grad()
        loss.backward()

        if X_adv.grad is None:
             raise RuntimeError("Gradient of input is None during PGD. Check model/computation graph.")

        grad_sign = X_adv.grad.sign()

        # Gradient ascent step
        X_adv_perturbed = X_adv.detach() + alpha * grad_sign

        # Projection back into epsilon-ball (L_infinity)
        eta = X_adv_perturbed - X_original
        eta = torch.clamp(eta, -epsilon, epsilon) # Clamp perturbation
        X_adv = X_original + eta # Apply clamped perturbation to original

        # Clamp to valid data range [0, 1]
        X_adv = torch.clamp(X_adv, 0, 1).detach() # Detach after each step

    return X_adv

###################################################################################
# Load the dataset
###################################################################################
data_train = np.load("../datasets/AZ-Task/AZ-task-23fam_train.npz") 
data_test = np.load("../datasets/AZ-Task/AZ-task-23fam_test.npz") 
# Ember and AZ
X_train = data_train["X_train"]
y_train_raw = data_train["Y_train"]
y_train = y_train_raw
X_test = data_test["X_test"]
y_test_raw = data_test["Y_test"]
y_test = y_test_raw


X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) 
X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)


num_classes = len(np.unique(y_train)) 
print(num_classes)

# Normalize to [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=512) # This is for Amplitude Embedding
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


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
"""
----Modular Weights QMLP----
This function will create modular weights for the QMLP based on 
the number of qubits and the number of layers are required.
"""
def modular_w(n_qubits,n_layers):
    shapes = {}
    for n in range(n_layers):
        shapes[f"rot_layer_{n}"] = (n_qubits, 3) 
        shapes[f"crx_layer_{n}"] = (n_qubits, 1) 
    return shapes

# Call the function to have the weights for the model
weight_shapes = modular_w(n_qubits, n_layers)
###########################################################################################################################################################################

# Main circuit for the QMLP structure
@qml.qnode(dev, interface="torch")
def qnode(inputs,**weights_kwargs):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    for n in range(n_layers):
        
        for i in range(n_qubits):
            qml.Rot(*weights_kwargs[f"rot_layer_{n}"][i], wires=i)
        for i in range(n_qubits):
            qml.CRX(weights_kwargs[f"crx_layer_{n}"][i][0], wires=[i, (i + 1) % n_qubits])
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

####################################################################################
qlayer = TorchLayer(qnode, weight_shapes)

class drebin(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        self.fc = nn.Linear(n_qubits, num_classes)
    def forward(self,x):
        x = x.to(next(self.parameters()).device)
        out = self.qlayer(x)  # Batched input directly
        out = self.fc(out.to(x.device))
        return F.log_softmax(out, dim=1)


####################################################################################
# Train Loop for the models
####################################################################################

# best_acc = 0.0
# for run_id  in range (1,4):
#     print(f' Training Run {run_id}')
#     # best_model = f'{best_model_prefix}{run_id}-layer{n_layers}.pt'
#     manual_seed = 42 + run_id
#     torch.manual_seed(manual_seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(manual_seed)
#         torch.cuda.manual_seed_all(manual_seed)
#     np.random.seed(manual_seed)
#     random.seed(manual_seed)

#     best_acc = 0.0
#     model = drebin().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
#     best_model = f'{best_model_prefix}{run_id}-layer{n_layers}{noise_afterfix}-{encoding}.pt'

#     start_time = time.time()
#     for epoch in range(1, epochs + 1):
#         train(model, device, train_loader, optimizer, epoch)
#         acc = test(model, device, test_loader)
#         if acc > best_acc:
#             best_acc = acc
#             torch.save(model.state_dict(), best_model)
#     end_time = time.time()
#     print(f"Training Time: {end_time - start_time:.2f} seconds")
#     print(f"Best Accuracy for Run {run_id}: {best_acc:.2f}%")

######################################################################
# EVALUATION
#######################################################################

attack_types_to_evaluate = ['none', 'fgsm', 'pgd']

for run_id in range (1,4):
    # names for the results
    best_model = f'{best_model_prefix}{run_id}-layer{n_layers}{noise_afterfix}-{encoding}.pt'
    
    cm_name = f'{cm_name_prefix}{run_id}-layer{n_layers}{noise_afterfix}-{encoding}.png'
    filename = f'{filename_prefix}_runs-layer{n_layers}{noise_afterfix}-{encoding}.csv' 
    if os.path.exists(best_model):
        print("Loading previous model...")
        model = drebin().to(device)
        # model.load_state_dict(torch.load(best_model))
        model.load_state_dict(torch.load(best_model, map_location=torch.device('cpu')))
        PGD_RANDOM_START = True # Recommended for stronger PGD
        PGD_NUM_ITER = 10 
        # epsilons for attacks
        epsilons = [0, 0.01, 0.1, 0.15]
        # epsilons = [0.01, 0.1, 0.15]
        all_results_for_run = [] 
        print(f'Evaluating {best_model}')
        for current_attack in attack_types_to_evaluate :
            print(f"\n--- Evaluating Model Run {run_id} with {current_attack.upper()} Attack ---")
            results = {}
            for eps in epsilons:
                if current_attack == 'none' and eps != 0:
                    continue 

                PGD_ALPHA = eps / PGD_NUM_ITER * 1.25
                print(f"\n>>> Evaluating for ε = {eps} (Attack: {current_attack})")
                metrics = evaluation(
                    model, test_loader, epsilon=eps, num_classes=num_classes,
                    attack_type=current_attack,
                    pgd_alpha=PGD_ALPHA,         
                    pgd_num_iter=PGD_NUM_ITER,   
                    pgd_random_start=PGD_RANDOM_START 
                )
                results[eps] = metrics
                for k, v in metrics.items():
                    print(f"{k.capitalize():<10}: {v:.4f}")
            for eps in epsilons:
                if eps in results: # Only add if data exists for this epsilon
                    # Append a dictionary for each row, including run_id, epsilon, and attack_type
                    row_data = {
                        'Run ID': run_id,
                        'Epsilon': eps,
                        'Attack Type': current_attack,
                        **results[eps] # Unpack all metrics (accuracy, precision, etc.)
                    }
                    all_results_for_run.append(row_data)
            ############################################
            # Graph
            ############################################
            plt.figure(figsize=(8, 5))
            # plt.plot(epsilons, [results[eps]['accuracy'] for eps in epsilons], marker='o')
            plt.plot([e for e in epsilons if e in results], 
                     [results[e]['accuracy'] for e in epsilons if e in results], 
                     marker='o')
            plt.title(f"Adversarial Robustness of QMLP ({current_attack}) -- run {run_id}")
            plt.xlabel(f"Epsilon ({current_attack})")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.tight_layout()
            # plt.savefig(f"qcnn_{current_attack}_accuracy_plot_run{run_id}_numlayers{n_layers}-{encoding}.png", dpi=300)
        final_csv_filename = f'{filename_prefix}_run{run_id}-layer{n_layers}_{encoding}_all_attacks.csv' 
        write_header = not os.path.exists(final_csv_filename) or os.stat(final_csv_filename).st_size == 0
        with open(final_csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if all_results_for_run:
                header_keys = list(all_results_for_run[0].keys())
                ordered_header = ['Run ID', 'Epsilon', 'Attack Type'] + \
                                [k for k in header_keys if k not in ['Run ID', 'Epsilon', 'Attack Type']]
                writer.writerow(ordered_header)
            
            for row_data in all_results_for_run:
                writer.writerow([row_data.get(k, '') for k in ordered_header])
            print(f"All results for Run {run_id} saved to {final_csv_filename}")

else: 
    print(f'{best_model} not found')
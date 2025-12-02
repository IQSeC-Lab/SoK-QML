import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc3(x), dim=1)

transform = transforms.ToTensor()

train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = 20000
val_size = 2000
train_dataset, val_dataset, _ = random_split(train_full, [train_size, val_size, len(train_full) - train_size - val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def pgd_attack(model, data, target, epsilon=0.3, alpha=0.01, iters=40, criterion=None):
    original_data = data.clone().detach()
    perturbed_data = data.clone().detach().requires_grad_(True)

    for _ in range(iters):
        output = model(perturbed_data)
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()

        perturbed_data = perturbed_data + alpha * perturbed_data.grad.sign()
        
        perturbation = torch.clamp(perturbed_data - original_data, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(original_data + perturbation, 0, 1).detach().requires_grad_(True)

    return perturbed_data

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def test_pgd_attack(model, test_loader, epsilon=0.3, alpha=0.01, iters=40):
    model.eval()
    criterion = nn.NLLLoss()

    all_preds = []
    all_targets = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        perturbed_data = pgd_attack(model, data, target, epsilon, alpha, iters, criterion)
        output = model(perturbed_data)
        pred = output.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds) * 100
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    cm = confusion_matrix(all_targets, all_preds)
    fnr_list, fpr_list = [], []

    for i in range(10):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr_list.append(fnr)
        fpr_list.append(fpr)

    avg_fnr = np.mean(fnr_list)
    avg_fpr = np.mean(fpr_list)

    print(f"[PGD] ε = {epsilon:.3f} | Acc: {acc:.2f}% | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | FNR: {avg_fnr:.4f} | FPR: {avg_fpr:.4f}")

    return acc, precision, recall, f1, avg_fnr, avg_fpr

model = MLP()  
model.load_state_dict(torch.load('MLP_mnist1.pth'))
model.to(device)
model.eval()

import pandas as pd

epsilons = [0.0, 0.01, 0.1, 0.15]
pgd_results = []

for eps in epsilons:
    print(f"Testing PGD attack with ε = {eps}")
    acc, prec, rec, f1, fnr, fpr = test_pgd_attack(model, test_loader, epsilon=eps, alpha=0.01, iters=40)

    pgd_results.append({
        'Epsilon': eps,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'FNR': fnr,
        'FPR': fpr
    })

# Save to CSV
df = pd.DataFrame(pgd_results)
df.to_csv("pgd_attack_results.csv", index=False)
print("\nPGD evaluation saved to pgd_attack_results.csv")


epsilons = [0.0, 0.01, 0.1, 0.15]

accuracies = []
precisions = []
recalls = []
f1_scores = []
fprs = []
fnrs = []

for eps in epsilons:
    print(f"Testing PGD attack with epsilon = {eps}")
    acc, prec, rec, f1, fnr, fpr = test_pgd_attack(model, test_loader, epsilon=eps, alpha=0.01, iters=40)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    fprs.append(fpr)
    fnrs.append(fnr)

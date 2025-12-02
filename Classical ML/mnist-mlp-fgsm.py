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

def fgsm_attack(model, data, target, epsilon, criterion):
    data.requires_grad = True
    output = model(data)
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data.detach()

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np

def test_fgsm_attack(model, test_loader, epsilon=0.3):
    model.eval()
    criterion = nn.NLLLoss()

    all_preds = []
    all_targets = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        perturbed_data = fgsm_attack(model, data, target, epsilon, criterion)

        with torch.no_grad():
            output = model(perturbed_data)

        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    cm = confusion_matrix(all_targets, all_preds)
    FP = cm.sum(axis=0) - np.diag(cm)   
    FN = cm.sum(axis=1) - np.diag(cm)   
    TP = np.diag(cm)                    
    TN = cm.sum() - (FP + FN + TP)      

    fpr = np.mean(FP / (FP + TN + 1e-10))  
    fnr = np.mean(FN / (FN + TP + 1e-10))

    print(f"[FGSM] Epsilon = {epsilon} | "
          f"Acc: {acc:.2f}% | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
          f"FPR: {fpr:.4f} | FNR: {fnr:.4f}")

    return acc, precision, recall, f1, fpr, fnr


model = MLP()  
model.load_state_dict(torch.load('MLP_mnist1.pth'))
model.to(device)
model.eval()

epsilons = [0.0, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]

accuracies = []
precisions = []
recalls = []
f1_scores = []
fprs = []
fnrs = []

for eps in epsilons:
    print(f"\nTesting FGSM with epsilon = {eps}")
    acc, prec, rec, f1, fpr, fnr = test_fgsm_attack(model, test_loader, epsilon=eps)
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    fprs.append(fpr)
    fnrs.append(fnr)

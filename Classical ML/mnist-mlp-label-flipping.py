import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = 20000
val_size = 2000
train_dataset, val_dataset, _ = random_split(train_full, [train_size, val_size, len(train_full) - train_size - val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import copy
import random
from torch.utils.data import Dataset, DataLoader

def dataset_to_list(dataset):
    return [(img, label) for img, label in dataset]

test_data_list = dataset_to_list(test_dataset)

def poison_data(data_list, poison_fraction=0.5):
    poisoned = []
    for img, label in data_list:
        if random.random() < poison_fraction:
            poisoned.append((img, (label + 1) % 10))  
        else:
            poisoned.append((img, label))
    return poisoned

class PoisonedDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    print(f"Test Accuracy: {100. * correct / total:.2f}%")

test_data_list = [(img, label) for img, label in test_dataset]
poisoned_test_data = poison_data(test_data_list, poison_fraction=0.5)
poisoned_test_dataset = PoisonedDataset(poisoned_test_data)

clean_test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=32, shuffle=False)

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
    
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np


def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = 100. * (all_preds == all_targets).sum() / len(all_targets)
    print(f"Accuracy: {acc:.2f}%")

    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    print(f"Precision (macro): {precision:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

    cm = confusion_matrix(all_targets, all_preds)
    fnr_list = []
    fpr_list = []

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

    print(f"\nAverage FNR: {avg_fnr:.4f}")
    print(f"Average FPR: {avg_fpr:.4f}")

model = MLP()
model.load_state_dict(torch.load('MLP_mnist1.pth'))
model.to(device)
model.eval()

print("Testing on poisoned test data:")
test_model(model, poisoned_test_loader)

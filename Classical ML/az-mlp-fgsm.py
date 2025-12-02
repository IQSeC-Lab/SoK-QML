import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_PATH = "/home/mimon/QML_CML_Adv/CML/MLP/AZ_MLP/AZ_23/A_Z Handwritten Data.csv"
MODEL_PATHS = [
    "./saved_runs/best_mlp_run1.pth",
    "./saved_runs/best_mlp_run2.pth",
    "./saved_runs/best_mlp_run3.pth"
]
EPSILONS = [0.0, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
BATCH_SIZE = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, num_classes=26):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc3(x), dim=1)


def load_dataset():
    df = pd.read_csv(DATA_PATH).astype("float32")
    X = df.drop("0", axis=1).values
    y = df["0"].astype(int).values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(-1, 1, 28, 28)
    return X, y


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

def test_fgsm_attack(model, test_loader, epsilon=0.3):
    model.eval()
    criterion = nn.NLLLoss()
    all_preds = []
    all_targets = []

    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
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

def visualize_multiple_epsilons(model, test_loader, epsilons, num_samples=6):
    """Show original, adversarial, and perturbation for each epsilon."""
    data_batch, target_batch = next(iter(test_loader))
    data_batch, target_batch = data_batch.to(DEVICE), target_batch.to(DEVICE)
    criterion = nn.NLLLoss()

    with torch.no_grad():
        pred_clean = model(data_batch).argmax(dim=1)

    fig_rows = num_samples
    fig_cols = len(epsilons) * 2 + 1  # Original + (Adv, Perturb) per epsilon
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 2, fig_rows * 2))

    orig_np = data_batch.detach().cpu().numpy()
    labels_np = target_batch.detach().cpu().numpy()
    pred_clean_np = pred_clean.detach().cpu().numpy()

    for row in range(num_samples):
        # Column 0: Original
        axes[row, 0].imshow(orig_np[row].squeeze(), cmap='gray')
        axes[row, 0].set_title(f"T:{labels_np[row]}\nC:{pred_clean_np[row]}")
        axes[row, 0].axis('off')

    for col_idx, eps in enumerate(epsilons):
        adv_data = fgsm_attack(model, data_batch, target_batch, eps, criterion)
        with torch.no_grad():
            pred_adv = model(adv_data).argmax(dim=1)

        adv_np = adv_data.detach().cpu().numpy()
        pred_adv_np = pred_adv.detach().cpu().numpy()

        for row in range(num_samples):
            # Adversarial image
            axes[row, 1 + col_idx * 2].imshow(adv_np[row].squeeze(), cmap='gray')
            axes[row, 1 + col_idx * 2].set_title(f"Eps:{eps}\nAdv:{pred_adv_np[row]}")
            axes[row, 1 + col_idx * 2].axis('off')

            # Perturbation
            diff = np.abs(adv_np[row] - orig_np[row])
            axes[row, 2 + col_idx * 2].imshow(diff.squeeze(), cmap='inferno')
            axes[row, 2 + col_idx * 2].set_title("Perturb")
            axes[row, 2 + col_idx * 2].axis('off')

    plt.tight_layout()
    plt.show()



def visualize_adversarial(model, test_loader, epsilon, num_samples=6):
    data_batch, target_batch = next(iter(test_loader))
    data_batch, target_batch = data_batch.to(DEVICE), target_batch.to(DEVICE)
    criterion = nn.NLLLoss()

    with torch.no_grad():
        pred_clean = model(data_batch).argmax(dim=1)

    adv_data = fgsm_attack(model, data_batch, target_batch, epsilon, criterion)
    with torch.no_grad():
        pred_adv = model(adv_data).argmax(dim=1)

    orig = data_batch.detach().cpu().numpy()
    adv = adv_data.detach().cpu().numpy()
    labels = target_batch.detach().cpu().numpy()
    pred_c = pred_clean.detach().cpu().numpy()
    pred_a = pred_adv.detach().cpu().numpy()

    fig, axes = plt.subplots(num_samples, 3, figsize=(6, num_samples*2))
    for i in range(num_samples):
        axes[i, 0].imshow(orig[i].squeeze(), cmap='gray')
        axes[i, 0].set_title(f"T:{labels[i]} / C:{pred_c[i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(adv[i].squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Adv:{pred_a[i]}")
        axes[i, 1].axis('off')

        diff = np.abs(adv[i] - orig[i])
        axes[i, 2].imshow(diff.squeeze(), cmap='inferno')
        axes[i, 2].set_title("Perturbation")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    X, y = load_dataset()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    for idx, model_path in enumerate(MODEL_PATHS):
        print(f"\n=== Testing Model {idx+1} ===")
        model = MLP(num_classes=len(np.unique(y))).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        for eps in EPSILONS:
            acc, prec, rec, f1, fpr, fnr = test_fgsm_attack(model, test_loader, epsilon=eps)

    final_model_path = MODEL_PATHS[0]  # pick model
    model = MLP(num_classes=len(np.unique(y))).to(DEVICE)
    model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
    model.eval()

    eps_for_vis = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
    print(f"\nVisualizing adversarial samples for epsilons={eps_for_vis}")
    visualize_multiple_epsilons(model, test_loader, epsilons=eps_for_vis, num_samples=8)
    
if __name__ == "__main__":
    main()

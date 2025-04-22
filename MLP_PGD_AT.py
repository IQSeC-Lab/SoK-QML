# -*- coding: utf-8 -*-

"""MLP_FGSM_Adversarial Training

Original file is located at
    https://colab.research.google.com/drive/1x1JthiDpLlN3EAcgRj3cwJiJ0QbOyPSD?usp=sharing
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred, h_2
    

INPUT_DIM = 28 * 28
OUTPUT_DIM = 10
model = MLP(INPUT_DIM, OUTPUT_DIM)


# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(28*28, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def plot_images(images):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure()
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap='bone')
        ax.axis('off')


N_IMAGES = 25
images = [image for image, label in [train_data[i] for i in range(N_IMAGES)]]
plot_images(images)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Train the model
def train(model, train_loader, test_loader, optimizer, criterion, device, epochs):
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output, _ = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Calculate average training loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)

        # Evaluate on test data
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f} - "
              f"Test Loss: {test_loss:.4f} - "
              f"Test Accuracy: {test_acc:.4f}")
        

# Evaluate model accuracy on clean images
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images.view(images.shape[0], -1))
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


epochs = 5
train(model, train_loader, test_loader, optimizer, criterion, device, epochs)


# Save the model's state dictionary
torch.save(model.state_dict(), 'mnist_mlp_model.pth')

model.load_state_dict(torch.load('mnist_mlp_model.pth'))
model.eval()  # Set the model to evaluation mode

# Evaluate on the test dataset
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"Test Accuracy: {test_acc:.4f}")



#---PGD Attack---
def pgd_attack(model, images, labels, eps=0.1, alpha=None, iters=40):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    ori_images = images.clone().detach()

    if alpha is None:
        alpha = eps / iters 

    images.requires_grad = True
    model.eval()

    for _ in range(iters):
        outputs, _ = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        images.requires_grad = True

    return images



from tqdm import tqdm
def train_adv_on_the_fly(model, train_loader, optimizer, criterion, device, attack_func, epsilon, pgd_iters):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader, desc="Training Progress"):
        images, labels = images.to(device), labels.to(device)

        adv_images = attack_func(model, images, labels, eps=epsilon, iters=pgd_iters)

        optimizer.zero_grad()
        outputs, _ = model(adv_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs, _ = model(images)

            loss = criterion(outputs, labels)  
            
            loss_total += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_total / len(loader), correct / total

# Evaluate on adversarial data
def evaluate_on_adv(model, loader, attack_func, criterion, device, epsilon, pgd_iters):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_func(model, images, labels, eps=epsilon, iters=pgd_iters)

        outputs, _ = model(adv_images)  
        outputs = outputs 
        
        loss = criterion(outputs, labels)
        loss_total += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return loss_total / len(loader), correct / total



import torch.optim as optim

defensive_model = MLP(28*28, 10).to(device)
optimizer = optim.Adam(defensive_model.parameters(), lr=0.001)

epochs = 5
epsilon = 0.1
pgd_iters = 10

for epoch in range(epochs):
    print(f"\nEpoch [{epoch+1}/{epochs}]")

    train_loss = train_adv_on_the_fly(defensive_model, train_loader, optimizer, criterion, device, pgd_attack, epsilon, pgd_iters)
    
    clean_loss, clean_acc = evaluate(defensive_model, test_loader, criterion, device)
    adv_loss, adv_acc = evaluate_on_adv(defensive_model, test_loader, pgd_attack, criterion, device, epsilon, pgd_iters)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Clean Accuracy: {clean_acc * 100:.2f}%")
    print(f"Adversarial Accuracy (eps={epsilon}): {adv_acc * 100:.2f}%")



epsilon_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
adv_accuracies = []

for eps in epsilon_list:
    _, adv_acc = evaluate_on_adv(defensive_model, test_loader, pgd_attack, criterion, device, epsilon=eps, pgd_iters=20)
    adv_accuracies.append(adv_acc)
    print(f"Epsilon: {eps:.2f}, Accuracy: {adv_acc*100:.2f}%")

import matplotlib.pyplot as plt
plt.plot(epsilon_list, adv_accuracies, marker='o')
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Adversarial Accuracy vs Epsilon")
plt.grid(True)
plt.show()
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


# Define FGSM Attack
def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    # Flatten images before passing them through the model
    outputs, _ = model(images)

    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Compute perturbation
    perturbation = epsilon * images.grad.sign()
    perturbed_images = images + perturbation
    perturbed_images = torch.clamp(perturbed_images, 0, 1)  

    return perturbed_images.detach()


# Generate adversarial examples
def generate_adversarial_data(model, loader, attack_func, epsilon, device='cpu'):
    adv_examples = []
    adv_labels = []
    model.eval()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        perturbed_images = attack_func(model, images, labels, epsilon)
        adv_examples.append(perturbed_images)
        adv_labels.append(labels)
    return torch.cat(adv_examples, dim=0), torch.cat(adv_labels, dim=0)

adv_test_images, adv_test_labels = generate_adversarial_data(model, test_loader, fgsm_attack, epsilon=0.1, device=device)

# Create a DataLoader from the new tensors
adv_test_dataset = torch.utils.data.TensorDataset(adv_test_images, adv_test_labels)
adv_test_loader = DataLoader(adv_test_dataset, batch_size=128, shuffle=False)

# evaluate() returns avg_loss and accuracy, unpack it
adv_loss, adv_acc = evaluate(model, adv_test_loader, criterion, device)
print(f"Adversarial Accuracy: {adv_acc * 100:.2f}%")


import matplotlib.pyplot as plt

epsilon_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracies = []

for epsilon in epsilon_values:
    # Generate adversarial examples
    adv_test_images, adv_test_labels = generate_adversarial_data(model, test_loader, fgsm_attack, epsilon, device)

    # Create a DataLoader for adversarial examples
    adv_test_dataset = torch.utils.data.TensorDataset(adv_test_images, adv_test_labels)
    adv_test_loader = DataLoader(adv_test_dataset, batch_size=128, shuffle=False)

    # Evaluate model accuracy on adversarial examples
    adv_loss, adv_acc = evaluate(model, adv_test_loader, criterion, device)  
    accuracies.append(adv_acc)
    print(f"Epsilon: {epsilon:.1f}, Accuracy: {adv_acc:.4f}")

plt.plot(epsilon_values, accuracies, marker='o')
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epsilon")
plt.grid(True)
plt.show()


# Effect on images with different noise rates
epsilon_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

test_iter = iter(test_loader)
images, labels = next(test_iter)

# Select a single image to visualize
image_index = 0
image = images[image_index]
label = labels[image_index]

# Create a figure with subplots
fig, axes = plt.subplots(1, len(epsilon_values), figsize=(15, 3))

# Iterate through epsilon values
for i, epsilon in enumerate(epsilon_values):
    perturbed_image = fgsm_attack(model, image.unsqueeze(0), label.unsqueeze(0), epsilon).squeeze(0)

    axes[i].imshow(perturbed_image.cpu().numpy().reshape(28, 28), cmap='gray')
    axes[i].set_title(f"Epsilon: {epsilon}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# Defense: Adversarial Training 
defensive_model = MLP(28*28, 10).to(device) 
defensive_optimizer = optim.Adam(defensive_model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    # Generate adversarial examples using training data
    adv_train_images, adv_train_labels = generate_adversarial_data(model, train_loader, fgsm_attack, epsilon=0.1, device=device)

    # Create a DataLoader for adversarial training examples
    adv_train_dataset = torch.utils.data.TensorDataset(adv_train_images, adv_train_labels)
    adv_train_loader = DataLoader(adv_train_dataset, batch_size=128, shuffle=True)

    train(defensive_model, train_loader, test_loader, defensive_optimizer, criterion, device, epochs=1)

    train(defensive_model, adv_train_loader, test_loader, defensive_optimizer, criterion, device, epochs=1)

    # Evaluate the defensive model's performance
    def_loss, def_acc = evaluate(defensive_model, adv_test_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{epochs}] - Defense Accuracy: {def_acc * 100:.2f}%")

print(f"Final Defense Accuracy: {def_acc * 100:.2f}%")


# Evaluate on clean test data
clean_loss, clean_acc = evaluate(defensive_model, test_loader, criterion, device)
print(f"Defense Accuracy on Clean Data: {clean_acc * 100:.2f}%")

# Evaluate on adversarial test data
adv_loss, adv_acc = evaluate(defensive_model, adv_test_loader, criterion, device)
print(f"Defense Accuracy on Attacked Data: {adv_acc * 100:.2f}%")



#Accuracy under different noise rates (epsilon from 0.0 to 1.0)
import matplotlib.pyplot as plt

epsilon_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracies = []

for epsilon in epsilon_values:
    adv_test_images, adv_test_labels = generate_adversarial_data(model, test_loader, fgsm_attack, epsilon, device)

    # Create a DataLoader for adversarial examples
    adv_test_dataset = torch.utils.data.TensorDataset(adv_test_images, adv_test_labels)
    adv_test_loader = DataLoader(adv_test_dataset, batch_size=128, shuffle=False)

    # Evaluate model accuracy on adversarial examples
    adv_loss, adv_acc = evaluate(defensive_model, adv_test_loader, criterion, device)  
    accuracies.append(adv_acc)
    print(f"Epsilon: {epsilon:.1f}, Accuracy: {adv_acc:.4f}")

# Plot the relationship
plt.plot(epsilon_values, accuracies, marker='o')
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epsilon for Defended Model")
plt.grid(True)
plt.show()
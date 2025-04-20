# -*- coding: utf-8 -*-

"""MLP_FGSM_Adversarial Training

Original file is located at
    https://colab.research.google.com/drive/1awK5oX1tcjBd-7u8CwHgRXt-FTOllEwG?usp=sharing
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
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
    


# FGSM Attack
def fgsm_attack(model, images, labels, epsilon, apply_masking=False, noise_factor=0.1):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs, _ = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    grad = images.grad.data

    # Add gradient masking
    if apply_masking:
        noise = noise_factor * torch.randn_like(grad)
        grad = grad + noise

    perturbation = epsilon * grad.sign()
    perturbed_images = images + perturbation
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images.detach()



# Generating Adversarial Examples
def generate_adversarial_data(model, loader, attack_func, epsilon, device='cpu', apply_masking=False, noise_factor=0.1):
    adv_examples = []
    adv_labels = []
    model.eval()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        perturbed_images = attack_func(
            model, images, labels, epsilon,
            apply_masking=apply_masking, noise_factor=noise_factor
        )

        adv_examples.append(perturbed_images)
        adv_labels.append(labels)

    return torch.cat(adv_examples, dim=0), torch.cat(adv_labels, dim=0)



# Training normal model
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, train_loader, test_loader, optimizer, criterion, device, epochs)

# Generating adversarial examples for normal model
adv_test_images, adv_test_labels = generate_adversarial_data(model, test_loader, fgsm_attack, epsilon=0.1, device=device)
# Evaluate adversarial examples
adv_loss, adv_acc = evaluate(model, DataLoader(torch.utils.data.TensorDataset(adv_test_images, adv_test_labels), batch_size=128, shuffle=False), criterion, device)
print(f"Adversarial Accuracy (Normal Model): {adv_acc:.2%}")


# Gradient Masking Defense
class GradientMaskingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        noise = torch.randn_like(grad_output) * 0.1 
        grad_input = grad_output + noise
        return grad_input
    

class MLPWithGradientMasking(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        h_1 = F.relu(self.input_fc(x))
        h_1 = GradientMaskingFunction.apply(h_1)  
        h_2 = F.relu(self.hidden_fc(h_1))
        h_2 = GradientMaskingFunction.apply(h_2)  

        y_pred = self.output_fc(h_2)
        return y_pred, h_2
    


# Training normal model
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, train_loader, test_loader, optimizer, criterion, device, epochs)

# Generate adversarial examples for normal model
adv_test_images, adv_test_labels = generate_adversarial_data(model, test_loader, fgsm_attack, epsilon=0.1, device=device)
# Evaluate adversarial examples
adv_loss, adv_acc = evaluate(model, DataLoader(torch.utils.data.TensorDataset(adv_test_images, adv_test_labels), batch_size=128, shuffle=False), criterion, device)
print(f"Adversarial Accuracy (Normal Model): {adv_acc:.2%}")

# Training masked model
# Instantiate the MLPWithGradientMasking class and assign it to model_with_masking
model_with_masking = MLPWithGradientMasking(INPUT_DIM, OUTPUT_DIM).to(device)
optimizer = optim.Adam(model_with_masking.parameters(), lr=0.001)
train(model_with_masking, train_loader, test_loader, optimizer, criterion, device, epochs)

# Generate adversarial examples for masked model
adv_test_images_masked, adv_test_labels_masked = generate_adversarial_data(model_with_masking, test_loader, fgsm_attack, epsilon=0.1, device=device)
# Evaluate adversarial examples
adv_loss_masked, adv_acc_masked = evaluate(model_with_masking,
                                           DataLoader(torch.utils.data.TensorDataset(adv_test_images_masked,
                                                                                     adv_test_labels_masked),
                                                      batch_size=128, shuffle=False), criterion, device)
print(f"Adversarial Accuracy (Masked Model): {adv_acc_masked:.2%}")



# Function to compute adversarial accuracy for various epsilon values for model_with_masking
def evaluate_accuracy_vs_epsilon_masked(model, test_loader, attack_func, epsilons, device):
    acc_list = []

    for epsilon in epsilons:
        # Generate adversarial examples
        adv_test_images, adv_test_labels = generate_adversarial_data(
            model, test_loader, attack_func, epsilon, device
        )

        # Create DataLoader for the adversarial examples
        adv_test_loader = DataLoader(torch.utils.data.TensorDataset(adv_test_images, adv_test_labels),
                                     batch_size=128, shuffle=False)

        # Evaluate adversarial accuracy
        adv_loss, adv_acc = evaluate(model, adv_test_loader, criterion, device)
        acc_list.append(adv_acc)
        print(f"Adversarial Accuracy at epsilon {epsilon:.1f}: {adv_acc * 100:.2f}%")

    return acc_list

# Epsilon values to test
epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Evaluate masked model with different epsilons
masked_accuracies = evaluate_accuracy_vs_epsilon_masked(
    model_with_masking, test_loader, fgsm_attack, epsilons, device
)

plt.figure(figsize=(10, 6))
plt.plot(epsilons, masked_accuracies, label='Masked Model', marker='o', color='b')

plt.title('Adversarial Accuracy vs Epsilon for Masked Model')
plt.xlabel('Epsilon (perturbation strength)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()
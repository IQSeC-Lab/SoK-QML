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
    


# Define FGSM Attack
def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs, _ = model(images)

    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Compute perturbation
    perturbation = epsilon * images.grad.sign()
    perturbed_images = images + perturbation
    perturbed_images = torch.clamp(perturbed_images, 0, 1)  

    return perturbed_images.detach()



# Generating Adversarial Examples (without masking)
def generate_adversarial_data(model, loader, attack_func, epsilon, device='cpu'):
    adv_examples = []
    adv_labels = []
    model.eval()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        perturbed_images = attack_func(
            model, images, labels, epsilon
        )

        adv_examples.append(perturbed_images)
        adv_labels.append(labels)

    return torch.cat(adv_examples, dim=0), torch.cat(adv_labels, dim=0)



# Load the pre-trained normal model
model = MLP(input_dim=784, output_dim=10)  
model.load_state_dict(torch.load('mnist_mlp_model.pth')) 
model.to(device) 
model.eval()  

# Generate adversarial examples for the normal model
adv_test_images, adv_test_labels = generate_adversarial_data(model, test_loader, fgsm_attack, epsilon=0.1, device=device)

# Evaluate adversarial examples on the normal model
adv_loss, adv_acc = evaluate(model, DataLoader(torch.utils.data.TensorDataset(adv_test_images, adv_test_labels), batch_size=128, shuffle=False), criterion, device)
print(f"Adversarial Accuracy (Normal Model): {adv_acc:.2%}")


# Gradient Masking Defense
class GradientMasking(torch.autograd.Function):
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
        h_1 = GradientMasking.apply(h_1)  
        h_2 = F.relu(self.hidden_fc(h_1))
        h_2 = GradientMasking.apply(h_2) 

        y_pred = self.output_fc(h_2)
        return y_pred, h_2
    


# Training masked model
model_with_masking = MLPWithGradientMasking(INPUT_DIM, OUTPUT_DIM).to(device)
optimizer = optim.Adam(model_with_masking.parameters(), lr=0.001)
train(model_with_masking, train_loader, test_loader, optimizer, criterion, device, epochs)

# Generate adversarial examples for masked model
adv_test_images_masked, adv_test_labels_masked = generate_adversarial_data(model_with_masking, test_loader, fgsm_attack, epsilon=0.1, device=device)

adv_loss_masked, adv_acc_masked = evaluate(model_with_masking,
                                           DataLoader(torch.utils.data.TensorDataset(adv_test_images_masked,
                                                                                     adv_test_labels_masked),
                                                      batch_size=128, shuffle=False), criterion, device)
print(f"Adversarial Accuracy (Masked Model): {adv_acc_masked:.2%}")



# Testing with Different Noise Rates
import matplotlib.pyplot as plt
epsilon_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracies = []

for epsilon in epsilon_values:
    adv_test_images, adv_test_labels = generate_adversarial_data(model_with_masking, test_loader, fgsm_attack, epsilon=epsilon, device=device)
    
    adv_loss_masked, adv_acc_masked = evaluate(model_with_masking,
                                               DataLoader(torch.utils.data.TensorDataset(adv_test_images, adv_test_labels),
                                                          batch_size=128, shuffle=False), criterion, device)
    
    accuracies.append(adv_acc_masked)

for epsilon, accuracy in zip(epsilon_values, accuracies):
  print(f"Epsilon: {epsilon:.1f}, Accuracy: {accuracy:.2%}")

# Plotting the accuracies
plt.plot(epsilon_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('Adversarial Accuracy vs Epsilon (Model with Gradient Masking)')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(epsilon_values)  
plt.show()

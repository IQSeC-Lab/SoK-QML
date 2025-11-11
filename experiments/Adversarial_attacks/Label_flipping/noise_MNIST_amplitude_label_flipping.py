####################################################################################################
# 1. IMPORTS & SETUP
####################################################################################################
import pennylane as qml
from pennylane import numpy as np
from pennylane_qiskit import AerDevice
from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
import idx2numpy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statistics
import os
import time
from datetime import datetime

####################################################################################################
# 2. LOAD & PREPROCESS MNIST (9-QUBIT AMPLITUDE ENCODING)
####################################################################################################
X_train = idx2numpy.convert_from_file('./MNIST/train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file('./MNIST/train-labels-idx1-ubyte')
X_test = idx2numpy.convert_from_file('./MNIST/t10k-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file('./MNIST/t10k-labels-idx1-ubyte')

X_train = X_train.astype(np.float32).reshape(X_train.shape[0], -1)
X_test = X_test.astype(np.float32).reshape(X_test.shape[0], -1)

X_total = np.vstack([X_train, X_test])
y_total = np.concatenate([y_train, y_test])

scaler = MinMaxScaler()
X_total_scaled = scaler.fit_transform(X_total)
X_total_reduced = X_total_scaled[:, :512]

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_total_reduced, y_total, test_size=0.2, stratify=y_total, random_state=42
)

X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_final, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_final, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

####################################################################################################
# 3. TRAINING & EVALUATION FUNCTION
####################################################################################################
def train_and_evaluate(noise_prob, run_id, trial, results_dir):
    n_qubits = 9
    dev = qml.device('default.mixed', wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):
            qml.DepolarizingChannel(noise_prob, wires=i)
        for i in range(n_qubits):
            qml.RY(weights[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(n_qubits):
            qml.DepolarizingChannel(noise_prob, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    from pennylane.qnn import TorchLayer
    weight_shapes = {"weights": n_qubits}
    qlayer = TorchLayer(quantum_circuit, weight_shapes)
    model = nn.Sequential(qlayer, nn.Linear(n_qubits, 10))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    log_file = os.path.join(results_dir, f"log_noise{noise_prob}_trial{trial}.txt")
    with open(log_file, 'w') as f:
        for epoch in range(30):
            model.train()
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            f.write(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}\n")

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        preds = model(X_test_tensor)
    end_time = time.time()
    inference_time = end_time - start_time

    predicted_classes = torch.argmax(preds, dim=1)
    acc = (predicted_classes == y_test_tensor).float().mean().item()

    model_path = os.path.join(results_dir, f"model_noise{noise_prob}_trial{trial}.pt")
    torch.save(model.state_dict(), model_path)

    with open(log_file, 'a') as f:
        f.write(f"Inference Time: {inference_time:.4f} seconds\n")

    return acc, inference_time

####################################################################################################
# 4. RUN EXPERIMENTS FOR MULTIPLE NOISE LEVELS
####################################################################################################
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
base_results_dir = f"results_qml_{timestamp}"
os.makedirs(base_results_dir, exist_ok=True)

noise_levels = [0.0, 0.01, 0.03, 0.05, 0.07, 0.1]
mean_accuracies = []
std_accuracies = []
mean_times = []

for noise in noise_levels:
    print(f"Running 3 trials with depolarizing noise = {noise}")
    acc_list = []
    time_list = []
    noise_dir = os.path.join(base_results_dir, f"noise_{noise}")
    os.makedirs(noise_dir, exist_ok=True)
    for trial in range(1, 4):
        acc, inference_time = train_and_evaluate(noise, timestamp, trial, noise_dir)
        acc_list.append(acc)
        time_list.append(inference_time)
    mean_acc = statistics.mean(acc_list)
    std_acc = statistics.stdev(acc_list)
    mean_time = statistics.mean(time_list)
    mean_accuracies.append(mean_acc)
    std_accuracies.append(std_acc)
    mean_times.append(mean_time)
    print(f"Mean Accuracy: {mean_acc*100:.2f}%, Std: {std_acc*100:.2f}%, Avg Inference Time: {mean_time:.4f} sec\n")

####################################################################################################
# 5. PLOT ACCURACY VS. NOISE WITH ERROR BARS
####################################################################################################
plt.figure(figsize=(8, 5))
plt.errorbar(noise_levels, [a * 100 for a in mean_accuracies], 
             yerr=[s * 100 for s in std_accuracies], fmt='-o', capsize=5)
plt.title("Accuracy vs. Depolarizing Noise Strength (with Std Dev)")
plt.xlabel("Depolarizing Probability")
plt.ylabel("Test Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(base_results_dir, "accuracy_vs_noise.png")
plt.savefig(plot_path)
plt.show()

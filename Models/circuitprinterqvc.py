import pennylane as qml
import jax.numpy as jnp  # JAX for differentiable operations
import numpy as np_classic  # Standard NumPy for non-differentiable operations like data loading/shuffling
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # To split the filtered data
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import csv
import gzip
import struct
import os
import argparse
from pennylane.drawer import draw_mpl
import inspect
import matplotlib.pyplot as plt

from pennylane.templates.layers import StronglyEntanglingLayers
parser = argparse.ArgumentParser(description="Configure a Quantum Machine Learning model.")

parser.add_argument(
    "-l", "--layers",
    type=int,
    default=1,
    help="Number of layers in the QML model (default: 1)"
)

args = parser.parse_args()


encoding = "Angle"
dataset = "MNIST"
noise = "noiseless"
# --- Global Configurations ---
num_qubits = 9
dev = qml.device("default.qubit", wires=num_qubits) # Or "lightning.gpu" if on GPU server
num_ansatz_layers = args.layers
print(f"Number of anzats layers: {num_ansatz_layers}")
metrics_name = f"{noise}_{encoding}_binary_classifier_{num_ansatz_layers}_{dataset}_metrics"
epochs = 30
batch_size = 64
opt = qml.AdamOptimizer(stepsize=0.01)



# --- MNIST Data Loading ---
def load_mnist_ubyte(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    with gzip.open(labels_path, 'rb') as lbpath:
        magic, num_items = struct.unpack('>II', lbpath.read(8))
        labels = np_classic.frombuffer(lbpath.read(), dtype=np_classic.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num_items, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np_classic.frombuffer(imgpath.read(), dtype=np_classic.uint8).reshape(len(labels), rows * cols)
    return images, labels

# --- IMPORTANT: Set the path to your MNIST dataset directory ---
mnist_path = '../datasets/mnist' # <--- *** YOU MUST CHANGE THIS LINE ***

try:
    X_full, y_full = load_mnist_ubyte(mnist_path, kind='train')
    X_test_full_original, y_test_full_original = load_mnist_ubyte(mnist_path, kind='t10k')
except FileNotFoundError as e:
    print(f"Error: MNIST file not found. Please check your 'mnist_path' setting.")
    raise e

# Flatten and Normalize
X_full_flat = X_full.reshape(X_full.shape[0], -1) / 255.0
X_test_flat_original = X_test_full_original.reshape(X_test_full_original.shape[0], -1) / 255.0

print(f"Loaded MNIST full train shape: {X_full_flat.shape}")
print(f"Loaded MNIST full test shape: {X_test_flat_original.shape}")

# --- Binary Classification Specific Data Filtering ---
# Choose two digits to classify, e.g., '0' and '1'
digit_0 = 0
digit_1 = 1

# Filter training data
train_indices = np_classic.where((y_full == digit_0) | (y_full == digit_1))
X_train_filtered = X_full_flat[train_indices]
y_train_filtered_labels = y_full[train_indices]

# Filter testing data
test_indices = np_classic.where((y_test_full_original == digit_0) | (y_test_full_original == digit_1))
X_test_filtered = X_test_flat_original[test_indices]
y_test_filtered_labels = y_test_full_original[test_indices]


# Map labels to -1 and 1 for the QVC
# Assuming digit_0 maps to -1 and digit_1 maps to 1
y_train_mapped = jnp.where(y_train_filtered_labels == digit_0, -1, 1)
y_test_mapped = jnp.where(y_test_filtered_labels == digit_0, -1, 1)

print(f"\nFiltered training data for {digit_0} vs {digit_1}: {X_train_filtered.shape}")
print(f"Filtered testing data for {digit_0} vs {digit_1}: {X_test_filtered.shape}")


# --- Dimensionality Reduction (PCA) ---
pca = PCA(n_components=num_qubits)
X_train_reduced = pca.fit_transform(X_train_filtered)
X_test_reduced = pca.transform(X_test_filtered) # Use transform on test set based on train fit

# Scale features to be between 0 and 2*pi for AngleEmbedding
scaler = MinMaxScaler(feature_range=(0, 2 * jnp.pi))
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

print(f"Reduced and scaled training data shape: {X_train_scaled.shape}")
print(f"Reduced and scaled testing data shape: {X_test_scaled.shape}")



@qml.qnode(dev, interface="jax")
def qvc_binary_classifier(params, features):
    # qml.AngleEmbedding(features, wires=range(num_qubits), rotation='Y')
    for layer in range(num_ansatz_layers):
        qml.AngleEmbedding(features, wires=range(num_qubits), rotation='Y')
        for wire in range(num_qubits):
            qml.Rot(*params[layer, wire], wires=wire)
        
        # Linear entanglement pattern (ring-style)
        for wire in range(num_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        qml.CNOT(wires=[num_qubits - 1, 0])  # Close the ring

    return (
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2))
    )

def cost_function_binary(params, features, targets):
    all_expvals = jnp.array([qvc_binary_classifier(params, f) for f in features])
    predictions_summed_per_sample = jnp.sum(all_expvals, axis=1) 
    predictions_avg = predictions_summed_per_sample / 3.0  
    loss = jnp.mean((predictions_avg - targets) ** 2)
    return loss 

# initial_params_binary = jnp.array(
#     np_classic.random.uniform(low=0, high=2 * jnp.pi, size=(num_ansatz_layers, num_qubits, 3))
# )
initial_params_binary = jnp.array(
    np_classic.random.uniform(low=0, high=2 * jnp.pi, size=(num_ansatz_layers, num_qubits, 3))
)

print(f"Initialized binary classifier parameters with shape: {initial_params_binary.shape}")


# --- Training Loop Function ---
def train_epoch_binary(epoch, current_params, X_train, y_train, optimizer_instance, batch_s):
    total_loss_epoch = 0.0
    correct_epoch = 0.0
    total_epoch = 0.0

    permutation = np_classic.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    
    print(f"Epoch {epoch} | Processing batches...")

    for i in range(0, X_train.shape[0], batch_s):
        X_batch = X_train_shuffled[i:i + batch_s]
        y_batch = y_train_shuffled[i:i + batch_s]

        # batch_loss_val, current_params = optimizer_instance.step_and_cost(
        #     cost_function_binary, current_params, features=X_batch, targets=y_batch
        # )
        # print(f"this is current batch loss{batch_loss_val.shape}")
        # print("Raw loss:", batch_loss_val)
        # total_loss_epoch += float(batch_loss_val)
        updated_params, batch_loss_val = optimizer_instance.step_and_cost(
            cost_function_binary, current_params, features=X_batch, targets=y_batch
        )
        current_params = updated_params # Update the parameters for the next iteration

        # print(f"this is current batch loss{batch_loss_val.shape}") # This should now print '()' for scalar
        # print("Raw loss:", batch_loss_val) # This should now print a single number
        total_loss_epoch += float(batch_loss_val) # This conversion will now succeed

        batch_predictions_sum = jnp.array([jnp.sum(jnp.array(qvc_binary_classifier(current_params, f))) for f in X_batch])
        batch_predictions_avg = batch_predictions_sum / 3.0
        batch_predicted = jnp.sign(batch_predictions_avg)
        
        total_epoch += len(y_batch)
        correct_epoch += jnp.sum(batch_predicted == y_batch).item()
        
    num_batches = X_train.shape[0] / batch_s
    final_epoch_avg_loss = total_loss_epoch / num_batches
    final_epoch_acc = 100 * correct_epoch / total_epoch
    return current_params, final_epoch_avg_loss, final_epoch_acc


# --- Evaluation Function ---
def evaluate_binary(params, X_test, y_test):
    print("\n--- Evaluating on Test Set ---")
    # predictions_sum = jnp.array([qvc_binary_classifier(params, f) for f in X_test])
    predictions_sum = jnp.array([jnp.sum(jnp.array(qvc_binary_classifier(params, f))) for f in X_test])

    predictions_avg = predictions_sum / 3.0
    predicted_labels = jnp.sign(predictions_avg) # Convert to -1 or 1

    correct = jnp.sum(predicted_labels == y_test).item()
    total = len(y_test)
    accuracy = (correct / total) * 100
    
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("-" * 50)
    return accuracy, correct, total

# --- Main Training Loop ---
current_params_binary = initial_params_binary
train_costs = []
train_accuracies = []
test_accuracies = []


print("circuit printer")
# Get weight shapes for the StronglyEntanglingLayers template
weight_shapes = {"params": StronglyEntanglingLayers.shape(n_layers=num_ansatz_layers, n_wires=num_qubits)}

style = "black_white"
dummy_input = jnp.zeros(num_qubits, dtype=jnp.float32)
dummy_weights = jnp.zeros(weight_shapes["params"], dtype=jnp.float32)

qml.drawer.use_style(style)
fig, ax = draw_mpl(qvc_binary_classifier, show_all_wires=True)(dummy_weights, dummy_input)
fig.set_size_inches(16, 8)  # Wide layout
plt.tight_layout()
plt.savefig("qvc_structure.png", dpi=300)
# plt.show()
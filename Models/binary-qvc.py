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



def evaluate_full_metrics(params, X_data, y_true_full_labels, digit_0, digit_1):
    print("\n--- Performing Full Metrics Evaluation ---")

    # 1. Get predictions from the QVC
    predictions_sum = jnp.array([qvc_binary_classifier(params, f) for f in X_data])
    predictions_avg = predictions_sum / 3.0
    
    # Convert continuous predictions to binary class labels (-1 or 1)
    y_pred_mapped = jnp.sign(predictions_avg)

    # Convert true labels to -1 or 1 to match QVC output
    y_true_mapped = jnp.where(y_true_full_labels == digit_0, -1, 1)

    # Convert JAX arrays to classic NumPy for scikit-learn
    y_true_np = np_classic.array(y_true_mapped)
    y_pred_np = np_classic.array(y_pred_mapped)

    # Convert labels from -1/1 to 0/1 for scikit-learn metrics that expect it
    # E.g., for AUC, precision/recall/f1 when `pos_label=1`
    # Let's map -1 -> 0, 1 -> 1
    y_true_01 = np_classic.where(y_true_np == 1, 1, 0)
    y_pred_01 = np_classic.where(y_pred_np == 1, 1, 0)

    # Get raw scores for AUC if possible (predictions_avg are continuous)
    # Ensure predictions_avg are also standard NumPy for roc_auc_score
    y_scores_np = np_classic.array(predictions_avg)

    # 2. Calculate Metrics
    acc = accuracy_score(y_true_01, y_pred_01)
    
    # Calculate loss (MSE as used in training)
    test_loss = jnp.mean((predictions_avg - y_true_mapped)**2).item() 
    precision_macro = precision_score(y_true_01, y_pred_01, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_01, y_pred_01, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_01, y_pred_01, average='macro', zero_division=0)

    # Confusion Matrix for False Positives/Negatives
    tn, fp, fn, tp = confusion_matrix(y_true_01, y_pred_01).ravel()
    fpr_macro = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Positive Rate
    fnr_macro = fn / (fn + tp) if (fn + tp) > 0 else 0 # False Negative Rate (Miss Rate)

    rescaled_scores = (y_scores_np - y_scores_np.min()) / (y_scores_np.max() - y_scores_np.min())
    try:
        roc_auc = roc_auc_score(y_true_01, rescaled_scores)
    except ValueError: 
        roc_auc = np_classic.nan 
    try:
        pr_auc = average_precision_score(y_true_01, rescaled_scores)
    except ValueError: 
        pr_auc = np_classic.nan 


    metrics = {
        'accuracy': acc,
        'loss': test_loss,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'fpr': fpr_macro,
        'fnr': fnr_macro,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

    # Print Metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Loss (MSE): {metrics['loss']:.4f}")
    print(f"Precision (Macro): {metrics['precision']:.4f}")
    print(f"Recall (Macro): {metrics['recall']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1']:.4f}")
    print(f"False Positive Rate (FPR): {metrics['fpr']:.4f}")
    print(f"False Negative Rate (FNR): {metrics['fnr']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print("-----------------------------------")

    return metrics


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

mnist_path = '../datasets/mnist' 

try:
    X_full, y_full = load_mnist_ubyte(mnist_path, kind='train')
    X_test_full_original, y_test_full_original = load_mnist_ubyte(mnist_path, kind='t10k')
except FileNotFoundError as e:
    print(f"Error: MNIST file not found. Please check your 'mnist_path' setting.")
    raise e

X_full_flat = X_full.reshape(X_full.shape[0], -1) / 255.0
X_test_flat_original = X_test_full_original.reshape(X_test_full_original.shape[0], -1) / 255.0

print(f"Loaded MNIST full train shape: {X_full_flat.shape}")
print(f"Loaded MNIST full test shape: {X_test_flat_original.shape}")

# ###################### Binary Classification Specific Data Filtering ######################
digit_0 = 0
digit_1 = 1
train_indices = np_classic.where((y_full == digit_0) | (y_full == digit_1))
X_train_filtered = X_full_flat[train_indices]
y_train_filtered_labels = y_full[train_indices]
test_indices = np_classic.where((y_test_full_original == digit_0) | (y_test_full_original == digit_1))
X_test_filtered = X_test_flat_original[test_indices]
y_test_filtered_labels = y_test_full_original[test_indices]
y_train_mapped = jnp.where(y_train_filtered_labels == digit_0, -1, 1)
y_test_mapped = jnp.where(y_test_filtered_labels == digit_0, -1, 1)

print(f"\nFiltered training data for {digit_0} vs {digit_1}: {X_train_filtered.shape}")
print(f"Filtered testing data for {digit_0} vs {digit_1}: {X_test_filtered.shape}")


# --- Dimensionality Reduction (PCA) ---
pca = PCA(n_components=num_qubits)
X_train_reduced = pca.fit_transform(X_train_filtered)
X_test_reduced = pca.transform(X_test_filtered)

# Scale features to be between 0 and 2*pi for AngleEmbedding
scaler = MinMaxScaler(feature_range=(0, 2 * jnp.pi))
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

print(f"Reduced and scaled training data shape: {X_train_scaled.shape}")
print(f"Reduced and scaled testing data shape: {X_test_scaled.shape}")

########################################################################################
# qnode
########################################################################################
@qml.qnode(dev, interface="jax")
def qvc_binary_classifier(params, features):
    for layer in range(num_ansatz_layers):
        qml.AngleEmbedding(features, wires=range(num_qubits), rotation='Y')
        for wire in range(num_qubits):
            qml.Rot(*params[layer, wire], wires=wire)
        for wire in range(num_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        qml.CNOT(wires=[num_qubits - 1, 0]) 
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

initial_params_binary = jnp.array(
    np_classic.random.uniform(low=0, high=2 * jnp.pi, size=(num_ansatz_layers, num_qubits, 3))
)
print(f"Initialized binary classifier parameters with shape: {initial_params_binary.shape}")


########################################################################################
# TRAINING 
########################################################################################
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

        updated_params, batch_loss_val = optimizer_instance.step_and_cost(
            cost_function_binary, current_params, features=X_batch, targets=y_batch
        )
        current_params = updated_params 
        total_loss_epoch += float(batch_loss_val) 
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

print("\nStarting binary classification training...")

for epoch_num in range(1, epochs + 1):
    current_params_binary, avg_epoch_cost, avg_epoch_acc = train_epoch_binary(
        epoch_num, current_params_binary, X_train_scaled, y_train_mapped, opt, batch_size
    )
    train_costs.append(avg_epoch_cost)
    train_accuracies.append(avg_epoch_acc)

    test_acc, correct_test, total_test = evaluate_binary(current_params_binary, X_test_scaled, y_test_mapped)
    test_accuracies.append(test_acc)

    print(f"\n--- Epoch {epoch_num} Summary ---")
    print(f"  Training Avg Cost: {avg_epoch_cost:.4f}")
    print(f"  Training Accuracy: {avg_epoch_acc:.2f}%")
    print(f"  Test Accuracy - EPOCH {epoch_num} -: {test_acc:.2f}%")
    print("=" * 60)

print("\nBinary classification training complete!")
print(f"Final training cost: {train_costs[-1]:.4f}")
print(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")
print("\n--- Final Evaluation with Full Metrics ---")
final_metrics = evaluate_full_metrics(current_params_binary, X_test_scaled, y_test_filtered_labels, digit_0, digit_1)

metrics_save_path_csv = f'./{metrics_name}.csv'
try:
    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(metrics_save_path_csv)
    
    with open(metrics_save_path_csv, 'a', newline='') as f: # 'a' for append mode
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(final_metrics.keys()) # Write header only once
        writer.writerow(final_metrics.values()) # Write the metrics data
    print(f"Evaluation metrics appended to: {metrics_save_path_csv}")
except Exception as e:
    print(f"Error saving metrics to CSV: {e}")

model_save_path = f'./{metrics_name}.npy'
params_to_save = np_classic.asarray(current_params_binary)

try:
    np_classic.save(model_save_path, params_to_save)
    print(f"\nModel parameters saved successfully to: {model_save_path}")
except Exception as e:
    print(f"Error saving model parameters: {e}")


# --- Plotting ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, epochs + 1), np_classic.array(train_costs))
plt.title("Training Cost Evolution")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(1, epochs + 1), np_classic.array(train_accuracies))
plt.title("Training Accuracy Evolution")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(1, epochs + 1), np_classic.array(test_accuracies))
plt.title("Test Accuracy Evolution")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.tight_layout()
# plt.show()
plt.savefig(f"testAcc{metrics_name}.png", dpi=300)

# --- Visualize Sample Predictions ---
# Function to get a single prediction for visualization
def predict_single_binary_sample(params, features_single_sample):
    # predictions_sum = qvc_binary_classifier(params, features_single_sample)
    # predictions_avg = predictions_sum / 3.0
    predictions_sum = jnp.sum(jnp.array(qvc_binary_classifier(params, features_single_sample)))
    predictions_avg = predictions_sum / 3.0

    predicted_label = digit_0 if jnp.sign(predictions_avg) == -1 else digit_1
    return predicted_label

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes.flat):
    idx = np_classic.random.randint(0, len(X_test_scaled))
    
    # Need to map back original full images for display
    original_idx_in_full_test_set = test_indices[0][idx] # Get the index in the *original* test set
    image_original_dim = X_test_full_original[original_idx_in_full_test_set].reshape(28, 28)
    true_label = y_test_full_original[original_idx_in_full_test_set]
    
    predicted_label = predict_single_binary_sample(current_params_binary, X_test_scaled[idx])

    ax.imshow(image_original_dim, cmap='gray')
    ax.set_title(f"True: {true_label}\nPred: {predicted_label}")
    ax.axis('off')
plt.suptitle(f"Binary Classification Sample Predictions ({digit_0} vs {digit_1})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{metrics_name}.png", dpi=300)
# plt.show()
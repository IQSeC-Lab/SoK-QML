import pennylane as qml
import jax.numpy as jnp  # Import JAX's NumPy for array creation and JAX-compatible ops
import numpy as np_classic  # Use standard NumPy for non-differentiable ops (like data loading)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # Corrected import path for PCA
from sklearn.preprocessing import MinMaxScaler

import gzip
import struct
import os  # To handle file paths



mnist_path = '../datasets/mnist' 
num_qubits = 9
dev = qml.device("default.qubit", wires=num_qubits)
print(f"Using {num_qubits}-qubit device: {dev.name}")


# Step 4: Load and Preprocess MNIST Data (From Local Files)
def load_mnist_ubyte(path, kind='train'):
    """Load MNIST images or labels from ubyte files."""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        # Read magic number and number of items
        magic, num_items = struct.unpack('>II', lbpath.read(8))
        # Use np_classic for raw data loading
        labels = np_classic.frombuffer(lbpath.read(), dtype=np_classic.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        # Read magic number, number of images, rows, cols
        magic, num_items, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        # Use np_classic for raw data loading
        images = np_classic.frombuffer(imgpath.read(), dtype=np_classic.uint8).reshape(len(labels), rows * cols)

    return images, labels



# Load data using the helper function
try:
    X_train_full, y_train_full = load_mnist_ubyte(mnist_path, kind='train')
    X_test_full, y_test_full = load_mnist_ubyte(mnist_path, kind='t10k')  # 't10k' for test set
except FileNotFoundError as e:
    print(f"Error: MNIST file not found. Please check your 'mnist_path' setting and ensure the files are there.")
    print(f"Expected files in '{mnist_path}': train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz")
    raise e  # Re-raise to stop execution if files are missing


# Flatten images (already done by load_mnist_ubyte)
X_train_flat = X_train_full
X_test_flat = X_test_full

print(f"Loaded MNIST flat train shape: {X_train_flat.shape}")
print(f"Loaded MNIST flat test shape: {X_test_flat.shape}")

# Normalize pixel values to [0, 1]
X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_flat / 255.0

# --- Dimensionality Reduction (Crucial for 9-qubit limit) ---
# PCA (Principal Component Analysis) to reduce 784 features down to num_qubits (9).
pca = PCA(n_components=num_qubits)
X_train_reduced = pca.fit_transform(X_train_norm)
X_test_reduced = pca.transform(X_test_norm)

# Scale features to be between 0 and 2*pi for AngleEmbedding
scaler = MinMaxScaler(feature_range=(0, 2 * jnp.pi))
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

print(f"Reduced and scaled training data shape: {X_train_scaled.shape}")
print(f"Reduced and scaled testing data shape: {X_test_scaled.shape}")


# --- Phase 2: Quantum Circuit Design (The QVC for Each Binary Classifier) ---

# Step 5: Define the Variational Circuit (Ansatz) for a Binary Classifier
num_ansatz_layers = 3  # You can experiment with this number

@qml.qnode(dev, interface="jax")  # Specify JAX interface for better performance
def qvc_binary_classifier(params, features):
    # Ensure features match num_qubits
    if len(features) != num_qubits:
        raise ValueError(f"Features vector length ({len(features)}) must match number of qubits ({num_qubits}).")

    # Step 1: Data Encoding Layer
    # Features will be JAX arrays due to the `interface="jax"`
    qml.AngleEmbedding(features, wires=range(num_qubits), rotation='Y')

    # Step 2: Variational Layers (trainable part)
    # Params are already JAX arrays
    qml.StronglyEntanglingLayers(weights=params, wires=range(num_qubits))

    # Step 3: Measurement
    # Return the SUM of expectation values. Division will happen classically.
    return qml.expval(qml.PauliZ(0) + qml.PauliZ(1) + qml.PauliZ(2))

# Step 6: Initialize Circuit Parameters for All Classifiers
num_classes = 10  # Digits 0-9 for MNIST

# Initialize random parameters for each of the 10 binary classifiers
# The shape is (number of layers, number of qubits, 3 angles per qubit)
# Use jnp.array to ensure they are JAX arrays from the start
all_params_multiclass = [
    jnp.array(np_classic.random.uniform(low=0, high=2 * jnp.pi, size=(num_ansatz_layers, num_qubits, 3)))
    for _ in range(num_classes)
]
print(f"Initialized {len(all_params_multiclass)} sets of parameters, each with shape: {all_params_multiclass[0].shape}")


# --- Phase 3: Training the Multiclass Classifier (One-vs-All Strategy) ---

# Step 7: Define the Cost Function for a Single Binary Classifier
def cost_function_binary(params, features, targets):
    # The QNode output is a JAX array due to interface="jax"
    # Convert list of outputs to jnp.array
    predictions_sum = jnp.array([qvc_binary_classifier(params, f) for f in features])
    predictions_avg = predictions_sum / 3.0  # Perform division here!
    # targets should also be JAX array for operations with predictions_avg
    loss = jnp.mean((predictions_avg - targets)**2)  # Use jnp.mean for consistency
    return loss

# Step 8: Choose and Initialize the Optimizer
opt = qml.AdamOptimizer(stepsize=0.01)  # Optimizer for each binary classifier
epochs = 30  # Number of training epochs for each classifier
batch_size = 64  # Mini-batch size

# Step 9: Train All Classifiers (Optimization Loop)
overall_avg_costs = []
overall_test_accuracies = []

print("\nStarting multiclass (One-vs-All) training...")
print("This will train 10 separate binary classifiers.")
print("\n" + "="*80)
print(f"{'Epoch':<5} | {'Class':<5} | {'Batch_in_Class':<12} | {'Samples_Processed_in_Batch':<25} | {'Batch_Acc':<10} | {'Batch_Loss':<12}")
print("="*80)
for epoch in range(epochs):
    epoch_avg_cost = 0.0

    # Train each binary classifier
    for class_idx in range(num_classes):
        # Create binary labels for the current classifier (1 if it's the target class, -1 otherwise)
        # Use jnp.where to ensure targets are JAX arrays
        y_train_class_binary = jnp.where(y_train_full == class_idx, 1, -1)

        # Shuffle data for stochastic gradient descent
        # Permutation can use np_classic as it's just indices
        permutation = np_classic.random.permutation(X_train_scaled.shape[0])
        X_train_shuffled = X_train_scaled[permutation]
        y_train_shuffled = y_train_class_binary[permutation]

        current_params = all_params_multiclass[class_idx]

        # Mini-batch training for the current classifier
        for i in range(0, X_train_scaled.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            # opt.step expects JAX arrays for differentiable parameters and inputs
            current_params = opt.step(cost_function_binary, current_params, features=X_batch, targets=y_batch)

        all_params_multiclass[class_idx] = current_params  # Update the parameters for this classifier

        # Calculate cost for the current classifier on the full training set
        current_classifier_cost = cost_function_binary(current_params, X_train_scaled, y_train_class_binary)
        epoch_avg_cost += current_classifier_cost / num_classes  # Accumulate average cost

    overall_avg_costs.append(epoch_avg_cost)

    # Evaluate overall test accuracy after all classifiers are updated in this epoch
    # Function to get predictions from all classifiers
    def get_multiclass_predictions(features_set, params_list):
        all_scores = []
        for class_idx_eval in range(num_classes):
            current_params_eval = params_list[class_idx_eval]
            # Get the raw output (sum of scores) for this classifier
            # The list comprehension results in Python floats/JAX arrays, then converted to jnp.array
            scores_sum_for_class = jnp.array([qvc_binary_classifier(current_params_eval, f) for f in features_set])
            # Divide by 3.0 to get the average score
            scores_avg_for_class = scores_sum_for_class / 3.0
            all_scores.append(scores_avg_for_class)
        
        # Stack scores: shape (num_classes, num_samples)
        all_scores = jnp.array(all_scores)
        # The predicted class is the one whose classifier gave the highest score
        predicted_classes = jnp.argmax(all_scores, axis=0) # Use jnp.argmax
        return predicted_classes

    final_test_predictions = get_multiclass_predictions(X_test_scaled, all_params_multiclass)
    
    overall_test_accuracy = jnp.mean(jnp.array(final_test_predictions) == jnp.array(y_test_full)) # Ensure all are JAX arrays for comparison
    overall_test_accuracies.append(overall_test_accuracy)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:4d} | Avg Cost: {epoch_avg_cost:.4f} | Overall Test Acc: {overall_test_accuracy:.4f}")

print("\nMulticlass classification training complete!")
print(f"Final overall training average cost: {overall_avg_costs[-1]:.4f}")
print(f"Final overall test accuracy: {overall_test_accuracies[-1]:.4f}")


# --- Phase 4: Evaluation and Visualization ---

# Step 10: Plot Training Progress
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Convert costs to classic numpy for plotting if they are JAX arrays
plt.plot(range(epochs), np_classic.array(overall_avg_costs))
plt.title("Multiclass Average Cost Evolution")
plt.xlabel("Epoch")
plt.ylabel("Average Cost")
plt.grid(True)

plt.subplot(1, 2, 2)
# Convert accuracies to classic numpy for plotting
plt.plot(range(epochs), np_classic.array(overall_test_accuracies))
plt.title("Multiclass Overall Test Accuracy Evolution")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 11: Visualize Sample Predictions
# Function to get multiclass predictions using the trained models
def predict_multiclass_sample(features_set_single_sample):
    # This function is meant for a single sample prediction for visualization
    all_scores = []
    for class_idx_eval in range(num_classes):
        current_params_eval = all_params_multiclass[class_idx_eval]
        # Get the raw output (sum of scores) for this classifier for a single sample
        scores_sum_for_class = qvc_binary_classifier(current_params_eval, features_set_single_sample)
        # Divide by 3.0 to get the average score
        scores_avg_for_class = scores_sum_for_class / 3.0
        all_scores.append(scores_avg_for_class)
    
    all_scores = jnp.array(all_scores)
    predicted_class = jnp.argmax(all_scores)
    return predicted_class

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes.flat):
    idx = np_classic.random.randint(0, len(X_test_scaled)) # Use np_classic for random int
    image_original_dim = X_test_full[idx].reshape(28, 28)
    true_label = y_test_full[idx]
    
    # Predict for one sample using the dedicated function
    predicted_label = predict_multiclass_sample(X_test_scaled[idx])

    ax.imshow(image_original_dim, cmap='gray')
    ax.set_title(f"True: {true_label}\nPred: {predicted_label}")
    ax.axis('off')
plt.suptitle("Multiclass Classification Sample Predictions (9-Qubit PCA-Reduced MNIST)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
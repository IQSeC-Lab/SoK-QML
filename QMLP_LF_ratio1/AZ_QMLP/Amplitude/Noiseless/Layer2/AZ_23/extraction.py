import numpy as np

# Load the AZ-Class dataset
train_data = np.load("AZ-Class-Task_23_families_train.npz")
test_data = np.load("AZ-Class-Task_23_families_test.npz")

# See what arrays are inside
print("Train Keys:", train_data.files)
print("Test Keys:", test_data.files)

# Check shapes of arrays
for key in train_data.files:
    print(f"train[{key}] shape: {train_data[key].shape}")

for key in test_data.files:
    print(f"test[{key}] shape: {test_data[key].shape}")

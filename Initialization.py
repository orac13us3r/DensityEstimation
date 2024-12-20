import numpy as np
import scipy.io

# Load the training data
train_data = scipy.io.loadmat('train_data.mat')
X_train = train_data['x']  # Assuming 'X' is the key for the image data
# Load and normalize test data
test_data = scipy.io.loadmat('test_data.mat')
X_test = test_data['x']  # Assuming 'X' is the key for the image data

# Reshape the data if necessary
# If X_train is not already in the shape (n_samples, 784), reshape it
def reshape_data(X, target_shape=(-1, 784)):
    if X.shape[1:] != target_shape[1:]:
        return X.reshape(target_shape)
    return X

X_train = reshape_data(X_train)
X_test = reshape_data(X_test)

# Compute mean and standard deviation for each feature
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Function to normalize data
def normalize_data(X, mean, std):
    return (X - mean) / std

# Normalize training data
X_train_normalized = normalize_data(X_train, mean, std)
X_test_normalized = normalize_data(X_test, mean, std)

print("Shape of normalized training data:", X_train_normalized.shape)
print("Shape of normalized test data:", X_test_normalized.shape)

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def perform_pca(X):
    # Compute the covariance matrix
    cov_matrix = np.cov(X.T)

    # Perform eigen analysis
    eigenvalues = np.linalg.eig(cov_matrix)
    eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvectors by decreasing eigenvalues
    index = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]

    # The columns of eigenvectors are the principal components
    principal_components = eigenvectors
    return principal_components, eigenvalues

# Perform PCA on the normalized training data
principal_components = perform_pca(X_train_normalized)
eigenvalues = perform_pca(X_train_normalized)

# Print the shape of the principal components matrix and first 10 eigen values
print("Shape of principal components:", principal_components.shape)
print("First 10 eigenvalues:", eigenvalues[:10])

def project_data(X, principal_components, n_components=2):
    return np.dot(X, principal_components[:, :n_components]) #dot product

# Project data onto the first two principal components
X_train_projected = project_data(X_train_normalized, principal_components)
X_test_projected = project_data(X_test_normalized, principal_components)

train_data = scipy.io.loadmat('train_data.mat')
test_data = scipy.io.loadmat('test_data.mat')
y_train = train_data['y'].ravel()  # Assuming 'y' is the key for labels, this compressed the multidimensional data into 1D
y_test = test_data['y'].ravel()

# Plotting
plt.figure(figsize=(11, 5))

# Plot training data
plt.subplot(121)
for class_label in np.unique(y_train):
    mask = y_train == class_label
    plt.scatter(X_train_projected[mask, 0], X_train_projected[mask, 1],
                label=f'Class {class_label}', alpha=0.6)
plt.title('Training Data Projection')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()

# Plot testing data
plt.subplot(122)
for class_label in np.unique(y_test):
    mask = y_test == class_label
    plt.scatter(X_test_projected[mask, 0], X_test_projected[mask, 1],
                label=f'Class {class_label}', alpha=0.6)
plt.title('Testing Data Projection')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()

plt.tight_layout()
plt.show()

# Observe the distribution
print("Observation:")
print("1. Separation of clusters across a 2D space.")
print("2. Shape and Definition")
print("3. Symmetry")

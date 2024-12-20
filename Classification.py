import numpy as np
from scipy.stats import multivariate_normal

def bayesian_classifier(X, gaussian_params, prior_probs):
    classes = list(gaussian_params.keys())
    posteriors = np.zeros((X.shape[0], len(classes)))

    for i, c in enumerate(classes):
        likelihood = multivariate_normal(gaussian_params[c]['mean'], gaussian_params[c]['cov']).pdf(X)
        posteriors[:, i] = likelihood * prior_probs[c]

    return np.argmax(posteriors, axis=1)

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

# Set prior probabilities (assumed to be equal as per the project description)
prior_probs = {0: 0.5, 1: 0.5}

# Classify training data
gaussian_params = estimate_gaussian_parameters(X_train_projected, y_train)
y_train_pred = bayesian_classifier(X_train_projected, gaussian_params, prior_probs)
train_accuracy = calculate_accuracy(y_train, y_train_pred)

# Classify testing data
y_test_pred = bayesian_classifier(X_test_projected, gaussian_params, prior_probs)
test_accuracy = calculate_accuracy(y_test, y_test_pred)

print(f"Training set accuracy: {train_accuracy:.2f}%")
print(f"Testing set accuracy: {test_accuracy:.2f}%")

# Visualize the decision boundary
plt.figure(figsize=(10, 8))

# Plot the training data points
for c in np.unique(y_train):
    mask = y_train == c
    plt.scatter(X_train_projected[mask, 0], X_train_projected[mask, 1],
                label=f'Class {c}', alpha=0.6)

# Create a grid of points
x, y = np.mgrid[-6:6:.01, -6:6:.01]
pos = np.dstack((x, y))

# Classify each point in the grid
grid_z = bayesian_classifier(pos.reshape(-1, 2), gaussian_params, prior_probs)
grid_z = grid_z.reshape(x.shape)

# Plot the decision boundary
plt.contour(x, y, grid_z, levels=[0.9], colors='k', linestyles='dashed')

plt.title('Bayesian Decision Boundary')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True)
plt.show()

print("\nObservation:")
print("1. Compare the training and testing accuracies.")

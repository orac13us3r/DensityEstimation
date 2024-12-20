import numpy as np
from scipy.stats import multivariate_normal

def estimate_gaussian_parameters(X, y):
    classes = np.unique(y) #unique parameters in y for reduction
    params = {}

    for c in classes:
        X_c = X[y == c]
        mean = np.mean(X_c, axis=0)
        cov = np.cov(X_c, rowvar=False)
        params[c] = {'mean': mean, 'cov': cov}

    return params

# Estimate parameters for each class using the projected training data
gaussian_params = estimate_gaussian_parameters(X_train_projected, y_train)

# Print the estimated parameters
for c, params in gaussian_params.items():
    print(f"\nClass {c} parameters:")
    print(f"Mean: {params['mean']}")
    print(f"Covariance matrix:\n{params['cov']}")

# Visualize the estimated distributions
plt.figure(figsize=(10, 8))

# Plot the training data points
for c in np.unique(y_train):
    mask = y_train == c
    plt.scatter(X_train_projected[mask, 0], X_train_projected[mask, 1],
                label=f'Class {c} data', alpha=0.7)

# Plot the estimated distributions
x, y = np.mgrid[-6:6:.1, -6:6:.1]
pos = np.dstack((x, y))

for c, params in gaussian_params.items():
    rv = multivariate_normal(params['mean'], params['cov'])
    plt.contour(x, y, rv.pdf(pos), levels=5, cmap=f'Blues' if c == 0 else 'Reds', alpha=0.5)

plt.title('2D Gaussian Distributions for Each Class')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True)
plt.show()

print("\nObservation:")
print("1. Alignment of Estimation against data points.")
print("2. Shape and orientation of the contours for each class.")
print("3. Consider how well the Gaussian assumption fits the observed data distribution.")

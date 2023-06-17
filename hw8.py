import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate a random dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.0)

# Create SVM classifiers
hard_svm = SVC(kernel='linear', C=1e6)
soft_svm = SVC(kernel='linear', C=0.5)
kernel_svm = SVC(kernel='rbf', gamma='scale')

# Fit the SVM classifiers
hard_svm.fit(X, y)
soft_svm.fit(X, y)
kernel_svm.fit(X, y)

# Calculate accuracy for each SVM classifier
hard_svm_accuracy = accuracy_score(y, hard_svm.predict(X))
soft_svm_accuracy = accuracy_score(y, soft_svm.predict(X))
kernel_svm_accuracy = accuracy_score(y, kernel_svm.predict(X))

print("Hard SVM Accuracy:", hard_svm_accuracy)
print("Soft SVM Accuracy:", soft_svm_accuracy)
print("Kernel SVM Accuracy:", kernel_svm_accuracy)

# Define the plot boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Create a meshgrid of points to plot the decision boundaries
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Predict the labels for each point in the meshgrid
hard_svm_Z = hard_svm.predict(np.c_[xx.ravel(), yy.ravel()])
soft_svm_Z = soft_svm.predict(np.c_[xx.ravel(), yy.ravel()])
kernel_svm_Z = kernel_svm.predict(np.c_[xx.ravel(), yy.ravel()])

# Reshape the predictions to match the meshgrid shape
hard_svm_Z = hard_svm_Z.reshape(xx.shape)
soft_svm_Z = soft_svm_Z.reshape(xx.shape)
kernel_svm_Z = kernel_svm_Z.reshape(xx.shape)

# Plot the decision boundaries and the data points
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.contourf(xx, yy, hard_svm_Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Hard SVM')

plt.subplot(1, 3, 2)
plt.contourf(xx, yy, soft_svm_Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Soft SVM')

plt.subplot(1, 3, 3)
plt.contourf(xx, yy, kernel_svm_Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Kernel SVM')

plt.tight_layout()
plt.show()

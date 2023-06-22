import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

# Read the dataset
data = pd.read_csv('d6.csv')

# Drop the ID column
data = data.drop('Customer ID', axis=1)

# Perform one-hot encoding on the Gender column
onehot_encoder = OneHotEncoder(sparse=False)
gender_encoded = onehot_encoder.fit_transform(data[['Gender']])
data['Gender_Female'] = gender_encoded[:, 0]
data['Gender_Male'] = gender_encoded[:, 1]
data = data.drop('Gender', axis=1)

# Apply k-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(data)

# Plot the clusters
plt.scatter(data['Annual Income ($)'], data['Spending Score (0-100)'], c=labels, cmap='viridis')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (0-100)')
plt.title('Customer Clusters')
plt.show()

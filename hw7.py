import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the customer data from CSV file
df = pd.read_csv('d7.csv')

# Extract the features for clustering
X = df.iloc[:, 0:3].values

# Perform k-means++ clustering with 2 clusters
k = 2
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Add the cluster labels as a new column in the dataframe
df['Cluster'] = labels

# Visualize the clustered data
plt.scatter(df['Annual Income ($)'], df['Spending Score (0-100)'], c=labels, cmap='viridis')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (0-100)')
plt.title('K-Means++ Clustering')
plt.show()

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from CSV file
df = pd.read_csv('customer_churn_dataset.csv')

# Drop the 'Customer ID' column
df = df.drop('Customer ID', axis=1)

# Perform one-hot encoding for the 'Gender' feature
df_encoded = pd.get_dummies(df, columns=['Gender'])

# Separate the features (X) and the target variable (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Create and train the KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# Calculate the accuracy of the model on the training data
y_train_pred = model.predict(X)
train_accuracy = accuracy_score(y, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

# Create a test datapoint
test_datapoint = pd.DataFrame({
    'Age': [35],
    'Monthly Charges': [50],
    'Total Charges': [500],
    'Number of Calls': [100],
    'Number of Messages': [20],
    'Gender_Male': [1],
    'Gender_Female': [0]
})

# Reorder the columns in the test datapoint to match the order of the training data
test_datapoint = test_datapoint[X.columns]

# Predict the churn status of the test datapoint
prediction = model.predict(test_datapoint)

print(f"Churn Prediction: {prediction}")

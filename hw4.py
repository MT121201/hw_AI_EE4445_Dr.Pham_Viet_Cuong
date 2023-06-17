import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Read the dataset from the CSV file
df = pd.read_csv('d4.csv')

# Split the dataset into input features (X) and target variable (y)
X = df.drop('Sales Revenue', axis=1)
y = df['Sales Revenue']
X = X.values
y = y.values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict sales revenue for a new input
new_input = [[1000, 500, 200]]  # Example input values
predicted_revenue = model.predict(new_input)
print(f"Predicted Sales Revenue: {predicted_revenue}")

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        activation = 1 if summation > 0 else 0
        return activation

    def train(self, training_inputs, labels, num_epochs):
        for _ in range(num_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# Training examples
training_inputs = np.array([
    [10, 5],   # Cat with size 10 and weight 5
    [5, 3],    # Cat with size 5 and weight 3
    [8, 7],    # Cat with size 8 and weight 7
    [12, 9],   # Dog with size 12 and weight 9
    [7, 4],    # Dog with size 7 and weight 4
    [11, 8]    # Dog with size 11 and weight 8
])
labels = np.array([0, 0, 0, 1, 1, 1])  # 0 for cats, 1 for dogs

perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels, num_epochs=10)

# Make predictions
test_inputs = np.array([
    [6, 2],    # Cat with size 6 and weight 2
    [9, 6]     # Dog with size 9 and weight 6
])

for inputs in test_inputs:
    prediction = perceptron.predict(inputs)

    if prediction == 0:
        print(f"Inputs {inputs}: Classified as a cat.")
    else:
        print(f"Inputs {inputs}: Classified as a dog.")

import numpy as np

# Define the objective function f(x, y)
def f(x, y):
    return x**2 + y**4 - 2*x*y + x - y

# Define the gradient of the objective function
def grad_f(x, y):
    grad_x = 2*x - 2*y + 1
    grad_y = 4*y**3 - 2*x - 1
    return np.array([grad_x, grad_y])

# Define the gradient descent method
def gradient_descent(x_init, y_init, learning_rate, num_iterations):
    x = x_init
    y = y_init
    for i in range(num_iterations):
        gradient = grad_f(x, y)
        x -= learning_rate * gradient[0]
        y -= learning_rate * gradient[1]
    return x, y

# Set the initial points and learning rates
initial_points = [(0, 0), (2, -1), (-3, 2)]
learning_rates = [0.1, 0.01, 0.001]

# Apply gradient descent for each combination of initial point and learning rate
for point in initial_points:
    for lr in learning_rates:
        x_opt, y_opt = gradient_descent(point[0], point[1], lr, num_iterations=100)
        print(f"Initial point: {point}, Learning rate: {lr}")
        print(f"Optimal solution: x = {x_opt}, y = {y_opt}, f(x, y) = {f(x_opt, y_opt)}")
        print()

import numpy as np

# Given dataset
data = np.array([
    [3.14, 65126],
    [1.73, 45567],
    [2.10, 71234],
    [2.59, 36487],
    [3.70, 50211],
    [2.84, 62238],
    [3.95, 26745],
    [1.47, 37456],
    [3.32, 41567],
    [0.82, 56982],
    [1.19, 30756],
    [2.92, 58009],
    [2.28, 46433],
    [0.44, 75211],
    [1.78, 59654],
    [1.25, 47612],
    [2.91, 41102],
    [3.82, 45781],
    [1.64, 65327],
    [3.08, 38940]
])

# Perform min-max scaling
normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Print the normalized data
print("Normalized Data:")
for row in normalized_data:
    print(row)

import numpy as np

# Training data, last column: bias
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Weights
np.random.seed(1)
w_1 = np.random.uniform(size=(3, 4))
w_2 = np.random.uniform(size=(4, 1))
# Epochs
epochs = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return np.subtract(x, np.square(x))

for epoch in range(epochs):
    # Layers
    l1 = sigmoid(np.dot(X, w_1))
    l2 = sigmoid(np.dot(l1, w_2))

    # Error
    l2_error = y - l2
    l2_delta = l2_error * sigmoid_prime(l2)
    l1_error = np.dot(l2_delta, w_2.T)
    l1_delta = l1_error * sigmoid_prime(l1)

    # Update weights
    w_1 += np.dot(X.T, l1_delta)
    w_2 += np.dot(l1, l2_delta)

    # Output
    output = np.round(l2)

print(output)

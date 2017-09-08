import numpy as np

# Training data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Weights
w_1 = np.random.uniform(size=(2, 4))
w_2 = np.random.uniform(size=(4, 1))
# Bias
bias = np.random.uniform(size=(4, 4))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for epoch in range(10000):
    # Layers
    l1 = np.add(np.dot(X, w_1), bias)
    l1 = sigmoid(l1)
    l2 = np.dot(l1, w_2)
    l2 = sigmoid(l2)

    # Loss
    error = np.subtract(y, l2)
    loss = np.sum(np.square(error))
    deriv = -2 * np.sum(error)

    # TODO: Optimize

print(deriv)

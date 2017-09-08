import numpy as np


def sigmoid(x, deriv=False):
    '''
    Sigmoid is a type of non-linear function  that
    maps any value to a value between 0 and 1.  If
    the deriv=True flag is passed on, the function
    would instead calculate  the derivative of the
    function which is used in the error back prop.
    '''
    if deriv == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# Input data
# 4 training examples with 3 input nodes
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Output data
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed set so that it will return the same random numbers each time
np.random.seed(1)

# Initialise weights to random values
# 3x4 matrix of weights - 3 input nodes and 4 hidden nodes
syn0 = 2 * np.random.random((3, 4)) - 1
# 4x1 matrix of weights - 4 hidden nodes and 1 output node
syn1 = 2 * np.random.random((4, 1)) - 1

# Training loop
for i in range(100000):
    # Layers
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(11, syn1))

    # Error rate
    l2_error = y - l2

    # Output error rate for every 10,000 epoch
    if (i % 10000 == 0):
        print("Error {}".format(str(np.mean(np.abs(l2_error)))))

    # Error deltas
    l2_delta = l2_error * sigmoid(l2, deriv=True)
    l1_error = np.dot(l2_delta, syn1.T)
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    # Update weights
    syn0 += np.dot(l0.T, l1_delta)
    syn1 += np.dot(l1.T, l2_delta)


# Print output
print("Output after training")
print(l2)

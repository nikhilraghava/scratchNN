import numpy as np


class NeuralNetwork():
    def __init__(self):
        # Seed random number generator
        np.random.seed(1)
        # Training data
        self.X = np.array([[0, 0, 1],
                           [0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])

        self.y = np.array([[0],
                           [1],
                           [1],
                           [0]])

        # Weights
        self.w_1 = np.random.uniform(size=(3, 4))
        self.w_2 = np.random.uniform(size=(4, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return np.subtract(x, np.square(x))

    def train_network(self, inputs, outputs, epochs):
        inpt = inputs
        outpt = outputs
        # Training loop
        for epoch in range(epochs):
            # Layers
            l1 = self.sigmoid(np.dot(inpt, self.w_1))
            l2 = self.sigmoid(np.dot(l1, self.w_2))

            # Error
            l2_error = outpt - l2
            l2_delta = l2_error * self.sigmoid_derivative(l2)
            l1_error = np.dot(l2_delta, self.w_2.T)
            l1_delta = l1_error * self.sigmoid_derivative(l1)

            # Update weights
            self.w_1 += np.dot(inpt.T, l1_delta)
            self.w_2 += np.dot(l1, l2_delta)

        # Return final output
        return l2

    def save_model(self):
        weights = np.array([self.w_1, self.w_2])
        np.save('model.npy', weights)

    def run_saved_model(self, input):
        weights = np.load('model.npy')
        # Load weights
        w_1 = weights[0]
        w_2 = weights[1]
        # Layers
        l1 = self.sigmoid(np.dot(input, w_1))
        l2 = self.sigmoid(np.dot(l1, w_2))
        # Return output
        return l2


if __name__ == '__main__':
    nn = NeuralNetwork()
    X, y = nn.X, nn.y
    nn.train_network(X, y, 60000)
    nn.save_model()
    output = nn.run_saved_model(X)
    print(output)


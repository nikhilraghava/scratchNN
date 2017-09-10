# ScratchNN

ScratchNN is a neural network implemented using Numpy, from scratch. We train the model to map inputs to outputs of the Exclusive-OR gate. Line by line explanation of the code can be found below.

## Implementation

Before we define our model, let's take a look at our training data. For this model, we will only be having 4 rows of training data. The training data for our model is the 2 input XOR gate's truth table. 

| A  | B  | Output |
|:--:|:--:|:------:|
|0   |0   |0       |
|0   |1   |1       |
|1   |0   |1       |
|1   |1   |0       |

Now let's look at a diagrammatic representation of our model.

<div align="center">
    <br><img src="https://cldup.com/zJzDxXKT-z.png"><br>
</div>

From our table, we know that we have 2 inputs: A and B that are being mapped to an output. So we define 2 input nodes and an output node with a hidden layer consisting of 4 hidden nodes in the middle. The number of hidden nodes in the hidden layer can vary and it is up to us to decide. Now let's implement this model with the help of Numpy.

```python
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
```

Here, we define a class, `NeuralNetwork` and then initialize the training data and weights. Notice that our input data, `X`, has 3 columns instead of 2. The extra column is to accomodate the bias. The weight, `self.w_1` is a `3x4` matrix between the input layer and the output layer. It is a `3x4` matrix because we have 3 input nodes: 2 data input nodes,an extra node for the bias and 4 hidden layer nodes. The weight, `self.w_2` is a `4x1` matrix betweeen the hidden layer and the output layer. It is a `4x1` matrix because we have 4 hidden layer nodes and an output node. Both the weights are matricies with random values between 0 and 1. Next we define our sigmoid function.

```python
def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```

The function follows the logistic function's equation.

<div align="center">
    <br><img src="https://cldup.com/6x0KSbiNuV.png" width="311.5" height="151.5"><br>
</div>

The sigmoid function helps us to map any value to a value between 0 and 1.This is useful when we have a binary classification task. Now we need to define the derivative of the sigmoid function.

```python
def sigmoid_derivative(self, x):
        return np.subtract(x, np.square(x))
```

The derivative of the sigmoid is useful when we are computing the loss during the backpropagation. The calculation to obtain the derivative of the sigmoid function:

<div align="center">
    <br><img src="https://cldup.com/KBcewAjNuR.png"><br>
</div>

Now let us define our model.

```python
def train_network(self, inpt, output, epochs):
        # Training loop
        for epoch in range(epochs):
            # Layers
            l1 = self.sigmoid(np.dot(inpt, self.w_1))
            l2 = self.sigmoid(np.dot(l1, self.w_2))

            # Error
            l2_error = output - l2
            l2_delta = l2_error * self.sigmoid_derivative(l2)
            l1_error = np.dot(l2_delta, self.w_2.T)
            l1_delta = l1_error * self.sigmoid_derivative(l1)

            # Update weights
            self.w_1 += np.dot(inpt.T, l1_delta)
            self.w_2 += np.dot(l1, l2_delta)

        # Return final output
        return l2
```
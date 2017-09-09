# ScratchNN

ScratchNN is a neural network implemented using Numpy, from scratch. We train the model to map the inputs to outputs of the Exclusive-NOR gate. Line by line explanation of the code can be found below.

## Implementation

Before we define our model, let's take a look at our training data. For this model, we will only be having 4 rows of training data. The training data for our model is just four lines of the XNOR gate's truth table. 

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

From our table, we know that we have 2 inputs: A and B that are being mapped to an output. So we define 2 input nodes and an output node with a hidden layer consisting of 4 hidden nodes in the middle. Now let's implement this model with the help of Numpy.

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

Here, we define a class, `NeuralNetwork` and then initialize the training data and the weights. The weight, `self.w_1` is the weight between the input layer and the hidden layer and it is a `3x4` matrix (3 input nodes, 2 data input nodes and an additional node to accommodate the bias) of random numbers between `0` and `1`. The weight, `self.w_2` is the weight between the hidden layer and the output layer and it is a `4x1` matrix (4 hidden nodes and an output node) of random numbers between `0` and `1`. Now we need to define our sigmoid function.
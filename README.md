# ScratchNN

ScratchNN is a neural network implemented using Numpy, from scratch. We train the model to map the inputs and outputs of the Exclusive-NOR gate. Line by line explaination of the code can be found below.

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

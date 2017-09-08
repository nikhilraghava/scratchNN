# scratchNN


scracthNN is a neural network implemented using Numpy, from scratch. We train the model to map the inputs and outputs of the Exclusive-NOR gate. Line by line explaination of the code can be found below.

## Implementation

Before we type our code, let's get our training data. For this model, we will only be having 4 instances of training data. 

| A  | B  | C  | Output |
|:--:|:--:|:--:|:------:|
|0   |0   |1   |0       |
|0   |1   |1   |1       |
|1   |0   |1   |1       |
|1   |1   |1   |1       |

The actual truth for the XNOR gate is longer but we will only be training using 4 examples. So now let's import `numpy as np` and get started.

```python
import numpy as np


def sigmoid(x, deriv=False):
    if deriv == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
```
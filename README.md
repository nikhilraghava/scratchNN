# ScratchNN

Artificial Neural Networks are statistical learning models inspired by biological neural networks (aka our brain). When we learn new concepts, we form an abstract understanding about them and apply it at a later time to solve/understand a different concept or problem. In this repository, we will be building a neural network that is able to understand the XOR gate's truth table and predict outputs based on a given set of inputs.

## Introduction to Neural Networks

Before we create an "artificial brain" that is capable of understanding the XOR gate's truth table, we first need to understand how our own brains work. Our brain is made up of millions of neurons and are connected to each other via synapses. Similarly, our own artificial neural network is made up of nodes, which are similar to neurons, and inter-node connections, which are similar to synapses. A neural network has three layers to it: an input layer, a hidden layer, and an output layer. There can be `n` number of hidden layers in a neural network and if the number of hidden layers is two or more, the neural network is known as a deep neural network. Hidden layers are "hidden" because they are not visible as network output. When we train a neural network we are essentially making the neural network "understand" how much each individual input influences the output. For example, if we design a neural network to predict if it would rain on a given day using temperature, humidity and wind speed as its inputs, then the goal of training would be to make the neural network "understand" how much each of those factors contributes individually to cause rain. So humidity might be a more dominant factor as compared to temperature or wind speed in causing rain. 

This is called "weighting" and when we train, our goal is to tune these weights to achieve the most accurate "understanding" of how each of those inputs influence the output. Sometimes, weighing those inputs might just not be enough for us to "understand" all our data. For example, if we know that wind speed makes a 20% contribution towards the likelihood of it raining on a given day, and if there were cases where there was no wind and it rained on a certain day, then our model, who just depended on weighing these inputs would not know that there is still a chance that it would rain on a non-windy day. To solve this we use what's known as a bias. A bias helps tune our model further and avoid potential discrepancies caused by just weighing our inputs. 

Neural networks are only able to handle numerical data and when we weigh and add biases to numerical data, we get numbers that don't necessarily mean anything unless they are passed through an activation function. An activation function limits and maps numerical outputs to values between a certain range (in our case, 0 and 1). So a number closer to 0 indicates that it would not rain and a number closer to 1 would indicate that it would rain.

## Data and Architecture

Our training data: a 2-input XOR gate's truth table.

| A  | B  | Output |
|:--:|:--:|:------:|
|0   |0   |0       |
|0   |1   |1       |
|1   |0   |1       |
|1   |1   |0       |

From our truth table, we know that we need two input nodes for our input layer and an output node for our output layer. To keep things simple, we will only be having a single hidden layer with four hidden nodes. An extra node will be placed at the input layer to accomodate bias for our neural network.

<div align="center">
    <br><img src="https://cldup.com/i2VxUILC0S.png" width="500" height="291"><br><br>
</div>

Our first layer, which is actually our hidden layer is represented using the following equation:

<div align="center">
    <br><img src="https://cldup.com/0tGOWR_4nT.png" width="275.7" height="24.9"><br><br>
</div>

Where `sigmoid` is defined as:

<div align="center">
    <br><img src="https://cldup.com/3F7AHo-uqa.png" width="206.1" height="52.5"><br><br>
</div>

Our second layer, which is our output layer is represented using the following equation:

<div align="center">
    <br><img src="https://cldup.com/Yh6ygfmoKW.png" width="321" height="24.9"><br><br>
</div>

Since we use a random set of initial weights, the output is almost always going to off the intended output, so we need a way to adjust our weights and this process is known as back propagation. In order to improve our model, we first need to quantify how wrong our predictions are and then adjust our weights in such a way that the difference between the output of our model and the intended output decreases over time.


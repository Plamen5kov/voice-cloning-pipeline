# Lab: L-Layer Deep Neural Network

## Overview

In this lab, you will build a deep neural network with as many layers as you want. This is the foundation for understanding how deep learning models work.

## Learning Objectives

By the end of this assignment, you'll be able to:
- Use non-linear units like ReLU to improve your model
- Build a deeper neural network (with more than 1 hidden layer)
- Implement an easy-to-use neural network class
- Understand forward and backward propagation in deep networks

## Structure

The lab is divided into the following sections:

1. **Initialization** - Initialize parameters for 2-layer and L-layer networks
2. **Forward Propagation** - Implement the forward pass through the network
3. **Cost Function** - Compute the cross-entropy cost
4. **Backward Propagation** - Implement backpropagation to compute gradients
5. **Update Parameters** - Update weights and biases using gradient descent

## Files

- `l_layer_neural_network.ipynb` - Main notebook with exercises
- `dnn_utils.py` - Helper functions for activation functions
- `testCases.py` - Test data for exercises
- `public_tests.py` - Public test functions
- `requirements.txt` - Python dependencies

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Open the Jupyter notebook and follow the instructions:

```bash
jupyter notebook l_layer_neural_network.ipynb
```

Complete each exercise by filling in the code sections marked with `# YOUR CODE STARTS HERE` and `# YOUR CODE ENDS HERE`.

## Notation

- Superscript $[l]$ denotes a quantity associated with the $l^{th}$ layer
- Superscript $(i)$ denotes a quantity associated with the $i^{th}$ example
- Lowerscript $i$ denotes the $i^{th}$ entry of a vector

## Tips

- Read the instructions carefully for each exercise
- Use the provided test cases to verify your implementation
- Don't modify the function signatures or add extra print statements
- Keep track of matrix dimensions to avoid shape errors

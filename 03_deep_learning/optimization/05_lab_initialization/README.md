# Lab 05: Initialization

## Overview

This lab explores the critical importance of weight initialization in deep neural networks. You'll implement and compare three different initialization methods:

- **Zero Initialization**: All weights initialized to zero (fails to break symmetry)
- **Random Initialization**: Large random values (can slow convergence)
- **He Initialization**: Scaled random initialization (recommended for ReLU networks)

## Learning Objectives

- Understand the importance of breaking symmetry in neural networks
- Implement different initialization strategies
- Observe how initialization affects training convergence and performance
- Learn why He initialization works well for ReLU activation functions

## Files

- `initialization.ipynb` - Main notebook with exercises and experiments
- `init_utils.py` - Helper functions for forward/backward propagation
- `public_tests.py` - Test cases for your implementations
- `README.md` - This file

## Setup

Install required dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

## Exercises

### Exercise 1: Zero Initialization
Implement `initialize_parameters_zeros()` to initialize all parameters to zero. You'll discover why this doesn't work well.

### Exercise 2: Random Initialization
Implement `initialize_parameters_random()` to initialize weights with large random values. Observe the trade-offs.

### Exercise 3: He Initialization
Implement `initialize_parameters_he()` using the He initialization formula. This works best for ReLU networks.

## Key Concepts

### Why Zero Initialization Fails
When all weights are zero, all neurons in each layer learn the same features (symmetry problem). The network becomes no more powerful than a linear classifier.

### Random Initialization
Breaking symmetry is essential, but very large random values can cause:
- Vanishing/exploding gradients
- Slow convergence
- High initial cost

### He Initialization
Scales weights by $\sqrt{\frac{2}{n^{[l-1]}}}$ where $n^{[l-1]}$ is the number of neurons in the previous layer. This helps maintain:
- Appropriate variance in activations
- Efficient gradient flow
- Fast convergence with ReLU activations

## Expected Results

| Model | Train Accuracy | Comment |
|-------|---------------|---------|
| Zero Initialization | ~50% | Fails to break symmetry |
| Large Random | ~83% | Too large weights |
| He Initialization | ~99% | Recommended method |

## Running the Lab

1. Open `initialization.ipynb` in Jupyter or VS Code
2. Run the setup cells to import libraries and load data
3. Implement the three initialization functions
4. Run the model with each initialization method
5. Compare results and decision boundaries

## Tips

- Pay attention to the shapes of weight matrices: `(layers_dims[l], layers_dims[l-1])`
- Bias vectors can be initialized to zeros
- Use `np.random.seed(3)` as specified to get consistent results
- The He scaling factor is $\sqrt{\frac{2}{\text{previous layer size}}}$

## Further Reading

- [He et al., 2015 - Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
- [Xavier/Glorot Initialization](http://proceedings.mlr.press/v9/glorot10a.html)
- Understanding the difficulty of training deep feedforward neural networks

## Assignment Notes

This is part of the Deep Learning Specialization coursework. Make sure to:
- Not add extra print statements in graded functions
- Keep function parameters unchanged
- Avoid using global variables
- Submit only after all tests pass

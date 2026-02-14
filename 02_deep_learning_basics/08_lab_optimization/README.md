# Lab 08: Optimization Methods

## Overview

This lab explores advanced optimization methods that can speed up neural network training and improve convergence. You'll implement and compare different optimization algorithms including Gradient Descent, Momentum, RMSProp, and Adam.

## Learning Objectives

By the end of this lab, you will be able to:

- Apply optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random mini-batches to accelerate convergence and improve optimization
- Implement learning rate decay and scheduling techniques
- Compare the performance of different optimization algorithms
- Understand the trade-offs between different optimization methods

## Topics Covered

### 1. Gradient Descent
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Parameter update mechanics

### 2. Mini-Batch Gradient Descent
- Creating random mini-batches
- Shuffling and partitioning training data
- Benefits of mini-batch processing

### 3. Momentum
- Exponentially weighted averages
- Velocity initialization
- Smoothing gradient descent updates
- Hyperparameter tuning (beta)

### 4. Adam Optimization
- Adaptive Moment Estimation
- First and second moment estimates
- Bias correction
- Combining momentum and RMSProp

### 5. Learning Rate Decay
- Exponential learning rate decay
- Fixed interval scheduling
- Applying decay to different optimizers

## Files in This Lab

- `optimization.ipynb` - Main notebook with exercises and implementations
- `opt_utils_v1a.py` - Utility functions for forward/backward propagation
- `testCases.py` - Test cases for exercises
- `public_tests.py` - Public test functions
- `requirements.txt` - Python package dependencies
- `images/` - Visualizations and diagrams

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Open the Jupyter notebook:
```bash
jupyter notebook optimization.ipynb
```

## Exercises

The lab includes the following graded exercises:

1. **Exercise 1**: `update_parameters_with_gd` - Implement gradient descent parameter updates
2. **Exercise 2**: `random_mini_batches` - Create random mini-batches from training data
3. **Exercise 3**: `initialize_velocity` - Initialize velocity for momentum
4. **Exercise 4**: `update_parameters_with_momentum` - Implement momentum-based updates
5. **Exercise 5**: `initialize_adam` - Initialize Adam optimizer variables
6. **Exercise 6**: `update_parameters_with_adam` - Implement Adam optimizer
7. **Exercise 7**: `update_lr` - Implement exponential learning rate decay
8. **Exercise 8**: `schedule_lr_decay` - Implement fixed interval learning rate scheduling

## Key Concepts

### Mini-Batch Gradient Descent
- Uses subsets of training data for each update
- Balances speed of SGD with stability of batch GD
- Typical mini-batch sizes: 16, 32, 64, 128

### Momentum
- Reduces oscillations in gradient descent
- Uses exponentially weighted average of past gradients
- Common beta values: 0.8 to 0.999 (typically 0.9)

### Adam
- Combines benefits of Momentum and RMSProp
- Computes adaptive learning rates for each parameter
- Typically requires less hyperparameter tuning
- Default values: beta1=0.9, beta2=0.999, epsilon=1e-8

### Learning Rate Decay
- Gradually reduces learning rate during training
- Allows for larger steps early, finer adjustments later
- Helps achieve better convergence

## Expected Results

When comparing optimization methods on the "moons" dataset:

| Method | Accuracy | Convergence Speed |
|--------|----------|-------------------|
| Gradient Descent | ~71% | Slow |
| Momentum | ~71% | Medium |
| Adam | ~94% | Fast |
| GD + LR Decay | ~94% | Medium-Fast |
| Momentum + LR Decay | ~95% | Fast |

## Tips for Success

1. **Include sufficient context**: When using `oldString` in edits, include 3-5 lines before and after
2. **Check shapes**: Verify tensor dimensions match expected values
3. **Test incrementally**: Run tests after implementing each exercise
4. **Visualize results**: Use the provided plotting functions to compare methods
5. **Experiment**: Try different hyperparameters to see their effects

## Common Pitfalls

- Forgetting to initialize velocity/momentum variables
- Incorrect indexing when creating mini-batches
- Not handling the last mini-batch when size doesn't divide evenly
- Forgetting bias correction in Adam
- Setting learning rate decay too aggressive (goes to zero too quickly)

## Next Steps

After completing this lab, you'll be ready to:
- Apply these optimization techniques to real-world problems
- Fine-tune hyperparameters for your specific use cases
- Understand when to use each optimization method
- Implement custom optimization strategies

## References

- Adam paper: https://arxiv.org/pdf/1412.6980.pdf
- Deep Learning Specialization by Andrew Ng
- Batch Normalization: https://arxiv.org/abs/1502.03167

## Notes

- This lab uses a simplified 3-layer neural network for demonstration
- The "moons" dataset is used for binary classification
- All random operations are seeded for reproducibility
- Images referenced in the notebook should be placed in the `images/` folder

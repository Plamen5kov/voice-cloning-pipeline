# Lab 07: Gradient Checking

## Overview

This lab teaches you how to implement **gradient checking** to verify that your backpropagation implementation is correct. Gradient checking is a critical debugging technique that can catch subtle bugs in your neural network implementation.

## Learning Objectives

By the end of this lab, you will:

- Understand the mathematical foundation of gradient checking
- Implement 1-dimensional gradient checking
- Implement N-dimensional gradient checking for neural networks
- Learn when and how to use gradient checking in practice

## Problem Statement

You are building a deep learning model for fraud detection in mobile payments. Because this is mission-critical, you need to verify that your backpropagation implementation is correct. Gradient checking provides mathematical proof that your gradients are computed correctly.

## Theory

Gradient checking uses the numerical approximation of gradients:

$$\frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon}$$

By comparing this numerical approximation with your backpropagation gradients, you can verify correctness.

## Exercises

### Exercise 1: Forward Propagation (1D)
Implement forward propagation for the simple function $J(\theta) = \theta \cdot x$

### Exercise 2: Backward Propagation (1D)
Implement backward propagation to compute $\frac{\partial J}{\partial \theta} = x$

### Exercise 3: Gradient Check (1D)
Implement gradient checking to verify the 1D backward propagation

### Exercise 4: Gradient Check (N-dimensional)
Implement gradient checking for a full neural network with multiple layers

## Key Concepts

### When to Use Gradient Checking
- ✅ During development to verify backprop implementation
- ✅ After making changes to backprop code
- ❌ Not during training (too slow)
- ❌ Not with dropout (gradient check doesn't work with dropout)

### Interpreting Results
- **Difference < 10⁻⁷**: Excellent! Your gradients are correct
- **Difference ~ 10⁻⁵**: Warning - might have a bug
- **Difference > 10⁻³**: Error - definitely has a bug

## Network Architecture

The N-dimensional gradient check uses a 3-layer neural network:

```
Input (4) → [Linear → ReLU] → (5) → [Linear → ReLU] → (3) → [Linear → Sigmoid] → (1)
```

## Files

- `gradient_checking.ipynb` - Main assignment notebook
- `gc_utils.py` - Helper functions (sigmoid, relu, vector conversions)
- `testCases.py` - Test case data
- `public_tests.py` - Public test functions
- `clean_notebook.py` - Utility to clean notebook outputs

## Running the Lab

1. Open `gradient_checking.ipynb` in Jupyter
2. Run all cells sequentially
3. Complete the exercises marked with `# YOUR CODE STARTS HERE`
4. Verify your solutions pass the test cases

## Expected Outputs

- **Exercise 1**: J = 8
- **Exercise 2**: dtheta = 3
- **Exercise 3**: difference ≈ 7.81e-11 (gradient check passes)
- **Exercise 4**: difference ≈ 0.285 (gradient check fails due to intentional bugs in backward_propagation_n)

Note: Exercise 4 is designed to fail initially. You'll need to debug and fix the errors in `backward_propagation_n()`.

## Common Mistakes

1. **Not using the correct epsilon**: Use 1e-7, not larger values
2. **Comparing scalars incorrectly**: Gradient check compares vectors using L2 norm
3. **Using gradient check with dropout**: This won't work - disable dropout first
4. **Running gradient check during training**: Too slow - only use for verification

## Tips

- Start with 1D gradient checking to understand the concept
- The N-dimensional case applies the same principle to all parameters
- The intentional bugs in `backward_propagation_n` are in `dW2` and `db1`
- Use the colored output messages to know if your implementation is correct

## Next Steps

After mastering gradient checking, you'll be ready to:
- Confidently implement complex neural network architectures
- Debug backpropagation issues systematically
- Move on to optimization algorithms and hyperparameter tuning

## References

- Deep Learning Specialization (Coursera)
- Gradient Checking and Advanced Optimization (Andrew Ng)

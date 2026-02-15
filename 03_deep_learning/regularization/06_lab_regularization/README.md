# Lab 06: Regularization

## Overview

This lab explores regularization techniques to prevent overfitting in deep neural networks. You'll implement and compare two powerful regularization methods:

- **L2 Regularization**: Penalizes large weights by adding a regularization term to the cost function
- **Dropout**: Randomly deactivates neurons during training to prevent co-adaptation

## Learning Objectives

- Understand the problem of overfitting in deep learning
- Implement L2 regularization in cost computation and backpropagation
- Implement dropout for forward and backward propagation
- Compare the effectiveness of different regularization techniques
- Learn when and how to apply each regularization method

## Problem Context

You're working with the French Football Corporation to optimize goalkeeper positioning. The model predicts where the goalkeeper should kick the ball based on historical game data.

## Files

- `regularization.ipynb` - Main notebook with exercises
- `reg_utils.py` - Helper functions for neural network operations
- `testCases.py` - Test case data
- `public_tests.py` - Public test functions
- `datasets/` - Dataset directory (data.mat file needed)
- `README.md` - This file

## Setup

Install required dependencies:

```bash
pip install numpy matplotlib scikit-learn scipy
```

### Dataset

This lab requires a dataset file `datasets/data.mat`. The dataset contains:
- Training data: positions on a football field
- Labels: whether French players (blue=1) or opponents (red=0) hit the ball

**Note**: You may need to create or download the dataset separately. The data represents 2D coordinates on a football field.

## Exercises

### Exercise 1: compute_cost_with_regularization
Implement the L2 regularization cost:
$$J_{regularized} = J_{cross-entropy} + \frac{1}{m} \frac{\lambda}{2} \sum_l \sum_k \sum_j W_{k,j}^{[l]2}$$

### Exercise 2: backward_propagation_with_regularization
Modify backpropagation to include L2 regularization gradients:
$$\frac{dW^{[l]}}{dJ} = \frac{dW^{[l]}}{dJ_{cross-entropy}} + \frac{\lambda}{m} W^{[l]}$$

### Exercise 3: forward_propagation_with_dropout
Implement dropout in forward propagation:
1. Create dropout mask $D^{[l]}$ with same shape as $A^{[l]}$
2. Set entries to 1 with probability `keep_prob`, 0 otherwise
3. Multiply $A^{[l]}$ by $D^{[l]}$ to shut down neurons
4. Divide by `keep_prob` to maintain expected value (inverted dropout)

### Exercise 4: backward_propagation_with_dropout
Implement dropout in backward propagation:
1. Apply same dropout mask $D^{[l]}$ to $dA^{[l]}$
2. Scale by dividing by `keep_prob`

## Key Concepts

### L2 Regularization
- **Purpose**: Penalize large weights to prevent overfitting
- **Effect**: Smoother decision boundaries
- **Hyperparameter**: $\lambda$ controls regularization strength
- **Trade-off**: Too large $\lambda$ can cause underfitting (high bias)

### Dropout
- **Purpose**: Prevent neurons from co-adapting by randomly dropping them
- **Effect**: Forces network to learn robust features
- **Implementation**: Use only during training, not testing
- **Inverted Dropout**: Scale activations by `1/keep_prob` during training

### Why L2 Works
- Smaller weights → simpler model
- Penalizing $W^2$ drives weights to smaller values
- Output changes more slowly as input changes
- More generalized model

### Why Dropout Works
- Each iteration trains a different sub-network
- Neurons become less sensitive to specific other neurons
- Prevents over-reliance on particular features
- Acts like ensemble learning

## Expected Results

| Model | Train Accuracy | Test Accuracy | Comment |
|-------|---------------|---------------|---------|
| No Regularization | ~95% | ~91.5% | Overfitting |
| L2 Regularization (λ=0.7) | ~94% | ~93% | Reduced overfitting |
| Dropout (keep_prob=0.86) | ~93% | ~95% | Best generalization |

## Important Notes

### Common Mistakes
- **Don't** use dropout during testing/inference
- **Don't** forget to scale by `keep_prob` in dropout
- **Don't** apply dropout to input or output layers
- **Do** apply dropout only to hidden layers

### Best Practices
1. Start with L2 regularization as baseline
2. Tune $\lambda$ using a validation set
3. Try dropout if L2 isn't sufficient
4. Common `keep_prob` values: 0.5, 0.8, 0.9
5. Can combine L2 and dropout, but this lab explores them separately

## Running the Lab

1. Open `regularization.ipynb` in Jupyter or VS Code
2. Ensure dataset is in `datasets/data.mat`
3. Run setup cells to import libraries
4. Implement the four exercises
5. Train models with different regularization settings
6. Compare decision boundaries and accuracies

## Tips

- L2 regularization affects only weight updates (dW), not biases (db)
- Dropout mask must be consistent between forward and backward pass
- Use `np.random.seed()` for reproducible dropout patterns (testing only)
- The regularization term in cost is: $\frac{\lambda}{2m} \sum ||W^{[l]}||^2$
- Remember inverted dropout: multiply by $\frac{1}{keep\_prob}$ during training

## Framework Integration

Modern frameworks handle regularization automatically:
- **TensorFlow**: `tf.nn.dropout()`, `kernel_regularizer`
- **PyTorch**: `nn.Dropout()`, weight_decay parameter
- **Keras**: `Dropout()` layer, `kernel_regularizer` argument

## Further Reading

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html) (Srivastava et al., 2014)
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html) (Glorot & Bengio, 2010)
- L2 Regularization vs Weight Decay

## Assignment Notes

This is part of the Deep Learning Specialization coursework. Make sure to:
- Not add extra print statements in graded functions
- Keep function parameters unchanged
- Avoid using global variables
- Test with provided test cases before submission

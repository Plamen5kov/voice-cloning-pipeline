# Lab 9: Introduction to TensorFlow

## Overview

This lab introduces TensorFlow 2.x as a deep learning framework for building neural networks more efficiently than with pure NumPy. You'll learn the fundamentals of TensorFlow operations and build your first complete neural network using the framework.

## Learning Objectives

By the end of this lab, you will be able to:

- Use `tf.Variable` to modify the state of a variable
- Explain the difference between a variable and a constant in TensorFlow
- Implement forward propagation using TensorFlow operations
- Use `tf.GradientTape` for automatic differentiation
- Train a neural network on a TensorFlow dataset
- Work with TensorFlow datasets and data pipelines

## Prerequisites

- Understanding of neural networks and their components (weights, biases, activations)
- Familiarity with forward and backward propagation
- Basic knowledge of gradient descent and optimization
- NumPy experience (covered in previous labs)

## Lab Contents

### Part 1: TensorFlow Basics
- Package imports and version checking
- Understanding TensorFlow tensors vs NumPy arrays
- Working with `tf.Variable` and `tf.constant`

### Part 2: Basic Operations with GradientTape
- **Exercise 1**: Implementing a linear function
- **Exercise 2**: Computing the sigmoid activation
- **Exercise 3**: Creating one-hot encodings
- **Exercise 4**: Initializing parameters with GlorotNormal

### Part 3: Building Your First Neural Network
- **Exercise 5**: Implementing forward propagation
- **Exercise 6**: Computing total loss with categorical cross-entropy
- Training the complete model on hand sign digit dataset (0-5)

## Dataset

This lab uses the **Hand Sign Digits dataset**, which consists of:
- Images of hand signs representing digits 0-5
- Image dimensions: 64x64x3 (RGB images)
- Training set: 1080 images
- Test set: 120 images

You'll need to download the dataset files:
- `train_signs.h5`
- `test_signs.h5`

Place these files in the `datasets/` folder.

## Setup Instructions

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: This lab was originally designed for TensorFlow 2.3, but has been updated to work with TensorFlow 2.16+ (the versions currently available). The code is fully compatible with modern TensorFlow 2.x versions.

2. Download the dataset files (see datasets/README.md for instructions)

3. Open the notebook:
   ```bash
   jupyter notebook tensorflow_intro.ipynb
   ```

## Key Concepts

### TensorFlow Variables vs Constants
- **tf.Variable**: Mutable state that can be modified during training (used for weights and biases)
- **tf.constant**: Immutable tensors that cannot be changed after creation

### GradientTape
TensorFlow's automatic differentiation system that records operations for computing gradients during backpropagation. This eliminates the need to manually implement backward propagation.

### TensorFlow Datasets
TensorFlow's `tf.data.Dataset` API provides efficient data loading and preprocessing:
- Lazy evaluation (generator-based)
- Built-in batching and prefetching
- Easy data transformations with `.map()`

### Architecture
The neural network you'll build has the following architecture:
- **Layer 1**: 25 units with ReLU activation
- **Layer 2**: 12 units with ReLU activation
- **Layer 3**: 6 units (output layer, one per class)
- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam

## Tips for Success

1. **Don't add extra print statements** - The autograder expects specific output
2. **Don't modify function parameters** - Keep function signatures as provided
3. **Use local variables, not global** - Follow the function scopes
4. **Read error messages carefully** - TensorFlow errors can be verbose but informative
5. **Check tensor shapes** - Use `.shape` to debug dimension mismatches

## Expected Results

After training for 100 epochs, you should see:
- Initial training accuracy: ~17%
- Final training accuracy: >80%
- Loss decreasing steadily
- Test accuracy improving over time

## Common Issues

### Issue: "Grader Error: Grader feedback not found"
**Solution**: Make sure you haven't:
- Added extra print statements
- Added extra code cells
- Changed function parameters
- Used global variables where not intended

### Issue: Tensor shape mismatches
**Solution**: 
- Check if you need to transpose with `tf.transpose()`
- Verify matrix multiplication order
- Use `print(tensor.shape)` to debug

### Issue: Dataset not found
**Solution**: Ensure the HDF5 files are in the `datasets/` folder

## Additional Resources

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Introduction to Gradients and Automatic Differentiation](https://www.tensorflow.org/guide/autodiff)
- [GradientTape API Documentation](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
- [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)

## Next Steps

After completing this lab, you'll be ready to:
- Build more complex neural network architectures with TensorFlow
- Experiment with different optimizers and learning rates
- Apply transfer learning with pre-trained models
- Deploy TensorFlow models for production use

---

**Note**: This lab is adapted from deep learning coursework and serves as an introduction to practical TensorFlow implementation in the voice cloning pipeline project.

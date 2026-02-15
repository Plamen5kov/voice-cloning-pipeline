# Numerical Approximation of Gradients

## Overview

Gradient checking is a technique that helps verify if your implementation of backpropagation is correct. Before implementing gradient checking, we need to understand how to numerically approximate gradients.

## The Problem

When implementing backpropagation, you might write all the equations but not be 100% sure if you've got all the details right. Gradient checking helps catch these potential bugs.

## One-Sided Difference (Less Accurate)

### Basic Approach

Given a function $f(\theta) = \theta^3$, we can approximate the derivative at a point $\theta$ by:

1. Nudging $\theta$ to the right by a small amount $\epsilon$
2. Computing the slope of the resulting triangle

### Formula

$$g(\theta) \approx \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}$$

### Example

With $\theta = 1$ and $\epsilon = 0.01$:

$$\frac{f(1.01) - f(1)}{0.01} = \frac{1.01^3 - 1^3}{0.01} = 3.0301$$

The true derivative is $g(\theta) = 3\theta^2 = 3$ when $\theta = 1$.

**Approximation error:** $0.0301$

## Two-Sided Difference (More Accurate)

### Better Approach

Instead of just nudging to the right, we nudge both left and right:
- $\theta + \epsilon$ (e.g., 1.01)
- $\theta - \epsilon$ (e.g., 0.99)

This creates a larger triangle that gives a much better approximation of the gradient.

### Formula

$$g(\theta) \approx \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$$

### Example

With $\theta = 1$ and $\epsilon = 0.01$:

$$\frac{f(1.01) - f(0.99)}{2 \times 0.01} = \frac{1.01^3 - 0.99^3}{0.02} = 3.0001$$

**Approximation error:** $0.0001$

### Why Two-Sided Is Better

The two-sided difference effectively uses two triangles (upper right and lower left), giving a more balanced estimate of the derivative.

## Mathematical Theory (Optional)

### Formal Definition

For very small values of $\epsilon$, the formal definition of a derivative is:

$$\frac{df}{d\theta} = \lim_{\epsilon \to 0} \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$$

### Error Analysis

**Two-sided difference error:**
$$\text{Error} = O(\epsilon^2)$$

**One-sided difference error:**
$$\text{Error} = O(\epsilon)$$

Since $\epsilon < 1$, we have $\epsilon^2 \ll \epsilon$, making the two-sided difference much more accurate.

### Example Error Comparison

For $\epsilon = 0.01$:
- Two-sided error: $O(0.0001)$ 
- One-sided error: $O(0.01)$ 

The two-sided difference is **100x more accurate**.

## Practical Considerations

### Computational Cost

The two-sided difference method runs **twice as slow** as the one-sided method because it requires computing $f(\theta + \epsilon)$ and $f(\theta - \epsilon)$.

### Recommendation

Despite being slower, the two-sided difference is **worth using** for gradient checking because:
- It's much more accurate
- It gives greater confidence that your backpropagation implementation is correct
- The extra computational cost is only incurred during debugging, not during training

## Key Takeaways

1. **Two-sided difference formula** provides a more accurate numerical approximation of gradients
2. Use $\epsilon = 0.01$ (or similar small value) for numerical approximation
3. The formula $\frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$ should be very close to the true derivative
4. This technique forms the foundation for **gradient checking** to verify backpropagation implementations

## Next Steps

In the next lesson, we'll see how to use this numerical approximation technique to implement gradient checking and verify whether your backpropagation implementation is correct or if there might be bugs to fix.

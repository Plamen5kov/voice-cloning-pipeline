# Gradient Checking

## Overview

Gradient checking (grad check) is a debugging technique that helps verify if your implementation of backpropagation is correct. It can save significant time by catching bugs in backprop implementations.

## Why Gradient Checking?

When implementing backpropagation, it's easy to make mistakes in:
- Computing derivatives
- Implementing chain rule correctly
- Handling matrix dimensions
- Computing partial derivatives for each layer

Gradient checking provides a numerical way to verify that your analytical gradients are correct.

## Implementation Steps

### Step 1: Reshape Parameters into Vectors

Your neural network has parameters:
- $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}, ..., W^{[L]}, b^{[L]}$

**Reshape all parameters into a giant vector $\theta$:**

```python
# Reshape matrices W into vectors and concatenate
theta = concatenate([
    W[1].reshape(-1),  # Flatten W[1]
    b[1].reshape(-1),  # Flatten b[1]
    W[2].reshape(-1),  # Flatten W[2]
    b[2].reshape(-1),  # Flatten b[2]
    # ... continue for all layers
    W[L].reshape(-1),
    b[L].reshape(-1)
])
```

Now the cost function becomes:
$$J(W^{[1]}, b^{[1]}, ..., W^{[L]}, b^{[L]}) \rightarrow J(\theta)$$

### Step 2: Reshape Gradients into Vectors

Similarly, reshape all gradients into a giant vector $d\theta$:

```python
# Reshape gradient matrices into vectors and concatenate
dtheta = concatenate([
    dW[1].reshape(-1),  # dW[1] has same dimension as W[1]
    db[1].reshape(-1),  # db[1] has same dimension as b[1]
    dW[2].reshape(-1),
    db[2].reshape(-1),
    # ... continue for all layers
    dW[L].reshape(-1),
    db[L].reshape(-1)
])
```

Note: $dW^{[i]}$ has the same dimensions as $W^{[i]}$, and $db^{[i]}$ has the same dimensions as $b^{[i]}$.

### Step 3: Compute Numerical Approximation

For each component $i$ of $\theta$, compute the numerical gradient:

$$d\theta_{\text{approx}}[i] = \frac{J(\theta_1, \theta_2, ..., \theta_i + \epsilon, ..., \theta_n) - J(\theta_1, \theta_2, ..., \theta_i - \epsilon, ..., \theta_n)}{2\epsilon}$$

**Implementation:**

```python
for i in range(len(theta)):
    # Perturb theta[i] by +epsilon
    theta_plus = theta.copy()
    theta_plus[i] += epsilon
    J_plus = compute_cost(theta_plus)
    
    # Perturb theta[i] by -epsilon
    theta_minus = theta.copy()
    theta_minus[i] -= epsilon
    J_minus = compute_cost(theta_minus)
    
    # Compute numerical gradient
    dtheta_approx[i] = (J_plus - J_minus) / (2 * epsilon)
```

This should give:
$$d\theta_{\text{approx}}[i] \approx d\theta[i]$$

where $d\theta[i] = \frac{\partial J}{\partial \theta_i}$ from backpropagation.

### Step 4: Compare Vectors

After the loop, you have two vectors:
- $d\theta_{\text{approx}}$ - numerical gradients
- $d\theta$ - analytical gradients from backpropagation

**Compute the relative difference:**

$$\text{difference} = \frac{\|d\theta_{\text{approx}} - d\theta\|_2}{\|d\theta_{\text{approx}}\|_2 + \|d\theta\|_2}$$

Where $\|v\|_2$ is the Euclidean norm (L2 norm):
$$\|v\|_2 = \sqrt{\sum_i v_i^2}$$

**Why normalize?**
- The denominator turns this into a ratio
- Handles cases where vectors are very small or very large
- Makes the metric scale-independent

## Interpreting Results

### Recommended Epsilon Value

Use $\epsilon = 10^{-7}$

### Error Thresholds

| Difference Value | Interpretation | Action |
|-----------------|----------------|--------|
| $\leq 10^{-7}$ | ✅ Excellent | Implementation is very likely correct |
| $\sim 10^{-5}$ | ⚠️ Warning | Double-check; might be okay, but verify components |
| $\sim 10^{-3}$ | ❌ Error | Likely has a bug; debug required |
| $> 10^{-3}$ | 🚨 Critical | Definitely has a bug; serious debugging needed |

### Debugging Tips

If the difference is too large:

1. **Check individual components:** Look at each element of $d\theta_{\text{approx}}[i]$ vs $d\theta[i]$
2. **Find the largest differences:** Identify which $i$ has the biggest discrepancy
3. **Trace back to the parameter:** Determine if the error is in $dW^{[l]}$ or $db^{[l]}$ for a specific layer
4. **Review that layer's backprop:** Focus debugging on the layer with the error

## Typical Workflow

```
1. Implement forward propagation
2. Implement backward propagation
3. Run gradient checking
   ├─ High difference (e.g., 10^-3)? → Debug backprop
   ├─ Continue debugging
   └─ Low difference (e.g., 10^-7)? → Implementation correct! ✓
4. Proceed with training
```

## Example Code Structure

```python
def gradient_check(parameters, gradients, X, y, epsilon=1e-7):
    """
    Check if backpropagation gradients are correct
    """
    # 1. Reshape parameters to vector
    theta = dictionary_to_vector(parameters)
    
    # 2. Reshape gradients to vector
    dtheta = gradients_to_vector(gradients)
    
    # 3. Compute numerical gradients
    dtheta_approx = np.zeros(theta.shape)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += epsilon
        J_plus = compute_cost(vector_to_dictionary(theta_plus), X, y)
        
        theta_minus = theta.copy()
        theta_minus[i] -= epsilon
        J_minus = compute_cost(vector_to_dictionary(theta_minus), X, y)
        
        dtheta_approx[i] = (J_plus - J_minus) / (2 * epsilon)
    
    # 4. Compute relative difference
    numerator = np.linalg.norm(dtheta_approx - dtheta)
    denominator = np.linalg.norm(dtheta_approx) + np.linalg.norm(dtheta)
    difference = numerator / denominator
    
    # 5. Check result
    if difference < 1e-7:
        print("✅ Gradient check passed!")
    else:
        print(f"❌ Gradient check failed: {difference}")
        
    return difference
```

## Key Takeaways

1. **Gradient checking verifies backpropagation** by comparing analytical gradients with numerical approximations
2. **Reshape all parameters and gradients** into giant vectors for easier comparison
3. **Use two-sided difference** for accurate numerical gradient approximation
4. **Set $\epsilon = 10^{-7}$** for numerical gradient computation
5. **Aim for difference $\leq 10^{-7}$** to be confident in your implementation
6. **Debug layer by layer** if gradient check fails by examining individual components
7. **Gradient checking is for debugging only** - too slow to use during actual training

## Next Steps

In the next lesson, we'll discuss practical implementation notes and tips for gradient checking, including when to use it and important considerations.

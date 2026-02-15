# Optimization Algorithms

A comprehensive guide to optimization algorithms used in deep learning, covering key concepts, formulas, and practical examples.

## Table of Contents

1. [Mini-batch Gradient Descent](#mini-batch-gradient-descent)
2. [Exponentially Weighted Averages](#exponentially-weighted-averages)
3. [Bias Correction in Exponentially Weighted Averages](#bias-correction-in-exponentially-weighted-averages)
4. [Gradient Descent with Momentum](#gradient-descent-with-momentum)
5. [RMSprop](#rmsprop)
6. [Adam Optimization Algorithm](#adam-optimization-algorithm)
7. [Learning Rate Decay](#learning-rate-decay)
8. [The Problem of Local Optima](#the-problem-of-local-optima)

---

## Mini-batch Gradient Descent

### Overview

Mini-batch gradient descent splits the training set into smaller batches and performs gradient descent on each batch. This provides a balance between the speed of stochastic gradient descent and the stability of batch gradient descent.

### Key Concepts

- **Batch Gradient Descent**: Use all training examples in each iteration
- **Stochastic Gradient Descent (SGD)**: Use 1 training example per iteration
- **Mini-batch Gradient Descent**: Use mini-batches of size $m$ (typically 64, 128, 256, 512, or 1024)

### Notation

- Training set: $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})\}$
- Mini-batch $t$: $X^{\{t\}}, Y^{\{t\}}$
- Mini-batch size: typically 64 to 512
- Number of mini-batches: $\text{num\_batches} = \lfloor \frac{m}{\text{batch\_size}} \rfloor$

### Algorithm

```
For t = 1, ..., num_batches:
    1. Forward propagation on X^{t}
       Z^[1] = W^[1] X^{t} + b^[1]
       A^[1] = g^[1](Z^[1])
       ...
       A^[L] = g^[L](Z^[L])
    
    2. Compute cost J^{t}:
       J^{t} = (1/batch_size) * Σ L(ŷ^(i), y^(i)) + regularization_term
    
    3. Backpropagation to compute gradients with respect to J^{t}
    
    4. Update parameters:
       W^[l] = W^[l] - α * dW^[l]
       b^[l] = b^[l] - α * db^[l]
```

### Example

```python
import numpy as np

def mini_batch_gradient_descent(X, Y, parameters, learning_rate=0.01, 
                                mini_batch_size=64, num_epochs=100):
    """
    Perform mini-batch gradient descent
    
    Arguments:
    X -- training data (n_x, m)
    Y -- training labels (1, m)
    parameters -- dictionary containing W and b for all layers
    learning_rate -- learning rate α
    mini_batch_size -- size of mini-batches
    num_epochs -- number of passes through the training set
    """
    m = X.shape[1]  # number of training examples
    costs = []
    
    for epoch in range(num_epochs):
        # Shuffle training data
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        
        # Create mini-batches
        num_complete_batches = m // mini_batch_size
        
        for k in range(num_complete_batches):
            # Get mini-batch k
            start = k * mini_batch_size
            end = (k + 1) * mini_batch_size
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]
            
            # Forward propagation
            AL, caches = forward_propagation(X_batch, parameters)
            
            # Compute cost
            cost = compute_cost(AL, Y_batch)
            
            # Backward propagation
            grads = backward_propagation(AL, Y_batch, caches)
            
            # Update parameters
            parameters = update_parameters(parameters, grads, learning_rate)
        
        # Handle remaining examples (if mini_batch_size doesn't divide m)
        if m % mini_batch_size != 0:
            start = num_complete_batches * mini_batch_size
            X_batch = X_shuffled[:, start:]
            Y_batch = Y_shuffled[:, start:]
            
            AL, caches = forward_propagation(X_batch, parameters)
            cost = compute_cost(AL, Y_batch)
            grads = backward_propagation(AL, Y_batch, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
        
        if epoch % 10 == 0:
            costs.append(cost)
            print(f"Epoch {epoch}: Cost = {cost}")
    
    return parameters, costs
```

### When to Use Different Batch Sizes

- **Small training set (m ≤ 2000)**: Use batch gradient descent
- **Typical mini-batch sizes**: 64, 128, 256, 512, 1024 (powers of 2)
- **Memory constraints**: Reduce batch size if running out of GPU/CPU memory

### Advantages

- Faster than batch gradient descent on large datasets
- More stable than stochastic gradient descent
- Better utilization of vectorization
- Works well with GPU optimization

---

## Exponentially Weighted Averages

### Overview

Exponentially weighted averages (also called exponentially weighted moving averages) are used to compute approximations of averages over a sequence of values, giving more weight to recent observations.

### Formula

$$v_t = \beta v_{t-1} + (1 - \beta) \theta_t$$

Where:
- $v_t$: Exponentially weighted average at time $t$
- $\theta_t$: Current observation at time $t$
- $\beta$: Weight parameter (typically 0.9, 0.95, 0.98, 0.99)
- $v_0 = 0$ (initialization)

### Interpretation

The exponentially weighted average $v_t$ approximates the average over approximately $\frac{1}{1-\beta}$ days/time steps.

- $\beta = 0.9$: Average over ~10 values
- $\beta = 0.95$: Average over ~20 values  
- $\beta = 0.98$: Average over ~50 values
- $\beta = 0.99$: Average over ~100 values

### Mathematical Expansion

$$v_t = (1-\beta)\theta_t + (1-\beta)\beta\theta_{t-1} + (1-\beta)\beta^2\theta_{t-2} + ... + (1-\beta)\beta^{t-1}\theta_1 + \beta^t v_0$$

This shows that recent observations have exponentially more weight than older ones.

### Example

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_exponentially_weighted_average(data, beta=0.9):
    """
    Compute exponentially weighted average
    
    Arguments:
    data -- array of observations
    beta -- weight parameter
    
    Returns:
    v -- exponentially weighted averages
    """
    v = np.zeros(len(data))
    v[0] = 0
    
    for t in range(1, len(data)):
        v[t] = beta * v[t-1] + (1 - beta) * data[t]
    
    return v

# Example: Temperature data
days = np.arange(1, 101)
temperatures = 20 + 5 * np.sin(days / 10) + np.random.randn(100) * 2

# Compute with different beta values
v_09 = compute_exponentially_weighted_average(temperatures, beta=0.9)
v_095 = compute_exponentially_weighted_average(temperatures, beta=0.95)
v_098 = compute_exponentially_weighted_average(temperatures, beta=0.98)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(days, temperatures, 'o', alpha=0.3, label='Actual Temperature')
plt.plot(days, v_09, label='β = 0.9 (~10 day avg)', linewidth=2)
plt.plot(days, v_095, label='β = 0.95 (~20 day avg)', linewidth=2)
plt.plot(days, v_098, label='β = 0.98 (~50 day avg)', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title('Exponentially Weighted Averages with Different β Values')
plt.grid(True, alpha=0.3)
plt.show()
```

### Implementation Efficiency

Exponentially weighted averages are memory-efficient:
- Only need to store one number $v_t$ (not entire history)
- Simple update formula
- Fast computation

---

## Bias Correction in Exponentially Weighted Averages

### The Problem

When initializing $v_0 = 0$, the early estimates of the exponentially weighted average are biased toward zero, especially when $\beta$ is large (close to 1).

### Why Bias Occurs

$$v_1 = \beta \cdot 0 + (1-\beta)\theta_1 = (1-\beta)\theta_1$$

If $\beta = 0.98$ and $\theta_1 = 40$:
$$v_1 = (1-0.98) \cdot 40 = 0.02 \cdot 40 = 0.8$$

This is much smaller than the actual value!

### Bias Correction Formula

$$v_t^{\text{corrected}} = \frac{v_t}{1 - \beta^t}$$

Where:
- $v_t$: Original exponentially weighted average
- $\beta^t$: $\beta$ raised to the power $t$
- As $t \to \infty$, $\beta^t \to 0$, so the correction factor approaches 1

### Why It Works

The expected value of $v_t$ without bias correction:
$$E[v_t] = (1-\beta^t) E[\theta]$$

Dividing by $(1-\beta^t)$ removes this bias:
$$E[v_t^{\text{corrected}}] = E[\theta]$$

### Example

```python
def compute_ewa_with_bias_correction(data, beta=0.9):
    """
    Compute exponentially weighted average with bias correction
    
    Arguments:
    data -- array of observations
    beta -- weight parameter
    
    Returns:
    v_corrected -- bias-corrected exponentially weighted averages
    v_uncorrected -- uncorrected exponentially weighted averages
    """
    v = np.zeros(len(data))
    v_corrected = np.zeros(len(data))
    
    for t in range(len(data)):
        if t == 0:
            v[t] = (1 - beta) * data[t]
        else:
            v[t] = beta * v[t-1] + (1 - beta) * data[t]
        
        # Apply bias correction
        v_corrected[t] = v[t] / (1 - beta**(t+1))
    
    return v_corrected, v

# Example with high beta value
data = np.array([40, 49, 45, 47, 48, 46, 50, 45, 44, 48])
beta = 0.98

v_corrected, v_uncorrected = compute_ewa_with_bias_correction(data, beta)

print("Time Step | Observation | Uncorrected | Corrected")
print("-" * 55)
for t in range(len(data)):
    print(f"{t+1:^9} | {data[t]:^11.2f} | {v_uncorrected[t]:^11.2f} | {v_corrected[t]:^9.2f}")

# Output shows how bias correction fixes the initial underestimation
```

### When to Use Bias Correction

- **Use it**: In the early stages of training (first few iterations)
- **Skip it**: After many iterations when $\beta^t \approx 0$
- **Common practice**: Often used in Adam optimizer during initial iterations

---

## Gradient Descent with Momentum

### Overview

Gradient descent with momentum computes exponentially weighted averages of gradients and uses that to update parameters. This helps smooth out oscillations and speeds up convergence.

### Motivation

Standard gradient descent can oscillate, especially when:
- There are steep dimensions (high gradient variance)
- There are gentle dimensions (slow progress)

Momentum helps by:
- Dampening oscillations in steep directions
- Accelerating progress in gentle directions

### Algorithm

**On iteration $t$:**

1. Compute $dW$, $db$ on current mini-batch

2. Compute momentum terms:
   $$v_{dW} = \beta v_{dW} + (1-\beta) dW$$
   $$v_{db} = \beta v_{db} + (1-\beta) db$$

3. Update parameters:
   $$W = W - \alpha v_{dW}$$
   $$b = b - \alpha v_{db}$$

### Hyperparameters

- $\alpha$: Learning rate (needs to be tuned)
- $\beta$: Momentum parameter (typically 0.9)
- Initialize: $v_{dW} = 0$, $v_{db} = 0$

### Alternative Formulation

Some implementations omit the $(1-\beta)$ term:

$$v_{dW} = \beta v_{dW} + dW$$
$$W = W - \alpha v_{dW}$$

This changes the scale of $v_{dW}$ but can be compensated by adjusting $\alpha$.

### Example

```python
def initialize_momentum(parameters):
    """
    Initialize velocity for momentum gradient descent
    
    Arguments:
    parameters -- dictionary containing W and b for all layers
    
    Returns:
    v -- dictionary containing velocity for all parameters
    """
    v = {}
    L = len(parameters) // 2  # number of layers
    
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    
    return v


def update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01):
    """
    Update parameters using gradient descent with momentum
    
    Arguments:
    parameters -- dictionary containing W and b
    grads -- dictionary containing dW and db
    v -- dictionary containing velocity for all parameters
    beta -- momentum hyperparameter
    learning_rate -- learning rate α
    
    Returns:
    parameters -- updated parameters
    v -- updated velocities
    """
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        # Compute momentum
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        
        # Update parameters
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]
    
    return parameters, v


# Example usage
def train_with_momentum(X, Y, layer_dims, learning_rate=0.01, beta=0.9, 
                       num_epochs=1000, mini_batch_size=64):
    """
    Train neural network with momentum
    """
    parameters = initialize_parameters(layer_dims)
    v = initialize_momentum(parameters)
    
    for epoch in range(num_epochs):
        # Mini-batch creation
        mini_batches = create_mini_batches(X, Y, mini_batch_size)
        
        for mini_batch in mini_batches:
            X_batch, Y_batch = mini_batch
            
            # Forward propagation
            AL, caches = forward_propagation(X_batch, parameters)
            
            # Compute cost
            cost = compute_cost(AL, Y_batch)
            
            # Backward propagation
            grads = backward_propagation(AL, Y_batch, caches)
            
            # Update parameters with momentum
            parameters, v = update_parameters_with_momentum(
                parameters, grads, v, beta, learning_rate
            )
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}")
    
    return parameters
```

### Intuition

Think of a ball rolling down a hill:
- Gradient ($dW$) is the force applied
- Velocity ($v_{dW}$) accumulates over time
- Ball gains momentum in consistent directions
- Ball slows down in oscillating directions

### Benefits

- Faster convergence than standard gradient descent
- Reduces oscillations
- Works well with mini-batch gradient descent
- Typical $\beta = 0.9$ works well for most cases

---

## RMSprop

### Overview

RMSprop (Root Mean Square propagation) adapts the learning rate for each parameter based on the magnitude of recent gradients. It helps deal with the problem of different learning rates needed for different parameters.

### Algorithm

**On iteration $t$:**

1. Compute $dW$, $db$ on current mini-batch

2. Compute exponentially weighted average of squared gradients:
   $$S_{dW} = \beta S_{dW} + (1-\beta) dW^2$$
   $$S_{db} = \beta S_{db} + (1-\beta) db^2$$
   
   Note: Squaring is element-wise

3. Update parameters:
   $$W = W - \alpha \frac{dW}{\sqrt{S_{dW}} + \varepsilon}$$
   $$b = b - \alpha \frac{db}{\sqrt{S_{db}} + \varepsilon}$$

### Hyperparameters

- $\alpha$: Learning rate (needs tuning)
- $\beta$: Decay rate (typically 0.9, 0.99, or 0.999)
- $\varepsilon$: Small constant for numerical stability (typically $10^{-8}$)
- Initialize: $S_{dW} = 0$, $S_{db} = 0$

### How It Works

- **Large gradients** → Large $S_{dW}$ → Smaller effective learning rate
- **Small gradients** → Small $S_{dW}$ → Larger effective learning rate

This helps:
- Prevent overshooting in steep directions
- Accelerate learning in gentle directions

### Example

```python
def initialize_rmsprop(parameters):
    """
    Initialize S (squared gradients) for RMSprop
    
    Arguments:
    parameters -- dictionary containing W and b for all layers
    
    Returns:
    s -- dictionary containing squared gradients for all parameters
    """
    s = {}
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    
    return s


def update_parameters_with_rmsprop(parameters, grads, s, beta=0.999, 
                                   learning_rate=0.001, epsilon=1e-8):
    """
    Update parameters using RMSprop
    
    Arguments:
    parameters -- dictionary containing W and b
    grads -- dictionary containing dW and db
    s -- dictionary containing squared gradients
    beta -- decay rate for moving average of squared gradients
    learning_rate -- learning rate α
    epsilon -- small constant for numerical stability
    
    Returns:
    parameters -- updated parameters
    s -- updated squared gradients
    """
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        # Compute exponentially weighted average of squared gradients
        s["dW" + str(l)] = beta * s["dW" + str(l)] + (1 - beta) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta * s["db" + str(l)] + (1 - beta) * np.square(grads["db" + str(l)])
        
        # Update parameters
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)] / (np.sqrt(s["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)] / (np.sqrt(s["db" + str(l)]) + epsilon)
    
    return parameters, s


# Example training with RMSprop
def train_with_rmsprop(X, Y, layer_dims, learning_rate=0.001, beta=0.999, 
                       num_epochs=1000, mini_batch_size=64):
    """
    Train neural network with RMSprop
    """
    parameters = initialize_parameters(layer_dims)
    s = initialize_rmsprop(parameters)
    
    for epoch in range(num_epochs):
        mini_batches = create_mini_batches(X, Y, mini_batch_size)
        
        for mini_batch in mini_batches:
            X_batch, Y_batch = mini_batch
            
            # Forward propagation
            AL, caches = forward_propagation(X_batch, parameters)
            
            # Compute cost
            cost = compute_cost(AL, Y_batch)
            
            # Backward propagation
            grads = backward_propagation(AL, Y_batch, caches)
            
            # Update parameters with RMSprop
            parameters, s = update_parameters_with_rmsprop(
                parameters, grads, s, beta, learning_rate
            )
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}")
    
    return parameters
```

### Visualization Example

```python
# Demonstrate RMSprop behavior
def demonstrate_rmsprop():
    # Simulate gradients with different magnitudes
    iterations = 100
    dW_steep = 10 * np.ones(iterations)  # Large gradients
    dW_gentle = 0.1 * np.ones(iterations)  # Small gradients
    
    s_steep = 0
    s_gentle = 0
    beta = 0.9
    alpha = 0.1
    epsilon = 1e-8
    
    effective_lr_steep = []
    effective_lr_gentle = []
    
    for i in range(iterations):
        # RMSprop for steep dimension
        s_steep = beta * s_steep + (1 - beta) * dW_steep[i]**2
        eff_lr_steep = alpha / (np.sqrt(s_steep) + epsilon)
        effective_lr_steep.append(eff_lr_steep)
        
        # RMSprop for gentle dimension
        s_gentle = beta * s_gentle + (1 - beta) * dW_gentle[i]**2
        eff_lr_gentle = alpha / (np.sqrt(s_gentle) + epsilon)
        effective_lr_gentle.append(eff_lr_gentle)
    
    plt.figure(figsize=(10, 5))
    plt.plot(effective_lr_steep, label='Steep dimension (large gradients)', linewidth=2)
    plt.plot(effective_lr_gentle, label='Gentle dimension (small gradients)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Effective Learning Rate')
    plt.legend()
    plt.title('RMSprop: Adaptive Learning Rates')
    plt.grid(True, alpha=0.3)
    plt.show()

demonstrate_rmsprop()
```

### Benefits

- Adapts learning rate automatically for each parameter
- Works well with non-stationary objectives
- Effective for RNNs and mini-batch learning
- Reduces need for manual learning rate tuning

---

## Adam Optimization Algorithm

### Overview

**Adam** (Adaptive Moment Estimation) combines the best properties of momentum and RMSprop. It's one of the most popular optimization algorithms in deep learning.

### Algorithm

**On iteration $t$:**

1. Compute $dW$, $db$ on current mini-batch

2. Compute momentum (first moment):
   $$v_{dW} = \beta_1 v_{dW} + (1-\beta_1) dW$$
   $$v_{db} = \beta_1 v_{db} + (1-\beta_1) db$$

3. Compute RMSprop (second moment):
   $$S_{dW} = \beta_2 S_{dW} + (1-\beta_2) dW^2$$
   $$S_{db} = \beta_2 S_{db} + (1-\beta_2) db^2$$

4. Bias correction:
   $$v_{dW}^{\text{corrected}} = \frac{v_{dW}}{1 - \beta_1^t}$$
   $$v_{db}^{\text{corrected}} = \frac{v_{db}}{1 - \beta_1^t}$$
   $$S_{dW}^{\text{corrected}} = \frac{S_{dW}}{1 - \beta_2^t}$$
   $$S_{db}^{\text{corrected}} = \frac{S_{db}}{1 - \beta_2^t}$$

5. Update parameters:
   $$W = W - \alpha \frac{v_{dW}^{\text{corrected}}}{\sqrt{S_{dW}^{\text{corrected}}} + \varepsilon}$$
   $$b = b - \alpha \frac{v_{db}^{\text{corrected}}}{\sqrt{S_{db}^{\text{corrected}}} + \varepsilon}$$

### Hyperparameters

- $\alpha$: Learning rate (needs tuning, typically 0.001)
- $\beta_1$: Momentum decay rate (default: 0.9)
- $\beta_2$: RMSprop decay rate (default: 0.999)
- $\varepsilon$: Small constant (default: $10^{-8}$)

**Default values work well in practice!**

### Example Implementation

```python
def initialize_adam(parameters):
    """
    Initialize v and s for Adam optimizer
    
    Arguments:
    parameters -- dictionary containing W and b for all layers
    
    Returns:
    v -- dictionary containing momentum for all parameters
    s -- dictionary containing RMSprop for all parameters
    """
    v = {}
    s = {}
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, 
                                beta1=0.9, beta2=0.999, 
                                learning_rate=0.001, epsilon=1e-8):
    """
    Update parameters using Adam optimizer
    
    Arguments:
    parameters -- dictionary containing W and b
    grads -- dictionary containing dW and db
    v -- dictionary containing momentum
    s -- dictionary containing RMSprop
    t -- current iteration number (for bias correction)
    beta1 -- momentum decay rate
    beta2 -- RMSprop decay rate
    learning_rate -- learning rate α
    epsilon -- small constant for numerical stability
    
    Returns:
    parameters -- updated parameters
    v -- updated momentum
    s -- updated RMSprop
    """
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    
    for l in range(1, L + 1):
        # Compute momentum (first moment)
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        
        # Bias correction for momentum
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)
        
        # Compute RMSprop (second moment)
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])
        
        # Bias correction for RMSprop
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)
        
        # Update parameters
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
    
    return parameters, v, s


# Example training with Adam
def train_with_adam(X, Y, layer_dims, learning_rate=0.001, beta1=0.9, beta2=0.999,
                   num_epochs=1000, mini_batch_size=64):
    """
    Train neural network with Adam optimizer
    """
    parameters = initialize_parameters(layer_dims)
    v, s = initialize_adam(parameters)
    t = 0  # iteration counter
    
    for epoch in range(num_epochs):
        mini_batches = create_mini_batches(X, Y, mini_batch_size)
        
        for mini_batch in mini_batches:
            t += 1  # Increment iteration counter
            X_batch, Y_batch = mini_batch
            
            # Forward propagation
            AL, caches = forward_propagation(X_batch, parameters)
            
            # Compute cost
            cost = compute_cost(AL, Y_batch)
            
            # Backward propagation
            grads = backward_propagation(AL, Y_batch, caches)
            
            # Update parameters with Adam
            parameters, v, s = update_parameters_with_adam(
                parameters, grads, v, s, t, beta1, beta2, learning_rate
            )
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}")
    
    return parameters
```

### Complete Example with Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_optimizers(X, Y, layer_dims, num_epochs=1000):
    """
    Compare different optimization algorithms
    """
    # Train with different optimizers
    print("Training with standard gradient descent...")
    params_gd, costs_gd = train_with_gd(X, Y, layer_dims, learning_rate=0.01, num_epochs=num_epochs)
    
    print("\nTraining with momentum...")
    params_momentum, costs_momentum = train_with_momentum(X, Y, layer_dims, learning_rate=0.01, num_epochs=num_epochs)
    
    print("\nTraining with RMSprop...")
    params_rmsprop, costs_rmsprop = train_with_rmsprop(X, Y, layer_dims, learning_rate=0.001, num_epochs=num_epochs)
    
    print("\nTraining with Adam...")
    params_adam, costs_adam = train_with_adam(X, Y, layer_dims, learning_rate=0.001, num_epochs=num_epochs)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(costs_gd, label='Gradient Descent', linewidth=2)
    plt.plot(costs_momentum, label='Momentum', linewidth=2)
    plt.plot(costs_rmsprop, label='RMSprop', linewidth=2)
    plt.plot(costs_adam, label='Adam', linewidth=2)
    plt.xlabel('Epochs (x10)')
    plt.ylabel('Cost')
    plt.title('Optimization Algorithm Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
```

### Why Adam is Popular

1. **Combines best of both worlds**: Momentum + RMSprop
2. **Bias correction**: Handles initialization bias
3. **Adaptive learning rates**: Different rate for each parameter
4. **Good default hyperparameters**: Usually works without tuning
5. **Robust**: Works well across different types of neural networks

### When to Use Adam

- **Default choice**: Start with Adam for most deep learning problems
- **Works well for**: Large datasets, high-dimensional parameter spaces
- **Alternatives to consider**:
  - SGD with momentum: Sometimes generalizes better (needs more tuning)
  - RMSprop: Good for RNNs
  - AdaGrad: For sparse gradients

---

## Learning Rate Decay

### Overview

Learning rate decay gradually reduces the learning rate during training. This allows the algorithm to take larger steps early in training (fast learning) and smaller steps later (fine-tuning).

### Why Use Learning Rate Decay?

**Early in training:**
- Large learning rate → Fast convergence
- Can make big jumps toward optimum

**Later in training:**
- Large learning rate → Oscillation around optimum
- Small learning rate → Fine-tune and converge precisely

### Common Decay Schedules

#### 1. Step Decay

$$\alpha = \alpha_0 \cdot \text{decay\_rate}^{\lfloor \frac{\text{epoch}}{k} \rfloor}$$

Where:
- $\alpha_0$: Initial learning rate
- $k$: Decay frequency (e.g., every 10 epochs)
- decay_rate: Multiplicative factor (e.g., 0.5, 0.9)

**Example:**
```python
def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_drop=10):
    """
    Step decay: reduce learning rate by 'drop' every 'epochs_drop' epochs
    """
    return initial_lr * (drop ** np.floor(epoch / epochs_drop))

# Example: α₀ = 0.1, drop every 10 epochs by factor of 0.5
epochs = np.arange(0, 100)
lrs = [step_decay(e) for e in epochs]

plt.figure(figsize=(10, 5))
plt.plot(epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Step Decay Schedule')
plt.grid(True, alpha=0.3)
plt.show()
```

#### 2. Exponential Decay

$$\alpha = \alpha_0 e^{-kt}$$

Where:
- $k$: Decay constant
- $t$: Epoch number

```python
def exponential_decay(epoch, initial_lr=0.1, k=0.05):
    """
    Exponential decay
    """
    return initial_lr * np.exp(-k * epoch)
```

#### 3. Time-based Decay (1/t Decay)

$$\alpha = \frac{\alpha_0}{1 + \text{decay\_rate} \cdot t}$$

```python
def time_based_decay(epoch, initial_lr=0.1, decay_rate=0.01):
    """
    Time-based decay (inverse time decay)
    """
    return initial_lr / (1 + decay_rate * epoch)
```

#### 4. Polynomial Decay

$$\alpha = \alpha_0 \left(1 - \frac{t}{T}\right)^p$$

Where:
- $T$: Total number of epochs
- $p$: Power (typically 0.5 or 1.0)

```python
def polynomial_decay(epoch, total_epochs, initial_lr=0.1, power=1.0):
    """
    Polynomial decay
    """
    return initial_lr * ((1 - epoch / total_epochs) ** power)
```

#### 5. Cosine Annealing

$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

```python
def cosine_annealing(epoch, total_epochs, max_lr=0.1, min_lr=0.001):
    """
    Cosine annealing schedule
    """
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
```

#### 6. Warm Restarts (Cosine Annealing with Restarts)

Periodically restart the learning rate to escape local minima.

```python
def cosine_annealing_with_restarts(epoch, initial_cycle_length=10, max_lr=0.1, min_lr=0.001, mult=2):
    """
    Cosine annealing with warm restarts (SGDR)
    """
    # Determine which cycle we're in
    cycle = 0
    cycle_length = initial_cycle_length
    epoch_in_cycle = epoch
    
    while epoch_in_cycle >= cycle_length:
        epoch_in_cycle -= cycle_length
        cycle_length *= mult
        cycle += 1
    
    # Cosine annealing within current cycle
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch_in_cycle / cycle_length))
```

### Comparison of Decay Schedules

```python
def plot_decay_schedules():
    """
    Visualize different learning rate decay schedules
    """
    epochs = np.arange(0, 100)
    total_epochs = 100
    
    schedules = {
        'Step Decay': [step_decay(e) for e in epochs],
        'Exponential Decay': [exponential_decay(e) for e in epochs],
        'Time-based Decay': [time_based_decay(e) for e in epochs],
        'Polynomial Decay': [polynomial_decay(e, total_epochs) for e in epochs],
        'Cosine Annealing': [cosine_annealing(e, total_epochs) for e in epochs],
        'Cosine w/ Restarts': [cosine_annealing_with_restarts(e) for e in epochs]
    }
    
    plt.figure(figsize=(14, 8))
    for name, schedule in schedules.items():
        plt.plot(epochs, schedule, label=name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Decay Schedule Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

plot_decay_schedules()
```

### Implementation with Training Loop

```python
def train_with_learning_rate_decay(X, Y, layer_dims, initial_lr=0.1, 
                                   decay_type='step', num_epochs=1000):
    """
    Train with learning rate decay
    """
    parameters = initialize_parameters(layer_dims)
    v, s = initialize_adam(parameters)
    t = 0
    costs = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        # Compute current learning rate based on decay schedule
        if decay_type == 'step':
            current_lr = step_decay(epoch, initial_lr)
        elif decay_type == 'exponential':
            current_lr = exponential_decay(epoch, initial_lr)
        elif decay_type == 'time_based':
            current_lr = time_based_decay(epoch, initial_lr)
        elif decay_type == 'cosine':
            current_lr = cosine_annealing(epoch, num_epochs, initial_lr)
        else:
            current_lr = initial_lr
        
        learning_rates.append(current_lr)
        
        # Mini-batch training
        mini_batches = create_mini_batches(X, Y, mini_batch_size=64)
        
        for mini_batch in mini_batches:
            t += 1
            X_batch, Y_batch = mini_batch
            
            # Forward and backward propagation
            AL, caches = forward_propagation(X_batch, parameters)
            cost = compute_cost(AL, Y_batch)
            grads = backward_propagation(AL, Y_batch, caches)
            
            # Update with current learning rate
            parameters, v, s = update_parameters_with_adam(
                parameters, grads, v, s, t, learning_rate=current_lr
            )
        
        costs.append(cost)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}, LR = {current_lr:.6f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(costs, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cost')
    ax1.set_title('Training Cost')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(learning_rates, linewidth=2, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'Learning Rate Schedule ({decay_type})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return parameters
```

### Manual Decay

Some practitioners manually adjust the learning rate based on validation performance:

```python
def train_with_manual_lr_adjustment(X_train, Y_train, X_val, Y_val, 
                                   layer_dims, patience=10):
    """
    Reduce learning rate when validation loss plateaus
    """
    parameters = initialize_parameters(layer_dims)
    learning_rate = 0.01
    best_val_cost = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(1000):
        # Training step
        AL_train, caches = forward_propagation(X_train, parameters)
        cost_train = compute_cost(AL_train, Y_train)
        grads = backward_propagation(AL_train, Y_train, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Validation
        AL_val, _ = forward_propagation(X_val, parameters)
        cost_val = compute_cost(AL_val, Y_val)
        
        # Check if validation improved
        if cost_val < best_val_cost:
            best_val_cost = cost_val
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Reduce learning rate if no improvement
        if epochs_without_improvement >= patience:
            learning_rate *= 0.5
            epochs_without_improvement = 0
            print(f"Epoch {epoch}: Reducing LR to {learning_rate:.6f}")
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Cost = {cost_train:.4f}, Val Cost = {cost_val:.4f}")
    
    return parameters
```

### Best Practices

1. **Start without decay**: Get a baseline with constant learning rate
2. **Monitor training**: Observe when training starts to plateau
3. **Choose appropriate schedule**: 
   - Step decay: Simple and interpretable
   - Cosine annealing: Smooth decay, popular in computer vision
   - Warm restarts: Can help escape local minima
4. **Hyperparameter tuning**: Decay rate matters as much as initial learning rate
5. **Combine with early stopping**: Monitor validation performance

---

## The Problem of Local Optima

### Overview

In classical optimization (convex problems), local optima are a major concern. However, in high-dimensional deep learning, the landscape is different and local optima are often not the main problem.

### Classical View (Outdated for Deep Learning)

**Traditional concern**: Getting stuck in local minima
- Gradient is zero: $\nabla J = 0$
- Surrounded by higher cost values
- Cannot escape with gradient descent

<img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 200'%3E%3Ctext x='200' y='100' text-anchor='middle' font-size='16'%3E[2D loss surface with local minima]%3C/text%3E%3C/svg%3E" width="400"/>

### Modern Understanding (High-Dimensional Spaces)

In high-dimensional spaces (millions of parameters), true local minima are **extremely rare**.

#### Why Local Minima are Rare

For a point to be a local minimum in $n$-dimensional space:
- All $n$ eigenvalues of the Hessian matrix must be positive
- Probability of this happening decreases exponentially with $n$

**Example**: For 20,000 parameters
- Probability that all directions curve upward ≈ $0.5^{20000} \approx 0$ (astronomically small)

### Saddle Points: The Real Problem

**Saddle point**: Gradient is zero, but it's not a minimum
- Some directions curve upward (local minimum in that direction)
- Some directions curve downward (local maximum in that direction)
- Result: A "plateau" where learning slows down

```
           ↑ (curves up - looks like minimum)
           |
    -------•------- → (curves down - looks like maximum)
           |
           ↓
```

#### Mathematical Definition

At a saddle point $\theta^*$:
- $\nabla J(\theta^*) = 0$ (gradient is zero)
- Hessian has both positive and negative eigenvalues

### Plateaus

**Plateaus**: Regions where the gradient is very small but non-zero
- Training slows down dramatically
- Can last for many epochs
- Eventually, the algorithm escapes

**Why plateaus occur**:
- Near saddle points
- Long, flat regions in the loss surface
- Symmetries in the network

### Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_saddle_point():
    """
    Visualize a saddle point in 2D
    """
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Saddle function: z = x² - y²
    Z = X**2 - Y**2
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.scatter([0], [0], [0], color='red', s=100, label='Saddle Point')
    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_zlabel('Cost J')
    ax1.set_title('Saddle Point (3D View)')
    ax1.legend()
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.scatter([0], [0], color='red', s=100, zorder=5, label='Saddle Point')
    ax2.set_xlabel('θ₁')
    ax2.set_ylabel('θ₂')
    ax2.set_title('Saddle Point (Contour View)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_saddle_point()


def compare_saddle_vs_minimum():
    """
    Compare saddle point vs local minimum
    """
    x = np.linspace(-2, 2, 100)
    
    # Local minimum: z = x²
    local_min = x**2
    
    # Along one direction of saddle: z = x²
    saddle_direction1 = x**2
    
    # Along another direction of saddle: z = -x²
    saddle_direction2 = -x**2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Local minimum
    ax1.plot(x, local_min, linewidth=3, color='blue')
    ax1.scatter([0], [0], color='red', s=100, zorder=5)
    ax1.set_xlabel('θ')
    ax1.set_ylabel('Cost J')
    ax1.set_title('Local Minimum\n(Curves up in all directions)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Saddle point
    ax2.plot(x, saddle_direction1, linewidth=3, label='Direction 1 (curves up)', color='blue')
    ax2.plot(x, saddle_direction2, linewidth=3, label='Direction 2 (curves down)', color='orange')
    ax2.scatter([0], [0], color='red', s=100, zorder=5, label='Saddle Point')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('Cost J')
    ax2.set_title('Saddle Point\n(Curves up in some directions, down in others)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_saddle_vs_minimum()
```

### How Optimization Algorithms Handle This

#### 1. Momentum
- Builds velocity through plateaus
- Can "roll through" saddle points
- Doesn't get stuck as easily

#### 2. Adam
- Adapts learning rate per parameter
- Some parameters might have larger gradients (help escape)
- Momentum component also helps

#### 3. Stochastic Nature
- Mini-batch randomness adds noise
- Noise can help escape saddle points
- Different mini-batches give different gradient directions

### Practical Implications

**What to do:**
1. ✅ **Don't worry too much about local minima** in high dimensions
2. ✅ **Use momentum-based optimizers** (Momentum, RMSprop, Adam)
3. ✅ **Be patient with plateaus** - training might slow down but will likely escape
4. ✅ **Monitor training curves** - distinguish between plateau and convergence
5. ✅ **Use mini-batches** - stochasticity helps escape saddle points

**Signs you're at a plateau (not converged)**:
- Gradient magnitude is small but non-zero
- Training cost decreases very slowly
- Eventually starts decreasing faster again

**Signs you've converged**:
- Gradient magnitude approaches zero
- Training cost stops decreasing
- Validation cost has stopped improving (with early stopping)

### Example: Detecting Plateaus

```python
def detect_plateau(costs, window=50, threshold=1e-5):
    """
    Detect if training is on a plateau
    
    Arguments:
    costs -- list of cost values over time
    window -- number of recent iterations to check
    threshold -- maximum allowed cost change to be considered plateau
    
    Returns:
    is_plateau -- boolean indicating if we're on a plateau
    """
    if len(costs) < window:
        return False
    
    recent_costs = costs[-window:]
    cost_change = abs(recent_costs[-1] - recent_costs[0]) / recent_costs[0]
    
    return cost_change < threshold


# Example usage in training loop
def train_with_plateau_detection(X, Y, layer_dims, num_epochs=1000):
    """
    Training with plateau detection
    """
    parameters = initialize_parameters(layer_dims)
    v, s = initialize_adam(parameters)
    costs = []
    t = 0
    
    for epoch in range(num_epochs):
        mini_batches = create_mini_batches(X, Y, 64)
        
        for mini_batch in mini_batches:
            t += 1
            X_batch, Y_batch = mini_batch
            
            AL, caches = forward_propagation(X_batch, parameters)
            cost = compute_cost(AL, Y_batch)
            grads = backward_propagation(AL, Y_batch, caches)
            
            parameters, v, s = update_parameters_with_adam(
                parameters, grads, v, s, t
            )
        
        costs.append(cost)
        
        # Check for plateau every 50 epochs
        if epoch > 0 and epoch % 50 == 0:
            if detect_plateau(costs, window=50):
                print(f"Epoch {epoch}: On a plateau (cost = {cost:.6f})")
            else:
                print(f"Epoch {epoch}: Making progress (cost = {cost:.6f})")
    
    return parameters, costs
```

### Summary Table

| Concept | What it is | Problem? | How to handle |
|---------|-----------|----------|---------------|
| **Local Minimum** | Point where gradient is zero, curves up in all directions | Rare in high dimensions | Not a major concern |
| **Saddle Point** | Point where gradient is zero, but some directions curve down | Common in high dimensions | Momentum, stochasticity |
| **Plateau** | Flat region with very small gradients | Slows training significantly | Be patient, use Adam, adjust learning rate |

### Key Takeaway

**The main challenge in deep learning optimization is not getting stuck in local optima, but rather:**
1. Navigating through saddle points
2. Escaping plateaus efficiently
3. Dealing with high variance in mini-batch gradients

Modern optimization algorithms (especially Adam) are designed to handle these challenges effectively.

---

## Summary and Recommendations

### Quick Reference Guide

| Algorithm | When to Use | Typical Hyperparameters | Pros | Cons |
|-----------|-------------|------------------------|------|------|
| **Mini-batch GD** | Always (as base) | batch_size = 64-512 | Fast, vectorized | Needs good learning rate |
| **Momentum** | When oscillations occur | β = 0.9 | Smooths updates, faster | One more hyperparameter |
| **RMSprop** | RNNs, adaptive needs | β = 0.999 | Adapts per parameter | Can be unstable |
| **Adam** | Default choice | β₁=0.9, β₂=0.999, α=0.001 | Robust, adaptive, fast | Sometimes overfits |
| **LR Decay** | Long training runs | Varies by schedule | Better convergence | Needs tuning |

### Recommended Workflow

1. **Start simple**: Mini-batch GD with fixed learning rate
2. **Add Adam**: Use default hyperparameters (β₁=0.9, β₂=0.999, ε=10⁻⁸)
3. **Tune learning rate**: Try α ∈ {0.1, 0.01, 0.001, 0.0001}
4. **Add LR decay**: If training for many epochs
5. **Monitor carefully**: Watch for plateaus vs convergence
6. **Don't worry about local optima**: Focus on saddle points and plateaus

### Further Reading

- Adam paper: Kingma & Ba, 2014
- RMSprop: Hinton's Coursera lecture
- Momentum: Polyak, 1964
- Learning rate schedules: Review recent papers on cosine annealing, warm restarts

---

*Document created: February 13, 2026*

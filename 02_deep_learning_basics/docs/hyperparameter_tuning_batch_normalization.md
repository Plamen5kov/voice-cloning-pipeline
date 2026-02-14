# Hyperparameter Tuning and Batch Normalization

## Table of Contents
1. [Random Sampling for Hyperparameters](#random-sampling-for-hyperparameters)
2. [Using an Appropriate Scale to Pick Hyperparameters](#using-an-appropriate-scale-to-pick-hyperparameters)
3. [Hyperparameter Tuning in Practice](#hyperparameter-tuning-in-practice)
4. [Normalizing Activations in a Network](#normalizing-activations-in-a-network)
5. [Fitting Batch Norm into a Neural Network](#fitting-batch-norm-into-a-neural-network)
6. [Why Does Batch Norm Work](#why-does-batch-norm-work)
7. [Batch Norm at Test Time](#batch-norm-at-test-time)
8. [Softmax Regression](#softmax-regression)
9. [Training a Softmax Classifier](#training-a-softmax-classifier)
10. [Deep Learning Frameworks](#deep-learning-frameworks)
11. [TensorFlow](#tensorflow)
12. [Gradient Tape and More](#gradient-tape-and-more)

---

## Random Sampling for Hyperparameters

### Why Random Sampling Matters

When training deep learning models, choosing the right hyperparameters is crucial for achieving optimal performance. Random sampling is a powerful strategy for exploring the hyperparameter space efficiently.

### The Problem with Grid Search

**Grid Search Approach:**
```python
# Grid search example (NOT recommended)
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
hidden_units = [50, 100, 150]

# This creates only 3^3 = 27 combinations
# Tests only 3 discrete values per hyperparameter
for lr in learning_rates:
    for batch in batch_sizes:
        for units in hidden_units:
            train_model(lr, batch, units)
```

**Problems:**
- ❌ Limited exploration (only 3 values per hyperparameter)
- ❌ Wastes resources if one hyperparameter doesn't matter
- ❌ May miss optimal values between grid points
- ❌ Exponentially expensive as dimensions increase

### Random Search: A Better Alternative

**Random Search Approach:**
```python
# Random search example (✅ Recommended)
import numpy as np

n_trials = 100

for trial in range(n_trials):
    # Sample hyperparameters randomly from continuous distributions
    learning_rate = 10 ** np.random.uniform(-4, -1)  # [0.0001, 0.1]
    batch_size = np.random.choice([32, 64, 128, 256])
    hidden_units = np.random.randint(50, 200)
    dropout_rate = np.random.uniform(0.1, 0.5)
    
    # Train model with these hyperparameters
    model = train_model(learning_rate, batch_size, hidden_units, dropout_rate)
    
    # Track performance
    performance = evaluate_model(model)
    save_results(trial, hyperparameters, performance)
```

**Advantages:**
- ✅ Explores many more values per hyperparameter (100 different values)
- ✅ More likely to find good values for important hyperparameters
- ✅ Can sample from continuous distributions
- ✅ Scales well to high-dimensional spaces
- ✅ Easy to parallelize

### Why Random Search Works Better

**Example Scenario:**
- 2 hyperparameters: learning_rate (very important), batch_size (less important)
- Budget: 9 experiments

**Grid Search (3×3):**
```
batch_size:  32    64    128
           +-----+-----+-----+
    0.001  |  ×  |  ×  |  ×  |  learning_rate
    0.01   |  ×  |  ×  |  ×  |
    0.1    |  ×  |  ×  |  ×  |
           +-----+-----+-----+

Result: Tests only 3 learning rate values
```

**Random Search (9 random points):**
```
batch_size
    |  ×      ×
    |     ×        ×
    |  ×    ×
    |        ×   ×
    |   ×
    +------------------
         learning_rate

Result: Tests 9 different learning rate values!
```

### Understanding the 10**r Formula

One of the most important techniques in random sampling is using **logarithmic scale** for certain hyperparameters. The formula `10**r` is key to this approach.

#### Why Use 10**r?

**The Problem with Uniform Sampling:**
```python
# BAD: Uniform sampling for learning rate
learning_rate = np.random.uniform(0.0001, 1.0)

# Result: 90% of samples will be between 0.1 and 1.0
# Only 10% between 0.0001 and 0.1
# This wastes trials on the high end and under-explores the low end
```

**The Solution: Logarithmic Sampling with 10**r:**
```python
# GOOD: Logarithmic sampling
r = np.random.uniform(-4, 0)  # Sample exponent uniformly
learning_rate = 10**r          # Transform to get final value

# Result: Equal exploration across orders of magnitude
# 25% of samples in [0.0001, 0.001]
# 25% of samples in [0.001, 0.01]
# 25% of samples in [0.01, 0.1]
# 25% of samples in [0.1, 1.0]
```

#### Step-by-Step Explanation

**Step 1: Identify the range**
```
We want learning_rate ∈ [0.0001, 1]
In scientific notation: [10^-4, 10^0]
```

**Step 2: Sample the exponent uniformly**
```python
r = np.random.uniform(-4, 0)
# r could be: -3.7, -2.1, -0.5, etc.
```

**Step 3: Compute 10^r**
```python
learning_rate = 10**r

Examples:
If r = -3.7  →  learning_rate = 10^-3.7 ≈ 0.0002
If r = -2.1  →  learning_rate = 10^-2.1 ≈ 0.0079
If r = -0.5  →  learning_rate = 10^-0.5 ≈ 0.316
```

#### Visual Comparison

```
Uniform Sampling (BAD):
[0.0001]────────────────────────────────────[1.0]
   |                                           |
   ├──┤ 10% of samples here
       ├────────────────────────────────────┤ 90% of samples here
   
Logarithmic Sampling (GOOD):
[0.0001]─────[0.001]─────[0.01]─────[0.1]─────[1.0]
   |           |           |          |          |
   ├─────┤     ├─────┤     ├────┤     ├────┤
   25%         25%         25%        25%
```

#### More Examples

**Beta for Exponentially Weighted Averages:**
```python
# For β ∈ [0.9, 0.999] (corresponding to averaging 10-1000 examples)
# Sample 1-β on log scale

r = np.random.uniform(-3, -1)  # Sample exponent for 1-β
one_minus_beta = 10**r         # 1-β ∈ [0.001, 0.1]
beta = 1 - one_minus_beta      # β ∈ [0.9, 0.999]

# Why? Because β near 0.999 is very sensitive:
# β = 0.9   → averages ~10 examples
# β = 0.99  → averages ~100 examples  (10x difference)
# β = 0.999 → averages ~1000 examples (10x difference)
```

**L2 Regularization Parameter:**
```python
# For λ ∈ [0.000001, 0.01] = [10^-6, 10^-2]
r = np.random.uniform(-6, -2)
lambda_reg = 10**r

# Explores: 0.000001, 0.00001, 0.0001, 0.001, 0.01 evenly
```

**Number of Hidden Units (NO log scale needed):**
```python
# For hidden_units ∈ [50, 200]
# Use uniform sampling (linear scale is fine here)
hidden_units = np.random.randint(50, 200)

# Why no log scale? Because difference between 50 and 100 units
# has similar impact as difference between 150 and 200 units
```

#### Quick Reference: When to Use 10**r

| Hyperparameter | Use Log Scale? | Reason |
|----------------|----------------|--------|
| Learning rate (α) | ✅ Yes | Spans multiple orders of magnitude |
| Beta (β for momentum/Adam) | ✅ Yes (for 1-β) | Small changes near high values matter a lot |
| L2 regularization (λ) | ✅ Yes | Spans multiple orders of magnitude |
| Dropout rate | ❌ No | Typically [0.1, 0.5], linear scale is fine |
| Number of layers | ❌ No | Discrete, small range (1-10) |
| Number of units | ❌ No | Linear impact on model capacity |
| Batch size | ❌ No (usually) | Typically powers of 2: 16, 32, 64, 128, 256 |

#### Implementation Template

```python
def sample_hyperparameters():
    """Sample hyperparameters using appropriate scales."""
    
    # LOG SCALE hyperparameters (use 10**r)
    r_lr = np.random.uniform(-4, 0)
    learning_rate = 10**r_lr  # [0.0001, 1]
    
    r_lambda = np.random.uniform(-6, -2)
    l2_lambda = 10**r_lambda  # [0.000001, 0.01]
    
    r_one_minus_beta = np.random.uniform(-3, -1)
    beta = 1 - 10**r_one_minus_beta  # [0.9, 0.999]
    
    # LINEAR SCALE hyperparameters (use uniform/randint)
    dropout = np.random.uniform(0.1, 0.5)
    hidden_units = np.random.randint(50, 500)
    n_layers = np.random.randint(2, 6)
    batch_size = np.random.choice([32, 64, 128, 256])
    
    return {
        'learning_rate': learning_rate,
        'l2_lambda': l2_lambda,
        'beta': beta,
        'dropout': dropout,
        'hidden_units': hidden_units,
        'n_layers': n_layers,
        'batch_size': batch_size
    }

# Example usage
for trial in range(100):
    hp = sample_hyperparameters()
    print(f"Trial {trial}: lr={hp['learning_rate']:.6f}, "
          f"λ={hp['l2_lambda']:.6f}, β={hp['beta']:.4f}")
```

### Practical Implementation

#### Basic Random Search

```python
import numpy as np
from collections import defaultdict

def random_hyperparameter_search(n_trials=50, n_epochs=10):
    """
    Perform random hyperparameter search.
    
    Args:
        n_trials: Number of random configurations to try
        n_epochs: Number of epochs to train each configuration
    
    Returns:
        Best hyperparameters and their performance
    """
    results = []
    
    for trial in range(n_trials):
        # Sample hyperparameters
        hp = {
            'learning_rate': 10 ** np.random.uniform(-4, -1),
            'batch_size': np.random.choice([16, 32, 64, 128, 256]),
            'n_hidden_layers': np.random.randint(1, 6),
            'hidden_units': np.random.randint(32, 512),
            'dropout_rate': np.random.uniform(0.0, 0.5),
            'l2_reg': 10 ** np.random.uniform(-6, -2),
            'optimizer': np.random.choice(['adam', 'sgd', 'rmsprop']),
        }
        
        # If optimizer is SGD, sample momentum
        if hp['optimizer'] == 'sgd':
            hp['momentum'] = np.random.uniform(0.8, 0.99)
        
        print(f"Trial {trial+1}/{n_trials}: Testing {hp}")
        
        # Train model
        model, history = train_model(hp, n_epochs)
        
        # Evaluate
        val_accuracy = history['val_accuracy'][-1]
        
        # Store results
        results.append({
            'hyperparameters': hp,
            'val_accuracy': val_accuracy,
            'history': history
        })
        
        print(f"  → Validation Accuracy: {val_accuracy:.4f}")
    
    # Find best configuration
    best = max(results, key=lambda x: x['val_accuracy'])
    
    print(f"\nBest hyperparameters (acc={best['val_accuracy']:.4f}):")
    for key, value in best['hyperparameters'].items():
        print(f"  {key}: {value}")
    
    return best, results
```

#### Advanced: Random Search with Early Stopping

```python
def random_search_with_early_stopping(n_trials=50, max_epochs=100, patience=5):
    """
    Random search with early stopping to save time.
    """
    results = []
    
    for trial in range(n_trials):
        hp = sample_hyperparameters()
        
        # Train with early stopping
        model, history, stopped_epoch = train_with_early_stopping(
            hp, 
            max_epochs=max_epochs,
            patience=patience
        )
        
        results.append({
            'hyperparameters': hp,
            'val_accuracy': max(history['val_accuracy']),
            'stopped_at_epoch': stopped_epoch,
            'training_time': history['training_time']
        })
    
    return results
```

### Sampling Strategies for Different Hyperparameters

#### 1. **Continuous Hyperparameters**

```python
# Learning rate (log-uniform)
learning_rate = 10 ** np.random.uniform(-5, 0)  # [1e-5, 1]

# Dropout rate (uniform)
dropout = np.random.uniform(0.0, 0.5)  # [0, 0.5]

# L2 regularization (log-uniform)
l2_lambda = 10 ** np.random.uniform(-6, -2)  # [1e-6, 1e-2]

# Momentum (uniform, but close to 1)
momentum = np.random.uniform(0.85, 0.99)  # [0.85, 0.99]
```

#### 2. **Discrete Hyperparameters**

```python
# Number of layers
n_layers = np.random.randint(1, 6)  # [1, 2, 3, 4, 5]

# Hidden units (log-scale for powers of 2)
units_power = np.random.randint(4, 10)  # [4, 5, 6, 7, 8, 9]
hidden_units = 2 ** units_power  # [16, 32, 64, 128, 256, 512]

# Batch size (powers of 2)
batch_size = np.random.choice([16, 32, 64, 128, 256])

# Activation function
activation = np.random.choice(['relu', 'tanh', 'sigmoid', 'elu'])
```

#### 3. **Conditional Hyperparameters**

```python
# Sample optimizer first
optimizer = np.random.choice(['adam', 'sgd', 'rmsprop'])

# Then sample optimizer-specific parameters
if optimizer == 'adam':
    beta1 = np.random.uniform(0.85, 0.95)  # Default: 0.9
    beta2 = np.random.uniform(0.99, 0.9999)  # Default: 0.999
elif optimizer == 'sgd':
    momentum = np.random.uniform(0.0, 0.99)
    nesterov = np.random.choice([True, False])
elif optimizer == 'rmsprop':
    rho = np.random.uniform(0.85, 0.95)  # Default: 0.9
```

### Complete Example: MNIST Hyperparameter Search

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

def sample_hyperparameters():
    """Sample random hyperparameters for MNIST."""
    return {
        'learning_rate': 10 ** np.random.uniform(-4, -2),
        'batch_size': np.random.choice([32, 64, 128]),
        'n_hidden': np.random.randint(1, 4),
        'hidden_units': np.random.choice([64, 128, 256, 512]),
        'dropout': np.random.uniform(0.1, 0.5),
        'optimizer': np.random.choice(['adam', 'sgd', 'rmsprop'])
    }

def build_model(hp):
    """Build model with given hyperparameters."""
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    # Add hidden layers
    for _ in range(hp['n_hidden']):
        model.add(keras.layers.Dense(hp['hidden_units'], activation='relu'))
        model.add(keras.layers.Dropout(hp['dropout']))
    
    # Output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Compile
    model.compile(
        optimizer=hp['optimizer'],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Load data
(X_train, y_train), (X_val, y_val) = keras.datasets.mnist.load_data()
X_train, X_val = X_train / 255.0, X_val / 255.0

# Random search
n_trials = 20
best_acc = 0
best_hp = None

for trial in range(n_trials):
    print(f"\n{'='*60}")
    print(f"Trial {trial + 1}/{n_trials}")
    print(f"{'='*60}")
    
    # Sample hyperparameters
    hp = sample_hyperparameters()
    print("Hyperparameters:")
    for key, value in hp.items():
        print(f"  {key:15s}: {value}")
    
    # Build and train
    model = build_model(hp)
    
    history = model.fit(
        X_train, y_train,
        batch_size=hp['batch_size'],
        epochs=10,
        validation_data=(X_val, y_val),
        verbose=0
    )
    
    # Evaluate
    val_acc = max(history.history['val_accuracy'])
    
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    
    # Track best
    if val_acc > best_acc:
        best_acc = val_acc
        best_hp = hp
        print("★ NEW BEST!")

print(f"\n{'='*60}")
print("BEST CONFIGURATION")
print(f"{'='*60}")
print(f"Accuracy: {best_acc:.4f}")
for key, value in best_hp.items():
    print(f"  {key:15s}: {value}")
```

### Tips for Effective Random Sampling

1. **Start with Wide Ranges**
   - Begin with broad parameter ranges
   - Narrow down based on initial results

2. **Use Appropriate Scales**
   - Log-scale for learning rates, regularization
   - Linear scale for dropout, number of layers

3. **Parallelize When Possible**
   - Run multiple trials simultaneously on different GPUs
   - Each trial is independent

4. **Track All Results**
   - Save all hyperparameter combinations and results
   - Visualize to find patterns

5. **Iterate and Refine**
   - Use coarse-to-fine strategy (covered in next section)
   - Zoom in on promising regions

### Visualization of Results

```python
import matplotlib.pyplot as plt

def plot_hyperparameter_impact(results):
    """Visualize how each hyperparameter affects performance."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Extract data
    lrs = [r['hyperparameters']['learning_rate'] for r in results]
    accs = [r['val_accuracy'] for r in results]
    
    # Plot learning rate impact (log scale)
    axes[0].scatter(lrs, accs, alpha=0.6)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Learning Rate Impact')
    
    # Plot batch size impact
    batch_sizes = [r['hyperparameters']['batch_size'] for r in results]
    axes[1].scatter(batch_sizes, accs, alpha=0.6)
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Batch Size Impact')
    
    # Add more plots for other hyperparameters...
    
    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png')
    plt.show()
```

### Summary: Random Sampling Best Practices

✅ **DO:**
- Use random sampling instead of grid search
- Sample from appropriate distributions (log for learning rate)
- Run enough trials (50-200 depending on budget)
- Track and visualize all results
- Use coarse-to-fine strategy
- Parallelize across multiple GPUs/machines

❌ **DON'T:**
- Use grid search for more than 2-3 hyperparameters
- Sample all hyperparameters uniformly
- Stop after just a few trials
- Ignore less important hyperparameters
- Forget to set random seeds for reproducibility

---

## Using an Appropriate Scale to Pick Hyperparameters

When tuning hyperparameters, it's crucial to use the right scale for sampling values. Not all hyperparameters should be sampled uniformly.

### Random Sampling vs. Appropriate Scale

**Uniform Sampling (for some hyperparameters):**
- Number of hidden units: Can sample uniformly from [50, 100]
- Number of layers: Can sample uniformly from discrete values [2, 3, 4, 5]

**Logarithmic Scale (for most hyperparameters):**

Many hyperparameters should be sampled on a logarithmic scale:

#### Learning Rate (α)
```
Instead of: α ∈ [0.0001, 1] uniformly
Use: α = 10^r where r ∈ [-4, 0] uniformly

Python example:
r = -4 * np.random.rand()  # r in [-4, 0]
alpha = 10**r
```

**Why logarithmic scale?**
- Explores both small and large values more efficiently
- For α ∈ [0.0001, 1], uniform sampling would give 90% of samples in [0.1, 1]
- Logarithmic sampling gives equal attention to [0.0001, 0.001], [0.001, 0.01], etc.

#### Exponentially Weighted Averages (β)

For β in momentum, RMSprop, or Adam:

```
β ∈ [0.9, 0.999]  # corresponds to averaging over 10 to 1000 examples

Use: 1 - β on logarithmic scale
1 - β ∈ [0.001, 0.1]

Python example:
r = np.random.uniform(-3, -1)  # for 1-β in [0.001, 0.1]
beta = 1 - 10**r
```

**Why this matters:**
- β = 0.9 averages over ~10 examples: (1/(1-β) ≈ 10)
- β = 0.999 averages over ~1000 examples
- Small changes near 0.999 have huge impact on behavior

### General Rule

Use logarithmic scale when:
- The hyperparameter spans multiple orders of magnitude
- Small changes in certain ranges have disproportionate effects
- You want to explore low values as thoroughly as high values

---

## Hyperparameter Tuning in Practice

### The Evolving Nature of Hyperparameter Tuning

**Hyperparameters to tune (in order of importance):**

1. **Most important:** Learning rate (α)
2. **Second tier:**
   - β (momentum term) - typically 0.9 is good default
   - Mini-batch size
   - Number of hidden units
3. **Third tier:**
   - Number of layers
   - Learning rate decay
4. **Fourth tier:**
   - Adam parameters: β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸ (rarely tuned)

### Search Strategies

#### 1. Random Search vs. Grid Search

**Grid Search (❌ Not recommended):**
```
Learning rates: [0.001, 0.01, 0.1]
Hidden units: [50, 100, 150]
→ Tests only 3 values per hyperparameter
```

**Random Search (✅ Recommended):**
```python
# Sample random combinations
for i in range(n_trials):
    learning_rate = 10 ** np.random.uniform(-4, 0)
    hidden_units = np.random.randint(50, 200)
    # Try this combination
```

**Why random search is better:**
- Explores more values for each hyperparameter
- More likely to find the important hyperparameter's optimal value
- If one hyperparameter matters more, random search explores it better

#### 2. Coarse to Fine Sampling

```
Step 1: Sample broadly across the entire range
        [-----------------------------]
        Find best region: [-----***-----]

Step 2: Zoom in and sample more densely in the best region
        [***]
        
Step 3: Continue refining until convergence
```

### Practical Considerations

#### Re-test Hyperparameters Occasionally

- Hyperparameters don't transfer perfectly across:
  - Different datasets
  - Different computational resources (GPUs, CPUs)
  - Different time periods (data distribution may shift)

**Recommendations:**
- Re-evaluate hyperparameters every few months
- When data or infrastructure changes significantly
- When moving to a new problem domain

#### Two Approaches to Training

**1. Babysitting One Model (Panda approach)**
- Limited computational resources
- Monitor training day by day
- Adjust learning rate, parameters based on performance
- Common in academia or small teams

**2. Training Many Models in Parallel (Caviar approach)**
- Sufficient computational resources
- Train many models with different hyperparameters simultaneously
- Pick the best performing model
- Common in industry with large clusters

---

## Normalizing Activations in a Network

### The Concept

Just as we normalize input features X to speed up learning, we can normalize activation values in hidden layers.

### Input Normalization (Review)

```python
# Normalize inputs
X_normalized = (X - μ) / σ

Where:
μ = mean of features
σ = standard deviation of features
```

**Benefits:**
- Makes cost function more symmetric
- Speeds up optimization
- Makes learning rate less sensitive

### Batch Normalization (Normalizing Hidden Layer Activations)

**Key Idea:** Normalize the values z^[l] (or a^[l]) for each layer

```
For layer l:
z^[l] = W^[l] * a^[l-1] + b^[l]

Normalize z^[l] to have:
- Mean = 0
- Variance = 1
(or other desired mean and variance)
```

### Batch Norm Algorithm

For a mini-batch of m examples:

```
Given z^(1), z^(2), ..., z^(m) for layer l

1. Compute mean:
   μ = (1/m) Σ z^(i)

2. Compute variance:
   σ² = (1/m) Σ (z^(i) - μ)²

3. Normalize:
   z_norm^(i) = (z^(i) - μ) / √(σ² + ε)
   
   (ε is small constant like 10^-8 for numerical stability)

4. Scale and shift:
   z̃^(i) = γ * z_norm^(i) + β
   
   where γ and β are learnable parameters
```

### Why Scale and Shift (γ and β)?

We don't always want mean = 0, variance = 1. Examples:

```
If activation is sigmoid:
- Mean = 0, variance = 1 keeps values in linear region
- May want to use non-linear region

Solution:
γ and β allow the network to learn:
- Any mean: controlled by β
- Any variance: controlled by γ

Special case:
If γ = √(σ² + ε) and β = μ
Then z̃ = z (identity transformation)
```

**Note:** With batch norm, the bias term b^[l] is not needed (gets zeroed out by mean subtraction). The β parameter in batch norm replaces it.

---

## Fitting Batch Norm into a Neural Network

### Network Architecture with Batch Norm

**Without Batch Norm:**
```
X → [W^[1], b^[1]] → z^[1] → a^[1] → [W^[2], b^[2]] → z^[2] → a^[2] → ... → ŷ
```

**With Batch Norm:**
```
X → [W^[1]] → z^[1] → BN(z^[1], β^[1], γ^[1]) → z̃^[1] → a^[1] → [W^[2]] → z^[2] → BN(z^[2], β^[2], γ^[2]) → z̃^[2] → a^[2] → ... → ŷ
```

### Parameters

**For each layer l:**
- W^[l]: Weight matrix (shape: n^[l] × n^[l-1])
- β^[l]: Batch norm shift parameter (shape: n^[l] × 1)
- γ^[l]: Batch norm scale parameter (shape: n^[l] × 1)
- ~~b^[l]~~: Bias term (removed when using batch norm)

### Implementation with Mini-Batches

```python
for t in range(num_iterations):
    # Get mini-batch
    X_batch = X[:, t*batch_size:(t+1)*batch_size]
    
    # Forward prop on mini-batch
    for l in range(1, L+1):
        z[l] = W[l] @ a[l-1]  # No bias term
        z_norm[l] = batch_norm(z[l])
        z_tilde[l] = gamma[l] * z_norm[l] + beta[l]
        a[l] = activation(z_tilde[l])
    
    # Compute cost on mini-batch
    
    # Backward prop
    # Compute dW[l], d_beta[l], d_gamma[l]
    
    # Update parameters
    W[l] = W[l] - alpha * dW[l]
    beta[l] = beta[l] - alpha * d_beta[l]
    gamma[l] = gamma[l] - alpha * d_gamma[l]
```

### Working with Optimization Algorithms

Batch norm works with:
- Gradient descent
- Momentum
- RMSprop
- Adam

**For Adam:**
```python
# Parameters include W, β, γ for each layer
# Adam maintains v and s for all parameters

# Update W
v_dW = beta1 * v_dW + (1 - beta1) * dW
s_dW = beta2 * s_dW + (1 - beta2) * dW^2
W = W - alpha * v_dW / (sqrt(s_dW) + epsilon)

# Update β (batch norm parameter, not Adam's beta!)
v_dbeta = beta1 * v_dbeta + (1 - beta1) * d_beta
s_dbeta = beta2 * s_dbeta + (1 - beta2) * d_beta^2
beta = beta - alpha * v_dbeta / (sqrt(s_dbeta) + epsilon)

# Update γ
v_dgamma = beta1 * v_dgamma + (1 - beta1) * d_gamma
s_dgamma = beta2 * s_dgamma + (1 - beta2) * d_gamma^2
gamma = gamma - alpha * v_dgamma / (sqrt(s_dgamma) + epsilon)
```

---

## Why Does Batch Norm Work

### 1. Normalizes Input Distribution

Similar to normalizing input features X:
- Makes optimization landscape smoother
- Allows higher learning rates
- Makes training less sensitive to weight initialization

### 2. Reduces Internal Covariate Shift

**Internal Covariate Shift:** The distribution of inputs to a layer changes as the parameters of previous layers change during training.

**Example:**
```
Training a cat classifier:
- Early layers learn to detect cats
- If all training images are black cats
- Later layers learn patterns specific to black cats
- If test set has colored cats, performance drops

With batch norm:
- Even if earlier layer outputs shift
- Batch norm re-centers and re-scales
- Later layers see more stable distributions
```

### 3. Slight Regularization Effect

**How batch norm regularizes:**

Each mini-batch is normalized using statistics (μ, σ²) computed on that mini-batch only:
- Adds noise to the values within each mini-batch
- Each example is processed differently depending on which mini-batch it's in
- Similar to dropout (but weaker effect)

**Regularization strength:**
- Larger mini-batch → less noise → less regularization
- Smaller mini-batch → more noise → more regularization

**Note:** Don't rely on batch norm for regularization. Use:
- L2 regularization
- Dropout
- Data augmentation

### 4. Allows Each Layer to Learn More Independently

```
Without batch norm:
Layer 3 depends heavily on exact values from layers 1 and 2

With batch norm:
Layer 3 receives normalized inputs
More robust to changes in earlier layers
Can learn more independently
```

---

## Batch Norm at Test Time

### The Challenge

During training:
```
μ and σ² are computed from mini-batch
Each mini-batch has its own statistics
```

At test time:
```
May process one example at a time
Can't compute meaningful mini-batch statistics
Need a different approach
```

### Solution: Exponentially Weighted Average

**During Training:**
Keep a running average of μ and σ² across mini-batches:

```python
# Initialize
μ_running = 0
σ²_running = 0

# During training (for each mini-batch)
μ_batch = mean(z_batch)
σ²_batch = variance(z_batch)

# Update running averages (exponentially weighted)
μ_running = beta * μ_running + (1 - beta) * μ_batch
σ²_running = beta * σ²_running + (1 - beta) * σ²_batch

# Typical: beta = 0.9 or 0.99
```

**At Test Time:**
Use the running averages instead of batch statistics:

```python
# Normalize using running statistics
z_norm = (z - μ_running) / sqrt(σ²_running + ε)

# Scale and shift
z_tilde = γ * z_norm + β
```

### Implementation Notes

Most deep learning frameworks handle this automatically:
```python
# TensorFlow
layer = tf.keras.layers.BatchNormalization()
# Automatically switches behavior for training vs inference

# PyTorch
model.train()  # Use batch statistics
model.eval()   # Use running statistics
```

### Alternative: Use Entire Training Set

Some implementations calculate μ and σ² from entire training set (not common):
```python
# After training
μ_final = mean(all_z_in_training_set)
σ²_final = variance(all_z_in_training_set)
```

---

## Softmax Regression

### Generalization of Logistic Regression

**Logistic Regression:** Binary classification (2 classes)
**Softmax Regression:** Multi-class classification (C classes)

### The Softmax Layer

For C classes, the output layer has C units:

```
z^[L] = W^[L] * a^[L-1] + b^[L]    # Shape: (C, 1)

Activation (softmax):
a^[L]_i = e^(z^[L]_i) / Σ(j=1 to C) e^(z^[L]_j)

Where:
- a^[L]_i is the probability of class i
- Σ a^[L]_i = 1 (probabilities sum to 1)
```

### Example: Image Classification (4 classes)

```
Classes:
0 = Cat
1 = Dog  
2 = Bird
3 = Other

Output:
z^[L] = [5.0]    →    a^[L] = [0.842]  = P(Cat)
        [2.0]          [0.042]  = P(Dog)
        [1.0]          [0.016]  = P(Bird)
        [0.1]          [0.006]  = P(Other)
                       -----
                       1.000

Prediction: class 0 (Cat) with 84.2% confidence
```

### Softmax Computation

```python
def softmax(z):
    """
    z: vector of size (C, 1)
    returns: probability vector of size (C, 1)
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)
```

**Properties:**
- Output is a valid probability distribution
- If one z_i >> others, softmax ≈ 1 for that class
- If all z_i are similar, probabilities are spread out

### Special Cases

**C = 2 (Binary classification):**
```
Softmax reduces to logistic regression:

a_1 = e^(z_1) / (e^(z_1) + e^(z_2))
a_2 = e^(z_2) / (e^(z_1) + e^(z_2))

This is equivalent to:
a_1 = σ(z_1 - z_2)
a_2 = 1 - a_1
```

---

## Training a Softmax Classifier

### Loss Function

For a single example with C classes:

```
True label: y = [0, 0, 1, 0]^T  (one-hot encoded, class 2)
Prediction: ŷ = [0.1, 0.2, 0.6, 0.1]^T

Loss (cross-entropy):
L(ŷ, y) = -Σ(j=1 to C) y_j * log(ŷ_j)

Since y is one-hot (only y_k = 1 for true class k):
L(ŷ, y) = -log(ŷ_k)

For this example (k = 2):
L = -log(0.6) ≈ 0.51
```

**Intuition:**
- Minimizing loss → maximizing log(ŷ_k)
- Maximizing log(ŷ_k) → maximizing ŷ_k (probability of correct class)
- Want ŷ_k as close to 1 as possible

### Cost Function

For m training examples:

```
J(W, b) = (1/m) Σ(i=1 to m) L(ŷ^(i), y^(i))
        = -(1/m) Σ(i=1 to m) Σ(j=1 to C) y_j^(i) * log(ŷ_j^(i))
```

### Gradient Computation

**Derivative of loss with respect to z^[L]:**

```
dz^[L] = ŷ - y

Example:
y = [0, 0, 1, 0]^T  (true label: class 2)
ŷ = [0.1, 0.2, 0.6, 0.1]^T

dz^[L] = [0.1, 0.2, -0.4, 0.1]^T
```

**Remarkably simple!** Same form as logistic regression.

### Backpropagation

```python
# Forward propagation
z_L = W_L @ a_L_minus_1 + b_L
a_L = softmax(z_L)  # Output layer

# Compute loss
loss = -np.sum(y * np.log(a_L))

# Backward propagation
dz_L = a_L - y
dW_L = (1/m) * dz_L @ a_L_minus_1.T
db_L = (1/m) * np.sum(dz_L, axis=1, keepdims=True)

# Continue backprop to earlier layers...
```

### Implementation Tips

**Numerical Stability:**
```python
# Naive implementation (can overflow)
def softmax_naive(z):
    return np.exp(z) / np.sum(np.exp(z))

# Stable implementation
def softmax_stable(z):
    z_shifted = z - np.max(z)  # Subtract max for stability
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)
```

**Vectorization:**
```python
# For mini-batch of m examples
# Z: (C, m)
# Y: (C, m) one-hot encoded labels

# Forward
A = softmax(Z, axis=0)  # Apply softmax along class dimension

# Loss
loss = -np.sum(Y * np.log(A)) / m

# Backward
dZ = A - Y
```

---

## Deep Learning Frameworks

### Why Use Frameworks?

**Manual Implementation Challenges:**
- Complex gradient computations
- Efficient GPU operations
- Numerical stability issues
- Memory management
- Debugging difficulties

**Framework Benefits:**
- Automatic differentiation
- Optimized operations
- Easy experimentation
- Community support
- Production deployment

### Choosing a Framework

**Criteria to consider:**
1. **Ease of programming** (development speed)
2. **Running speed** (training and inference)
3. **Truly open source** (governance, licensing)

### Popular Frameworks (2024)

#### 1. **TensorFlow / Keras**
```
Pros:
✓ Industry standard
✓ Excellent deployment tools (TF Serving, TF Lite)
✓ Strong mobile/edge support
✓ Keras provides high-level API

Cons:
✗ Steeper learning curve for advanced features
✗ More verbose for research
```

#### 2. **PyTorch**
```
Pros:
✓ Intuitive, Pythonic API
✓ Dynamic computation graphs
✓ Popular in research
✓ Great debugging experience

Cons:
✗ Deployment historically harder (improving)
✗ Less mature mobile support
```

#### 3. **JAX**
```
Pros:
✓ Functional programming style
✓ Automatic vectorization
✓ High performance
✓ Composable transformations

Cons:
✗ Smaller ecosystem
✗ Steeper learning curve
```

#### 4. **Others**
- **MXNet**: Scalable, multi-language
- **Caffe**: Fast, production-focused (legacy)
- **Chainer**: Dynamic graphs (legacy)

### Framework Evolution

**Trend:** Frameworks are converging:
- PyTorch added TorchScript for deployment
- TensorFlow 2.0 adopted eager execution
- Both support ONNX for model exchange

**Recommendation:**
- **Learning:** Start with PyTorch or Keras
- **Production:** TensorFlow or PyTorch (both mature)
- **Research:** PyTorch or JAX
- **Mobile:** TensorFlow Lite

---

## TensorFlow

### Installation

```bash
# CPU version
pip install tensorflow

# GPU version (same package, auto-detects GPU)
pip install tensorflow

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Basic Concepts

#### 1. **Tensors**

```python
import tensorflow as tf

# Creating tensors
x = tf.constant([1, 2, 3])  # 1D tensor
y = tf.constant([[1, 2], [3, 4]])  # 2D tensor
z = tf.zeros((3, 3))  # 3x3 matrix of zeros

# Tensor properties
print(x.shape)  # Shape
print(x.dtype)  # Data type
print(x.numpy())  # Convert to NumPy array
```

#### 2. **Variables**

```python
# Variables are mutable tensors (for model parameters)
W = tf.Variable(tf.random.normal((3, 2)))
b = tf.Variable(tf.zeros((2,)))

# Update variables
W.assign(W + 0.1)  # Add 0.1 to all elements
W.assign_add(0.1)  # Equivalent
```

#### 3. **Operations**

```python
# Element-wise operations
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

c = a + b  # [5, 7, 9]
d = a * b  # [4, 10, 18]

# Matrix operations
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])

C = tf.matmul(A, B)  # Matrix multiplication
D = A @ B  # Equivalent using @ operator
```

### Building Models with Keras

#### Sequential API (Simple)

```python
from tensorflow import keras

# Define model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

# Predict
predictions = model.predict(X_new)
```

#### Functional API (Flexible)

```python
# For complex architectures (multi-input, multi-output, skip connections)
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

#### Custom Model (Full Control)

```python
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dropout = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.dense3 = keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x)
        x = self.dense2(x)
        return self.dense3(x)

model = CustomModel()
```

### Common Layers

```python
# Dense (fully connected)
keras.layers.Dense(units=64, activation='relu')

# Convolutional
keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')
keras.layers.MaxPooling2D(pool_size=2)

# Recurrent
keras.layers.LSTM(units=128, return_sequences=True)
keras.layers.GRU(units=64)

# Normalization
keras.layers.BatchNormalization()
keras.layers.LayerNormalization()

# Regularization
keras.layers.Dropout(0.5)
keras.layers.GaussianNoise(0.1)

# Flatten
keras.layers.Flatten()
```

### Optimizers

```python
# SGD
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Adam
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# RMSprop
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

# Learning rate schedules
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.96
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

---

## Gradient Tape and More

### TensorFlow's GradientTape

The `GradientTape` API provides automatic differentiation for custom training loops.

#### Basic Usage

```python
import tensorflow as tf

# Define a simple function
x = tf.Variable(3.0)

# Record operations for automatic differentiation
with tf.GradientTape() as tape:
    y = x ** 2  # y = 9

# Compute gradient dy/dx
dy_dx = tape.gradient(y, x)
print(dy_dx)  # 6.0 (derivative of x^2 at x=3 is 2*3 = 6)
```

#### Training a Model

```python
# Define model, loss, optimizer
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.MeanSquaredError()

# Training loop
for epoch in range(epochs):
    for X_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(X_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Update weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

#### Multiple Gradients

```python
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as tape:
    z = x**2 + y**2  # z = 4 + 9 = 13

# Compute both gradients
dz_dx, dz_dy = tape.gradient(z, [x, y])
print(dz_dx)  # 4.0 (2x at x=2)
print(dz_dy)  # 6.0 (2y at y=3)
```

#### Persistent Tape

```python
x = tf.Variable(3.0)

# Persistent tape can be used multiple times
with tf.GradientTape(persistent=True) as tape:
    y = x ** 2
    z = x ** 3

dy_dx = tape.gradient(y, x)  # 6.0
dz_dx = tape.gradient(z, x)  # 27.0

# Must delete persistent tapes manually
del tape
```

#### Higher-Order Gradients

```python
x = tf.Variable(3.0)

# Compute second derivative
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x ** 3
    
    dy_dx = tape1.gradient(y, x)  # First derivative: 3x^2

d2y_dx2 = tape2.gradient(dy_dx, x)  # Second derivative: 6x
print(d2y_dx2)  # 18.0 (6*3)
```

### Advanced TensorFlow Features

#### 1. **Custom Layers**

```python
class CustomDense(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        # Create weights when we know input shape
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Use it
layer = CustomDense(32)
```

#### 2. **Custom Training Loops**

```python
class CustomTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_acc = keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    
    @tf.function  # Compile to graph for performance
    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            loss = self.loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss.update_state(loss)
        self.train_acc.update_state(y, predictions)
        
        return loss
    
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_acc.reset_states()
            
            for X_batch, y_batch in dataset:
                loss = self.train_step(X_batch, y_batch)
            
            print(f'Epoch {epoch}: Loss={self.train_loss.result():.4f}, '
                  f'Acc={self.train_acc.result():.4f}')
```

#### 3. **Callbacks**

```python
# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)

# TensorBoard logging
tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

# Learning rate scheduling
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3
)

# Train with callbacks
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stop, checkpoint, tensorboard, reduce_lr]
)
```

#### 4. **Data Pipeline (tf.data)**

```python
# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# Apply transformations
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Data augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# Use in training
model.fit(dataset, epochs=10)
```

#### 5. **Mixed Precision Training**

```python
# Use mixed precision for faster training on modern GPUs
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Build model (automatically uses float16 for computations)
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax', dtype='float32')  # Keep output in float32
])

# Loss scaling for numerical stability
optimizer = keras.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
```

#### 6. **Distributed Training**

```python
# Distribute training across multiple GPUs
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Create model within strategy scope
with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Training automatically distributed
model.fit(train_dataset, epochs=10)
```

### Best Practices

1. **Use `@tf.function` for performance:**
```python
@tf.function
def train_step(X, y):
    # Compiles to graph for faster execution
    ...
```

2. **Profile your code:**
```python
# Use TensorFlow Profiler
tf.profiler.experimental.start('logdir')
# ... training code ...
tf.profiler.experimental.stop()
```

3. **Check for memory leaks:**
```python
# Clear session periodically
keras.backend.clear_session()
```

4. **Use appropriate data types:**
```python
# Use float32 by default (float16 for mixed precision)
# Use int32/int64 for indices
```

5. **Validate input shapes:**
```python
model.build(input_shape=(None, 784))
model.summary()
```

---

## Summary

### Key Takeaways

1. **Hyperparameter Tuning:**
   - Use logarithmic scale for learning rate, β
   - Random search > Grid search
   - Coarse to fine sampling

2. **Batch Normalization:**
   - Normalizes hidden layer activations
   - Speeds up training, allows higher learning rates
   - Has slight regularization effect
   - Use running averages at test time

3. **Softmax Regression:**
   - Generalization of logistic regression to C classes
   - Uses cross-entropy loss
   - Gradient: dz^[L] = ŷ - y

4. **Deep Learning Frameworks:**
   - TensorFlow/Keras for production
   - PyTorch for research
   - Choose based on use case

5. **TensorFlow:**
   - GradientTape for automatic differentiation
   - Keras for high-level model building
   - tf.data for efficient data pipelines
   - Callbacks for training control

### Further Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Adam Paper](https://arxiv.org/abs/1412.6980)

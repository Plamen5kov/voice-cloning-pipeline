# Practical Hyperparameter Tuning & Deep Learning Guide

A focused, pragmatic guide based on essential deep learning concepts for real-world practice.

---

## 1. Hyperparameter Search Strategy

### Random Search vs Grid Search

**✅ USE: Random Search**
```python
import numpy as np

# Random search - explores more values efficiently
for trial in range(25):  # Even 25 trials covers a lot
    learning_rate = 10 ** np.random.uniform(-4, -1)
    hidden_units = np.random.randint(64, 512)
    dropout = np.random.uniform(0.1, 0.5)
    
    # Train and evaluate
```

**❌ AVOID: Grid Search**
- Exponentially expensive (k^n combinations)
- Wastes resources on less important hyperparameters
- Only tests few values per hyperparameter

**Why Random is Better:**
- Explores many more values for the most important hyperparameter
- You don't know in advance which hyperparameters matter most
- More efficient use of computational budget

---

## 2. Hyperparameter Priority (Limited Resources)

### Focus Your Effort Here:

**1st Priority - MUST TUNE:**
- **Learning Rate (α)** - Biggest impact on performance

**2nd Priority - Should tune if possible:**
- Number of hidden units
- Mini-batch size
- β (momentum) - if using momentum (default 0.9 is often good)

**3rd Priority - Nice to have:**
- Number of layers
- Learning rate decay

**4th Priority - Almost never tune:**
- Adam parameters (β₁=0.9, β₂=0.999, ε=10⁻⁸)
- Weight initialization variance

### Practical Action Plan:

**With very limited resources (5-10 experiments):**
```python
# Focus ONLY on learning rate
for trial in range(10):
    lr = 10 ** np.random.uniform(-4, -1)
    train_model(lr=lr)
```

**With moderate resources (25-50 experiments):**
```python
# Add architecture exploration
for trial in range(50):
    lr = 10 ** np.random.uniform(-4, -1)
    hidden_units = np.random.choice([64, 128, 256, 512])
    batch_size = np.random.choice([32, 64, 128])
    train_model(lr=lr, hidden_units=hidden_units, batch_size=batch_size)
```

---

## 3. Log Scale for Learning Rate

### The Formula

For learning rate α ∈ [0.0001, 1.0] = [10⁻⁴, 10⁰]:

```python
# Step 1: Sample exponent uniformly
r = np.random.uniform(-4, 0)

# Step 2: Compute α
alpha = 10**r

# Examples:
# r = -3.7  →  α ≈ 0.0002
# r = -2.1  →  α ≈ 0.0079
# r = -0.5  →  α ≈ 0.316
```

### When to Use Log Scale

| Hyperparameter | Use Log Scale? | Formula |
|----------------|----------------|---------|
| Learning rate (α) | ✅ Yes | `r = np.random.uniform(-4, 0); α = 10**r` |
| L2 regularization (λ) | ✅ Yes | `r = np.random.uniform(-6, -2); λ = 10**r` |
| Beta (β for momentum) | ✅ Yes (for 1-β) | `r = np.random.uniform(-3, -1); β = 1 - 10**r` |
| Dropout rate | ❌ No | `np.random.uniform(0.1, 0.5)` |
| Number of units | ❌ No | `np.random.randint(50, 500)` |
| Batch size | ❌ No | `np.random.choice([32, 64, 128, 256])` |

**Why?** Log scale ensures equal exploration across orders of magnitude.

---

## 4. Adam Optimizer Parameters

### Default Values (Use These!)

```python
optimizer = Adam(
    lr=0.001,        # ← ONLY tune this!
    beta1=0.9,       # ← Keep default
    beta2=0.999,     # ← Keep default
    epsilon=1e-8     # ← Keep default
)
```

**Key Insight:** Even with unlimited resources, β₁, β₂, and ε are rarely tuned. Defaults work excellently in practice.

**Priority:**
- β₁, β₂, ε are 4th tier (lowest priority)
- Save your limited resources for learning rate

---

## 5. Panda vs Caviar Approach

### Your Approach = Your Compute Budget

**Panda Approach 🐼** (Limited compute)
- Train ONE model at a time
- Monitor daily, adjust hyperparameters based on performance
- Babysit the training process
- Common in: Academia, small teams, individual researchers

**Caviar Approach 🐟** (Sufficient compute)
- Train MANY models in parallel
- Different hyperparameters simultaneously
- Pick best performer
- Common in: Industry, big tech companies

**Your choice depends entirely on available computational resources.**

---

## 6. When to Re-tune Hyperparameters

### ✅ Re-evaluate hyperparameters when:

1. **New data is added** - Data distribution may shift
2. **Computational resources change** - Different hardware affects optimal settings
3. **Every few months** - Regular practice
4. **Moving to new problem domain**

### Why?

Hyperparameters are NOT universal - they depend on:
- Dataset characteristics
- Hardware available
- Problem domain
- Time (data drifts)

**Action:** At minimum, re-tune learning rate when environment changes significantly.

---

## 7. Batch Normalization Essentials

### Key Parameters

**γ (gamma) and β (beta) are LEARNABLE PARAMETERS, not hyperparameters**

```python
# Batch norm algorithm
z_norm = (z - μ) / sqrt(σ² + ε)  # Normalize
z_tilde = γ * z_norm + β          # Scale and shift

# Where:
# γ controls variance
# β controls mean
# Both learned via backpropagation (like weights)
```

### Important Facts:

✅ **Drop b[l] (bias term)** when using batch norm - it gets zeroed out  
✅ **Keep W[l] (weights)** - still needed!  
✅ **ε (epsilon = 10⁻⁸)** prevents division by zero (numerical stability)  
✅ **One γ and one β PER HIDDEN UNIT** - not one per layer  

### Batch Norm at Test Time

**❌ WRONG:** Turn off batch norm  
**✅ CORRECT:** Use running averages instead of batch statistics

```python
# Training: Use batch statistics
z_norm = (z - μ_batch) / sqrt(σ²_batch + ε)

# Test: Use running averages (exponentially weighted)
z_norm = (z - μ_running) / sqrt(σ²_running + ε)

# Still apply scale and shift
z_tilde = γ * z_norm + β
```

**Frameworks handle this automatically:**
```python
# PyTorch
model.train()  # Uses batch statistics
model.eval()   # Uses running averages

# TensorFlow/Keras
model.fit()     # training=True
model.predict() # training=False
```

---

## 8. Deep Learning Framework Selection

### Three Main Criteria:

1. **Ease of programming** (development speed)
2. **Running speed** (training and inference)
3. **Truly open source** (governance, licensing)

### NOT Selection Criteria:

❌ Must use Python exclusively  
❌ Must run only on cloud  
❌ Must be implemented in C  

### Practical Recommendations:

- **Learning:** PyTorch or Keras
- **Production:** PyTorch or TensorFlow (both mature)
- **Research:** PyTorch or JAX
- **Mobile/Edge:** TensorFlow Lite

---

## 9. Quick Reference: Good Defaults

### Starting Point for Most Problems:

```python
# Optimizer
optimizer = 'Adam'
learning_rate = 0.001  # Start here, then tune

# Architecture
hidden_units = 128      # Try: 64, 128, 256, 512
batch_size = 64         # Try: 32, 64, 128
dropout = 0.2           # Try: 0.2-0.5 if overfitting

# Adam parameters (don't tune)
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Training
epochs = 50  # With early stopping
```

### Diagnostic Quick Guide:

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Cost increases/explodes | Learning rate too high | Reduce by 10x |
| Cost decreases very slowly | Learning rate too low | Increase by 2-5x |
| Cost decreases smoothly | Just right! ✓ | Keep it |
| Good train, poor validation | Overfitting | Add dropout/regularization |

---

## 10. Practical Workflow

### Step-by-Step Process:

1. **Start with good defaults** (see section 9)
2. **Train baseline model** - establish performance floor
3. **Tune learning rate** - Random search, 10-25 experiments
4. **If resources allow:** Explore architecture (units, layers, batch size)
5. **If overfitting:** Add regularization (dropout, L2)
6. **Re-evaluate** when data/resources change

### Coarse-to-Fine Strategy:

```
Step 1: Sample broadly
        [-----------------------------]
        Find best region: [-----***-----]

Step 2: Zoom in on best region
        [***]
        Sample more densely

Step 3: Refine until satisfied
```

---

## 11. Common Pitfalls to Avoid

❌ Spending too much time on low-priority hyperparameters  
❌ Using grid search with many hyperparameters  
❌ Using uniform sampling for learning rate  
❌ Thinking optimal hyperparameters transfer across datasets  
❌ Turning off batch norm at test time  
❌ Treating γ and β as hyperparameters to tune  
❌ Tuning Adam's β₁, β₂, ε when resources are limited  

---

## 12. Essential Code Snippets

### Random Hyperparameter Search

```python
import numpy as np

def sample_hyperparameters():
    """Sample hyperparameters using appropriate scales."""
    
    # LOG SCALE (use 10**r)
    r_lr = np.random.uniform(-4, -1)
    learning_rate = 10**r_lr  # [0.0001, 0.1]
    
    r_lambda = np.random.uniform(-6, -2)
    l2_lambda = 10**r_lambda  # [0.000001, 0.01]
    
    # LINEAR SCALE (use uniform/choice)
    dropout = np.random.uniform(0.1, 0.5)
    hidden_units = np.random.choice([64, 128, 256, 512])
    batch_size = np.random.choice([32, 64, 128])
    
    return {
        'learning_rate': learning_rate,
        'l2_lambda': l2_lambda,
        'dropout': dropout,
        'hidden_units': hidden_units,
        'batch_size': batch_size
    }

# Run search
best_score = 0
best_params = None

for trial in range(25):
    hp = sample_hyperparameters()
    score = train_and_evaluate(hp)
    
    if score > best_score:
        best_score = score
        best_params = hp
        print(f"New best: {score:.4f} with lr={hp['learning_rate']:.6f}")
```

### Batch Normalization Implementation

```python
def batch_norm_forward(z, gamma, beta, epsilon=1e-8):
    """Forward pass with batch normalization."""
    # Compute statistics
    mu = np.mean(z, axis=1, keepdims=True)
    sigma_sq = np.var(z, axis=1, keepdims=True)
    
    # Normalize
    z_norm = (z - mu) / np.sqrt(sigma_sq + epsilon)
    
    # Scale and shift
    z_tilde = gamma * z_norm + beta
    
    return z_tilde, (z_norm, mu, sigma_sq)
```

---

## Summary: Focus Areas for Practice

### 🎯 High Priority (Master These):

1. **Random search implementation** for hyperparameters
2. **Log scale sampling** for learning rate (10**r formula)
3. **Learning rate tuning** as first priority
4. **When to re-tune** hyperparameters
5. **Batch norm at test time** (running averages)

### 📚 Medium Priority (Understand Well):

6. Panda vs Caviar approaches
7. Adam optimizer defaults
8. Batch norm parameters (γ, β as learnable, not hyperparams)
9. Framework selection criteria

### 💡 Key Mindset:

- **Focus on learning rate** when resources are limited
- **Use random search**, not grid search
- **Use log scale** for hyperparameters spanning orders of magnitude
- **Defaults often work** for Adam parameters
- **Re-tune when environment changes**

---

*Created: February 14, 2026*
*Based on: Deep Learning Specialization concepts and practical experience*

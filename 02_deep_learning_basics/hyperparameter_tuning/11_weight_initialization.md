# Weight Initialization for Deep Networks

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 0:05 / 6:10

## Introduction

In the last video, you saw how very deep neural networks can have the problems of **vanishing and exploding gradients**.

**Partial solution:** Better, more careful choice of random initialization for your neural network.
- Doesn't solve it entirely
- But helps **a lot**!

## Starting Simple: Single Neuron

To understand weight initialization, let's start with initializing weights for a **single neuron**, then generalize to a deep network.

### Single Neuron Example

```
Input features:
  x₁ ──┐
  x₂ ──┤
  x₃ ──┼──→ [Neuron] ──→ a = g(z) ──→ ŷ
  x₄ ──┘
  
  Weights: w₁, w₂, w₃, w₄
```

**The computation:**

```
z = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + ... + wₙxₙ
a = g(z)
```

**Simplification:** Let's set b = 0 (ignore bias for now)

### The Key Insight: Variance Should Depend on n

**Problem:** We want z to be neither too big nor too small.

**Observation:**
- z is the **sum** of wixi terms
- The **larger n is**, the smaller you want wi to be
- If you're adding up **many terms**, you want each term to be **smaller**

### The Solution: Set Variance = 1/n

One reasonable approach:

```
Set variance of W to be 1/n
```

Where **n** = number of input features going into the neuron

**Why?** This prevents z from blowing up or becoming too small as you sum more terms.

## Generalizing to Deep Networks

### For a Single Layer

For layer l, the weight matrix W[l] can be initialized as:

```python
W[l] = np.random.randn(shape) * np.sqrt(1 / n[l-1])
```

Where:
- **shape** = dimensions of W[l]
- **n[l-1]** = number of units in layer l-1 (number of inputs to each neuron in layer l)

### Why n[l-1]?

```
Layer l-1        Layer l
(n[l-1] units) → (n[l] units)

Each unit in layer l receives n[l-1] inputs
Therefore, variance should be 1/n[l-1]
```

## Activation Function Specific Initialization

The best initialization depends on your activation function!

### 1. ReLU Activation (Most Common)

**For ReLU:** Use variance = **2/n**

```python
W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(2 / n[l-1])
```

**Where this comes from:**
- Paper by **He et al.**
- Works better than 1/n for ReLU
- Accounts for ReLU killing half the activations

**When to use:**
```python
if g(z) == ReLU(z):
    variance = 2 / n[l-1]
```

**This is the most common choice** since ReLU is the most common activation function!

### 2. Tanh Activation (Xavier Initialization)

**For tanh:** Use variance = **1/n**

```python
W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(1 / n[l-1])
```

**Where this comes from:**
- Called **"Xavier initialization"**
- Named after Xavier Glorot
- Theoretically derived for tanh activation

**When to use:**
```python
if g(z) == tanh(z):
    variance = 1 / n[l-1]  # Xavier
```

### 3. Yoshua Bengio Variant

**Alternative formula:**

```python
W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(2 / (n[l-1] + n[l]))
```

**Where this comes from:**
- Paper by **Yoshua Bengio** and colleagues
- Uses both input and output dimensions
- Has theoretical justification

**Formula:**
```
variance = 2 / (n[l-1] + n[l])
```

## Summary of Initialization Methods

| Activation Function | Recommended Variance | Name | Formula |
|---------------------|---------------------|------|---------|
| **ReLU** | 2/n[l-1] | He initialization | `np.sqrt(2/n[l-1])` |
| **tanh** | 1/n[l-1] | Xavier initialization | `np.sqrt(1/n[l-1])` |
| **Alternative** | 2/(n[l-1]+n[l]) | Bengio variant | `np.sqrt(2/(n[l-1]+n[l]))` |

## Implementation Examples

### ReLU Network (Most Common)

```python
# For each layer l with ReLU activation
W1 = np.random.randn(n1, nx) * np.sqrt(2 / nx)
W2 = np.random.randn(n2, n1) * np.sqrt(2 / n1)
W3 = np.random.randn(n3, n2) * np.sqrt(2 / n2)
# ... and so on

# General formula:
W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(2 / n[l-1])
```

### Tanh Network (Xavier)

```python
# For each layer l with tanh activation
W1 = np.random.randn(n1, nx) * np.sqrt(1 / nx)
W2 = np.random.randn(n2, n1) * np.sqrt(1 / n1)
W3 = np.random.randn(n3, n2) * np.sqrt(1 / n2)
# ... and so on

# General formula:
W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(1 / n[l-1])
```

## The Mathematical Intuition

### Why This Works

**Assumption:** Input features/activations are roughly:
- Mean ≈ 0
- Variance ≈ 1

**With proper initialization:**

```
If inputs have:  E[x] ≈ 0, Var(x) ≈ 1
And we set:      Var(w) = constant/n

Then z = Σ(wi·xi) will have:
  E[z] ≈ 0
  Var(z) ≈ 1
```

**Result:** z takes on a similar scale throughout the network!

### Without Proper Initialization

```
WRONG WAY: Just use np.random.randn(shape)

Layer 1:  Var(z) = n[0] × Var(w)     ← grows with n
Layer 2:  Var(z) = n[1] × Var(w)  
Layer 3:  Var(z) = n[2] × Var(w)
...
Layer L:  Var(z) = HUGE or tiny!     ← explodes or vanishes!
```

### With Proper Initialization

```
RIGHT WAY: np.random.randn(shape) * np.sqrt(constant/n)

Layer 1:  Var(z) ≈ 1
Layer 2:  Var(z) ≈ 1
Layer 3:  Var(z) ≈ 1
...
Layer L:  Var(z) ≈ 1     ← stays stable!
```

## How This Helps With Vanishing/Exploding Gradients

### The Goal

Try to set each weight matrix W[l] so that it's:
- ✓ Not too much bigger than 1
- ✓ Not too much less than 1
- ✓ Close to a scaled identity matrix

**Effect:** Gradients don't explode or vanish too quickly!

### Comparison

```
Bad Initialization (uniform random):
  Layer 1:  ||W|| ≈ random
  Layer 50: Activations EXPLODE or VANISH
  Gradients: Unusable

Good Initialization (variance = constant/n):
  Layer 1:  ||W|| ≈ 1
  Layer 50: Activations still reasonable
  Gradients: Usable for learning
```

## Tuning as a Hyperparameter

### The Variance is Tunable!

The variance parameter could be another hyperparameter to tune:

```python
# Add a tunable multiplier
variance_multiplier = 1.0  # hyperparameter to tune

W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(variance_multiplier * 2 / n[l-1])
```

**Example values to try:**
- `variance_multiplier = 0.5`
- `variance_multiplier = 1.0` (default)
- `variance_multiplier = 2.0`

### Priority for Tuning

**Andrew Ng's recommendation:**

```
Hyperparameter Priority:
  1. Learning rate α               ← First priority
  2. Hidden units, Mini-batch size
  3. Number of layers
  4. Learning rate decay
  ...
  N. Weight initialization variance ← Lower priority
```

**When to tune:**
- Not one of the first hyperparameters to tune
- But can have a modest to reasonable effect
- Sometimes helps a reasonable amount
- Usually lower priority relative to other hyperparameters

## Practical Guidelines

### Quick Decision Tree

```
What activation function are you using?

├─ ReLU (most common)
│  └─ Use: np.sqrt(2 / n[l-1])
│     ✓ Default choice
│     ✓ Works great in practice
│
├─ tanh
│  └─ Use: np.sqrt(1 / n[l-1])
│     ✓ Xavier initialization
│     ✓ Theoretically motivated
│
└─ Other or Mixed
   └─ Try: np.sqrt(2 / (n[l-1] + n[l]))
      Or experiment with variance multiplier
```

### General Recipe

1. **Use the formulas as default values** (starting point)
2. **Don't tune initially** - focus on other hyperparameters first
3. **If training is problematic**, consider tuning the variance multiplier
4. **For ReLU networks**, stick with 2/n[l-1] unless you have a reason to change

## Complete Example: 3-Layer Network

### Network Architecture

```
Input: 784 features (e.g., 28×28 image)
Layer 1: 128 units, ReLU
Layer 2: 64 units, ReLU
Layer 3: 10 units, softmax
```

### Initialization Code

```python
# Layer 1
n_x = 784
n_1 = 128
W1 = np.random.randn(n_1, n_x) * np.sqrt(2 / n_x)
b1 = np.zeros((n_1, 1))

# Layer 2
n_2 = 64
W2 = np.random.randn(n_2, n_1) * np.sqrt(2 / n_1)
b2 = np.zeros((n_2, 1))

# Layer 3
n_3 = 10
W3 = np.random.randn(n_3, n_2) * np.sqrt(2 / n_2)
b3 = np.zeros((n_3, 1))
```

### Expected Variance Scale

```
Layer 1: Var(z[1]) ≈ 1   (scaled by 2/784)
Layer 2: Var(z[2]) ≈ 1   (scaled by 2/128)
Layer 3: Var(z[3]) ≈ 1   (scaled by 2/64)

All layers maintain similar variance!
```

## Summary

### The Problem
- Deep networks suffer from vanishing/exploding gradients
- Random initialization can make this worse

### The Solution
- **Careful weight initialization** based on layer width
- Scale variance by 1/n or 2/n depending on activation

### Key Formulas

```
ReLU:  W[l] = np.random.randn(shape) * np.sqrt(2 / n[l-1])
tanh:  W[l] = np.random.randn(shape) * np.sqrt(1 / n[l-1])
Other: W[l] = np.random.randn(shape) * np.sqrt(2 / (n[l-1] + n[l]))
```

### Benefits
- ✓ Weights don't explode too quickly
- ✓ Weights don't decay to zero too quickly
- ✓ Can train reasonably deep networks
- ✓ Gradients remain in useful range
- ✓ Much faster training

### What This Doesn't Do
- ✗ Doesn't completely solve vanishing/exploding gradients
- ✗ Not a magic bullet
- ✓ But helps **a lot**!

## What's Next

When you train deep networks, this is another trick that will help you make your neural networks train **much more quickly**!

---

## Quick Reference Card

### Initialization Cheat Sheet

```python
# RELU (MOST COMMON - USE THIS!)
W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)

# TANH
W = np.random.randn(n_out, n_in) * np.sqrt(1 / n_in)

# ALTERNATIVE (BENGIO)
W = np.random.randn(n_out, n_in) * np.sqrt(2 / (n_in + n_out))

# BIAS (ALL CASES)
b = np.zeros((n_out, 1))
```

### Variance Table

| Method | Variance Formula | Best For | Paper |
|--------|-----------------|----------|-------|
| **He** | 2/n[l-1] | ReLU, Leaky ReLU | He et al. |
| **Xavier** | 1/n[l-1] | tanh, sigmoid | Glorot & Bengio |
| **Bengio variant** | 2/(n[l-1]+n[l]) | General | Bengio et al. |

### Why It Matters

```
Without proper initialization:
  → Activations: 10⁻¹⁵ or 10¹⁵  ⚠️
  → Gradients:   10⁻¹⁵ or 10¹⁵  ⚠️
  → Training:    Fails or very slow

With proper initialization:
  → Activations: Order of 1  ✓
  → Gradients:   Order of 1  ✓
  → Training:    Much faster ✓
```

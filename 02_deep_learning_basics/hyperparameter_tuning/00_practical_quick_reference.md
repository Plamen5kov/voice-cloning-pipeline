# Practical Quick Reference Guide

**Purpose:** Quick answers to common practical questions about neural network training, based on real scenarios and decision-making.

---

## Table of Contents
1. [Data Splitting](#data-splitting)
2. [Diagnosing Problems](#diagnosing-problems)
3. [Fixing High Bias (Underfitting)](#fixing-high-bias-underfitting)
4. [Fixing High Variance (Overfitting)](#fixing-high-variance-overfitting)
5. [Regularization](#regularization)
6. [Dropout](#dropout)
7. [Normalization](#normalization)
8. [Weight Initialization](#weight-initialization)
9. [Common Mistakes](#common-mistakes)

---

## Data Splitting

### Q: How do I split 10,000 examples?

**A: Use traditional ratios (60/20/20 or 70/30)**

```
For 10,000 examples:
Training:  6,000 (60%)
Dev:       2,000 (20%)
Test:      2,000 (20%)

Or:
Training:  7,000 (70%)
Test:      3,000 (30%)
```

**Only use modern ratios (98/1/1) for datasets > 1 million examples!**

### Q: Should dev and test sets come from the same distribution?

**A: YES - CRITICAL!**

```
✅ CORRECT:
Training:  Web images (different distribution OK)
Dev:       Phone camera images  }
Test:      Phone camera images  } ← SAME distribution

❌ WRONG:
Dev:   US phone cameras
Test:  European phone cameras  ← Different!
```

### Q: Do dev and test need the same number of examples?

**A: NO - Just need enough for their purpose**

```
For 10 million examples:
Training:  9,950,000 (99.5%)
Dev:          40,000 (0.4%)   ← Different sizes
Test:         10,000 (0.1%)   ← That's fine!

Each just needs to be big enough for its job.
```

---

## Diagnosing Problems

### Quick Diagnosis Chart

```
Training Error | Dev Error | Diagnosis
---------------|-----------|------------------
0.1%           | 11%       | HIGH VARIANCE (overfitting)
19%            | 21%       | HIGH BIAS (underfitting)
15%            | 16%       | HIGH BIAS
15%            | 30%       | HIGH BIAS + HIGH VARIANCE
0.5%           | 1%        | LOW BIAS + LOW VARIANCE ✓
```

### Step-by-Step Process

**Step 1: Check training error**
- High (>target)? → High bias problem
- Low? → Continue to step 2

**Step 2: Check gap between training and dev**
- Large gap (>2-3%)? → High variance problem
- Small gap? → You're good!

---

## Fixing High Bias (Underfitting)

### The Problem
- Model too simple to fit training data
- Training error is high

### Solutions (In Order of Effectiveness)

#### 1. ✅ Make Network Bigger
```python
# Instead of:
layers = [2, 3, 1]

# Try:
layers = [10, 8, 5, 1]  # More layers
# OR
layers = [20, 15, 1]    # More units per layer
```

**Effect:** Almost always helps!

#### 2. ✅ Train Longer
- More epochs
- Never hurts, doesn't always help

#### 3. ✅ Try Different Architecture
- Less systematic, but can help

### What NOT to Do

❌ **Get more data** - Won't help if you can't fit the data you have!
❌ **Add regularization** - Will make it worse!
❌ **Use dropout** - Will constrain the model more!

---

## Fixing High Variance (Overfitting)

### The Problem
- Model fits training data too well
- Large gap between training and dev error

### Solutions (In Order of Effectiveness)

#### 1. ✅✅✅ Get More Data (Best!)
```
Current: 10,000 examples
Try:     50,000 examples → Often solves the problem!
```

#### 2. ✅✅ Add/Increase Regularization

**L2 Regularization:**
```python
# Increase lambda
lambda = 0.01  → Dev error: 9%
lambda = 0.1   → Dev error: 6%
lambda = 1.0   → Dev error: 3%  ✓ Better!
```

**Dropout:**
```python
# Decrease keep_prob
keep_prob = 0.8  → Dev error: 8%
keep_prob = 0.5  → Dev error: 4%  ✓ Better!
```

#### 3. ✅✅ Data Augmentation
```python
# For images:
- Horizontal flips
- Random rotations
- Random crops
- Slight distortions
```

#### 4. ✅ Early Stopping
Stop training when dev error starts increasing

### What NOT to Do

❌ **Make network bigger** - Will overfit more!
❌ **Train longer** - Will overfit more!
❌ **Decrease regularization** - Opposite of what you need!

---

## Regularization

### L2 Regularization (Weight Decay)

#### How Lambda Affects Your Model

```
λ = 0      → No regularization → Can overfit
λ = 0.01   → Light regularization
λ = 0.1    → Moderate regularization
λ = 1.0    → Strong regularization
λ = 100    → Too much → Underfitting!

↑ Increase λ → ↑ More regularization → ↓ Less overfitting
↓ Decrease λ → ↓ Less regularization → ↑ More overfitting
```

#### Does Lambda Affect Test Predictions?

**NO - Only indirectly!**

**During Training:**
```python
J = cost + (λ/2m) × Σ||W||²  # Lambda used here
dW = gradient + (λ/m) × W
W = (1 - αλ/m) × W - α × gradient  # Weight decay
```

**During Testing:**
```python
ŷ = forward_prop(X, W, b)  # No lambda!
```

Lambda shapes W during training, but isn't in the prediction formula.

### What Increases Regularization?

```
Action                     | Increases Regularization?
---------------------------|-------------------------
✅ Increase λ              | YES
✅ Decrease keep_prob      | YES
✅ Data augmentation       | YES
✅ Early stopping          | YES
❌ Decrease λ              | NO (decreases it!)
❌ Increase keep_prob      | NO (decreases it!)
❌ Normalize inputs        | NO (for speed, not regularization)
```

---

## Dropout

### When to Use Dropout

```
✅ USE when:
- High variance (overfitting)
- Large network
- Computer vision (almost always)

❌ DON'T USE when:
- High bias (underfitting)
- No overfitting
- Need to monitor cost function J
```

### Keep_Prob Settings

```
Layer Type                  | keep_prob | Dropout Strength
----------------------------|-----------|------------------
Input layer                 | 1.0       | No dropout
Small layers (few params)   | 0.9-1.0   | Light/none
Medium layers               | 0.7       | Moderate
Large layers (many params)  | 0.5       | Strong
Output layer                | 1.0       | No dropout
```

### How Keep_Prob Affects Training

```
keep_prob: 0.5 → 0.6 (INCREASE)
           ↓
Effect:    LESS regularization
           ↓
Training:  LOWER error (fits better)
Dev:       Possibly HIGHER error (more overfitting)

keep_prob: 0.6 → 0.5 (DECREASE)
           ↓
Effect:    MORE regularization
           ↓
Training:  HIGHER error (fits worse)
Dev:       Possibly LOWER error (less overfitting)
```

### Inverted Dropout Implementation

#### During Training:
```python
# Step 1: Create mask
d = np.random.rand(a.shape[0], a.shape[1]) < keep_prob

# Step 2: Apply dropout
a = a * d

# Step 3: Scale up (INVERTED DROPOUT)
a = a / keep_prob

✅ Do BOTH dropout AND scaling!
```

#### During Testing:
```python
# NO dropout
# NO scaling
a = g(z)  # Just normal forward prop!

❌ Do NOT apply dropout at test time!
❌ Do NOT keep 1/keep_prob factor!
```

---

## Normalization

### Why Normalize Inputs?

**One reason only: SPEED UP TRAINING**

```
Purpose:
✅ Faster training
✅ Easier optimization
✅ Can use larger learning rates

NOT for:
❌ Reducing variance
❌ Regularization
❌ Visualization
```

### How It Helps

**Without Normalization:**
```
x₁ ∈ [1, 1000]  }  Very different
x₂ ∈ [0, 1]     }  scales

Result:
- Elongated cost function
- Small learning rate needed
- Many oscillating steps
- SLOW convergence
```

**With Normalization:**
```
x₁ ∈ [-1, 1]  }  Similar
x₂ ∈ [-1, 1]  }  scales

Result:
- Spherical cost function
- Larger learning rate possible
- Direct path to minimum
- FAST convergence ✓
```

### How to Normalize

```python
# Training set
μ = np.mean(X_train, axis=0)
σ² = np.var(X_train, axis=0)
X_train = (X_train - μ) / σ²

# Test set - use SAME μ and σ² from training!
X_test = (X_test - μ) / σ²  # ← Important!
```

### When to Normalize

**Always!**

> Just normalize anyway! It usually helps and pretty much never does any harm.

---

## Weight Initialization

### The Problem: Vanishing/Exploding Gradients

**With tanh activation:**

```
Symptom: Gradients near 0
Problem: Vanishing gradients
         Learning extremely slow
```

### Solutions

#### 1. ✅ Use Proper Initialization (Xavier for tanh)

```python
# For tanh activation:
W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(1 / n[l-1])
```

#### 2. ✅✅ Switch to ReLU (Better!)

```python
# Use ReLU instead of tanh
a = np.maximum(0, z)

# With He initialization:
W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(2 / n[l-1])
```

**Why ReLU is better:**
- No vanishing gradient problem
- Gradient is 1 for z > 0
- Much faster training

### Initialization Summary

| Activation | Variance | Name | Formula |
|------------|----------|------|---------|
| **ReLU** | 2/n[l-1] | He | `np.sqrt(2/n[l-1])` |
| **tanh** | 1/n[l-1] | Xavier | `np.sqrt(1/n[l-1])` |

---

## Common Mistakes

### Mistake 1: Using the Wrong Solution for the Problem

```
❌ WRONG:
High bias (can't fit training data)
→ Get more data (won't help!)

✅ CORRECT:
High bias → Bigger network
High variance → More data
```

### Mistake 2: Mixing Up Keep_Prob and Lambda

```
To INCREASE regularization:
✅ Increase λ (not decrease!)
✅ Decrease keep_prob (not increase!)

higher keep_prob = LESS dropout = LESS regularization
higher λ = MORE penalty = MORE regularization
```

### Mistake 3: Using Dropout When Not Needed

```
❌ WRONG:
Training error: 19%
Dev error: 21%
→ Add dropout (will make it worse!)

✅ CORRECT:
Small gap → No overfitting → Don't use dropout!
Only use dropout when overfitting (high variance)
```

### Mistake 4: Applying Dropout at Test Time

```
❌ WRONG:
# Test time
d = np.random.rand(...) < keep_prob
a = a * d / keep_prob

✅ CORRECT:
# Test time
a = g(z)  # Normal forward prop, no dropout!
```

### Mistake 5: Different μ and σ² for Test Set

```
❌ WRONG:
X_train = (X_train - μ_train) / σ²_train
X_test = (X_test - μ_test) / σ²_test  # Different!

✅ CORRECT:
# Calculate from training set
μ = np.mean(X_train)
σ² = np.var(X_train)

# Use SAME for both
X_train = (X_train - μ) / σ²
X_test = (X_test - μ) / σ²  # Same μ and σ²!
```

### Mistake 6: Wrong Dev/Test Distributions

```
❌ WRONG:
Dev:   High-quality images
Test:  Phone camera images  # Different!

✅ CORRECT:
Training: Web images (can differ)
Dev:      Phone images    }
Test:     Phone images    } Same distribution!
```

---

## Quick Decision Trees

### Decision Tree: What Should I Do?

```
START: Train model
   ↓
Is training error high?
   ├─ YES → HIGH BIAS
   │        ├─ Make network bigger
   │        ├─ More layers
   │        ├─ More units
   │        └─ Train longer
   │
   └─ NO → Check dev error
            ↓
       Is dev error much higher than training?
            ├─ YES → HIGH VARIANCE
            │        ├─ Get more data (best!)
            │        ├─ Add regularization (increase λ)
            │        ├─ Use dropout (decrease keep_prob)
            │        └─ Data augmentation
            │
            └─ NO → You're done! ✓
```

### Decision Tree: Should I Use Dropout?

```
Is training error much lower than dev error?
   ├─ YES → You have high variance
   │        ↓
   │     Are you working on computer vision?
   │        ├─ YES → Definitely use dropout
   │        └─ NO → Try other regularization first
   │
   └─ NO → DON'T use dropout!
           (You have high bias or are already good)
```

---

## Cheat Sheet: Symptoms & Solutions

| Symptoms | Problem | Solutions |
|----------|---------|-----------|
| Training error = 19%, Dev error = 21% | **High Bias** | Bigger network, more layers, train longer |
| Training error = 0.1%, Dev error = 11% | **High Variance** | More data, regularization, dropout |
| Training error = 15%, Dev error = 30% | **Both!** | Fix bias first, then variance |
| Gradients near 0 with tanh | **Vanishing Gradients** | Xavier init OR switch to ReLU |
| Training very slow | **Bad cost function** | Normalize inputs |
| Dropout makes training worse | **Wrong problem** | You have high bias, remove dropout! |

---

## Key Formulas

### Normalization
```python
μ = np.mean(X_train, axis=0)
σ² = np.var(X_train, axis=0)
X_normalized = (X - μ) / σ²
```

### L2 Regularization
```python
J = cost + (λ/2m) × Σ||W[l]||²
dW[l] = gradient + (λ/m) × W[l]
W[l] = (1 - αλ/m) × W[l] - α × gradient  # Weight decay
```

### Inverted Dropout (Training)
```python
d = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
a = a * d / keep_prob
```

### Weight Initialization
```python
# ReLU
W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(2/n[l-1])

# tanh
W[l] = np.random.randn(n[l], n[l-1]) * np.sqrt(1/n[l-1])
```

---

## Remember

1. **Diagnose first, then act** - Check training and dev errors before choosing solution
2. **Different problems need different solutions** - High bias ≠ high variance
3. **Normalize inputs always** - Speeds up training, never hurts
4. **Regularize only when overfitting** - Don't use dropout/high λ with high bias
5. **Dev and test must match** - Same distribution, critical!
6. **No dropout at test time** - And no scaling either!
7. **One μ and σ² for all** - Calculate from training, use for test too

---

**This guide is based on practical questions and real scenarios. Refer to individual topic files for detailed explanations.**

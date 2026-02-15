# Dropout Regularization

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 0:04 / 9:25

## Introduction

In addition to L2 regularization, another very powerful regularization technique is called **"dropout."**

## What is Dropout?

### The Problem: Overfitting

Let's say you train a neural network and it's overfitting. Here's what you do with dropout.

### The Dropout Process

**For each layer of the network:**
1. Set some probability of eliminating a node in the neural network
2. For each node, toss a coin
3. Have a certain chance (e.g., 0.5) of keeping each node
4. Have a certain chance (e.g., 0.5) of removing each node

### Visual Example

```
Original Network (overfitting):
Input → [H] [H] [H] [H] → [H] [H] [H] → [H] [H] → Output
        └─ Layer 1 ─┘      └ Layer 2 ┘   └ L3 ┘

After Dropout (50% keep probability):
Input → [H] [X] [H] [X] → [H] [X] [H] → [H] [X] → Output
        └─ Layer 1 ─┘      └ Layer 2 ┘   └ L3 ┘
        
        [H] = Hidden unit kept
        [X] = Hidden unit eliminated
```

### What Happens After Elimination

1. **Remove the node** and all its outgoing connections
2. You end up with a **much smaller, diminished network**
3. **Do backpropagation** training on this diminished network for one example
4. **On different examples**, toss a new set of coins
5. Keep a different set of nodes, eliminate different nodes
6. Train using a different neural network for each training example

### Why Does This Work?

While it might seem crazy to randomly drop nodes, this actually works!

**Intuition:** Because you're training a much smaller network on each example, you end up regularizing the network. These much smaller networks being trained prevent overfitting.

## Implementing Dropout: Inverted Dropout

There are several ways to implement dropout. The most common technique is called **inverted dropout**.

### Example: Implementing Dropout for Layer 3

Let's illustrate dropout for layer `l = 3`:

#### Step 1: Create Dropout Mask

```python
# Set dropout vector for layer 3
d3 = np.random.rand(a3.shape[0], a3.shape[1])  # Same shape as a3
d3 = d3 < keep_prob  # Boolean mask
```

Where:
- **`keep_prob`** = probability that a given hidden unit will be kept
- Example: `keep_prob = 0.8` means 80% chance of keeping, 20% chance of eliminating

#### Step 2: Apply Dropout Mask

```python
# Zero out elements
a3 = a3 * d3  # Element-wise multiplication
# Or equivalently: a3 *= d3
```

**What this does:**
- For every element of `d3` that equals zero (20% chance)
- The multiplication zeros out the corresponding element of `a3`

**Python note:** `d3` will be a boolean array (True/False), but the multiply operation works and interprets True as 1 and False as 0.

#### Step 3: Scale Up (Inverted Dropout)

```python
# Scale up to maintain expected value
a3 = a3 / keep_prob  # Or: a3 /= keep_prob
```

This is the **inverted dropout technique**.

## Why the Scaling Step?

### The Problem Without Scaling

Let's say you have **50 units** in the third hidden layer:
- `a3` is 50×1 dimensional (or 50×m if vectorized)
- With 80% keep probability, 20% elimination probability
- On average, you eliminate **10 units** (20% of 50)

### Impact on Next Layer

```python
z4 = W4 · a3 + b4
```

**Problem:** On expectation, `a3` will be reduced by 20% because 20% of elements are zeroed out.

### The Solution: Inverted Dropout

```python
a3 = a3 / keep_prob  # Divide by 0.8
```

**Effect:**
- This bumps the value back up by roughly 20%
- Corrects for the reduction
- Ensures the **expected value of `a3` remains the same**

### Benefits of Inverted Dropout

No matter what you set `keep_prob` to:
- `keep_prob = 0.8` (20% dropout)
- `keep_prob = 0.9` (10% dropout)  
- `keep_prob = 1.0` (no dropout)
- `keep_prob = 0.5` (50% dropout)

**Inverted dropout ensures that the expected value of `a3` remains unchanged.**

This makes **test time easier** because you have less of a scaling problem.

## Complete Inverted Dropout Implementation

```python
# Training time - Layer l
keep_prob = 0.8  # Example: 80% keep probability

# Step 1: Generate dropout mask
d_l = np.random.rand(a_l.shape[0], a_l.shape[1])
d_l = d_l < keep_prob

# Step 2: Apply mask (shut off neurons)
a_l = a_l * d_l

# Step 3: Scale up (inverted dropout)
a_l = a_l / keep_prob
```

## Important: Different Masks for Different Examples

### Across Training Examples

For different training examples, you zero out **different hidden units**.

### Across Iterations

If you make multiple passes through the same training set:
- **Iteration 1:** Zero out some hidden units (pattern A)
- **Iteration 2:** Zero out different hidden units (pattern B)
- Continue with different random patterns each iteration

**You should NOT keep zeroing out the same hidden units** for the same example across iterations.

### The `d` Vector Controls This

The vector `d` (or `d3` for layer 3) is used to decide what to zero out:
- Both in **forward propagation**
- And in **backpropagation**

## Dropout at Test Time

### What to Do at Test Time

**DO NOT use dropout at test time!**

```python
# Test time - No dropout
# Given test example x

a0 = x
z1 = W1 · a0 + b1
a1 = g1(z1)
z2 = W2 · a1 + b2
a2 = g2(z2)
# ... continue through all layers
# Make prediction ŷ
```

**Notice:**
- No dropout at test time
- No random coin flipping
- No random elimination of hidden units

### Why No Dropout at Test Time?

**Reason:** When making predictions at test time, you don't want your output to be random!

If you implemented dropout at test time, it would just add noise to your predictions.

### Alternative (Not Recommended)

In theory, you could:
1. Run the prediction process many times
2. Randomly drop out different hidden units each time
3. Average across all predictions

**Problems:**
- Computationally inefficient
- Gives roughly the same result as not using dropout
- Not worth the computational cost

### Why Inverted Dropout Makes Test Time Easy

Remember the step where we divided by `keep_prob`?

```python
a3 = a3 / keep_prob
```

**Effect:** This ensures that even when you don't implement dropout at test time:
- The scaling is already handled
- Expected values of activations don't change
- No need for extra scaling parameters at test time
- Test time is different from training time only in not dropping out units

## Summary of Dropout

### Training Time

```python
# For each layer with dropout:
d_l = np.random.rand(a_l.shape[0], a_l.shape[1]) < keep_prob
a_l *= d_l
a_l /= keep_prob  # Inverted dropout
```

### Test Time

```python
# NO dropout - just normal forward propagation
# No masking, no scaling
```

### Key Points

| Aspect | Details |
|--------|---------|
| **Purpose** | Regularization to prevent overfitting |
| **Training** | Randomly drop units each iteration |
| **Keep Probability** | Fraction of units to keep (e.g., 0.8 = keep 80%) |
| **Inverted Dropout** | Divide by keep_prob to maintain expected values |
| **Test Time** | NO dropout - use all units |
| **Randomness** | Different units dropped each iteration/example |

## Workflow Comparison

### With L2 Regularization
```
Define cost function J with penalty term
  ↓
Train network minimizing J
  ↓
Use same network at test time
```

### With Dropout
```
For each training example:
  ↓
Randomly eliminate different hidden units
  ↓
Train on diminished network
  ↓
At test time: Use full network (no dropout)
```

## Implementation Checklist

Training phase:
1. ✓ Set `keep_prob` for each layer
2. ✓ Generate random mask: `d = (np.random.rand(...) < keep_prob)`
3. ✓ Apply mask: `a *= d`
4. ✓ Scale up: `a /= keep_prob`
5. ✓ Use different mask for each iteration/example

Test phase:
1. ✓ Forward propagate normally
2. ✓ Do NOT apply dropout
3. ✓ Do NOT use random masks
4. ✓ Use all hidden units

## Why Inverted Dropout is Preferred

**Earlier dropout implementations:**
- Missed the `/ keep_prob` line
- Made test time more complicated
- Required extra scaling at test time

**Inverted dropout (modern standard):**
- Includes the `/ keep_prob` step
- Makes test time simple
- Most common implementation today
- **Recommended approach**

## Coming Up

**Question:** Why does dropout really work? What is dropout actually doing?

The next video will provide better intuition about what dropout is accomplishing under the hood.

---

## Quick Reference

### Inverted Dropout Formula

```python
# Layer l
keep_prob = 0.8  # Example

# Forward prop with dropout
d_l = np.random.rand(a_l.shape[0], a_l.shape[1]) < keep_prob
a_l = a_l * d_l / keep_prob

# Test time
# Just use a_l normally (no dropout)
```

### Keep Probability Examples

```
keep_prob = 1.0  → No dropout (keep 100%)
keep_prob = 0.9  → Drop 10% of units
keep_prob = 0.8  → Drop 20% of units
keep_prob = 0.5  → Drop 50% of units
keep_prob = 0.2  → Drop 80% of units (very aggressive)
```

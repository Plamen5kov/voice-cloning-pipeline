# Why Regularization Reduces Overfitting

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 0:06 / 7:09

## Introduction

Why does regularization help with overfitting? Why does it help with reducing variance problems? Let's go through a couple of examples to gain intuition about how it works.

## Recall: The Bias-Variance Spectrum

From our earlier discussions, we have three scenarios:

```
High Bias           Just Right          High Variance
(Underfit)         (Good fit)          (Overfit)
    
Simple model    Balanced model    Overly complex model
```

## The Setup: Overfitting Neural Network

Let's say we're fitting a large and deep neural network that is currently overfitting.

### Cost Function with Regularization

```
J(W, b) = (1/m) ∑ L(ŷ, y) + (λ/2m) ∑ ||W[l]||²F
                                     ↑
                              Frobenius norm penalty
```

What we did for regularization was add an extra term that **penalizes the weight matrices from being too large**.

**Question:** Why does shrinking the L2 norm (Frobenius norm) of parameters cause less overfitting?

## Intuition #1: Zeroing Out Hidden Units

### What Happens with Very Large λ?

If you crank the regularization parameter **λ to be really, really big**:

1. You'll be strongly incentivized to set weight matrices W to be reasonably close to zero
2. This will set weights so close to zero for many hidden units
3. Effectively **"zeroing out" a lot of hidden units**
4. Reducing the impact of these hidden units

### The Effect: Simpler Network

```
Original Complex Network:
[Input] → [Many Hidden Units] → [Many Hidden Units] → [Output]
  ↓ All units active and contributing

With High λ (Regularization):
[Input] → [Some units ≈ 0] → [Some units ≈ 0] → [Output]
  ↓ Many units have minimal impact

Result: Almost like a much smaller network
```

This becomes **almost like logistic regression**, but stacked multiple layers deep.

### Moving Along the Spectrum

```
λ too large:
  W ≈ 0 → Too simple → High Bias (left side)

λ = 0 (no regularization):
  Complex network → Overfitting → High Variance (right side)

λ just right:
  Balanced complexity → "Just Right" (middle)
```

**The intuition:** By cranking up λ to be really big, you move from the overfitting case (right) much closer to the high bias case (left).

Hopefully, there's an **intermediate value of λ** that results in the "just right" case in the middle.

### Clarification: What Really Happens

**Note:** The intuition of "completely zeroing out" hidden units isn't quite right.

**What actually happens:**
- It still uses **all the hidden units**
- But each hidden unit has a **much smaller effect**
- You end up with a **simpler network**
- Therefore **less prone to overfitting**

## Intuition #2: Keeping Activations in Linear Regime

### Assumption: Using tanh Activation Function

```
g(z) = tanh(z)

  1 |              _________
    |            /
    |           /
    |          /
  0 |_________/_____________ z
    |        /
    |       /
    |      /
    |_____/
 -1 |
    
    ←─────┬─────┬─────→
       Large  Small  Large
      negative  z  positive
         z          z
    
    Linear region (small z) → Roughly linear
    Large |z| values → Highly non-linear
```

### Key Observation

**So long as z is quite small** (taking on only a smallish range of values):
- You're using the **linear regime** of the tanh function
- The activation function is approximately linear

**When z wanders to larger or smaller values:**
- The activation function becomes **less linear**
- Non-linear behavior emerges

### How Regularization Enforces Linearity

#### Step 1: Large λ → Small Weights

```
Large λ → Weights W are penalized for being large
       → W becomes relatively small
```

#### Step 2: Small Weights → Small z

```
z = W·a + b

If W is small → z is also relatively small
(ignoring the effect of b for now)
```

#### Step 3: Small z → Linear Activation

```
If z takes on a small range of values:
  g(z) ≈ linear function

Each layer becomes roughly linear!
```

#### Step 4: Linear Layers → Linear Network

```
Layer 1: Linear
Layer 2: Linear
Layer 3: Linear
  ↓
Whole network = Linear function
```

**From Course 1:** If every layer is linear, then your **whole network is just a linear network**, even if it's very deep!

### The Implication

A deep network with linear activation functions:
- Can only compute a **linear function**
- Cannot fit very complicated, highly non-linear decision boundaries
- **Cannot overfit** to the same degree as a non-linear network

```
Regularization → Small W → Small z → Linear g(z) → Simple function
                                                    ↓
                                            Less overfitting
```

### Summary of Intuition #2

```
If λ is large:
  → W is small
  → z is small (takes on a small range of values)
  → g(z) is relatively linear (if using tanh)
  → Network computes something close to a linear function
  → Much simpler function (not highly non-linear)
  → Much less able to overfit
```

## Visual Comparison

### Without Regularization (Overfitting)

```
Highly non-linear decision boundary:
  - Wiggles around every training point
  - Captures noise in the data
  - Complex, non-linear activations throughout
```

### With Regularization

```
Smoother decision boundary:
  - More linear, less complex
  - Generalizes better
  - Activations closer to linear regime
```

## Implementation Tip: Debugging with Cost Function J

### Important Warning ⚠️

When implementing regularization, remember that you've **modified the definition of J**:

```
Old J (without regularization):
  J = (1/m) ∑ L(ŷ, y)

New J (with regularization):
  J = (1/m) ∑ L(ŷ, y) + (λ/2m) ∑ ||W[l]||²F
```

### Debugging Gradient Descent

One way to debug gradient descent:
- **Plot cost function J vs. number of iterations**
- You want to see J **decrease monotonically** after every iteration

### Common Mistake

❌ **Wrong:** Plotting the old definition of J (just the first term)
- You might not see monotonic decrease
- Misleading debugging information

✓ **Correct:** Plot the **new definition of J** (including regularization term)
- Both terms included: data loss + regularization penalty
- Should decrease monotonically

```
Plot this:  J = Loss + (λ/2m)||W||²F  ✓
Not this:   J = Loss only              ✗
```

Otherwise, you might not see J decrease monotonically on every single iteration!

## Summary: Two Intuitions for Why Regularization Works

### Intuition 1: Simpler Network
```
Large λ → Small W → Reduced impact of hidden units
       → Simpler network (fewer effective units)
       → Less capacity to overfit
```

### Intuition 2: Linear Regime
```
Large λ → Small W → Small z → Linear activations
       → Network behaves more linearly
       → Cannot fit complex non-linear boundaries
       → Less overfitting
```

## Key Takeaways

1. **Regularization reduces model complexity** without changing architecture
2. **Large λ** pushes toward high bias; **small λ** allows high variance
3. **Optimal λ** (tuned on dev set) balances the trade-off
4. **Two complementary intuitions:**
   - Reducing effective network size
   - Keeping activations in linear regime
5. **Always plot the complete J** (including regularization term) when debugging

## What's Next?

L2 regularization is the regularization technique used most often in training deep learning models.

However, in deep learning, there's another sometimes-used regularization technique called **dropout regularization**.

---

## Quick Visual Reference

### Effect of λ on Network Behavior

```
λ = 0 (No regularization)
├─ Large weights allowed
├─ Complex non-linear functions
├─ High capacity
└─ Risk: Overfitting (High Variance)

λ = Small
├─ Moderate weight penalty
├─ Balanced complexity
├─ Good capacity
└─ "Just Right" ✓

λ = Very Large
├─ Heavy weight penalty
├─ Near-linear functions
├─ Low capacity
└─ Risk: Underfitting (High Bias)
```

### The Regularization Sweet Spot

```
Model       ┃ Underfit  │  Good Fit  │  Overfit
Complexity  ┃           │            │
            ┃           │            │
Too Simple  ┃    ■      │            │
            ┃           │            │
Balanced    ┃           │     ★      │  ← Goal: Find this λ
            ┃           │            │
Too Complex ┃           │            │     ■
            ┃           │            │
            ┗━━━━━━━━━━┿━━━━━━━━━━━━┿━━━━━━━━━
                  Increase λ →
```

# Understanding Dropout

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 0:20 / 7:05

⚠️ **Note:** In the original video (around 2:40-2:50), there's an error in the dimensions stated. This document uses the **correct dimensions**:
- W[1] should be 7×3 (not 3×7)
- W[3] should be 3×7 (not 7×3)

**General rule:** Number of neurons in **previous layer** = number of **columns**; Number of neurons in **current layer** = number of **rows**.

## Introduction

Dropout does this seemingly crazy thing of randomly knocking out units in your network. Why does it work as a regularizer? Let's gain some better intuition.

## Intuition #1: Working with Smaller Networks

### From the Previous Video

Dropout randomly knocks out units in your network, so on every iteration you're working with a smaller neural network.

**Effect:** Using a smaller neural network seems like it should have a regularizing effect.

## Intuition #2: Can't Rely on Any One Feature

Let's look at dropout from the perspective of a single unit:

### The Perspective of One Unit

```
        Input 1 ──┐
        Input 2 ──┤
        Input 3 ──┼──→ [Hidden Unit] ──→ Output
        Input 4 ──┘
```

Consider a single unit (circled in purple in the video):
- This unit has **4 inputs**
- It needs to generate some meaningful output

### What Happens with Dropout

With dropout, the inputs can get randomly eliminated:
- Sometimes **two units** get eliminated
- Sometimes a **different unit** gets eliminated
- The pattern changes randomly

### The Effect: Spreading Out Weights

**What this means for the unit:**

❌ **Can't rely on any one feature** - any feature could go away at random
❌ **Can't rely on any one input** - any input could go away at random

**The unit's response:**

The unit will be **reluctant to put all its bets on just one input**.

Instead, it will be **motivated to spread out the weights**:
- Give a little bit of weight to each of the four inputs
- Don't put too much weight on any single input
- Distribute the importance across all inputs

### Connection to L2 Regularization

By spreading out the weights, this will tend to have an effect of **shrinking the squared norm of the weights**.

```
Spreading weights → Smaller individual weights → Smaller ||W||²
```

**Similar to L2 regularization:**
- Effect of implementing dropout is shrinking the weights
- Helps prevent overfitting

### Dropout as Adaptive L2 Regularization

Dropout can formally be shown to be an **adaptive form of L2 regularization**.

**Key difference:**
- The L2 penalty on different weights are **different**
- It depends on the **size of the activations** being multiplied into that weight
- Even more **adaptive** to the scale of different inputs

### Summary of Intuition #2

Dropout has a similar effect to L2 regularization:
- L2 regularization applied to different weights can be different
- More adaptive to the scale of different inputs
- Forces the network to not rely too heavily on any single feature

## Varying Keep-Prob by Layer

### Example Network Architecture

```
Input Layer:  3 features
Layer 1:      7 hidden units
Layer 2:      7 hidden units
Layer 3:      3 hidden units
Layer 4:      2 hidden units
Output:       1 unit
```

### Weight Matrix Dimensions (CORRECTED)

Using the correct dimensions:

```
W[1]: 7 × 3   (Layer 1 has 7 units, previous layer has 3)
W[2]: 7 × 7   (Layer 2 has 7 units, previous layer has 7) ← Largest!
W[3]: 3 × 7   (Layer 3 has 3 units, previous layer has 7)
W[4]: 2 × 3   (Layer 4 has 2 units, previous layer has 3)
W[5]: 1 × 2   (Output has 1 unit, previous layer has 2)
```

**Dimension Rule:** For weight matrix W[l]:
- **Rows** = number of units in current layer l
- **Columns** = number of units in previous layer (l-1)

### Choosing Different Keep-Prob Values

You can vary `keep_prob` by layer based on overfitting concerns:

#### Layer 2: Most Parameters → Lowest Keep-Prob

```python
keep_prob[2] = 0.5  # Most aggressive dropout
```

**Why?**
- W[2] is 7×7 - the **largest weight matrix**
- Has the most parameters
- More worried about overfitting
- Use **lower keep_prob** (more dropout)

#### Other Layers: Fewer Parameters → Higher Keep-Prob

```python
keep_prob[1] = 0.7
keep_prob[3] = 0.7
keep_prob[4] = 0.7
```

**Why?**
- Fewer parameters
- Less worried about overfitting
- Use **higher keep_prob** (less dropout)

#### Layers with No Overfitting Concern

```python
keep_prob[0] = 1.0  # Input layer - no dropout
keep_prob[5] = 1.0  # Output layer - no dropout
```

**Note:** `keep_prob = 1.0` means keeping every unit - you're really not using dropout for that layer.

### Strategy Summary

```
Layer Type              Keep-Prob    Dropout Strength
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Large layers (many params)  0.5      Strong dropout
Medium layers               0.7      Moderate dropout
Small layers                0.9      Light dropout
Input layer                 1.0      No dropout
Layers with few params      1.0      No dropout
```

### The Analogy to L2 Regularization

Lower `keep_prob` is like **cranking up the regularization parameter λ**:
- Regularize some layers more than others
- Layers with more parameters get more regularization

## Dropout on the Input Layer

### Can You Apply Dropout to Input?

**Yes, it's feasible:**
- Some chance of zeroing out one or more input features

### In Practice

**Rarely done:**
- Usually don't want to eliminate input features
- `keep_prob = 1.0` is quite common for the input layer
- Or use a very high value like `keep_prob = 0.9`

**Why?**
- Much less likely that you want to eliminate half of the input features
- Input features are the foundation of your model

### Summary: Input Layer Dropout

```
Common:       keep_prob = 1.0  (no dropout)
Occasionally: keep_prob = 0.9  (very light dropout)
Rare:         keep_prob = 0.5  (too aggressive for inputs)
```

## Hyperparameter Tuning Trade-offs

### More Flexibility = More Hyperparameters

If you vary `keep_prob` by layer:
- ✓ Can target regularization where needed most
- ✗ Gives you even more hyperparameters to search for using cross-validation

### Alternative: Simplified Approach

**Option 1:** Apply dropout to some layers, not others
- Have dropout on certain layers (e.g., largest layers)
- No dropout on other layers
- Just **one hyperparameter**: the `keep_prob` for layers where you apply dropout

**Option 2:** Use same `keep_prob` for all dropout layers
- Simpler to tune
- Just one value to search for

## Dropout in Computer Vision

### Very Frequently Used

Many of the first successful implementations of dropout were in **computer vision**.

**Why so common in computer vision?**

1. **Input size is huge** - all those pixels
2. **Almost never have enough data**
3. **Almost always overfitting**

### Common Practice

Computer vision researchers:
- Pretty much **always use dropout** almost as a default
- Has become standard practice

### Does This Generalize?

**Not always!**

The intuition from computer vision doesn't always generalize to other disciplines.

**Key principle to remember:**
- **Dropout is a regularization technique**
- It helps prevent overfitting
- **Unless your algorithm is overfitting, don't bother using dropout**

### Usage in Other Application Areas

Dropout is used **somewhat less often** in other application areas because:
- You might have enough data
- You might not be overfitting
- Other regularization techniques might work better

## Implementation Tips

### Big Downside: Cost Function J is Less Well-Defined

**The problem:**

On every iteration, you're randomly knocking off a bunch of nodes.

**Effect on debugging:**
- Cost function J is **no longer well-defined**
- Harder to double-check that gradient descent is working
- Can't easily verify that J is going downhill on every iteration
- **You lose the debugging tool** of plotting J vs. iterations

### Recommended Debugging Approach

#### Step 1: Turn Off Dropout

```python
# Set keep_prob = 1.0 for ALL layers
keep_prob = 1.0
```

#### Step 2: Run Code and Monitor J

```python
# Make sure J is monotonically decreasing
# Plot J vs iterations
# Verify gradient descent is working
```

```
Cost J
  ↓     ╲
  ↓       ╲
  ↓         ╲___
  ↓              ╲____
  ↓                    ╲____
  └────────────────────────────→ Iterations
  
  Should see smooth decrease
```

#### Step 3: Turn On Dropout

```python
# Now set keep_prob to desired values
keep_prob = [1.0, 0.7, 0.5, 0.7, 1.0]  # Example
```

#### Step 4: Hope for the Best

**Hope that:**
- You didn't introduce bugs during dropout implementation
- The training still works properly

**Challenge:**
- You need other ways (not plotting J) to verify code is working
- Harder to debug with dropout enabled

### Debugging Workflow

```
1. Implement dropout in code
        ↓
2. Turn dropout OFF (keep_prob = 1.0)
        ↓
3. Verify J decreases monotonically
        ↓
4. Verify gradient descent works correctly
        ↓
5. Turn dropout ON
        ↓
6. Train with dropout (harder to monitor)
```

## Summary: When to Use Dropout

### Use Dropout When:
- ✓ Your model is **overfitting**
- ✓ You have **high variance**
- ✓ You can't get more training data
- ✓ Working with very large models
- ✓ Computer vision problems (almost always)

### Don't Use Dropout When:
- ✗ Your model has **high bias** (underfitting)
- ✗ You have **enough data** and no overfitting
- ✗ L2 regularization is sufficient
- ✗ You need to carefully monitor cost function J

### Configuration Guidelines

| Layer Type | Typical Keep-Prob | Reasoning |
|------------|-------------------|-----------|
| **Input layer** | 1.0 (or 0.9) | Don't eliminate input features |
| **Small hidden layers** | 0.9 or 1.0 | Few parameters, less overfitting risk |
| **Large hidden layers** | 0.5 - 0.7 | Many parameters, higher overfitting risk |
| **Output layer** | 1.0 | Don't dropout output |

## Key Takeaways

1. **Dropout forces spreading of weights** - can't rely on any single feature
2. **Similar to adaptive L2 regularization** - different penalties for different weights
3. **Vary keep_prob by layer** - lower for layers with more parameters
4. **Very common in computer vision** - due to chronic data shortage
5. **Use only when overfitting** - it's a regularization technique
6. **Debugging is harder** - J is less well-defined with dropout
7. **Test without dropout first** - verify code works before adding complexity

## Coming Up

There are still a few more regularization techniques worth knowing. Let's talk about a few more such techniques in the next video.

---

## Quick Reference: Keep-Prob Strategy

```
Overfitting Concern    →    Keep-Prob Value
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Very high (large layer)  →  0.5 (50% dropout)
High                     →  0.6 - 0.7
Moderate                 →  0.8
Low                      →  0.9
None                     →  1.0 (no dropout)
```

### Dropout Effect on Weights

```
Without Dropout:
  w₁ = 10.0  ← Heavily relies on one feature
  w₂ = 0.1
  w₃ = 0.1
  w₄ = 0.2

With Dropout:
  w₁ = 3.0   ← Weights more spread out
  w₂ = 2.5
  w₃ = 2.8
  w₄ = 3.1
```

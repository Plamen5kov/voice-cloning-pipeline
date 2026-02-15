# Vanishing / Exploding Gradients

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 0:05 / 6:07

## Introduction

One of the problems of training neural networks, **especially very deep neural networks**, is vanishing and exploding gradients.

**What this means:**
- When training a very deep network
- Your derivatives (slopes) can get either:
  - **Very, very big** (exploding)
  - **Very, very small**, maybe even exponentially small (vanishing)
- This makes training **difficult**

In this video, we'll see:
1. What exploding and vanishing gradients really means
2. How careful random weight initialization can significantly reduce this problem

## Setting Up the Problem

### A Very Deep Neural Network

```
Input (x) → [Layer 1] → [Layer 2] → [Layer 3] → ... → [Layer L] → ŷ
             W[1],b[1]   W[2],b[2]   W[3],b[3]         W[L],b[L]
```

For visualization, let's say we have **only 2 hidden units per layer** (though it could be more).

**Parameters:** W[1], W[2], W[3], ..., W[L] and corresponding biases

### Simplifying Assumptions

To make the math clearer, let's assume:

1. **Linear activation function:** g(z) = z
2. **No bias:** b[l] = 0 for all layers

These assumptions allow us to see the core problem more clearly.

## Mathematical Analysis

### Forward Propagation with Simplified Network

With our assumptions, the output ŷ will be:

```
ŷ = W[L] · W[L-1] · W[L-2] · ... · W[3] · W[2] · W[1] · x
```

**Why?** Let's verify:

```
Layer 1:
  z[1] = W[1] · x + b[1] = W[1] · x     (since b = 0)
  a[1] = g(z[1]) = z[1]                 (since g(z) = z)
  → a[1] = W[1] · x

Layer 2:
  z[2] = W[2] · a[1] = W[2] · W[1] · x
  a[2] = g(z[2]) = z[2]
  → a[2] = W[2] · W[1] · x

Layer 3:
  a[3] = W[3] · W[2] · W[1] · x

...continuing this pattern...

Output:
  ŷ = W[L] · W[L-1] · ... · W[2] · W[1] · x
```

So ŷ is the **product of all weight matrices** applied to x.

## Case 1: Exploding Gradients (Weights > 1)

### Scenario: Each Weight Matrix Slightly Larger Than Identity

Let's say each weight matrix W[l] is:

```
W[l] = [1.5   0  ]
       [0    1.5 ]
       
= 1.5 × I  (where I is the identity matrix)
```

**Note:** Technically W[L] has different dimensions, but let's focus on the pattern for W[1] through W[L-1].

### What Happens to ŷ?

```
ŷ = W[L-1] · W[L-2] · ... · W[2] · W[1] · x

If each W[l] ≈ 1.5 × I:

ŷ ≈ (1.5 × I)^(L-1) · x
  = 1.5^(L-1) × I^(L-1) · x
  = 1.5^(L-1) × x
```

### The Exponential Growth

```
L = 2:   ŷ = 1.5^1 × x  = 1.5x
L = 3:   ŷ = 1.5^2 × x  = 2.25x
L = 5:   ŷ = 1.5^4 × x  = 5.06x
L = 10:  ŷ = 1.5^9 × x  = 38.4x
L = 20:  ŷ = 1.5^19 × x = 1,477x
L = 50:  ŷ = 1.5^49 × x = 2,448,641x   ⚠️⚠️
L = 100: ŷ = 1.5^99 × x = 4 × 10^17 x  ⚠️⚠️⚠️
L = 150: ŷ = 1.5^149 × x = HUGE!!!     ⚠️⚠️⚠️
```

**Result:** For a very deep neural network, the value of ŷ will **explode**!

It grows **exponentially** as a function of the number of layers L.

### Example with Input x = [1, 1]ᵀ

Let's trace the activations through the network:

```
Layer 0 (input):  a[0] = [1.0, 1.0]
Layer 1:          a[1] = [1.5, 1.5]
Layer 2:          a[2] = [2.25, 2.25]
Layer 3:          a[3] = [3.375, 3.375]
Layer 4:          a[4] = [5.06, 5.06]
...
Layer L:          a[L] = [HUGE, HUGE]

Activations EXPLODE! 💥
```

## Case 2: Vanishing Gradients (Weights < 1)

### Scenario: Each Weight Matrix Slightly Smaller Than Identity

Now let's say each weight matrix W[l] is:

```
W[l] = [0.5   0  ]
       [0    0.5 ]
       
= 0.5 × I
```

### What Happens to ŷ?

```
ŷ ≈ (0.5 × I)^(L-1) · x
  = 0.5^(L-1) × x
```

### The Exponential Decay

```
L = 2:   ŷ = 0.5^1 × x  = 0.5x
L = 3:   ŷ = 0.5^2 × x  = 0.25x
L = 5:   ŷ = 0.5^4 × x  = 0.0625x
L = 10:  ŷ = 0.5^9 × x  = 0.00195x
L = 20:  ŷ = 0.5^19 × x = 0.0000019x
L = 50:  ŷ = 0.5^49 × x = 1.78 × 10^-15 x  ⚠️⚠️
L = 100: ŷ = 0.5^99 × x = 1.58 × 10^-30 x  ⚠️⚠️⚠️
L = 150: ŷ = 0.5^149 × x ≈ 0                ⚠️⚠️⚠️
```

**Result:** The activations **decrease exponentially** as a function of L.

In a very deep network, activations end up **vanishing** to essentially zero!

### Example with Input x = [1, 1]ᵀ

```
Layer 0 (input):  a[0] = [1.0, 1.0]
Layer 1:          a[1] = [0.5, 0.5]
Layer 2:          a[2] = [0.25, 0.25]
Layer 3:          a[3] = [0.125, 0.125]
Layer 4:          a[4] = [0.0625, 0.0625]
...
Layer L:          a[L] ≈ [0, 0]

Activations VANISH! 🔻
```

## Visual Summary

### Exploding Activations (W > I)

```
Activation
Magnitude
    ↑
    │                                    •  ← Layer 150
    │                                   /
    │                               •  /
    │                              /  
 10⁶│                          •  /
    │                         /
    │                     •  /
 10³│                    /
    │                •  /
    │               /
  10│           •  /
    │          /
   1│      •  /
    │     /
    │  • /
    └──•──────────────────────────→ Layer
       1  10    50      100    150
       
     Exponential growth! (W = 1.5 × I)
```

### Vanishing Activations (W < I)

```
Activation
Magnitude
    ↑
   1│  •
    │   \
    │    \  •
0.1 │     \
    │      \
    │       \  •
10⁻³│        \
    │         \
    │          \  •
10⁻⁶│           \
    │            \
    │             \  •
10⁻¹⁵│             \___•___•___
    └────────────────────────────→ Layer
         1  10    50   100   150
         
       Exponential decay! (W = 0.5 × I)
```

## The Key Intuition

### Weights Slightly > Identity → Exploding

```
If W ≈ [1.1  0  ]  (just a bit > 1)
       [0   1.1 ]

Then with very deep network:
  Activations → 1.1^L → EXPLODES as L ↑
```

### Weights Slightly < Identity → Vanishing

```
If W ≈ [0.9  0  ]  (just a bit < 1)
       [0   0.9 ]

Then with very deep network:
  Activations → 0.9^L → VANISHES as L ↑
```

## Impact on Gradients

### Similar Problem for Gradients

The same reasoning applies to **derivatives/gradients**:

**During backpropagation:**
- Gradients also get multiplied by weight matrices
- They flow backwards through the network
- Same exponential compounding effect!

```
Exploding Gradients:  ∂J/∂W[1] ∝ W^L → HUGE
Vanishing Gradients:  ∂J/∂W[1] ∝ W^L → ~0
```

### Why This is a Problem

#### Exploding Gradients

```
Gradient is HUGE:
  → Weight update: W := W - α × (HUGE gradient)
  → Weights change drastically
  → Training becomes unstable
  → May diverge (NaN values)
```

#### Vanishing Gradients

```
Gradient is ~0:
  → Weight update: W := W - α × (tiny gradient)
  → Weights barely change
  → Learning is extremely slow
  → Early layers don't learn
  → Gradient descent takes tiny steps
```

**If gradients are exponentially smaller than L:**
- Gradient descent will take **tiny little steps**
- It will take a **long time** to learn anything
- Early layers essentially **freeze**

## Modern Deep Networks

### The Scale of the Problem

**Modern neural networks can be very deep:**

- Microsoft recently achieved great results with a **152-layer neural network**
- Many successful architectures have L = 50, 100, 150+ layers

**With such deep networks:**
- If W > I: activations/gradients increase exponentially
- If W < I: activations/gradients decrease exponentially

```
Example: L = 150

If W = 1.1 × I:  Factor = 1.1^150 = 3.8 × 10^6  (explodes!)
If W = 0.9 × I:  Factor = 0.9^150 = 6.5 × 10^-8 (vanishes!)
```

### Historical Context

**For a long time, this problem was a huge barrier to training deep neural networks.**

This is why deep learning took so long to become practical - the vanishing/exploding gradient problem made very deep networks nearly impossible to train!

## Summary

### The Problem

Deep networks suffer from:

| Issue | Cause | Effect |
|-------|-------|--------|
| **Exploding Gradients** | W slightly > I | Activations/gradients grow exponentially with depth |
| **Vanishing Gradients** | W slightly < I | Activations/gradients shrink exponentially with depth |

### Why It's a Problem

**Exploding:**
- 💥 Training becomes unstable
- 💥 Weights update too dramatically
- 💥 May diverge or produce NaN

**Vanishing:**
- 🔻 Learning becomes extremely slow
- 🔻 Early layers don't learn
- 🔻 Gradient descent stuck

### Mathematical Core

```
For L layers:

ŷ ∝ W^L

If W > 1:  W^L → ∞     (exponential growth)
If W < 1:  W^L → 0     (exponential decay)

Same applies to gradients in backprop!
```

## What's Next: The Partial Solution

**There's a partial solution** that doesn't completely solve this problem, but **helps a lot**:

### Careful Choice of Weight Initialization

How you initialize your weights can significantly reduce the vanishing/exploding gradient problem!

Let's explore this in the next video.

---

## Quick Reference

### The Pattern

```
Deep Network Depth:
  L = 10   → Manageable
  L = 50   → Problems start
  L = 100  → Serious issues
  L = 150+ → Critical without proper techniques

Compounding Effect:
  1.1^10  = 2.6      (acceptable)
  1.1^50  = 117      (problematic)
  1.1^100 = 13,781   (severe)
  1.1^150 = 1.6×10⁶  (catastrophic)

  0.9^10  = 0.35     (acceptable)
  0.9^50  = 0.005    (problematic)
  0.9^100 = 0.000027 (severe)
  0.9^150 ≈ 0        (catastrophic)
```

### Activation Trajectory

```
EXPLODING (W = 1.5 × I):
  Layer 1:   a = 1.5
  Layer 10:  a = 58
  Layer 50:  a = 2.4 million    ⚠️
  Layer 100: a = 4.4 × 10¹⁷    ⚠️⚠️
  
VANISHING (W = 0.5 × I):
  Layer 1:   a = 0.5
  Layer 10:  a = 0.001
  Layer 50:  a = 1.8 × 10⁻¹⁵   ⚠️
  Layer 100: a ≈ 0              ⚠️⚠️
```

### The Core Issue

**Multiplicative compounding over many layers** amplifies small deviations from identity:
- Slightly > 1 → Explosion
- Slightly < 1 → Vanishing
- Need W ≈ I for stable gradients through many layers

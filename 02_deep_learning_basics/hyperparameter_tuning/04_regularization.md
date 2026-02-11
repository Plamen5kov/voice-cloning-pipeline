# Regularization

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 6:17 / 9:42

## Introduction

If you suspect your neural network is overfitting your data (high variance problem), one of the first things you should try is **regularization**.

### Two Ways to Address High Variance

1. **Regularization** - Often helps prevent overfitting
2. **Get more training data** - Quite reliable but not always feasible or can be expensive

Adding regularization will often help to prevent overfitting or reduce variance in your network.

## L2 Regularization for Logistic Regression

### Standard Cost Function

For logistic regression, you minimize the cost function J:

```
J(w, b) = (1/m) ∑ L(ŷ(i), y(i))
```

Where:
- **w** is an nx-dimensional parameter vector
- **b** is a real number (scalar)
- **L** is the loss on individual predictions

### Adding L2 Regularization

To add regularization to logistic regression, you add:

```
J(w, b) = (1/m) ∑ L(ŷ(i), y(i)) + (λ/2m) ||w||²
```

Where:
- **λ (lambda)** = regularization parameter
- **||w||²** = L2 norm of w squared

### The L2 Norm

The norm of w squared is defined as:

```
||w||² = ∑(j=1 to nx) wj² = wᵀw
```

This is the square Euclidean norm (L2 norm) of the parameter vector w.

### Why Only Regularize w, Not b?

**Question:** Why regularize just w? Why not add something about b as well?

**Answer:**
- In practice, you *could* include b, but it's usually omitted
- **w** is typically a high-dimensional parameter vector (lots of parameters)
- **b** is just a single number
- Almost all parameters are in w rather than b
- Adding the b term won't make much difference in practice
- You can include it if you want, but it's not standard practice

## L1 Regularization (Alternative)

### L1 vs L2 Regularization

**L1 Regularization:**
```
J(w, b) = (1/m) ∑ L(ŷ(i), y(i)) + (λ/m) ∑|wj|
```

This is called the **L1 norm** of the parameter vector w.

### Effect of L1 Regularization

- **w will end up being sparse** (lots of zeros in the w vector)
- Some say this helps with **compressing the model** (parameters with zeros need less memory)
- In practice, L1 regularization helps only a little bit for compression
- **Not used that much** compared to L2

### L2 vs L1 in Practice

**L2 regularization is used much, much more often** when training neural networks.

## The Regularization Parameter (λ)

### What is Lambda?

- **λ (lambda)** is called the regularization parameter
- Another hyperparameter you need to tune

### How to Set Lambda

Usually set using:
- Development set
- Hold-out cross validation

**Process:**
1. Try a variety of values
2. See what does best in terms of:
   - Doing well on training set
   - Setting the norm of parameters to be small (prevents overfitting)

### Programming Note

⚠️ **Important:** In Python, `lambda` is a reserved keyword!

In programming exercises:
- Use `lambd` (without the 'a')
- This avoids clashing with Python's reserved keyword

## L2 Regularization for Neural Networks

### Neural Network Cost Function

For a neural network with L layers, the cost function is:

```
J(w[1], b[1], ..., w[L], b[L]) = (1/m) ∑(i=1 to m) L(ŷ(i), y(i))
```

Where:
- **L** (capital L) = number of layers
- Parameters: w[1], b[1] through w[L], b[L]

### Adding Regularization

To add regularization, you add the Frobenius norm:

```
J = (1/m) ∑ L(ŷ, y) + (λ/2m) ∑(l=1 to L) ||w[l]||²F
```

### The Frobenius Norm

The **Frobenius norm** of a matrix is defined as:

```
||w[l]||²F = ∑(i=1 to n[l]) ∑(j=1 to n[l-1]) (w[l]ij)²
```

Where:
- **w[l]** is an n[l] × n[l-1] dimensional matrix
- **n[l]** = number of units in layer l
- **n[l-1]** = number of units in layer l-1
- Sum of all elements of the matrix, squared

### Why "Frobenius Norm"?

**Why not just call it "L2 norm of a matrix"?**

For arcane linear algebra technical reasons, by convention, this is called the **Frobenius norm** (denoted with subscript F).

It simply means: **Sum of squares of all elements of a matrix**

## Implementing Gradient Descent with Regularization

### Without Regularization (Before)

```
dw[l] = (from backprop)
w[l] = w[l] - α × dw[l]
```

### With Regularization (After)

```
dw[l] = (from backprop) + (λ/m) × w[l]
w[l] = w[l] - α × dw[l]
```

Where:
- The new dw[l] is still the correct derivative of the cost function
- Now includes the regularization term

## Weight Decay: Alternative Name for L2 Regularization

### Why It's Called "Weight Decay"

Let's expand the update equation:

```
w[l] = w[l] - α × [dw[l] + (λ/m)w[l]]
     = w[l] - α × dw[l] - α(λ/m) × w[l]
     = w[l] - α(λ/m) × w[l] - α × dw[l]
     = w[l](1 - α(λ/m)) - α × dw[l]
```

### The Key Insight

```
w[l] = (1 - αλ/m) × w[l] - α × (backprop gradient)
        ↑
   This is slightly less than 1
```

**What's happening:**
1. You multiply the weight matrix w by `(1 - αλ/m)`
2. This number is slightly less than 1
3. So you're "decaying" the weights a little bit each step
4. Then you subtract the normal gradient from backprop

### The Formula

```
w[l] ← (1 - αλ/m) × w[l] - α × (gradient from backprop)
       └─────┬─────┘
         Decay factor
        (slightly < 1)
```

**This is why L2 regularization is also called "weight decay"** - you're multiplying weights by a number slightly less than 1, causing them to decay.

## Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Prevent overfitting / Reduce variance |
| **L2 Regularization** | Most common, adds λ/(2m) × ||w||² to cost |
| **L1 Regularization** | Less common, makes w sparse, adds λ/m × ∑\|wj\| |
| **Lambda (λ)** | Regularization parameter, tune using dev set |
| **Frobenius Norm** | For matrices: sum of squares of all elements |
| **Weight Decay** | Alternative name for L2 regularization |
| **Python Note** | Use `lambd` not `lambda` (reserved keyword) |

## Implementation Checklist

For neural networks with regularization:

1. ✓ Add (λ/2m) × ∑||w[l]||²F to cost function
2. ✓ Compute dw[l] from backprop
3. ✓ Add (λ/m) × w[l] to dw[l]
4. ✓ Update: w[l] = w[l] - α × dw[l]
5. ✓ Tune λ using dev set

## Coming Up

**Question:** Why does regularization prevent overfitting?

The next topic will provide intuition for how regularization actually prevents overfitting.

---

## Quick Reference: L2 Regularization Equations

### Logistic Regression
```
J(w, b) = (1/m) ∑ L(ŷ, y) + (λ/2m)||w||²

||w||² = ∑ wj² = wᵀw
```

### Neural Network
```
J = (1/m) ∑ L(ŷ, y) + (λ/2m) ∑(l=1 to L) ||w[l]||²F

||w[l]||²F = ∑∑ (w[l]ij)²
```

### Gradient Update
```
dw[l] = (∂J/∂w[l])backprop + (λ/m)w[l]

w[l] = w[l] - α × dw[l]
     = (1 - αλ/m)w[l] - α(∂J/∂w[l])backprop
```

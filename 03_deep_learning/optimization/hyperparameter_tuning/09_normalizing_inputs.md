# Normalizing Inputs

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 0:05 / 5:29

## Introduction

When training a neural network, one of the techniques to **speed up your training** is to normalize your inputs.

## Example: Training Set with Two Features

Let's say you have a training set with **two input features** (x is two-dimensional).

### Scatter Plot of Training Set (Before Normalization)

```
x₂
 ↑
 │    •     •
 │  •   •     •
 │    •   •        •
 │  •       •    •
 │    •   •    •
 │      •    •
 │  •     •
 │    •       •
 └────────────────────→ x₁
 
 Note: x₁ has much wider range than x₂
```

## Two Steps of Normalization

### Step 1: Zero Out the Mean

**Calculate the mean:**

```
μ = (1/m) Σᵢ x⁽ⁱ⁾
```

Where:
- **μ** is a vector (same dimensions as x)
- **m** is the number of training examples
- Sum over all training examples

**Subtract the mean:**

```
x := x - μ    (for every training example)
```

**Effect:** Move the training set until it has **zero mean**

```
BEFORE:                    AFTER:
x₂                         x₂
 ↑                          ↑
 │    •     •               │    •     •
 │  •   •     •             │  •   •     •
 │    •   •        •        │----•---•---•---- x₁
 │  •       •    •          │  •   •   •
 │    •   •    •            │    • •
 │      •    •              │  •   •
 │  •     •                 ↓
 └────────────→ x₁
 
 Data centered           Data centered
 off-center              at origin
```

### Step 2: Normalize the Variances

**Calculate the variance:**

```
σ² = (1/m) Σᵢ (x⁽ⁱ⁾)²    (element-wise squaring)
```

Where:
- **σ²** is a vector with the variances of each feature
- We've already subtracted the mean, so x⁽ⁱ⁾² gives us the variance
- Element-wise squaring: square each element of the vector

**Normalize by variance:**

```
x := x / σ²    (element-wise division)
```

**Effect:** Scale each feature so it has **variance = 1**

```
BEFORE (after mean subtraction):    AFTER:
x₂                                  x₂
 ↑                                   ↑
 │    •     •                        │  • • •
 │  •   •     •                      │ • • • •
 │----•---•---•---- x₁               │--•-•-•-- x₁
 │  •   •   •                        │ • • • •
 │    • •                            │  • • •
 │  •   •                            ↓
 ↓
 
x₁ much wider than x₂            Both x₁ and x₂ have
(different variances)            variance ≈ 1
```

## Complete Normalization Formula

### For Training Set

```python
# Step 1: Zero out mean
μ = (1/m) * Σ x⁽ⁱ⁾
x = x - μ

# Step 2: Normalize variance  
σ² = (1/m) * Σ (x⁽ⁱ⁾)²
x = x / σ²
```

### Important: Use Same μ and σ² for Test Set!

**Critical rule:** Use the **same μ and σ²** calculated from your training set to normalize your test set.

```python
# Training set
μ = calculate_mean(X_train)
σ² = calculate_variance(X_train)  # After mean subtraction
X_train = (X_train - μ) / σ²

# Test set - use SAME μ and σ² from training!
X_test = (X_test - μ) / σ²  # ← Use training set's μ and σ²
```

**Why?**
- You want your data (both training and test examples) to go through the **same transformation**
- Don't estimate μ and σ² separately on training and test sets
- Otherwise training and test sets would be normalized differently

## Why Do We Normalize Inputs?

### The Cost Function Without Normalization

Recall the cost function:

```
J(w, b) = (1/m) Σ L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
```

**If you use unnormalized input features** where features are on very different scales:
- Feature x₁ ranges from 1 to 1,000
- Feature x₂ ranges from 0 to 1

**The cost function will look like this:**

```
           w₂
            ↑
            │
        ╱───┼───╲
      ╱     │     ╲
    ╱       │       ╲
  ╱         │         ╲
╱           │           ╲
│           │           │  ← Very elongated
╲           │           ╱     "bowl shape"
  ╲         │         ╱
    ╲       │       ╱
      ╲     │     ╱
        ╲___│___╱
            │
────────────┼────────────→ w₁
            │
            
Very squished out, elongated cost function
Minimum is hard to find
```

### The Cost Function With Normalization

**If you normalize features** so they're on similar scales:

```
           w₂
            ↑
            │
         ___│___
       ╱    │    ╲
     ╱      │      ╲
    │       │       │  ← More spherical
    │       •       │     "bowl shape"
     ╲      │      ╱      (• = minimum)
       ╲____│____╱
            │
────────────┼────────────→ w₁
            │

More symmetric, easier to optimize
```

### Impact on Gradient Descent

#### Without Normalization (Elongated Cost Function)

```
w₂
 ↑         Gradient descent path:
 │           ╱│╲
 │          ╱ │ ╲
 │    •───→  │  ←───•
 │      ╲    │    ╱      Oscillates back
 │       ╲   │   ╱       and forth
 │    •───→  │  ←───•
 │      ╲    │    ╱
 │       ╲   │   ╱
 │         ⤋ │ ⤋
 │           •  ← Finally reaches minimum
 │              after many steps
 └────────────────────→ w₁
 
 Needs SMALL learning rate
 MANY oscillating steps
```

#### With Normalization (Spherical Cost Function)

```
w₂
 ↑
 │      
 │      Start •
 │            │
 │            │   Can go pretty much
 │            ↓   straight to minimum
 │            │
 │            │
 │            •  ← Minimum
 │
 └────────────────────→ w₁
 
 Can use LARGER learning rate
 FEWER, more direct steps
```

### The Intuition

**With normalized features:**
- ✓ Cost function is more **symmetric/spherical**
- ✓ Can take **larger steps** in gradient descent
- ✓ Doesn't need to **oscillate** back and forth
- ✓ Reaches minimum **faster**

**Without normalized features:**
- ✗ Cost function is **elongated**
- ✗ Must use **small learning rate** to avoid divergence
- ✗ Takes **many oscillating steps**
- ✗ **Slower** to reach minimum

## Mathematical Intuition

### Why Features on Different Scales Cause Problems

If features have very different ranges:

```
x₁ ∈ [1, 1000]     → w₁ needs to be very small (≈ 0.001)
x₂ ∈ [0, 1]        → w₂ needs to be much larger (≈ 1)
```

**Result:** Parameters w₁ and w₂ end up with very different scales

**Cost function becomes:**
- Very sensitive to changes in w₁
- Less sensitive to changes in w₂
- Creates elongated contours

### With Normalization

All features have similar scales:

```
x₁ ∈ [-1, 1]       → w₁ ≈ similar scale
x₂ ∈ [-1, 1]       → w₂ ≈ similar scale
```

**Result:** Parameters w₁ and w₂ can have similar scales

**Cost function becomes:**
- Similar sensitivity to all parameters
- More spherical contours
- Easier to optimize

## High-Dimensional Reality

**Note:** In practice, w is a **high-dimensional vector**.

The 2D plots don't convey all the intuitions correctly, but the rough intuition holds:

> **Your cost function will be more round and easier to optimize when your features are on similar scales.**

Not going from:
- ✗ Feature 1: [1, 1000]
- ✗ Feature 2: [0, 1]

But rather:
- ✓ All features: approximately [-1, 1]
- ✓ All features: similar variance

This makes your cost function J **easier and faster to optimize**.

## When is Normalization Important?

### Very Important When:

Features are on **dramatically different ranges**:

```
x₁: [1, 1000]      ← Range of ~1000
x₂: [0, 1]         ← Range of ~1
x₃: [0, 0.001]     ← Range of ~0.001

This REALLY hurts your optimization algorithm!
```

**Action:** Definitely normalize!

### Less Important When:

Features are on **fairly similar ranges**:

```
x₁: [0, 1]         ← Range of ~1
x₂: [-1, 1]        ← Range of ~2
x₃: [1, 2]         ← Range of ~1

These are fairly similar ranges
```

**Action:** Normalization less critical, but still recommended

### General Recommendation

**Just normalize anyway!**

Setting all features to:
- ✓ Zero mean (μ = 0)
- ✓ Unit variance (σ² = 1)

**Benefits:**
- Guarantees all features are on similar scales
- Usually helps learning algorithm run faster
- Pretty much **never does any harm**

**Even if not sure:** Often you'll do it anyway, because there's no downside!

## Summary

### The Two-Step Process

| Step | Operation | Formula | Effect |
|------|-----------|---------|--------|
| **1. Zero mean** | Subtract mean | x := x - μ | Centers data at origin |
| **2. Unit variance** | Divide by std dev | x := x / σ² | Scales all features to variance ≈ 1 |

### Key Rules

1. ✓ **Always use the same μ and σ²** for test set as you calculated from training set
2. ✓ **Normalize when features are on different scales** (e.g., 1-1000 vs 0-1)
3. ✓ **Consider normalizing even when scales are similar** (no harm, possible benefit)
4. ✓ **Normalization speeds up training** by making cost function easier to optimize

### Why It Works

```
Unnormalized → Elongated cost function → Small steps → Slow
    vs.
Normalized → Spherical cost function → Large steps → Fast
```

## Implementation Checklist

```python
# Training set
μ = np.mean(X_train, axis=0)
X_train_centered = X_train - μ
σ² = np.mean(X_train_centered ** 2, axis=0)
X_train_normalized = X_train_centered / σ²

# Test set (use SAME μ and σ²!)
X_test_centered = X_test - μ
X_test_normalized = X_test_centered / σ²

# Or combined:
X_train_normalized = (X_train - μ) / σ²
X_test_normalized = (X_test - μ) / σ²
```

## Real-World Examples of Features That Need Normalization

### Example 1: House Price Prediction

```
Features on VERY different scales:

x₁ = Square footage:        [500, 5000]      Range: ~4,500
x₂ = Number of bedrooms:    [1, 5]           Range: ~4
x₃ = Age of house (years):  [0, 100]         Range: ~100
x₄ = Distance to city (km): [0.1, 50]        Range: ~50
x₅ = Price per sqft:        [100, 1000]      Range: ~900

Without normalization, w₁ would need to be ~0.0001
while w₂ might be ~1.0 - huge difference!
```

### Example 2: Medical Diagnosis

```
Patient health metrics:

x₁ = Heart rate (bpm):          [40, 200]        Range: ~160
x₂ = Blood pressure (mmHg):     [80, 200]        Range: ~120
x₃ = Cholesterol (mg/dL):       [100, 400]       Range: ~300
x₄ = Age (years):               [0, 100]         Range: ~100
x₅ = Weight (kg):               [30, 150]        Range: ~120
x₆ = Height (cm):               [140, 210]       Range: ~70
x₇ = Blood glucose (mg/dL):     [70, 300]        Range: ~230

All different ranges - definitely need normalization!
```

### Example 3: E-Commerce Product Recommendation

```
Product features:

x₁ = Price ($):                 [0.99, 9999]     Range: ~10,000 ⚠️
x₂ = Number of reviews:         [0, 50000]       Range: ~50,000 ⚠️
x₃ = Star rating:               [1, 5]           Range: ~4
x₄ = Time on market (days):     [1, 3650]        Range: ~3,649
x₅ = Discount percentage:       [0, 90]          Range: ~90
x₆ = View count:                [0, 1000000]     Range: ~1,000,000 ⚠️

View count dominates without normalization!
```

### Example 4: Image Data (Computer Vision)

```
Raw pixel values vs. other features:

x₁ = Red channel:               [0, 255]         Range: 255
x₂ = Green channel:             [0, 255]         Range: 255
x₃ = Blue channel:              [0, 255]         Range: 255

After normalization: [0, 1] or [-1, 1]

If combined with other features:
x₄ = Image aspect ratio:        [0.5, 2.0]       Range: ~1.5
x₅ = File size (KB):            [10, 5000]       Range: ~4,990

Pixel values and file size on very different scales!
```

### Example 5: Financial Data

```
Stock/company metrics:

x₁ = Stock price:               [$1, $3000]      Range: ~$3,000
x₂ = Trading volume:            [100K, 100M]     Range: ~100M ⚠️
x₃ = Market cap (billions):     [0.01, 3000]     Range: ~3,000
x₄ = P/E ratio:                 [-50, 200]       Range: ~250
x₅ = Dividend yield (%):        [0, 15]          Range: ~15
x₆ = 52-week % change:          [-90, 500]       Range: ~590

Trading volume is orders of magnitude larger!
```

### Example 6: Climate/Weather Data

```
Sensor measurements:

x₁ = Temperature (°C):          [-40, 50]        Range: 90
x₂ = Pressure (hPa):            [950, 1050]      Range: 100
x₃ = Humidity (%):              [0, 100]         Range: 100
x₄ = Wind speed (km/h):         [0, 200]         Range: 200
x₅ = Rainfall (mm):             [0, 500]         Range: 500
x₆ = UV Index:                  [0, 15]          Range: 15
x₇ = Air quality (AQI):         [0, 500]         Range: 500

Pressure values are huge compared to UV index!
```

### Example 7: Social Media Analytics

```
User/post metrics:

x₁ = Follower count:            [0, 100M]        Range: ~100M ⚠️⚠️
x₂ = Like count:                [0, 10M]         Range: ~10M ⚠️
x₃ = Comment count:             [0, 500K]        Range: ~500K
x₄ = Post length (chars):       [1, 5000]        Range: ~5,000
x₅ = Engagement rate (%):       [0, 100]         Range: ~100
x₆ = Account age (days):        [1, 5000]        Range: ~5,000
x₇ = Hashtag count:             [0, 30]          Range: ~30

Follower count would completely dominate without normalization!
```

### Example 8: Autonomous Driving

```
Sensor data:

x₁ = Speed (km/h):              [0, 200]         Range: 200
x₂ = LIDAR distance (m):        [0.1, 300]       Range: ~300
x₃ = Steering angle (degrees):  [-45, 45]        Range: 90
x₄ = Throttle position (%):     [0, 100]         Range: 100
x₅ = GPS latitude:              [33.0, 48.0]     Range: 15
x₆ = GPS longitude:             [-124, -70]      Range: 54
x₇ = Accelerometer (m/s²):      [-20, 20]        Range: 40

GPS coordinates, LIDAR, and accelerometer all different!
```

### Example 9: Text/NLP Features

```
Document features:

x₁ = Word count:                [10, 100000]     Range: ~100,000 ⚠️
x₂ = Unique words:              [10, 10000]      Range: ~10,000
x₃ = Average word length:       [3, 10]          Range: ~7
x₄ = Sentence count:            [1, 5000]        Range: ~5,000
x₅ = TF-IDF score:              [0, 1]           Range: ~1
x₆ = Reading level:             [1, 20]          Range: ~19

Word count vs. TF-IDF score: 100,000 vs. 1!
```

### Example 10: Credit Scoring

```
Applicant features:

x₁ = Annual income ($):         [0, 1000000]     Range: ~1M ⚠️⚠️
x₂ = Credit score:              [300, 850]       Range: ~550
x₃ = Loan amount ($):           [1000, 500000]   Range: ~499K
x₄ = Employment length (months):[0, 480]         Range: ~480
x₅ = Number of credit cards:    [0, 20]          Range: ~20
x₆ = Debt-to-income ratio (%):  [0, 100]         Range: ~100
x₇ = Age:                       [18, 80]         Range: ~62

Income dwarfs everything else!
```

### Common Problematic Combinations

Notice these patterns that **require normalization:**

```
❌ Very Bad Combinations:

Money values     [0, 1M]        vs.  Ratings        [0, 5]
Counts/Volume    [0, 100M]      vs.  Percentages    [0, 100]
Pixel values     [0, 255]       vs.  Probabilities  [0, 1]
Distances        [0, 1000s]     vs.  Angles         [-180, 180]
Timestamps       [0, billions]  vs.  Categories     [0, 10]
```

### After Normalization

All features scaled to similar ranges:

```
Before:                          After:
Feature 1: [1, 1000000]    →    Feature 1: [-1.5, 1.5]     σ² ≈ 1
Feature 2: [0, 100]        →    Feature 2: [-1.2, 1.8]     σ² ≈ 1
Feature 3: [0, 1]          →    Feature 3: [-1.0, 1.0]     σ² ≈ 1
Feature 4: [-50, 200]      →    Feature 4: [-1.3, 1.4]     σ² ≈ 1

Now all features contribute equally to the model!
```

### The Key Principle

**Whenever you have features measured in different units or scales:**
- Dollars vs. counts
- Meters vs. percentages  
- Pixels vs. ratios
- Timestamps vs. categories

**You almost certainly need normalization!**

## What's Next

We've covered normalizing input features to speed up training. Next, let's keep talking about more ways to speed up the training of your neural network.

---

## Quick Reference

### When Features Are On Different Scales

```
BEFORE:                           AFTER:
Feature 1: [1, 1000]    →        Feature 1: [-1, 1]
Feature 2: [0, 1]       →        Feature 2: [-1, 1]
Feature 3: [-5, 5]      →        Feature 3: [-1, 1]

Cost function: Elongated         Cost function: Spherical
Training: Slow                   Training: Fast
```

### Normalization Benefits

| Aspect | Without Normalization | With Normalization |
|--------|----------------------|-------------------|
| **Cost function shape** | Elongated, asymmetric | Spherical, symmetric |
| **Learning rate** | Must be very small | Can be larger |
| **Convergence** | Many oscillating steps | More direct path |
| **Training speed** | Slow | Fast |
| **Difficulty** | Hard to tune | Easier to tune |

### The Formula Card

```
NORMALIZATION FORMULA:

Training:
  μ = (1/m) Σ x⁽ⁱ⁾
  σ² = (1/m) Σ (x⁽ⁱ⁾ - μ)²
  x_norm = (x - μ) / σ²

Test:
  x_test_norm = (x_test - μ) / σ²
  
  ⚠️ Use μ and σ² from TRAINING set!
```

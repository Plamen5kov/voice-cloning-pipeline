# Bias / Variance

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 8:17 / 8:46

## Introduction

Almost all really good machine learning practitioners tend to have a very sophisticated understanding of bias and variance. Bias and variance is one of those concepts that's **easy to learn but difficult to master**. Even if you think you've seen the basic concepts of bias and variance, it's often more nuanced than you'd expect.

### The Deep Learning Era Shift

In the deep learning era, there's been less discussion of what's called the **bias-variance trade-off**. You might have heard about this trade-off, but in the deep learning era, there's less of a trade-off. We still talk about bias and variance, but we just talk less about the bias-variance trade-off.

## Understanding Bias and Variance Visually

### Example: 2D Data Classification

Consider a dataset that you're trying to fit:

#### 1. High Bias (Underfitting)
- **Example:** Fitting a straight line to the data with logistic regression
- Not a very good fit to the data
- **Underfitting** the data
- Too simple to capture the pattern

#### 2. High Variance (Overfitting)
- **Example:** An incredibly complex classifier (e.g., deep neural network with many hidden units)
- Fits the data perfectly but doesn't generalize well
- **Overfitting** the data
- Too complex, captures noise

#### 3. Just Right
- Medium level of complexity
- Fits a curve that looks like a reasonable fit to the data
- Balanced between underfitting and overfitting

### The Challenge in High Dimensions

In a 2D example with just two features (x₁ and x₂), you can plot the data and visualize bias and variance. However, **in high-dimensional problems, you can't plot the data and visualize the decision boundary**.

Instead, we use a couple of different metrics to understand bias and variance.

## Diagnosing Bias and Variance

### Key Metrics

The two key numbers to look at:
1. **Training set error**
2. **Dev set (development set) error**

### Important Assumption

For cat picture classification (or similar tasks), let's assume:
- **Human-level performance ≈ 0% error** (people can do this nearly perfectly)
- **Bayes error (optimal error) ≈ 0%**
- Train and dev sets are drawn from the same distribution

## Diagnosis Examples

### Example 1: High Variance

| Metric | Error |
|--------|-------|
| Training set error | 1% |
| Dev set error | 11% |

**Diagnosis: High Variance**
- Doing very well on the training set
- Doing relatively poorly on the development set
- Overfitting the training set
- Not generalizing well to the holdout cross-validation set

### Example 2: High Bias

| Metric | Error |
|--------|-------|
| Training set error | 15% |
| Dev set error | 16% |

**Diagnosis: High Bias**
- Not doing well even on the training set
- Underfitting the data
- Performance on dev set is only 1% worse than training set
- Generalizing at a reasonable level, but baseline is poor

### Example 3: High Bias AND High Variance

| Metric | Error |
|--------|-------|
| Training set error | 15% |
| Dev set error | 30% |

**Diagnosis: High Bias + High Variance (Worst of Both Worlds)**
- **High bias:** Not doing well on training set
- **High variance:** Does even worse on dev set
- Not fitting training data well AND not generalizing well

### Example 4: Low Bias AND Low Variance (Ideal)

| Metric | Error |
|--------|-------|
| Training set error | 0.5% |
| Dev set error | 1% |

**Diagnosis: Low Bias + Low Variance**
- Excellent performance on training set
- Excellent generalization to dev set
- Users would be happy with only 1% error

## Important Subtlety: Bayes Error

The analysis above is predicated on the assumption that **human-level performance (or Bayes error) is nearly 0%**.

### When Bayes Error is Higher

If the optimal error (Bayes error) is much higher, say **15%**, then:
- A classifier with 15% training error would be perfectly reasonable
- You wouldn't say it has high bias
- It would have pretty low variance

**Example scenario:** Really blurry images where even humans can't classify well
- Bayes error might be much higher
- The analysis changes accordingly
- More sophisticated analysis needed (covered in later videos)

## The Diagnostic Process

### Step 1: Check Training Set Error
- **Tells you:** How well you're fitting the training data
- **Diagnoses:** Whether you have a **bias problem**

### Step 2: Check Dev Set Error Gap
- **Tells you:** How much higher error goes from training to dev set
- **Diagnoses:** Whether you have a **variance problem**
- **Measures:** How well you're generalizing from training set to dev set

## Visualizing High Bias + High Variance

### The Worst of Both Worlds

What does a classifier with both high bias and high variance look like?

**Characteristics:**
- **Mostly linear** (underfits the data) → High Bias
- **Weird flexibility in some regions** (overfits certain points) → High Variance

**Example:** 
- A mostly linear classifier that doesn't fit a quadratic-shaped decision boundary well (high bias)
- But has too much flexibility in the middle, overfitting individual outlier examples (high variance)

### Is This Realistic?

While this seems contrived in 2D:
- **In high-dimensional inputs**, you actually do get classifiers with:
  - High bias in some regions
  - High variance in other regions
- This is less contrived and more common in real applications

## Summary

### Diagnostic Approach

By examining your algorithm's errors, you can diagnose:

| Condition | Training Error | Dev Error | Diagnosis |
|-----------|---------------|-----------|-----------|
| Low training error, high dev error | Low (e.g., 1%) | High (e.g., 11%) | **High Variance** |
| High training error, moderate dev error | High (e.g., 15%) | Similar (e.g., 16%) | **High Bias** |
| High training error, very high dev error | High (e.g., 15%) | Much higher (e.g., 30%) | **High Bias + High Variance** |
| Low training error, low dev error | Very low (e.g., 0.5%) | Low (e.g., 1%) | **Low Bias + Low Variance** ✓ |

### What's Next?

Depending on whether your algorithm suffers from:
- **Bias**
- **Variance**
- **Both**
- **Neither**

There are different things you can try to improve it.

The next step is to learn a **basic recipe for machine learning** that lets you more systematically try to improve your algorithm depending on whether it has high bias or high variance issues.

---

## Key Takeaways

1. **Bias and variance are sophisticated concepts** that require deep understanding
2. **In the deep learning era**, there's less of a bias-variance trade-off
3. **Use training and dev set errors** to diagnose problems
4. **High bias** = underfitting (poor performance on training set)
5. **High variance** = overfitting (large gap between training and dev performance)
6. **You can have both** high bias and high variance simultaneously
7. **Analysis assumes** Bayes error ≈ 0% and same distribution for train/dev sets

# Basic Recipe for Machine Learning

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 1:08 / 6:21

## Introduction

In the previous video, you saw how looking at training error and dev set error can help you diagnose whether your algorithm has a bias or variance problem, or maybe both. This information lets you use a **basic recipe for machine learning** to much more systematically improve your algorithm's performance.

## The Basic Recipe: A Systematic Approach

When training a neural network, here's the basic recipe to follow:

### Step 1: Does Your Algorithm Have High Bias?

**How to evaluate:** Look at **training set/training data performance**

If it has **high bias** (not fitting the training set well), try:

1. **Pick a bigger network**
   - More hidden layers
   - More hidden units
   - ✓ Almost always helps

2. **Train longer**
   - Certainly never hurts
   - Doesn't always help, but worth trying

3. **Try more advanced optimization algorithms**
   - Will be covered later in the course

4. **Try a different neural network architecture** *(maybe it works, maybe it won't)*
   - Different architectures may be better suited for specific problems
   - Less systematic - requires experimentation
   - Parenthetical because it's not guaranteed to work

**Keep trying these until you can fit the training set pretty well.**

#### Important Note on Bias Reduction

Usually, if you have a **big enough network**, you should be able to fit the training data well, so long as:
- It's a problem that is possible for someone to do
- If a human can do well on the task
- Bayes error is not too high

*Exception:* If the image is very blurry or task is inherently impossible, it may be impossible to fit well.

### Step 2: Does Your Algorithm Have High Variance?

**How to evaluate:** Look at **dev set performance**

Ask: Are you able to generalize from pretty good training set performance to pretty good dev set performance?

If you have **high variance**, try:

1. **Get more data** *(if you can get it)*
   - ✓ Best way to solve high variance
   - Can only help
   - Sometimes you can't get more data

2. **Regularization**
   - Try to reduce overfitting
   - Will be covered in the next video

3. **Try a different neural network architecture** *(sometimes)*
   - Can reduce variance problem
   - Can also reduce bias problem
   - Less systematic - harder to be totally systematic

**Keep trying these until you achieve both low bias and low variance.**

Once you have low bias AND low variance → **You're done!** ✓

## Visual Flow Chart

```
╔═══════════════════════════════════════════════════════════════╗
║                  TRAIN INITIAL MODEL                          ║
╚═══════════════════════════════════════════════════════════════╝
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │  HIGH BIAS?                               │
        │  (Check Training Set Performance)         │
        └───────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │ YES                   │ NO
                ▼                       │
    ┌─────────────────────────┐         │
    │  Fix High Bias:         │         │
    │  • Bigger network       │         │
    │  • More hidden layers   │         │
    │  • Train longer         │         │
    │  • Try new architecture │         │
    └─────────────────────────┘         │
                │                       │
                └───────┐               │
                        │               │
                ┌───────┘               │
                │                       │
                └───────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │  HIGH VARIANCE?                           │
        │  (Check Dev Set Performance)              │
        └───────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │ YES                   │ NO
                ▼                       │
    ┌─────────────────────────┐         │
    │  Fix High Variance:     │         │
    │  • Get more data        │         │
    │  • Regularization       │         │
    │  • Try new architecture │         │
    └─────────────────────────┘         │
                │                       │
                └───────┐               │
                        │               │
                ┌───────┘               │
                │                       │
                └───────────────────────┘
                            │
                            ▼
╔═══════════════════════════════════════════════════════════════╗
║               ✓ DONE! (Low Bias + Low Variance)               ║
╚═══════════════════════════════════════════════════════════════╝
```

## Key Points to Notice

### 1. Different Problems Require Different Solutions

Depending on whether you have **high bias** or **high variance**, the set of things you should try could be quite different.

- Use the **training and dev set** to diagnose bias or variance
- Use that diagnosis to select the appropriate subset of things to try

**Example:** If you have a high bias problem, **getting more training data is actually not going to help** (or at least it's not the most efficient thing to do).

Being clear on your specific problem helps you focus on the most useful solutions.

### 2. The Bias-Variance Trade-off: Then vs. Now

#### Earlier Era of Machine Learning

- Lots of discussion about the **bias-variance trade-off**
- Most techniques would:
  - Increase bias and reduce variance, OR
  - Reduce bias and increase variance
- Didn't have many tools that just reduce bias OR just reduce variance without hurting the other

#### Modern Deep Learning / Big Data Era

We now have tools that can reduce one without significantly hurting the other:

| Action | Effect on Bias | Effect on Variance | Condition |
|--------|----------------|-------------------|-----------|
| **Bigger network** | ↓ Reduces | → Doesn't hurt | So long as you regularize appropriately |
| **More data** | → Doesn't hurt much | ↓ Reduces | Pretty much always |

**This is a huge advantage!** You can:
- Drive down bias without necessarily hurting variance
- Drive down variance without necessarily hurting bias

### 3. Why Deep Learning is So Useful for Supervised Learning

One of the big reasons deep learning has been so successful:
- **Much less of a tradeoff** between bias and variance
- Don't have to carefully balance bias and variance
- More options for reducing either one independently

## Important Considerations

### Training a Bigger Network

**Almost never hurts**, with one caveat:

- ✓ The main cost is **computational time**
- Must use **proper regularization**
- So long as you're regularizing, a bigger network is generally better

### When to Use What

| Problem | What NOT to Do | What to Do |
|---------|---------------|------------|
| High Bias | Get more data | Bigger network, train longer |
| High Variance | Make network bigger | Get more data, regularization |

## Regularization: A Preview

Regularization has been mentioned several times in this discussion:

- **Very useful technique for reducing variance**
- There is a **slight bias-variance trade-off** when using regularization
  - Might increase bias a little bit
  - Often not too much if you have a huge enough network
- Will be covered in detail in the next video

## Summary

### The Systematic Approach

1. **Train initial model**
2. **Diagnose the problem:**
   - High bias? → Check training set performance
   - High variance? → Check dev set performance
3. **Apply appropriate solution:**
   - For bias: Bigger network, train longer, different architecture
   - For variance: More data, regularization, different architecture
4. **Iterate until both bias and variance are acceptable**

### Modern Advantages

In the deep learning era:
- Less need to trade off bias vs. variance
- Can independently address each problem
- Bigger networks + more data + regularization = powerful combination
- Main limitation: computational resources and data availability

---

## Quick Reference Guide

### Diagnosis Table

| Training Error | Dev Error | Problem | Solution |
|---------------|-----------|---------|----------|
| High | Higher | High Bias + High Variance | Bigger network + More data |
| High | Similar | High Bias | Bigger network, train longer |
| Low | High | High Variance | More data, regularization |
| Low | Low | Good! | Deploy and celebrate 🎉 |

### Action Priority

**If High Bias:**
1. ↑ Increase network size first
2. ↑ Train longer
3. Try different architecture

**If High Variance:**
1. ↑ Get more data (if possible)
2. ↑ Apply regularization
3. Try different architecture

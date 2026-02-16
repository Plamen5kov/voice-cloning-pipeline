# Train/Dev/Test Strategy

## Core Concept
Properly structuring your datasets to ensure reliable evaluation and successful deployment.

## The Three Sets

### Training Set
**Purpose:** Learn parameters (weights, biases)  
**Usage:** Fit the model

### Dev Set (Validation Set/Hold-out Set)
**Purpose:** Tune hyperparameters and select best model  
**Usage:** Decide which model/architecture to use  
**Key:** Never train on this!

### Test Set
**Purpose:** Unbiased estimate of final performance  
**Usage:** Report final metrics after all decisions are made  
**Key:** Look at this only once at the very end!

## Critical Rule: Distributions Must Match

### ✅ Correct
```
Training:   Images scraped from web (different distribution OK)
Dev:        Images from your mobile app  }
Test:       Images from your mobile app  } ← SAME distribution
Production: Images from your mobile app  }
```

### ❌ Wrong
```
Dev:  Images from US users
Test: Images from Indian users  ← Different distribution!
```

**Why it matters:**  
If dev/test distributions differ, you optimize for the wrong thing!

## Set Sizes

### Traditional ML Era (< 10,000 examples)
```
70% Train / 30% Test
or
60% Train / 20% Dev / 10% Test
```

### Modern ML Era (> 1,000,000 examples)
```
98% Train / 1% Dev / 1% Test
or
99% Train / 0.5% Dev / 0.5% Test
```

**Why smaller dev/test?**
- 10,000 examples is enough to evaluate model performance
- If you have 1M examples, 1% = 10K examples for dev
- More data for training is better!

### Rules of Thumb

**Dev set size:**
- Large enough to detect meaningful differences between models
- Typical: 1,000 - 10,000 examples
- More if many hyperparameters to tune

**Test set size:**
- Large enough to give confidence in system performance
- Typical: 1,000 - 10,000 examples  
- More for safety-critical applications

**No test set?**
- OK for personal projects or rapid prototyping
- NOT OK for production systems
- You risk overfitting to your dev set

## When to Change Dev/Test Sets

### Scenario 1: Distribution Mismatch

**Problem:**
```
Dev/Test: High-quality professional photos
Production: Low-quality user smartphone photos
```

**Solution:** Change dev/test to match production!

### Scenario 2: Metric Doesn't Reflect Goals

**Example: Cat Classifier**

Initial metric:
```
Error = (1/m) * Σ(y_pred ≠ y_true)
```

Problem:
```
Algorithm A: 3% error (but shows pornographic images occasionally)
Algorithm B: 5% error (but never shows inappropriate content)
```

Metric says A is better, but B is actually better for users!

**Solution:** Change metric to penalize inappropriate content:
```
Error = (1/m) * Σ[ w_i * (y_pred ≠ y_true) ]

where w_i = 1 for normal examples
      w_i = 10 for inappropriate examples (or simply exclude them)
```

## The Optimization Process

### Two Step Process

**Step 1: Define metric and dev/test set**
This is your "target" 🎯

**Step 2: Aim for the target**
Train models and optimize the metric

### When Target is Wrong

If you realize your metric or dev/test set doesn't reflect real-world performance:

**Stop, change the target, then continue**

Don't keep optimizing for the wrong thing!

## Distribution Strategy for Specific Scenarios

### Multi-Region Application
```
Option 1: Mix all regions
  Dev/Test: 50% US + 30% Europe + 20% Asia

Option 2: Worst-case focus
  Dev/Test: Only from worst-performing region
```

### Data from Multiple Sources
```
Training: Can be from different sources
Dev/Test: Must match production distribution

Example (Speech Recognition):
  Training: Purchased speech + synthetic + web-scraped
  Dev/Test: Only real recordings from your app
```

### Imbalanced Classes
```
Don't balance dev/test artificially!
Match real-world distribution:
  If fraud is 0.1% in production → 0.1% in dev/test
```

## Common Questions

### Q: Can I use cross-validation instead of dev set?
**A:** Yes for small datasets, but dev set is more efficient for large datasets

### Q: What if I don't have enough data for separate test set?
**A:** Use only train/dev split, but be aware you may overfit to dev set over time

### Q: Should I ever retrain on dev + test?
**A:** Only after final evaluation, and only if deploying to same distribution

### Q: How often should distributions be updated?
**A:** Whenever production data distribution changes significantly

## Checklist

Before starting your ML project:

- [ ] Dev and test sets come from same distribution
- [ ] Distribution matches production/real-world usage
- [ ] Dev set large enough to distinguish models (typically 1K-10K examples)
- [ ] Test set large enough for confidence (typically 1K-10K examples)
- [ ] Metric aligns with business goals
- [ ] Have plan for updating sets if distribution shifts

## Example: Voice Cloning Application

```
Goal: Clone voices for audiobook narration

Training:
  - LibriSpeech (diverse speakers, clean audio)
  - Common Voice (crowd-sourced, various quality)
  - Your custom recordings

Dev/Test:
  - Only audiobook-style narration
  - Same microphone setup as production
  - Same background noise level as target environment
  - Representative speaker demographics

Metric:
  Optimizing: MOS (Mean Opinion Score) for naturalness
  Satisficing:
    - WER (Word Error Rate) < 5%
    - Inference time < 2x real-time
    - Voice similarity score > 0.8
```

## Key Takeaways

1. **Dev and test must have same distribution (that matches production)**
2. **Set sizes: Just enough to evaluate, modern datasets use much smaller %**
3. **Change sets/metrics when they don't reflect real goals**
4. **Define the target first, then aim for it**
5. **Production distribution > training distribution for dev/test**

---

**Related:** [Evaluation Metrics](evaluation_metrics.md), [Human-Level Performance](human_level_performance.md)

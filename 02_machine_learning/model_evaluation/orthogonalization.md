# Orthogonalization

## Core Concept
Clear, systematic approach to tuning ML systems where each "knob" affects only one aspect of performance.

## The Old TV Analogy

**Old TVs had separate knobs for:**
- Image position (left/right)
- Image position (up/down)
- Image width
- Image height
- Trapezoid distortion
- etc.

**Each knob controlled ONE thing**

**Good design:**  
Turn "width" knob → only width changes ✓

**Bad design:**  
Turn "width" knob → width AND height AND position all change ✗

## Orthogonalization in ML

### The Goal

**Design your tuning process so each change affects ONE specific problem**

This makes debugging and improvement systematic rather than random.

## The Chain of ML Performance

### Four Things to Get Right

```
1. Fit training set well
   ↓
2. Fit dev set well
   ↓
3. Fit test set well
   ↓
4. Perform well in real world
```

**Each stage has specific "knobs" (strategies) to turn.**

## The Orthogonal Knobs

### 1. Training Set Performance (Reduce Bias)

**Goal:** Training error ≈ Bayes error

**Knobs to turn:**
- ✅ Train bigger network
- ✅ Train longer / better optimization (Adam, learning rate schedule)
- ✅ Use better architecture (ResNet vs. plain net)
- ✅ Add more features

**Don't Use:**
- ❌ Regularization (makes training worse!)
- ❌ Get more data (won't help if model can't fit current data)

### 2. Dev Set Performance (Reduce Variance)

**Goal:** Dev error ≈ Training error

**Knobs to turn:**
- ✅ Regularization (L2, dropout, data augmentation)
- ✅ Get more training data
- ✅ Try simpler architecture

**Already Fixed:**
- ✓ Training set performance is good (if not, go back to step 1)

### 3. Test Set Performance

**Goal:** Test error ≈ Dev error

**If test >> dev:**
- ✅ Get bigger dev set

**Why this helps:**
- You may have overfit to small dev set
- Bigger dev set gives more reliable estimate

### 4. Real World Performance

**Goal:** Real world performance ≈ Test performance

**If real world >> test:**
- ✅ Change dev/test set distribution to match real world
- ✅ Change cost function to better reflect real priorities

**Example:**
- Test set: High-quality images
- Real world: Smartphone photos
- **Solution:** Collect smartphone photos for dev/test!

## The Anti-Pattern: Early Stopping

**Why early stopping is non-orthogonal:**

Early stopping uses the same knob to control TWO things:
1. Training set performance (stops before overfitting training)
2. Dev set performance (prevents overfitting to specific training set)

**Problem:**  
You can't separately tune:
- "Fit training set better"
- "Generalize to dev set better"

**Better approach:**
1. Train as long as needed to fit training set
2. Then use regularization to improve dev performance

This keeps the knobs orthogonal!

> Note: People still use early stopping because it's simple and works. But it's theoretically less clean.

## Orthogonalization in Practice

### Example: Image Classification

```
Current State:
  Training error: 8%
  Dev error: 12%
  Test error: 12%
  Real world: 15%
```

**Systematic Debugging:**

**Step 1: Training error too high?**
```
8% vs. Bayes (~5%) → Yes, bias problem!

Orthogonal action:
  → Train bigger model ✓
  → NOT: add regularization ✗ (wrong knob!)
  
Result: Training error → 6%
```

**Step 2: Dev error too high?**
```
12% vs. 6% training → Yes, variance problem!

Orthogonal action:
  → Add regularization ✓
  → Get more data ✓
  → NOT: bigger model ✗ (wrong knob!)
  
Result: Dev error → 8%
```

**Step 3: Test error too high?**
```
12% test vs. 8% dev → Yes, overfit to dev!

Orthogonal action:
  → Get bigger dev set ✓
  
Result: Test error → 8% (was just estimation error)
```

**Step 4: Real world worse than test?**
```
15% real vs. 8% test → Yes, distribution mismatch!

Orthogonal action:
  → Update dev/test to match real distribution ✓
  → Re-evaluate and repeat steps 1-3
```

## Benefits of Orthogonalization

### 1. **Systematic Debugging**
Know exactly which knob to turn for each problem

### 2. **Faster Iteration**
Don't waste time on wrong solutions

### 3. **Clearer Communication**
Team knows what each experiment is trying to fix

### 4. **Predictable Results**
Changing one thing doesn't break something else

## Common Pitfalls

### ❌ Non-Orthogonal Approaches

**Problem:** Using same technique for multiple goals
- Early stopping (affects both bias and variance)
- Changing architecture (can affect both underfitting and overfitting)

**Problem:** Conflicting changes
- Adding dropout AND bigger network simultaneously
- Changing cost function AND changing dev set at same time

### ✅ Good Practices

**Do one thing at a time:**
1. Fix training performance first
2. Then fix dev performance
3. Then fix test/real-world alignment

**Use clear cause-effect:**
- This knob → this problem
- That knob → that problem

## Orthogonalization Checklist

Before making a change:

- [ ] Which specific problem am I trying to fix?
  - Training error too high?
  - Gap between training and dev?
  - Gap between dev and test?
  - Gap between test and real world?

- [ ] Is this the right knob for that problem?
  - Or will it also affect other stages?

- [ ] Am I changing only one thing?
  - If changing multiple things, I won't know what worked!

- [ ] Have I fixed earlier stages first?
  - Don't worry about variance if you have bias issues!

## Applied Example: Voice Cloning

```
Problem: Voice cloning sounds unnatural

Measurements:
  Training MOS: 4.2 (humans: 4.5)
  Dev MOS: 3.8
  Test MOS: 3.7
  Real-world MOS: 3.2
```

**Orthogonal diagnosis:**

**Stage 1: Training (bias)**
```
4.2 vs 4.5 → Small gap, but could improve

Actions:
  → Try bigger model architecture ✓
  → Train for more epochs ✓
  Result: Training MOS → 4.4
```

**Stage 2: Dev (variance)**
```
3.8 vs 4.4 training → 0.6 gap, variance issue!

Actions:
  → Get more training voice samples ✓
  → Data augmentation (pitch shift, speed change) ✓
  Result: Dev MOS → 4.1
```

**Stage 3: Real-world alignment**
```
3.2 real vs. 3.7 test → Distribution mismatch!

Analysis:
  - Test set: Studio recordings
  - Real world: Phone call quality
  
Actions:
  → Collect dev/test with phone call quality ✓
  → Update metric to weight phone quality higher ✓
  Re-run evaluation pipeline
```

## Key Principles

1. **One knob, one problem** - Each tuning action targets specific stage
2. **Sequential improvement** - Fix training, then dev, then test, then deployment
3. **Avoid multi-purpose knobs** - They make debugging harder
4. **Measure at each stage** - Know which stage has the problem

## Orthogonalization + Human-Level Performance

**Combined power:**

```
1. Use human-level perf to identify bias vs. variance
2. Use orthogonalization to know which knob to turn
3. Systematically improve until satisfactory performance
```

**Workflow:**
```
Measure → Identify stage → Select orthogonal knob → Apply →Repeat
```

This makes ML development engineering rather than alchemy!

---

**Related:** [Human-Level Performance](human_level_performance.md), [Error Analysis](error_analysis.md), [Train/Dev/Test Strategy](train_dev_test_strategy.md)

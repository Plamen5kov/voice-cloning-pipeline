# Human-Level Performance

## Core Concept
Using human-level performance as a proxy for Bayes optimal error to guide ML development priorities.

## Why Human-Level Performance Matters

### Progress Curve of ML Systems

```
Performance
    │     ┌─────────── Bayes Optimal Error (theoretical best)
    │    ╱│
    100% ─────────────────────────
    │   ╱ │
    │  ╱  │  ← Rapid progress when below human-level
    │ ╱   │
    │╱____│  ← Slower progress near/above human-level
    ├─────┴──────────────────────────> Time
          ↑
    Human-level
```

### Two Reasons Human-Level is Special

1. **Proxy for Bayes Error**
   - Bayes optimal error = theoretical best possible
   - For tasks humans can do well, human-level ≈ Bayes optimal
   - Helps identify if problem is bias or variance

2. **Access to Human Tools**
   - Get labeled data from humans
   - Manual error analysis (humans can explain mistakes)
   - Better analysis of why algorithm fails

## Bayes Optimal Error

**Definition:** The best possible error rate for any classifier (perfect knowledge of underlying distribution)

**In practice:** Unknown, but can estimate using:
- Human-level performance (for perceptual tasks)
- Theoretical limits (for well-defined tasks)

## Avoidable Bias

### The Key Insight

Not all bias is bad! Some bias is unavoidable.

```
Scenario A:
  Human error:     1%  ← Approximate Bayes error
  Training error:  5%  
  Dev error:       6%
  
  Avoidable bias: 4% (5% - 1%)
  Variance:       1% (6% - 5%)
  → Focus on reducing BIAS!

Scenario B:
  Human error:     7.5%  ← Approximate Bayes error
  Training error:  8%
  Dev error:       10%
  
  Avoidable bias: 0.5% (8% - 7.5%)
  Variance:       2% (10% - 8%)
  → Focus on reducing VARIANCE!
```

### Avoidable vs. Unavoidable Bias

**Formula:**
```
Avoidable bias = Training error - Bayes error (approximated by human-level)
Variance = Dev error - Training error
```

**Unavoidable bias:** Gap between Bayes error and 0%  
(You can't do better than Bayes optimal!)

## Defining Human-Level Performance

### Multiple Levels of "Human"

For medical image diagnosis:
- Typical radiologist: 3% error
- Experienced radiologist: 2% error
- Team of experienced radiologists: 0.5% error

**Which to use as "human-level"?**

**Answer:** Depends on purpose!

### For Estimating Bayes Error
Use **best possible human performance**:
- Team of experienced radiologists: 0.5%
- This is closest to Bayes optimal error

### For Deployment Goals
Use **target human benchmark**:
- Match typical radiologist: 3% error (easier goal)
- Surpass experienced radiologist: 2% error (harder goal)
- Surpass team: 0.5% error (very hard!)

## Understanding the Gaps

```
                                    Performance
0%  ──────────────────────────────────────────────
    │                                           ↑
    ├─ Bayes Error (~HumanBest)                │ Unavoidable bias
    │                                           ↓
X%  ──────────────────────────────────────────────
    │                                           ↑
    ├─ Training Error                           │ Avoidable bias
    │                                           ↓
Y%  ──────────────────────────────────────────────
    │                                           ↑
    ├─ Dev Error                                │ Variance
    │                                           ↓
Z%  ──────────────────────────────────────────────
```

### Decision Tree

```
If (Training error - Bayes error) is large:
  → High avoidable bias
  → Fix: Bigger network, train longer, better architecture
  
If (Dev error - Training error) is large:
  → High variance
  → Fix: More data, regularization, simpler model
```

## Surpassing Human-Level Performance

### When ML Can Beat Humans

**Tasks where ML excels:**
- Structured data (not perceptual)
- Lots of training data available
- Pattern is learnable but too complex for humans

**Examples:**
- Online advertising
- Product recommendations
- Loan approvals
- Logistics routing

**Less common (but possible):**
- Specific perceptual tasks with lots of data
  - Speech recognition (with huge datasets)
  - Some medical imaging (specific, narrow tasks)

### Why It's Harder After Human-Level

1. **Harder to get labeled data** humans might disagree or be wrong)
2. **Error analysis less informative (humans can't explain optimal decisions)
3. **Bayes error estimate becomes unclear** (which causes bias/variance confusion)

### Beyond Human-Level

Once you surpass human-level:
- Still use train/dev error gap for variance
- But can't easily estimate avoidable bias
- Harder to know if you're near Bayes optimal

## Improving Your Model Performance

###  The Two Fundamental Assumptions

Supervised learning works when:

1. **Fit training set well** (low avoidable bias)
   - Training error ≈ Bayes error
   
2. **Training set performance generalizes** (low variance)
   - Dev error ≈ Training error

### Toolkit for Each Problem

**High Avoidable Bias? (Training error >> Bayes)**
- Train bigger model
- Train longer / better optimization
- Better architecture / hyperparameters
- More features

**High Variance? (Dev error >> Training error)**
- More training data
- Regularization (L2, dropout, data augmentation)
- Better architecture / hyperparameters

## Practical Workflow

### Step 1: Establish Human-Level Performance
```
Task: Image classification
  - Typical human: 5% error
  - Domain expert: 2% error
  - Team of experts: 1% error
  
Use 1% as Bayes error estimate
```

### Step 2: Measure Gaps
```
Training error: 8%
Dev error: 10%

Avoidable bias: 8% - 1% = 7%  ← LARGE!
Variance: 10% - 8% = 2%

→ Priority: Reduce bias first
```

### Step 3: Take Action
```
Since bias is the problem:
  ✓ Use bigger model
  ✓ Train longer
  ✓ Try different architecture
  
  ✗ Don't add more data yet (variance is small)
  ✗ Don't add regularization (makes bias worse!)
```

### Step 4: Iterate
```
After improvements:
  Training error: 3%
  Dev error: 6%
  
  Avoidable bias: 3% - 1% = 2%
  Variance: 6% - 3% = 3%  ← Now larger!
  
→ Now focus on variance (more data, regularization)
```

## Real-World Example: Speech Recognition

```
Human-level (professional transcriber): 5.9%

Model V1:
  Training: 10.6%
  Dev: 14.8%
  
  Analysis:
    Avoidable bias: 4.7%  ← Focus here first
    Variance: 4.2%

  Actions:
    - Bigger RNN model ✓
    - Train for more epochs ✓
    Result: Training drops to 7.3%

Model V2:
  Training: 7.3%
  Dev: 11.5%
  
  Analysis:
    Avoidable bias: 1.4%
    Variance: 4.2%  ← Now the bottleneck!
    
  Actions:
    - Add more training data ✓
    - Data augmentation (noise, speed) ✓
    Result: Dev drops to 8.1%

Model V3:
  Training: 6.2%
  Dev: 8.1%
  
  Surpassed human-level! 🎉
  (But now harder to know how much more improvement is possible)
```

## Key Takeaways

1. **Human-level performance ≈ Bayes error** for perceptual tasks
2. **Avoidable bias = Training - Bayes** tells you if model is underfitting
3. **Variance = Dev - Training** tells you if model is overfitting
4. **Use best human performance to estimate Bayes error**
5. **Progress slows near human-level** but can continue beyond
6. **Compare gaps to decide**: bias reduction vs. variance reduction

---

**Related:** [Error Analysis](error_analysis.md), [Orthogonalization](orthogonalization.md), [Train/Dev/Test Strategy](train_dev_test_strategy.md)

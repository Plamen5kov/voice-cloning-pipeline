# Data Mismatch

## Core Concept

When your training data comes from a different distribution than your dev/test data, you need special techniques to diagnose and address the mismatch.

### What Does "Different Distribution" Mean?

**Statistical Definition:**

A dataset comes from a **different distribution** when its statistical properties differ from another dataset. This can manifest in several ways:

**1. Input Features (X) differ:**
- Different ranges or scales of feature values
- Different noise patterns or quality levels
- Different correlations between features
- Different missing data patterns

**2. Output Labels (Y) differ:**
- Different class proportions (class imbalance)
- Different label definitions or annotation criteria
- Different error rates in labeling

**3. Joint Distribution P(X, Y) differs:**
- Different relationships between inputs and outputs
- Different conditional probabilities P(Y|X)

**Concrete Examples:**

| Training Distribution | Test/Production Distribution | What Differs |
|----------------------|------------------------------|--------------|
| High-quality internet images | Blurry phone camera photos | Image quality, lighting, noise |
| Studio audio recordings | Noisy car environment audio | Background noise, acoustic properties |
| Hospital A's MRI scans | Hospital B's MRI scans | Equipment calibration, imaging protocols |
| Product reviews (formal) | Social media posts (casual) | Vocabulary, sentence structure, slang |
| Daytime driving footage | Nighttime/rainy driving footage | Lighting conditions, visibility |

**Why It Matters:**

Your model learns patterns from the training distribution. When real-world data has different statistical properties, performance degrades because the model encounters:
- Feature values outside its training range
- Different noise levels or patterns it didn't learn to handle
- Different feature correlations it didn't observe
- Different class frequencies than it was optimized for

## The Problem

### Traditional Setup (Same Distribution)

```
Training: 200,000 cat images from web
Dev:      10,000 cat images from web  } Same distribution
Test:     10,000 cat images from web  }
```

**Simple diagnosis:**
- High training error → Bias problem
- High dev error (vs train) → Variance problem

### Mismatched Distribution Setup

```
Training: 200,000 cat images from web (high quality)
Dev:      10,000 cat images from mobile app (low quality) } Same distribution
Test:     10,000 cat images from mobile app (low quality) }

Production: Mobile app images
```

**Problem:** Training and dev/test have different distributions!

**New question:** Is poor dev performance due to:
1. **Variance** (model doesn't generalize)?
2. **Data mismatch** (different distribution)?

## Introducing Training-Dev Set

### Solution: Create a Training-Dev Set

```
Training:      200,000 images from web
Training-Dev:  10,000 images from web (held out from training)
Dev:           10,000 images from mobile app
Test:          10,000 images from mobile app
```

**Key insight:** Training-Dev has same distribution as Training but was never trained on.

### Error Analysis with Training-Dev

**Four key numbers:**
1. **Human-level error** (Bayes error proxy)
2. **Training error**
3. **Training-dev error**
4. **Dev error**

## Diagnosing with Training-Dev Set

### Case 1: Variance Problem (No Data Mismatch)

```
Human-level:     0%
Training:        1%
Training-Dev:    9%   ← Big jump!
Dev:            10%

Diagnosis: VARIANCE
- Training-dev is from same distribution as training
- Model doesn't generalize even within same distribution
- Data mismatch is NOT the problem

Solution: Standard variance fixes
- Get more training data (from web)
- Regularization
- Reduce model complexity
```

### Case 2: Data Mismatch Problem

```
Human-level:     0%
Training:        1%
Training-Dev:    1.5%  ← Small jump
Dev:            10%    ← Big jump!

Diagnosis: DATA MISMATCH
- Model generalizes fine on web images (training-dev is good)
- Model fails on mobile images (dev is bad)
- Distribution mismatch is the problem

Solution: Address data mismatch
- Get more mobile app data for training
- Data synthesis/augmentation
- Domain adaptation techniques
```

### Case 3: Both Variance and Data Mismatch

```
Human-level:     0%
Training:        1%
Training-Dev:    5%    ← Variance problem
Dev:            12%    ← Data mismatch on top

Diagnosis: BOTH PROBLEMS
- Training → Training-Dev gap: Variance (4%)
- Training-Dev → Dev gap: Data mismatch (7%)

Solution: Fix in order
1. First: Address variance (bigger problem)
2. Then: Address data mismatch
```

### Case 4: Bias Problem

```
Human-level:     0%
Training:       10%    ← High training error
Training-Dev:   11%
Dev:            12%

Diagnosis: HIGH BIAS
- Training error itself is too high
- Small gaps suggest mismatch/variance are minor

Solution: Standard bias fixes
- Bigger model
- Train longer
- Better architecture
```

## Complete Diagnostic Framework

### The Four Gaps

```
1. Avoidable Bias     = Training error - Human-level
2. Variance           = Training-Dev error - Training error
3. Data Mismatch      = Dev error - Training-Dev error
4. Test Overfitting   = Test error - Dev error
```

### Example Analysis

```
Human-level:     0.5%
Training:        3.0%  → Avoidable bias = 2.5%
Training-Dev:    3.5%  → Variance = 0.5%
Dev:             8.0%  → Data mismatch = 4.5%
Test:            8.1%  → Test overfitting = 0.1%

Priority ranking:
1. Data mismatch (4.5%) ← FOCUS HERE
2. Avoidable bias (2.5%)
3. Variance (0.5%)
4. Test overfitting (0.1%) ← Not a concern
```

## Addressing Data Mismatch

### Strategy 1: Collect More Target Distribution Data

**Most effective but often hardest**

```
Problem: Only 10,000 mobile app images
Solution: Collect 50,000 more mobile app images for training

Before:
  Training: 200,000 web + 10,000 mobile
  Dev: 10,000 mobile

After:
  Training: 200,000 web + 60,000 mobile
  Dev: 10,000 mobile
```

**When to use:**
- Target distribution data is acquirable
- Have budget/resources to collect
- Quality labeling is feasible

### Strategy 2: Artificial Data Synthesis

**Create synthetic data that matches target distribution**

#### Example 1: Speech Recognition with Car Noise

```
Problem: Model trained on quiet audio, deployed in cars
Mismatch: Clean audio vs car noise

Synthesis approach:
  Clean audio: "The quick brown fox"
  + Car noise: Engine hum, road noise
  = Synthetic car audio

Result: 10,000 hours of clean audio → 10,000 hours of in-car audio
```

**Pitfall:** Over-representing specific noise
```
❌ Use same 1-hour car recording for all 10,000 hours
   Model learns that specific car's noise, not general car noise

✅ Use diverse car noise: 100 different cars, various speeds, roads
   Model learns general car noise patterns
```

#### Example 2: Computer Vision with Image Effects

```
Problem: Training on professional photos, deploying on phone cameras
Mismatch: High-quality vs phone camera artifacts

Synthesis approach:
  Professional photo of bird
  + Phone camera effects: Lower resolution, motion blur, lens distortion
  = Synthetic phone photo

Techniques:
- Add realistic blur
- Reduce resolution
- Simulate lens distortion
- Add compression artifacts
- Vary lighting conditions
```

**Key principle:** Diverse synthesis > large but narrow synthesis

### Strategy 3: Domain Adaptation Techniques

**Make model robust to distribution shift**

#### Technique 1: Adversarial Training

```
Train model to be invariant to domain differences
- Main network: Classifies cat vs not-cat
- Domain classifier: Tries to detect if image is web vs mobile
- Train main network to fool domain classifier

Result: Features that work across both domains
```

#### Technique 2: Self-Training / Pseudo-Labeling

```
1. Train model on labeled source data (web images)
2. Apply to target data (mobile images)
3. Use high-confidence predictions as pseudo-labels
4. Retrain on original + pseudo-labeled data
5. Repeat
```

#### Technique 3: Fine-tuning on Target Domain

```
1. Pre-train on large source dataset (200K web images)
2. Fine-tune on small target dataset (10K mobile images)
3. Use lower learning rate for fine-tuning
4. Optionally freeze early layers
```

### Strategy 4: Manual Error Analysis

**Understand what's specifically failing**

```
Step 1: Sample 100 errors from dev set (mobile images)

Step 2: Categorize errors
- Blurry images: 45%
- Dark/underexposed: 30%
- Unusual angles: 15%
- Actually correct (label error): 10%

Step 3: Targeted solutions
- Blur → Add motion blur synthesis to training
- Dark → Add low-light augmentation
- Angles → Collect more varied viewpoint data
```

## Practical Guidelines

### When Training and Dev Distributions Differ

**✅ Do this:**
1. Always create Training-Dev set (10K examples from training distribution)
2. Use it to separate variance from data mismatch
3. Prioritize based on gap sizes
4. Accept that some mismatch is okay if costs are high

**❌ Don't do this:**
1. Assume poor dev performance is variance without checking
2. Collect more training data from wrong distribution
3. Over-synthesize from narrow data (same noise clip repeated)

### Data Synthesis Best Practices

```
✅ Good synthesis:
- Diverse parameters (100 different car noises)
- Realistic effects (matches target domain)
- Validates on real target data

❌ Bad synthesis:
- Repeated parameters (same noise clip)
- Unrealistic effects (too extreme)
- Never validates on real data
```

### Deciding How Much Target Data to Collect

```
Cost-benefit analysis:

Option A: Collect 50K more mobile images
  - Cost: $50,000, 2 months
  - Expected improvement: 8% → 4% error (4% absolute)
  
Option B: Improve model architecture
  - Cost: 2 weeks engineering
  - Expected improvement: 8% → 7% error (1% absolute)

Start with B (cheap, fast), then do A if needed
```

## Example Scenarios

### Scenario 1: Speech Recognition with Accents

```
Setup:
  Training: 10,000 hours US English
  Dev/Test: Indian English accent
  
Error analysis:
  Human-level: 5%
  Training: 6%
  Training-Dev: 7%
  Dev: 15%
  
Diagnosis:
  - Avoidable bias: 1% (6-5)
  - Variance: 1% (7-6)
  - Data mismatch: 8% (15-7) ← MAIN PROBLEM
  
Solutions (in order):
  1. Collect Indian English training data
  2. Use data augmentation with accent simulation
  3. Fine-tune model on Indian English subset
```

### Scenario 2: Medical Imaging with Different Scanners

```
Setup:
  Training: 100,000 images from Scanner A (high-end)
  Dev/Test: Scanner B images (lower-end, production)
  
Error analysis:
  Human-level: 2%
  Training: 3%
  Training-Dev: 4%
  Dev: 9%
  
Diagnosis:
  - Avoidable bias: 1%
  - Variance: 1%
  - Data mismatch: 5% ← MAIN PROBLEM
  
Solutions:
  1. Collect more Scanner B images (most effective)
  2. Simulate Scanner B artifacts on Scanner A images
  3. Use domain adaptation techniques
  
Note: Medical data is expensive/regulated
  → Synthesis might be only feasible option initially
```

### Scenario 3: Satellite vs Drone Imagery

```
Setup:
  Training: Satellite images (high altitude, lower resolution)
  Dev/Test: Drone images (low altitude, higher resolution)
  Task: Detect buildings
  
Challenge: Significant distribution mismatch

Solutions:
  1. Downsample drone images to satellite resolution (for training)
  2. Train multi-scale model
  3. Use both distributions in training
  4. Accept that model will need retraining for drone deployment
  
Decision: Might need separate models for each use case
```

## Key Takeaways

1. **Training-Dev set is essential** when distributions differ
   - Same distribution as training
   - Separates variance from data mismatch

2. **Four gaps to monitor:**
   - Avoidable bias (human → training)
   - Variance (training → training-dev)
   - Data mismatch (training-dev → dev)
   - Test overfitting (dev → test)

3. **Prioritize by gap size** - Fix biggest problem first

4. **Data mismatch solutions:**
   - Collect target distribution data (best)
   - Synthesize target distribution (practical)
   - Domain adaptation (advanced)
   - Accept mismatch if cost too high

5. **Synthesis pitfall:** Avoid over-representing narrow slice of data

6. **It's okay to have different training and dev distributions**
   - Common in real-world applications
   - Just measure and address the mismatch systematically

## Decision Framework

```
Do I have data mismatch?
├─ Training-dev error ≈ Dev error → NO, it's variance
└─ Training-dev error << Dev error → YES, data mismatch

How do I fix it?
├─ Can I collect target data? → YES: Collect more
├─ Can I synthesize realistically? → YES: Data synthesis  
├─ Is domain adaptation feasible? → YES: Advanced techniques
└─ Is mismatch cost acceptable? → YES: Accept and deploy with caution
```

---

**Related:** [Train/Dev/Test Strategy](train_dev_test_strategy.md), [Error Analysis](error_analysis.md), [Human-Level Performance](human_level_performance.md)

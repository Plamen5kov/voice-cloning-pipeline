# Practical ML Project Decisions Guide

A quick-reference guide for making common machine learning decisions, based on systematic evaluation principles.

## Quick Decision Trees

### 1. Should I Use a Single Evaluation Metric?

**YES - Always work toward a single evaluation metric!**

Benefits:
- Simplifies decision-making (Model A vs Model B is clear)
- Enhances development speed (no ambiguous comparisons)
- Team alignment (everyone knows what "better" means)

### 2. How Do I Handle Multiple Concerns?

```
Do you have multiple performance concerns (accuracy, speed, size)?
├─ YES → Choose ONE approach:
│   ├─ Option A (Recommended): Optimizing + Satisficing
│   │   Example: Maximize accuracy
│   │            Subject to: inference ≤ 100ms, size ≤ 10MB
│   │
│   └─ Option B: Weighted combination (F-beta score)
│       Example: F2 score (when false negatives are 2x worse than false positives)
│
└─ NO → Use single metric (accuracy, F1, etc.)
```

**Anti-pattern:** Trying to compare models with 2+ metrics simultaneously
```
❌ Model A: 92% accuracy, 5s runtime, 8MB
❌ Model B: 90% accuracy, 2s runtime, 12MB
   Which is better? You can't decide!
```

**Better approach:**
```
✅ Optimize: Accuracy
✅ Constraints: Runtime ≤ 10s, Size ≤ 10MB
   
   Model A: 92%, 5s, 8MB  ✓ WINNER
   Model B: 90%, 2s, 12MB ✗ Too large
```

---

### 3. My Model Has Too Many False Negatives - What Do I Do?

```
Problem: Model misses positive cases (birds present but not detected)

Quick fixes (in order):
1. Adjust classification threshold
   Default: threshold = 0.5
   Fix: Lower to 0.3 or 0.2 to catch more positives
   Trade-off: More false positives
   
2. Change your metric
   Use F2 score (weights recall 2x more than precision)
   Or: Optimize recall with precision ≥ threshold
   
3. Training-level fixes
   - Oversample positive examples
   - Use class weights (penalize missing positives more)
   - Collect more positive training data
```

**Decision:**
```
Is false negative MORE costly than false positive?
├─ YES → Use F2 score or optimize recall with precision constraint
├─ EQUAL → Use F1 score (balanced)
└─ NO → Use standard F1 or optimize precision
```

---

### 4. I Have New Data - Where Should It Go?

```
KEY QUESTIONS: 
1. Is this a NEW type of data you didn't have before?
2. Does it match your production distribution?

┌─────────────────────────────────────────────────────────────┐
│ Decision Flow                                                │
└─────────────────────────────────────────────────────────────┘
        │
        ├─ NEW distribution type that WILL be in production
        │   └─> Add proportionally to train/dev/test
        │       Example: You had ONLY daytime photos
        │                Now adding nighttime (production will have both)
        │                → Need nighttime in dev/test to measure performance
        │
        ├─ MORE data of SAME distribution you already have
        │   └─> Add to TRAINING ONLY
        │       Example: Already have security camera images in all sets
        │                Got 10,000 more security camera images
        │                → Training only (dev/test already representative)
        │
        └─ Different distribution, NOT production
            └─> Add to TRAINING ONLY
                Example: Production is security cameras
                         New data is professional photography
                         → Training only (helps generalization)
```

**Common mistakes:**

❌ **Mistake 1:** Adding new data evenly to train/dev/test when it's NOT production
```
Production: Security cameras only
Action: Added phone images to train/dev/test evenly
Problem: Dev/test no longer match production!
Fix: Phone images → training only
```

❌ **Mistake 2:** Only adding to training when it's a NEW type that IS production
```
Production: NOW includes nighttime images (didn't before)
Action: Added night images to training only
Problem: Can't measure performance on night images!
Fix: Add proportionally to train/dev/test (or at least some to dev/test)
```

❌ **Mistake 3:** Reshuffling everything when you get MORE of the same
```
Setup: Already have 10K security camera images split properly
Action: Got 5K more security cameras → reshuffled all 15K
Problem: Expensive, breaks reproducibility, unnecessary
Fix: Just add new 5K to training set
```

---

### 5. Bias vs Variance - What Should I Focus On?

**You MUST know Bayes error (human-level performance) first!**

```
Step 1: Get baseline measurements
├─ Human-level performance (Bayes error proxy): __%
├─ Training error: __%
└─ Dev error: __%

Step 2: Calculate gaps
├─ Avoidable Bias = Training error - Human-level
└─ Variance = Dev error - Training error

Step 3: Focus on bigger gap
```

### Example 1: High Bias Problem

```
Human-level: 0.1%
Training:    2.0%  → Gap = 1.9% (LARGE)
Dev:         2.1%  → Gap = 0.1% (small)

Diagnosis: HIGH BIAS (model can't fit training data)

Solutions:
✅ Bigger network (more layers, more units)
✅ Train longer (more epochs)
✅ Better architecture
✅ Reduce regularization (less dropout, less L2)
✅ Better optimization (Adam vs SGD, tune learning rate)

Don't do:
❌ Get more training data (won't help bias)
❌ Add regularization (makes bias worse)
```

### Example 2: High Variance Problem

```
Human-level: 0.1%
Training:    0.5%  → Gap = 0.4% (small)
Dev:         3.0%  → Gap = 2.5% (LARGE)

Diagnosis: HIGH VARIANCE (model doesn't generalize)

Solutions:
✅ Get more training data
✅ Regularization (dropout, L2, L1)
✅ Data augmentation
✅ Reduce model complexity
✅ Early stopping

Don't do:
❌ Bigger network (makes variance worse)
❌ Remove regularization (makes variance worse)
```

### Example 3: Both Problems

```
Human-level: 0.1%
Training:    2.0%  → Gap = 1.9% (large bias)
Dev:         5.0%  → Gap = 3.0% (large variance)

Diagnosis: BOTH HIGH BIAS AND HIGH VARIANCE

Solutions (in order):
1. Fix bias first (bigger network, train longer)
2. Then fix variance (more data, regularization)
3. Or use techniques that help both:
   - Better architecture
   - Batch normalization
   - Residual connections
```

---

### 6. Train/Dev/Test Error Patterns - What Do They Mean?

| Human | Train | Dev | Test | Diagnosis | Action |
|-------|-------|-----|------|-----------|--------|
| 0.1% | 2% | 2.1% | - | **High bias** | Bigger model, reduce regularization |
| 0.1% | 0.5% | 3% | - | **High variance** | More data, regularization |
| 0.1% | 2% | 5% | - | **Both bias & variance** | Fix bias first, then variance |
| 0.1% | 0.5% | 0.6% | 5% | **Distribution mismatch** | Dev and test from different sources |
| 4% | 4.1% | 4.2% | - | **Near optimal** | Model at Bayes error, limited improvement possible |
| 0.1% | 0.05% | 0.05% | - | **Surpassing human-level** | Near Bayes error, possible overfitting, most avoidable bias removed |
| 0.1% | 0.2% | 0.3% | 0.3% | **Excellent** | Deploy! |

---

### 7. Dev Set Problems

#### Problem: Dev and Test Errors Are Very Different

```
Scenario:
Training: 2%
Dev: 2.1%  ← Looks good!
Test: 7%   ← Wait, what?!

Diagnosis: DISTRIBUTION MISMATCH
- Dev and test are from different data sources
- You've been optimizing for the wrong thing

Solutions:
1. Verify dev and test came from same distribution
2. If not, rebuild both from target distribution
3. Shuffle and re-split to ensure consistency
```

#### Problem: Overfitting to Dev Set

```
Scenario:
You've tried 100+ hyperparameter combinations
Dev performance kept improving
Test performance is much worse

Diagnosis: DEV SET CONTAMINATION
- Used dev set too many times
- Essentially memorized what works on this specific set

Solutions:
✅ Get FRESH dev and test sets (best option)
✅ Get a BIGGER dev set (provides more reliable estimate)
✅ Use current test set as new dev
✅ Set aside new test set for final evaluation

Don't do:
❌ Adjust regularization (wrong problem - this is an evaluation issue, not a training issue)
```

### Important: ML vs Human-Level Performance vs Bayes Error

```
Key relationships:

Bayes Error (theoretical minimum) ≤ Human-level ≤ ML Performance

Can ML beat human-level? YES
✅ ML can surpass human performance on many tasks
✅ Examples: image recognition, game playing, some medical diagnosis

Can ML beat Bayes error? NO
❌ Bayes error is the theoretical best possible
❌ It represents irreducible error in the data
❌ If ML appears to beat it, you're likely overfitting

What happens when ML beats human-level?
- Most avoidable bias has been eliminated
- Close to Bayes error
- Further progress becomes much harder
- Risk of overfitting increases
```

---

## Common Scenarios: Bird Classification Example

### Scenario 0: When False Negatives Cost You the Contract

```
Situation:
Your system: 94% accuracy, few false positives
Competitor: 92% accuracy, fewer false negatives
Client prefers competitor despite your higher accuracy

Root cause: Wrong evaluation metric
You optimized for accuracy, but client cares about catching birds

Fix: Redefine your metric BEFORE further development

Option 1 (Recommended): F2 Score
  F2 = (1 + 4) × (P × R) / (4P + R)
  Weights recall 2x more than precision
  Single number that captures the tradeoff

Option 2: Optimizing + Satisficing
  Optimize: Recall (minimize false negatives)
  Constraint: Precision ≥ 0.70
  
Option 3: Weighted Error
  Cost = 1 × (false positives) + 10 × (false negatives)
  If missing a bird is 10x worse than false alarm

Don't do:
❌ "Consider both accuracy and false negative rate"
   (Two metrics = can't compare models clearly)
❌ "Pick false negative rate as the only metric"
   (Ignores false positives completely)

Key principle: Redefine metric to match business value, then develop
```

### Scenario 1: Accurate Overall but Missing Birds

```
Situation:
✓ 92% accuracy on dev set
✗ High false negative rate (missing many birds)
✗ Competitor's system preferred despite lower accuracy

Problem: Accuracy doesn't capture your concern (false negatives matter more)

Solution: Redefine your optimizing metric to include false negatives

1. Use F2 score (recommended)
   F2 = (1 + 2²) × (P × R) / (2² × P + R)
   This weights recall 2x more than precision
   Single number that captures both accuracy AND false negative concerns

2. Or use optimizing + satisficing:
   Optimize: Recall (minimize false negatives)
   Constraint: Precision ≥ 0.75 (keep quality acceptable)

3. Quick fix for existing model: Lower classification threshold
   Default: 0.5 → Try: 0.3
   More birds detected, some false alarms

Don't do:
❌ Just optimize for false negative rate alone (ignores false positives entirely)
❌ Ask team to "consider both accuracy and false negatives" (two metrics = unclear priority)
```

### Scenario 2: Adding Nighttime Bird Images

```
Situation:
Current: Only daytime photos in dataset
New: Nighttime photos collected
Production: Will NOW see both day and night (didn't before)

Question: Where does night data go?

Answer: Add proportionally to train/dev/test
1. Split nighttime data: 98% train / 1% dev / 1% test
2. Add to respective existing sets
3. Retrain and evaluate

Alternative (if you want fresh start):
1. Combine all data (day + night)
2. Randomly split into train/dev/test
3. Retrain from scratch

Why: Your dev/test MUST include night images if production will have them
      You need to measure performance on this new scenario
```

### Scenario 3: Found Professional Bird Photography Dataset

```
Situation:
Current: Security camera images (low quality, production)
New: Professional photos (high quality, different distribution)

Question: Where does it go?

Answer: TRAINING ONLY
- Training: Security cameras + professional photos
- Dev/Test: Security cameras ONLY (matches production)

Why: Diverse training helps generalization
     But dev/test must stay focused on production distribution
```

### Scenario 4: Adding New Bird Species (Distribution Shift)

```
Situation:
New species migrating into area
Performance degrading on new data
Only 1,000 images of new species

Wrong approaches:
❌ Add to training set only
   Problem: Can't measure performance on new species!
❌ Just do data augmentation
   Problem: Still can't evaluate properly
❌ Reshuffle everything
   Problem: Time-consuming, breaks reproducibility

Correct approach (PRIORITY ORDER):
1. FIRST: Define new evaluation metric using new dev/test set
   - Add new species to dev and test sets
   - This allows you to measure what matters NOW
   - Your old dev/test set no longer represents production
   
2. THEN: Add remaining data to training
   - Use the rest for training
   - Consider data augmentation to increase training data

3. THEN: Retrain and evaluate on new metric

Why this order:
- Evaluation comes before training
- If you can't measure it, you can't improve it
- The distribution has CHANGED, so your metric must change
```

### Scenario 5: Defining Human-Level Performance

```
Situation:
Need to estimate Bayes error for bird species classification
Multiple humans with different expertise levels

Data:
Bird expert #1: 0.3% error
Bird expert #2: 0.5% error
Normal person #1: 1.0% error
Normal person #2: 1.2% error

Question: What is "human-level performance"?

Answer: 0.3% (best expert performance)

Why:
✅ Closest practical estimate to Bayes error
✅ Represents what's achievable with perfect attention
✅ Optimistic benchmark helps identify avoidable bias

Don't use:
❌ 0.4% (average of experts) - Too conservative
❌ 0.75% (average of all four) - Way too conservative
❌ 1.0% (typical person) - Not the true limit
❌ 0.0% (perfect) - Unrealistic, not Bayes error

Impact on diagnosis:
With human-level = 0.3% and training error = 2.0%:
  → Avoidable bias = 1.7% (significant problem)
  
With human-level = 1.0% and training error = 2.0%:
  → Avoidable bias = 1.0% (smaller problem)
  
The choice matters for prioritization!
```

### Scenario 6: Long Training Time

```
Situation:
100,000,000 cat images
Training takes 2 weeks per experiment
Can't iterate quickly

Problem: Iteration speed vs accuracy tradeoff

Solutions (in priority order):

1. Use subset of data for rapid iteration
   ✅ Train on 10M images instead of 100M
   ✅ Get results in 1-2 days instead of 2 weeks
   ✅ Acceptable accuracy loss for development speed
   ✅ Use full dataset only for final model
   
2. Use distributed training if available
   ✅ 4 GPUs → ~3.5x speedup (not perfect 4x due to overhead)
   ✅ 8 GPUs → ~6x speedup
   
3. Start with smaller model
   ✅ Prototype with lightweight architecture
   ✅ Scale up once you've validated approach

4. Progressive data loading
   ✅ Train on 10M → evaluate → train on 30M → evaluate
   ✅ Stop early if not improving

Key insight: 
- Iteration speed is critical in development
- 10 experiments with 10M images beats 1 experiment with 100M
- Use full data only when you've converged on architecture/hyperparameters
```

---

## Checklist: Before Training

**Data Setup:**
```
☐ Dev and test sets from SAME distribution
☐ Dev and test sets match PRODUCTION distribution
☐ Training can be broader (diverse data is OK)
☐ Set sizes reasonable:
  - Large data: 98% train / 1% dev / 1% test
  - Small data: 60% train / 20% dev / 20% test
```

**Metrics:**
```
☐ Single evaluation metric defined (not 2+)
☐ If multiple concerns: defined optimizing vs satisficing
☐ Metric aligns with business goal (not just accuracy)
☐ Established human-level performance baseline
```

**Infrastructure:**
```
☐ Can iterate quickly (don't overengineer v1)
☐ Dev set evaluation is fast (use subset if needed)
☐ Error analysis process defined
☐ Version control for models and data
```

---

## Checklist: During Training

```
☐ Training error approaching human-level? 
  → If NO: Focus on bias (bigger model, train longer)
  
☐ Dev error close to training error?
  → If NO: Focus on variance (more data, regularization)
  
☐ Dev error acceptable for production?
  → If NO: Continue iteration
  → If YES: Evaluate on test set
  
☐ Test error similar to dev error?
  → If NO: Distribution mismatch or dev set contamination
  → If YES: Ready for production evaluation
```

---

## Quick Formulas

### Error Analysis Priority

```
Error Impact Score = (% of errors) × (ease of fix) × (1 / required time)

Example:
Category A: 45% of errors, easy fix, 1 week → Score: 0.45 × 1.0 × (1/1) = 0.45
Category B: 10% of errors, easy fix, 1 week → Score: 0.10 × 1.0 × (1/1) = 0.10
Category C: 30% of errors, hard, 3 months → Score: 0.30 × 0.3 × (1/12) = 0.0075

Priority: A → C → B
```

### F-Beta Score

```
Fβ = (1 + β²) × (precision × recall) / (β² × precision + recall)

β = 0.5: Precision weighted 2x more (fewer false positives matter)
β = 1.0: F1 score (balanced)
β = 2.0: Recall weighted 2x more (catching positives matters)
```

### Bias and Variance

```
Avoidable Bias = Training error - Bayes error
Variance = Dev error - Training error

Focus area = max(Avoidable Bias, Variance)
```

---

## Decision Matrix: What To Do Next

| Situation | What to Do | What NOT to Do |
|-----------|-----------|----------------|
| High training error (vs human) | Bigger model, reduce regularization | Add more data, increase regularization |
| Large train-dev gap | More data, regularization, augmentation | Bigger model, reduce regularization |
| Dev-test error mismatch | Check distributions, rebuild sets | Adjust hyperparameters |
| Too many false negatives | Redefine metric (F2 score), lower threshold | Use accuracy alone, ask team to "consider both" |
| New species to add | First: new dev/test with species. Then: train | Training only |
| Different distribution data | Add to training only | Add to dev/test |
| NEW production distribution | Define new evaluation metric first | Just add to training |
| MORE of same distribution | Add to training only | Reshuffle everything |
| Dev set overfitted (100+ tries) | Fresh dev/test OR bigger dev set | Just adjust regularization |
| Long training time (2 weeks) | Reduce training data size, iterate faster | Use all data every time |

---

## Anti-Patterns to Avoid

### ❌ Wrong: Multiple Metrics Without Structure
```python
# Confusing - which model is better?
model_a = {"accuracy": 0.92, "latency": 100, "f1": 0.88}
model_b = {"accuracy": 0.90, "latency": 50, "f1": 0.91}
```

### ✅ Right: Single Decision Criterion
```python
# Clear decision
def evaluate(model):
    if model.latency > 100:  # Satisficing constraint
        return 0  # Disqualified
    return model.f1  # Optimizing metric

best_model = max(models, key=evaluate)
```

---

### ❌ Wrong: Dev/Test Setup
```python
# Production: security cameras
train = load("security_cameras.pkl")
dev = load("phone_images.pkl")      # ← Different distribution!
test = load("professional_photos.pkl")  # ← Even more different!
```

### ✅ Right: Dev/Test Setup
```python
# Production: security cameras
all_camera_data = load("security_cameras.pkl")
train, dev, test = split(all_camera_data, [0.98, 0.01, 0.01])

# Optional: Add diverse data to training only (helps generalization)
train += load("phone_images.pkl")
train += load("professional_photos.pkl")

# Got more security camera images later? Add to training
train += load("more_security_cameras.pkl")  # No need to reshuffle
```

---

### ❌ Wrong: Ignoring Bayes Error
```python
train_error = 2.0%
dev_error = 2.1%
# "Variance is 0.1%, model is great!"
# But if human-level is 0.1%, you have 1.9% avoidable bias!
```

### ✅ Right: Full Analysis
```python
human_level = 0.1%
train_error = 2.0%
dev_error = 2.1%

bias = train_error - human_level  # 1.9% - FOCUS HERE
variance = dev_error - train_error  # 0.1% - OK
```

---

## When to Ship

```
✅ Ship when:
├─ Training error ≈ human-level (low avoidable bias)
├─ Dev error ≈ training error (low variance)
├─ Test error ≈ dev error (no distribution mismatch)
├─ Satisficing metrics met (latency, size, etc.)
└─ Error analysis shows remaining errors are acceptable

⚠️ Don't ship when:
├─ Large avoidable bias (train >> human)
├─ Large variance (dev >> train)
├─ Distribution mismatch (test >> dev)
├─ Haven't done error analysis
└─ Optimizing for wrong metric
```

---

## Resources

**Full conceptual guides:**
- [Evaluation Metrics](evaluation_metrics.md) - F1, satisficing vs optimizing
- [Train/Dev/Test Strategy](train_dev_test_strategy.md) - Distribution matching
- [Human-Level Performance](human_level_performance.md) - Bias analysis
- [Orthogonalization](orthogonalization.md) - Systematic tuning
- [Error Analysis](error_analysis.md) - Prioritizing improvements

**Visual guide:**
- [AI Learning Path](../../ai_mind_map/ai_complete_learning_path.html) - Interactive graph

---

## Summary: The Systematic Approach

```
1. Define Success
   └─> Single metric (optimizing + satisficing)

2. Establish Baseline
   └─> Human-level performance / Bayes error

3. Set Up Data Correctly
   └─> Dev/test match production
   └─> Training can be broader

4. Train Initial Model
   └─> Get baseline numbers

5. Diagnose Using Error Analysis
   ├─> High bias? → Bigger model, reduce regularization
   ├─> High variance? → More data, add regularization
   ├─> Distribution mismatch? → Fix data splits
   └─> Metric wrong? → Change metric

6. Make ONE orthogonal change
   └─> Measure impact

7. Iterate until satisfactory
   └─> Repeat steps 5-6

8. Final Test Set Evaluation
   └─> Confirm generalization

9. Deploy & Monitor
   └─> Track production metrics
```

**Remember:** Don't guess. Measure, analyze, then act systematically.

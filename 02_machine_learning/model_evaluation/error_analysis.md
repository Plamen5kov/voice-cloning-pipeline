# Error Analysis

## Core Concept
Systematically analyze errors to identify highest-impact improvements rather than guessing what to fix.

## The Problem: Where to Focus?

Imagine your cat classifier has 10% error (90% accuracy).

You notice it sometimes mistakes dogs for cats.

**Should you spend 3 months building dog-specific features?**

**Error analysis answers this!**

## Manual Error Analysis Process

### Step 1: Get a Sample of Errors

Take 100 mislabeled dev set examples (or all errors if < 100)

### Step 2: Create Error Categories

Look through errors and create categories:

| Image # | Dog | Great Cat | Blurry | Instagram Filter | Comments |
|---------|-----|-----------|--------|------------------|----------|
| 1 | ✓ | | | ✓ | Pitbull in sunglasses |
| 2 | | ✓ | | | Lion, not house cat |
| 3 | ✓ | | ✓ | | Blurry dog |
| 4 | | | | ✓ | Heavy filter on cat |
| ... | | | | | |
| **% of total** | **8%** | **43%** | **61%** | **12%** | |

### Step 3: Prioritize

**Key insight:** Even perfect dog detector only fixes 8% of errors!
- 10% error × 8% = 0.8% absolute improvement
- 90% accuracy → 90.8% accuracy

**Great cats are the problem!** 43% of errors
- Potential improvement: 4.3% absolute
- Much higher priority!

## Ceiling Analysis

**Estimate upper bound of each improvement:**

```
Current performance: 90%

If we perfectly solved:
  ✓ Dog problem: 90% → 90.8% (+0.8%)
  ✓ Great cat problem: 90% → 94.3% (+4.3%)
  ✓ Blurry images: 90% → 96.1% (+6.1%)
  ✓ Instagram filters: 90% → 91.2% (+1.2%)
```

**Priority ranking:**
1. Blurry images (6.1% potential)
2. Great cats (4.3% potential)
3. Instagram filters (1.2% potential)
4. Dog mislabeling (0.8% potential)

## Error Analysis for Multiple Ideas

### Example: Speech Recognition

You have 3 ideas to improve:
1. Fix car noise issues
2. Fix people speaking in background
3. Fix far-field audio (microphone far away)

**Error analysis:**

| Audio clip | Car noise | Cafe noise | Far-field | Other | Comments |
|------------|-----------|------------|-----------|-------|----------|
| 1 | ✓ | | | | Highway background |
| 2 | | ✓ | ✓ | | Coffee shop, user far from phone |
| 3 | | | ✓ | | Across the room |
| ... | | | | | |
| **% of errors** | **4%** | **12%** | **24%** | **60%** | |

**Conclusion:**
- Focus on far-field audio (24% of errors)
- Consider cafe noise (12%)
- Car noise is low priority (4%)
- Investigate "Other" category (60% is a lot!)

## Cleaning Mislabeled Data

### Training Set

**Mislabeled training examples:**
- Deep learning is quite robust to random errors
- Systematic errors are more problematic

**Should you fix training labels?**
- If errors are random: Probably not worth it (small effect)
- If errors are systematic: Yes, can hurt significantly

### Dev/Test Set

**Mislabeled dev/test examples:**
- Add a category in error analysis!

| Example | Dog | Blurry | **Mislabeled** | Comments |
|---------|-----|--------|----------------|----------|
| 1 | ✓ | | | |
| 2 | | | ✓ | Actually is a cat |
| 3 | | ✓ | | |
| **% errors** | **8%** | **61%** | **6%** | |

**Decision framework:**

If mislabeled examples are a large fraction:
```
Overall dev error: 10%
Errors due to:
  - Mislabeled: 0.6% (6% of 10%)
  - Other causes: 9.4%
  
If improving algorithm from 10% → 8%:
  - Mislabeled is now 0.6% / 8% = 7.5% of errors
  - May want to fix labels
```

**Guidelines for fixing labels:**
1. Apply same process to dev AND test sets
2. Consider examples algorithm got RIGHT too (not just errors)
3. Examine multiple sources of error, not just mislabeled

## The Two Fundamental Assumptions

For supervised learning to work well:

### Assumption 1: Fit Training Set Well
```
Training error ≈ Bayes error
or
Training error ≈ Human-level performance
```

**If violated:**
- Model has high bias
- Can't even fit the data it's trained on
- → Fix with bigger model, better optimization, better architecture

### Assumption 2: Training Performance Generalizes
```
Dev error ≈ Training error
```

**If violated:**
- Model has high variance
- Overfitting to training set
- → Fix with more data, regularization, data augmentation

## Systematic Improvement Workflow

### 1. Start with Error Analysis

```
Step 1: Collect errors from dev set
Step 2: Manually examine 100-200 errors
Step 3: Categorize error types
Step 4: Quantify percentage in each category
Step 5: Estimate ceiling for each fix
```

### 2. Prioritize Based on Impact

```
Priority = (% of errors) × (ease of fix) × (required resources)
```

High priority:
- Common errors (high %)
- Solvable errors (technical feasibility)
- Cheap to implement (engineering time)

### 3. Implement Top Priority Fix

**Make one change at a time!**
- Implement highest priority improvement
- Measure impact
- Run error analysis again (problems shift!)

### 4. Iterate

```
Measure → Analyze → Prioritize → Implement → Repeat
```

## Real-World Example: Bird Classification

```
Initial state:
  - 15% error on bird species classification
  - 1000 errors in dev set

Error analysis results:
```

| Category | % of Errors | Potential Gain | Priority |
|----------|-------------|----------------|----------|
| Similar-looking species | 45% | 6.75% | HIGH |
| Juvenile birds | 28% | 4.2% | MEDIUM |
| Partial occlusion | 18% | 2.7% | MEDIUM |
| Bad photo quality | 6% | 0.9% | LOW |
| Mislabeled | 3% | 0.45% | LOW |

**Action Plan:**
1. **First:** Add more training data for similar species pairs (45% impact)
2. **Second:** Data augmentation for juvenile birds (28% impact)
3. **Later:** Handle occlusion (18% impact)
4. **Skip for now:** Bad photos (only 6% impact)

## Best Practices

### ✅ Do This

- **Look at actual errors** - Numbers don't tell the whole story
- **Quantify everything** - Count percentage in each category
- **Start small** - Analyze 100 errors is enough for insight
- **Iterate quickly** - Don't spend months on low-impact items
- **Consider ease** - 10% impact taking 1 day beats 12% taking 3 months
- **Share insights** - Show team what errors look like

### ❌ Avoid This

- **Guessing priorities** - "I think X is the problem" without data
- **Fixing everything** - Focus on high-impact items first
- **Stopping after one analysis** - Problem distribution shifts as you improve
- **Ignoring systematic errors** - Random errors are fine, systematic ones aren't
- **Only looking at errors** - Sometimes need to check correct predictions too

## Building First System Quickly

### The Build-Measure-Learn Loop

```
1. Build initial system quickly (days/weeks, not months)
   ↓
2. Use it to establish baseline metrics
   ↓
3. Run error analysis
   ↓
4. Identify biggest problems
   ↓
5. Iterate and improve
   ↓
   (repeat steps 3-5)
```

**Don't over-engineer v1!**
- Get something working fast
- Let data guide your improvements
- Error analysis >> intuition

## Key Takeaways

1. **Error analysis is systematic, not random** - Count and categorize
2. **Ceiling analysis shows potential** - Maximum possible improvement
3. **Fix high-impact problems first** - Not most interesting ones
4. **One change at a time** - Know what worked
5. **Iterate based on data** - Error distribution changes as you improve
6. **Build fast, measure, learn** - Don't spend months on wrong thing

## Error Analysis Template

```markdown
### Error Analysis Report

**Date:** [Date]
**Model:** [Model version]
**Dataset:** [Dev set name]
**Total Errors:** [Count]
**Error Rate:** [Percentage]

**Error Categories:**

| Category | Count | % of Errors | Potential Fix | Est. Impact | Priority |
|----------|-------|-------------|---------------|-------------|----------|
| [Category 1] | X | Y% | [Approach] | Z% | HIGH |
| [Category 2] | X | Y% | [Approach] | Z% | MEDIUM |
| [Category 3] | X | Y% | [Approach] | Z% | LOW |

**Insights:**
- [Key finding 1]
- [Key finding 2]

**Recommended Actions:**
1. [Top priority fix]
2. [Second priority]
3. [Third priority]

**Next Steps:**
- [Specific task with timeline]
```

---

**Related:** [Orthogonalization](orthogonalization.md), [Human-Level Performance](human_level_performance.md), [Evaluation Metrics](evaluation_metrics.md)

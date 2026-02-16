# Evaluation Metrics

## Core Concept
Choosing and combining metrics to effectively evaluate and compare machine learning models.

## Single Number Evaluation Metric

**Problem:** Multiple metrics make it hard to compare models  
**Solution:** Combine metrics into a single number

### Example: Precision vs. Recall Tradeoff

Instead of tracking both separately:
```
Model A: Precision = 95%, Recall = 90%
Model B: Precision = 98%, Recall = 85%
Which is better? 🤔
```

Use **F1 Score** (harmonic mean):
```
Model A: F1 = 92.4%
Model B: F1 = 91.0%
Model A is better! ✅
```

### Formula
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

## Satisficing vs. Optimizing Metrics  

**Key Insight:** Not all metrics are equally important

### Definitions

**Optimizing Metric:** The metric you want to maximize/minimize
- Example: Maximize accuracy
- Example: Minimize error rate

**Satisficing Metric:** Any metric that must meet a threshold
- Example: Running time < 100ms
- Example: Model size < 10MB
- Example: Fairness score > 90%

### Pattern

```
1 Optimizing metric (what you're trying to optimize)
+ N Satisficing metrics (constraints that must be met)
= Clear decision criterion
```

### Example: Wake Word Detection

**Scenario:** Building "Alexa" / "Hey Google" style wake word detector

Metrics to consider:
- Accuracy (how often it correctly detects)
- False positive rate
- Latency (response time)
- Model size
- Power consumption

**Better approach:**
```
Optimizing: Maximize accuracy
Satisficing:
  - False positive rate < 1 per 24 hours
  - Latency < 50ms
  - Model size < 5MB
  - Power consumption < 100mW
```

Now you can clearly pick the most accurate model that meets all constraints!

## Combining Multiple Metrics

### Average (Simple but often wrong)
```
Average = (Metric1 + Metric2 + ... + MetricN) / N
```
**Problem:** Treats all metrics equally, which may not reflect real-world priorities

### Weighted Average
```
Combined = w1*Metric1 + w2*Metric2 + ... + wN*MetricN
```
Where weights sum to 1.0

**Better:** Reflects relative importance of each metric

### Max/Min of Set
Take worst-case performance across regions/categories:
```
Combined = min(Accuracy_US, Accuracy_China, Accuracy_India, ...)
```
**Use case:** Ensuring fairness across demographic groups

## Choosing the Right Metric

### Classification
- **Binary:** Precision, Recall, F1, ROC-AUC
- **Multi-class:** Macro-F1, Weighted-F1, Per-class accuracy
- **Imbalanced:** F1, Precision-Recall curve, ROC-AUC

### Regression
- **MAE** (Mean Absolute Error): Easy to interpret
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **R²:** Goodness of fit

### Ranking/Recommendations
- **Precision@K:** Precision in top K results
- **NDCG:** Normalized Discounted Cumulative Gain
- **MAP:** Mean Average Precision

## Practical Guidelines

1. **Start simple:** Single optimizing metric when possible
2. **Add constraints:** Use satisficing metrics for hard requirements
3. **Align with business goals:** Metric should reflect real-world value
4. **Make it interpretable:** Stakeholders should understand what it means
5. **Iterate:** Change metrics if they don't reflect true performance

## Common Mistakes

❌ Optimizing for multiple metrics simultaneously
❌ Choosing metrics that don't reflect business value
❌ Using accuracy on imbalanced datasets
❌ Forgetting about inference time/resource constraints
❌ Not considering fairness across subgroups

## Real-World Example: Content Moderation

**Bad approach:**
- Track: Accuracy, Precision, Recall, F1, Latency, False Positives, False Negatives
- Result: Can't decide which model is best

**Good approach:**
```
Optimizing: Maximize Precision (minimize false positives)
Satisficing:
  - Recall > 95% (catch at least 95% of violations)
  - Latency < 100ms (real-time moderation)
  - Fairness score > 90% across demographics
```

Now model selection is clear!

## Key Takeaway

**A good evaluation metric makes model comparison obvious and aligns with real-world goals.**

Use 1 optimizing metric + N satisficing metrics to simplify decisions while respecting constraints.

---

**Related:** [Train/Dev/Test Strategy](train_dev_test_strategy.md), [Error Analysis](error_analysis.md)

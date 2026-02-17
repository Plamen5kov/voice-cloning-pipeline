# Multi-Task Learning

## Core Concept

Train a single neural network to perform multiple related tasks simultaneously, allowing the network to share representations and learn from all tasks together.

## The Motivation

### Single-Task Learning (Traditional)

```
Task 1: Detect pedestrians → Model 1
Task 2: Detect cars → Model 2
Task 3: Detect stop signs → Model 3
Task 4: Detect traffic lights → Model 4

4 separate models, 4 separate training processes
```

### Multi-Task Learning

```
Single model with shared layers:

Input Image
    ↓
[Shared Convolutional Layers] ← Learn general features
    ↓
    ├→ [Task 1 Head] → Pedestrians (yes/no)
    ├→ [Task 2 Head] → Cars (yes/no)
    ├→ [Task 3 Head] → Stop signs (yes/no)
    └→ [Task 4 Head] → Traffic lights (yes/no)

1 model, simultaneous training on all tasks
```

**Key advantage:** Shared layers learn richer representations by seeing diverse training signals.

## Why Multi-Task Learning Works

### Shared Representations

**Lower layers learn general features useful across tasks**

```
Example: Autonomous driving

All tasks need to:
- Detect edges and shapes
- Understand spatial relationships
- Recognize object boundaries
- Process lighting conditions

By sharing these layers:
- Task 1 might teach the network about vertical edges (pedestrians)
- Task 2 teaches about rectangular shapes (cars)
- Task 3 teaches about circular shapes (signs)
- All tasks benefit from each other's lessons
```

### Regularization Effect

Training on multiple tasks acts as regularization:
- Prevents overfitting to any single task
- Forces network to learn general features
- Similar to ensemble methods

### Data Amplification

Each task contributes training signal:
```
Task 1: 1,000 examples
Task 2: 2,000 examples
Task 3: 1,500 examples
Task 4: 1,000 examples

Effective training signal: 5,500 training instances
(though not simply additive)
```

## When Multi-Task Learning Helps

### ✅ Makes Sense When:

**1. Tasks are related and share low-level features**

```
✅ Computer Vision: Object detection tasks
   - Pedestrians, cars, bikes, signs all need edge detection
   - All need texture understanding
   - All need spatial reasoning

✅ NLP: Multiple text analysis tasks
   - Named Entity Recognition
   - Part-of-speech tagging
   - Sentiment analysis
   All benefit from understanding grammar and word relationships

✅ Speech: Multiple audio tasks
   - Speech recognition
   - Speaker identification  
   - Emotion detection
   All need audio feature extraction
```

**2. Each task has somewhat limited data**

```
Example: Medical imaging

Task 1: Detect tumors (500 labeled images)
Task 2: Detect fractures (400 labeled images)
Task 3: Detect infections (300 labeled images)

Single-task: Each model trained on limited data
Multi-task: Model learns from all 1,200 images
           Each task benefits from others' data
```

**3. Tasks have similar data amounts**

```
✅ Good balance:
   Task 1: 1,000 examples
   Task 2: 800 examples
   Task 3: 1,200 examples
   Roughly similar → each task contributes meaningfully

❌ Poor balance:
   Task 1: 1,000,000 examples
   Task 2: 100 examples
   Task 3: 50 examples
   Task 1 dominates, others barely contribute
```

**4. Network is large enough to handle complexity**

```
With 4 tasks, you need:
- Sufficient capacity in shared layers
- Separate capacity for task-specific heads
- Large enough to not bottleneck

Rule of thumb: Bigger network than single-task equivalent
```

### ❌ Doesn't Help When:

**1. Tasks are unrelated**

```
❌ Object detection + Stock price prediction
   - No shared low-level features
   - Training on both may hurt both

❌ Image classification + Text generation
   - Different input modalities
   - No feature sharing benefit
```

**2. One task has overwhelming amount of data**

```
❌ Task 1: 10,000,000 examples
   Task 2: 1,000 examples

Problem: Task 1 dominates training
Solution: Either
- Use task weighting to balance
- Just train on Task 1, then transfer learn to Task 2
```

**3. Tasks conflict with each other**

```
❌ Classify images as sharp vs blurry
    AND classify as color vs black-and-white
    
These might conflict in what features to emphasize
May need separate models
```

## Implementation Architectures

### Hard Parameter Sharing (Most Common)

```
Architecture:

Input
  ↓
[Shared Conv Layer 1]    ← All tasks share
  ↓
[Shared Conv Layer 2]    ← All tasks share
  ↓
[Shared Conv Layer 3]    ← All tasks share
  ↓
  ├─ [Task 1 Layers] → Output 1
  ├─ [Task 2 Layers] → Output 2
  ├─ [Task 3 Layers] → Output 3
  └─ [Task 4 Layers] → Output 4

Pros:
- Reduces overfitting significantly
- Efficient (fewer parameters)
- Strong regularization

Cons:
- Tasks must be quite related
- All tasks go through same bottleneck
```

### Soft Parameter Sharing

```
Architecture:

Input
  ↓
  ├─ [Task 1 Column] → Output 1
  ├─ [Task 2 Column] → Output 2
  ├─ [Task 3 Column] → Output 3
  └─ [Task 4 Column] → Output 4
       ↕
  Cross-talk between columns (regularization)

Each task has its own model
But parameters are encouraged to be similar

Pros:
- More flexible
- Tasks can be less related
- Can handle different architectures per task

Cons:
- More parameters
- Less regularization
- More complex to implement
```

### Customized Sharing

```
Architecture (Example):

Input
  ↓
[Shared Early Layers]    ← All tasks benefit from low-level features
  ↓
  ├─ [Task 1&2 Shared] → ├─ Task 1 output
  │                       └─ Task 2 output
  │
  └─ [Task 3&4 Shared] → ├─ Task 3 output
                          └─ Task 4 output

Some tasks share more layers than others

Use when: Some tasks are more related to each other
```

## Loss Function Design

### Simple Weighted Sum

```python
# Multi-label classification (all binary outputs)
total_loss = (
    w1 * loss_task1 +  # e.g., w1 = 1.0
    w2 * loss_task2 +  # e.g., w2 = 0.5
    w3 * loss_task3 +  # e.g., w3 = 1.0
    w4 * loss_task4    # e.g., w4 = 0.8
)

# Weights can balance:
# - Task importance
# - Dataset sizes
# - Task difficulty
```

### Automatic Task Weighting

```python
# Learn task weights during training
# Based on homoscedastic uncertainty
# (Kendall et al., 2018)

# Each task has learned weight sigma
loss = (
    loss_task1 / (2 * sigma1**2) + log(sigma1) +
    loss_task2 / (2 * sigma2**2) + log(sigma2) +
    loss_task3 / (2 * sigma3**2) + log(sigma3) +
    loss_task4 / (2 * sigma4**2) + log(sigma4)
)

# Network learns optimal task weighting automatically
```

### Gradient Balancing

```python
# GradNorm: balance gradient magnitudes across tasks
# Ensures no single task dominates training

# Adjust task weights so gradient norms are balanced
```

## Practical Examples

### Example 1: Autonomous Driving

```
Single network for:
1. Object detection (cars, pedestrians, cyclists)
2. Lane detection
3. Drivable area segmentation
4. Traffic sign classification

Setup:
  Backbone: ResNet-50
  Shared: First 40 layers
  Task-specific: Last 10 layers + heads

Benefits:
- All tasks share visual understanding
- Pedestrian detector helps car detector (similar shapes)
- Lane detection helps drivable area
- More efficient than 4 separate models
```

### Example 2: Medical Diagnosis

```
Chest X-ray interpretation:
1. Detect pneumonia
2. Detect fractures
3. Detect cardiomegaly (enlarged heart)
4. Detect pleural effusion

Setup:
  Backbone: DenseNet-121
  Input: Same chest X-ray
  Output: 4 binary classifications

Benefits:
- Limited labeled data per condition
- All tasks need similar low-level features (lung anatomy)
- Regularization prevents overfitting
- Faster inference (one forward pass for all diagnoses)
```

### Example 3: NLP Multi-Task

```
Text understanding:
1. Named Entity Recognition (token classification)
2. Part-of-Speech Tagging (token classification)
3. Sentiment Analysis (sequence classification)
4. Question Answering (span extraction)

Setup:
  Backbone: BERT
  Shared: BERT encoder (all 12 layers)
  Task-specific: Different heads per task

Benefits:
- All tasks benefit from language understanding
- NER and POS tagging are highly related
- Sentiment analysis helps with understanding emotional context
```

## Training Strategies

### Strategy 1: Uniform Sampling

```python
# Sample equally from all tasks each batch

batch_size = 32
per_task = batch_size // num_tasks  # 8 per task

batch = []
for task in tasks:
    batch.extend(sample(task.data, per_task))

# Pros: Simple, fair to all tasks
# Cons: Ignores dataset size imbalance
```

### Strategy 2: Proportional Sampling

```python
# Sample proportional to dataset size

task_probs = [len(task.data) for task in tasks]
task_probs = task_probs / sum(task_probs)

batch = []
for task, prob in zip(tasks, task_probs):
    n_samples = int(batch_size * prob)
    batch.extend(sample(task.data, n_samples))

# Pros: Natural for imbalanced data
# Cons: Small tasks get little signal
```

### Strategy 3: Temperature Sampling

```python
# Scale dataset sizes with temperature

T = 0.5  # Temperature < 1 helps small datasets
task_probs = [len(task.data) ** (1/T) for task in tasks]
task_probs = task_probs / sum(task_probs)

# T = 1: Proportional (normal)
# T < 1: Upweight smaller datasets
# T > 1: Downweight smaller datasets

# Pros: Flexible, helps small tasks
# Cons: Need to tune T
```

### Strategy 4: Curriculum Learning

```python
# Start with easier tasks, gradually add harder ones

# Week 1-2: Train on Task 1 & 2 (easier)
# Week 3-4: Add Task 3
# Week 5+: Add Task 4

# Or start with easy examples from all tasks

# Pros: Can improve final performance
# Cons: More complex training schedule
```

## Practical Implementation

### PyTorch Example

```python
import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_per_task):
        super().__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Task-specific heads
        self.task1 = nn.Linear(256, num_classes_per_task[0])
        self.task2 = nn.Linear(256, num_classes_per_task[1])
        self.task3 = nn.Linear(256, num_classes_per_task[2])
        self.task4 = nn.Linear(256, num_classes_per_task[3])
        
    def forward(self, x):
        # Shared features
        features = self.shared(x)
        features = features.view(features.size(0), -1)
        
        # Task outputs
        out1 = self.task1(features)
        out2 = self.task2(features)
        out3 = self.task3(features)
        out4 = self.task4(features)
        
        return out1, out2, out3, out4

# Training loop
model = MultiTaskModel([2, 2, 2, 2])  # All binary tasks
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch in dataloader:
        images, labels1, labels2, labels3, labels4 = batch
        
        # Forward pass
        out1, out2, out3, out4 = model(images)
        
        # Compute losses
        loss1 = criterion(out1, labels1)
        loss2 = criterion(out2, labels2)
        loss3 = criterion(out3, labels3)
        loss4 = criterion(out4, labels4)
        
        # Weighted combination
        total_loss = (
            1.0 * loss1 +
            1.0 * loss2 +
            0.5 * loss3 +  # Less important task
            0.5 * loss4
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

### With Uncertainty Weighting

```python
class MultiTaskUncertainty(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # ... backbone and heads ...
        
        # Learnable task weights (log variance)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, x):
        # ... get task outputs ...
        return outputs, self.log_vars

# Training
outputs, log_vars = model(images)
losses = [criterion(out, label) for out, label in zip(outputs, labels)]

# Uncertainty-weighted loss
total_loss = sum([
    loss / (2 * torch.exp(log_var)) + log_var / 2
    for loss, log_var in zip(losses, log_vars)
])
```

## Common Pitfalls

### ❌ Pitfall 1: Unbalanced Task Difficulty

```
Problem:
  Task 1: 50% accuracy → loss = 0.7
  Task 2: 95% accuracy → loss = 0.05
  
  Task 1 dominates gradient, Task 2 barely trains

Solution:
  - Normalize losses
  - Use per-task learning rates
  - Use uncertainty weighting
```

### ❌ Pitfall 2: Conflicting Gradients

```
Problem:
  Task 1 gradient: Increase feature A
  Task 2 gradient: Decrease feature A
  Tasks fight each other

Solution:
  - Project conflicting gradients (PCGrad)
  - Use gradient surgery
  - Accept that some tasks may not benefit
```

### ❌ Pitfall 3: Negative Transfer

```
Problem:
  Multi-task model performs WORSE than single-task

Causes:
  - Tasks too unrelated
  - One task dominates training
  - Insufficient model capacity

Solution:
  - Verify tasks are actually related
  - Balance task contributions
  - Use larger model
  - Consider task clustering (subset of tasks together)
```

### ❌ Pitfall 4: Imbalanced Data

```
Problem:
  Task 1: 1,000,000 examples
  Task 2: 1,000 examples
  Task 2 barely gets trained

Solution:
  - Use temperature sampling (T < 1)
  - Oversample small task
  - Use higher loss weight for small task
```

## When to Use Multi-Task vs Transfer Learning

### Multi-Task Learning

```
✅ Use when:
- Multiple related tasks to solve simultaneously
- All tasks are important in production
- Tasks have similar data amounts
- Want single model for efficiency

Example: Self-driving car needs to detect pedestrians,
         cars, lanes, signs all at once
```

### Transfer Learning

```
✅ Use when:
- One main task that matters
- Source task has much more data
- Tasks can be done sequentially
- Want maximum performance on one task

Example: ImageNet (source) → Medical imaging (target)
```

### Both!

```
✅ Use both when:
1. Pre-train with multi-task on large dataset
2. Fine-tune on specific target task

Example:
  Pre-train: Multiple NLP tasks on Wikipedia
  Transfer: Fine-tune on your specific task
```

## Key Takeaways

1. **Multi-task learning shares representations** across related tasks

2. **Works best when:**
   - Tasks are related (share low-level features)
   - Data amounts are balanced
   - Network capacity is sufficient

3. **Benefits:**
   - Regularization (prevents overfitting)
   - Data amplification (all tasks learn from each other)
   - Efficiency (one model for multiple tasks)

4. **Key design decisions:**
   - How much sharing? (hard vs soft parameter sharing)
   - How to weight tasks? (manual vs automatic)
   - How to sample data? (uniform vs proportional)

5. **Watch out for:**
   - Negative transfer (worse than single-task)
   - Task imbalance (one dominates)
   - Conflicting gradients

6. **Common pattern:** Shared backbone + task-specific heads

7. **Multi-task ≠ Transfer learning**
   - Multi-task: Train multiple tasks together
   - Transfer: Pre-train one task, fine-tune on another

## Decision Framework

```
Should I use multi-task learning?

Are tasks related? (share low-level features)
├─ NO → Train separately or use transfer learning
└─ YES → Continue
    │
    Do all tasks matter in production?
    ├─ NO → Use transfer learning instead
    │        (pre-train on data-rich task, transfer to target)
    └─ YES → Continue
        │
        Are data amounts balanced? (within 10x)
        ├─ NO → Use task weighting or temperature sampling
        └─ YES → Continue
            │
            Do you have sufficient model capacity?
            ├─ NO → Use larger model
            └─ YES → Use multi-task learning! ✅
```

---

**Related:** [Transfer Learning](transfer_learning.md), [End-to-End Learning](end_to_end_learning.md), [Orthogonalization](orthogonalization.md)

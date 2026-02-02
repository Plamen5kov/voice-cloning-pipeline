# Deep Learning Concepts & Terminology

A comprehensive guide to understanding the fundamental concepts in machine learning and deep learning.

---

## Table of Contents
1. [Gradients](#gradients)
2. [Backpropagation](#backpropagation)
3. [Loss Functions](#loss-functions)
4. [Optimizers](#optimizers)
5. [Activation Functions](#activation-functions)
6. [Overfitting & Underfitting](#overfitting--underfitting)
7. [Regularization](#regularization)
8. [Normalization](#normalization)
9. [Learning Rate](#learning-rate)
10. [Epochs, Batches & Iterations](#epochs-batches--iterations)
11. [Forward & Backward Pass](#forward--backward-pass)
12. [Model Capacity](#model-capacity)
13. [Generalization](#generalization)
14. [Hyperparameters](#hyperparameters)

---

## Gradients

### What Are Gradients?

**Mathematically**: A gradient is the derivative of a function. It tells you **how much** a function's output changes when you change its input, and **in which direction**.

**In neural networks**: Gradients tell you how much each weight should change to reduce the loss (error).

### Simple Analogy

Imagine you're lost in foggy mountains and want to get to the valley (lowest point):

```
🏔️  You are here (high loss)
 ⬇️  Gradient points downhill
🌲  Go this direction to descend
⬇️  Keep following the slope
🏡  Valley (low loss) ✓
```

The **gradient** is like a compass that points downhill. In neural networks:
- **Loss** = your altitude (how wrong the model is)
- **Weights** = your position on the mountain
- **Gradient** = which direction to walk to descend
- **Learning rate** = how big a step to take

### How Gradients Work in Practice

```python
# 1. Forward pass - compute predictions
output = model(data)
loss = criterion(output, target)  # How wrong are we?

# 2. Backward pass - compute gradients
loss.backward()  # ← THE MAGIC HAPPENS HERE

# 3. Update weights using gradients
optimizer.step()  # weight = weight - learning_rate * gradient
```

### What `loss.backward()` Does

It computes: **"If I change weight W by a tiny amount, how much does the loss change?"**

For each parameter, it calculates:
```
gradient = ∂loss/∂weight
```

This uses the **chain rule** from calculus to propagate errors backward through the network.

### Concrete Example

Say your model predicts "7" but the true digit is "3":

```
Layer 2 (output):  [0.1, 0.1, 0.1, 0.05, ...., 0.8, 0.1]
                                    ↑true      ↑predicted
                                    digit 3    digit 7

Loss = high (we're wrong!)

Gradients computed backward:
- Layer 2 weights: "Increase weight for digit 3, decrease for digit 7"
- Layer 1 weights: "Adjust features that led to this mistake"
- All weights get a gradient telling them how to improve
```

### The Update Formula

```python
# What the optimizer does internally:
for each weight w in model:
    w = w - learning_rate * gradient
    
# Example with actual numbers:
w = 0.5          # Current weight value
gradient = 0.3   # Computed by backward()
lr = 0.001       # Learning rate

w_new = 0.5 - (0.001 * 0.3) = 0.4997  # Updated weight
```

### Why `optimizer.zero_grad()`?

Gradients **accumulate** by default in PyTorch:

```python
# Batch 1:
loss.backward()  # gradients = [0.1, 0.2, 0.3, ...]

# Batch 2 (without zero_grad):
loss.backward()  # gradients = [0.1+0.4, 0.2+0.1, 0.3+0.5, ...]  ❌ WRONG!

# Correct way:
optimizer.zero_grad()  # gradients = [0, 0, 0, ...]
loss.backward()         # gradients = [0.4, 0.1, 0.5, ...]  ✓
```

### Common Gradient Problems

1. **Vanishing Gradients**
   - Gradients become tiny (near 0) in deep networks
   - Early layers don't learn
   - Solution: ReLU activation, ResNet skip connections, better initialization

2. **Exploding Gradients**
   - Gradients become huge
   - Weights jump wildly, training diverges
   - Solution: Gradient clipping, batch normalization, lower learning rate

3. **Dead Neurons**
   - If ReLU neuron always outputs 0, its gradient is 0
   - It never learns (permanently "dead")
   - Solution: LeakyReLU, proper weight initialization

---

## Backpropagation

### What Is It?

**Backpropagation** = "backward propagation of errors" - the algorithm that efficiently computes gradients for all weights in a neural network.

### How It Works

1. **Forward pass**: Input flows through network, produces output
2. **Compute loss**: Compare output to true label
3. **Backward pass**: Error flows backward, computing gradients via chain rule
4. **Update weights**: Use gradients to adjust weights

### Why It's Revolutionary

Before backprop (1986), training neural networks was impractical. Backprop made it possible to:
- Compute gradients for millions of parameters efficiently
- Train networks with many layers
- Enable modern deep learning

### The Chain Rule

The magic of backprop is using calculus chain rule:

```
If z = f(g(x)), then:
dz/dx = (dz/dg) × (dg/dx)

In neural networks:
loss → layer3 → layer2 → layer1 → input

∂loss/∂layer1 = (∂loss/∂layer3) × (∂layer3/∂layer2) × (∂layer2/∂layer1)
```

PyTorch does this automatically with `loss.backward()`!

### Visual Example

```
Forward:
Input → [W1] → Hidden → [W2] → Output → Loss
  3       0.5     1.5      0.8     1.2     0.4

Backward (compute gradients):
Loss (0.4) → ∂L/∂W2=0.1 → ∂L/∂W1=0.05
     ←          ←              ←
```

---

## Loss Functions

### What Is a Loss Function?

A **loss function** (or cost function) measures how wrong your model's predictions are. Lower loss = better predictions.

### Common Loss Functions

#### 1. **CrossEntropyLoss** (Classification)
```python
criterion = nn.CrossEntropyLoss()
```

**When to use**: Multi-class classification (MNIST, ImageNet, etc.)

**What it does**: 
- Combines LogSoftmax + Negative Log Likelihood
- Penalizes confident wrong predictions heavily
- Outputs probability distribution over classes

**Math**:
```
Loss = -log(probability of correct class)

If true class has:
- 0.9 probability: Loss = -log(0.9) = 0.10 (good!)
- 0.1 probability: Loss = -log(0.1) = 2.30 (bad!)
```

#### 2. **MSELoss** (Regression)
```python
criterion = nn.MSELoss()
```

**When to use**: Predicting continuous values (prices, temperatures)

**What it does**: Mean Squared Error = average of (prediction - truth)²

**Math**:
```
Loss = (1/n) × Σ(prediction - truth)²
```

#### 3. **BCELoss** (Binary Classification)
```python
criterion = nn.BCELoss()
```

**When to use**: Two-class problems (spam/not spam, cat/dog)

**What it does**: Binary Cross Entropy

### Why Different Loss Functions?

- **Classification**: Need probability distributions → CrossEntropy
- **Regression**: Need continuous values → MSE, MAE
- **Binary**: Special efficient case → BCE

Using the wrong loss function won't work (can't use MSE for classification)!

---

## Optimizers

### What Is an Optimizer?

An **optimizer** is the algorithm that updates weights using gradients. It decides:
- How much to change each weight
- In what direction
- How to adapt over time

### Common Optimizers

#### 1. **SGD (Stochastic Gradient Descent)**
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Pros**: Simple, works well with momentum, sometimes generalizes better
**Cons**: Requires careful learning rate tuning
**When to use**: Simple problems, or when other optimizers overfit

**Update rule**:
```
weight = weight - learning_rate × gradient
```

#### 2. **Adam (Adaptive Moment Estimation)** ⭐ Most Popular
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Pros**: 
- Adapts learning rate per parameter
- Works well out-of-the-box
- Combines benefits of momentum + RMSprop
- Good default choice

**Cons**: Can overfit more than SGD, uses more memory

**When to use**: Almost always! Great default for most problems

**What makes it adaptive**:
- Tracks moving average of gradients (momentum)
- Tracks moving average of squared gradients (adaptive LR)
- Different effective learning rate per parameter

#### 3. **AdamW**
```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Pros**: Adam with better weight decay (regularization)
**When to use**: Training transformers, large models (BERT, GPT)

#### 4. **RMSprop**
```python
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```

**When to use**: Recurrent neural networks (RNNs)

### Optimizer Comparison

| Optimizer | Learning Rate | Speed | Memory | Best For |
|-----------|---------------|-------|--------|----------|
| SGD | Sensitive | Fast | Low | Simple problems |
| Adam | Forgiving | Fast | High | Most problems ⭐ |
| AdamW | Forgiving | Fast | High | Transformers |

### Key Hyperparameters

- **learning_rate**: How big a step (0.001 is common for Adam)
- **momentum**: Helps accelerate in consistent directions
- **weight_decay**: Regularization to prevent overfitting

---

## Activation Functions

### What Are Activation Functions?

**Activation functions** introduce non-linearity into neural networks. Without them, the network would just be linear regression, no matter how many layers!

### Why Non-Linearity Matters

```python
# Without activation (just linear):
layer1 = W1 × x
layer2 = W2 × layer1 = W2 × (W1 × x) = (W2×W1) × x

# This is equivalent to single layer!
# Can't learn complex patterns

# With activation:
layer1 = relu(W1 × x)  # Non-linear!
layer2 = W2 × layer1
# Now can learn complex, non-linear patterns
```

### Common Activation Functions

#### 1. **ReLU (Rectified Linear Unit)** ⭐ Most Popular
```python
relu = nn.ReLU()
```

**Formula**: `f(x) = max(0, x)`

**Graph**:
```
   |    /
   |   /
   |  /
___|_/_____
   |
```

**Pros**: 
- Simple and fast
- Doesn't saturate (no vanishing gradient for positive values)
- Sparse activation (many neurons output 0)
- Works extremely well in practice

**Cons**:
- "Dead ReLU" problem (neurons can get stuck at 0)
- Not zero-centered

**When to use**: Default choice for hidden layers in most networks

#### 2. **Tanh (Hyperbolic Tangent)**
```python
tanh = nn.Tanh()
```

**Formula**: `f(x) = (e^x - e^-x) / (e^x + e^-x)`

**Range**: [-1, 1]

**Graph**:
```
    1 |     ___
      |    /
    0 |___/
      |  /
   -1 |_/
```

**Pros**:
- Zero-centered (better than sigmoid)
- Smooth gradient

**Cons**:
- Saturates (vanishing gradient problem)
- Slower than ReLU

**When to use**: RNNs, older networks

#### 3. **Sigmoid**
```python
sigmoid = nn.Sigmoid()
```

**Formula**: `f(x) = 1 / (1 + e^-x)`

**Range**: [0, 1]

**Pros**:
- Outputs probability-like values
- Smooth

**Cons**:
- Strong saturation (gradients near 0 for large |x|)
- Not zero-centered
- Rarely used in hidden layers anymore

**When to use**: Binary classification output layer

#### 4. **LeakyReLU**
```python
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
```

**Formula**: `f(x) = max(0.01x, x)`

**Pros**:
- Fixes "dead ReLU" problem
- Allows small gradient when x < 0

**When to use**: When you experience dead ReLU issues

#### 5. **GELU (Gaussian Error Linear Unit)**
```python
gelu = nn.GELU()
```

**Pros**: 
- Used in transformers (BERT, GPT)
- Smooth approximation of ReLU

**When to use**: Transformer models

### Activation Function Comparison

| Function | Range | Saturates? | Speed | Best For |
|----------|-------|------------|-------|----------|
| ReLU | [0, ∞) | No (right) | Fast | Default choice ⭐ |
| Tanh | [-1, 1] | Yes | Slow | RNNs |
| Sigmoid | [0, 1] | Yes | Slow | Binary output |
| LeakyReLU | (-∞, ∞) | No | Fast | Fixing dead ReLU |
| GELU | (-∞, ∞) | No | Medium | Transformers |

### Where to Use Them

- **Hidden layers**: ReLU (or LeakyReLU, GELU)
- **Binary classification output**: Sigmoid
- **Multi-class output**: None (CrossEntropyLoss includes softmax)
- **Regression output**: None (linear)

---

## Overfitting & Underfitting

### The Goldilocks Problem of Machine Learning

#### Underfitting (Too Simple)
```
Model: y = mx + b (straight line)
Data: Complex curve

Result: High training error, high test error
Problem: Model not complex enough to capture patterns
```

**Signs**:
- Poor performance on both training and test sets
- Training and validation accuracy both low and similar
- Model too simple for the problem

**Solutions**:
- Use more complex model (more layers, more neurons)
- Train longer
- Remove regularization
- Add more features

#### Overfitting (Too Complex)
```
Model: Memorizes every training example
Data: Training set perfectly, test set poorly

Result: Low training error, HIGH test error
Problem: Model memorized noise, doesn't generalize
```

**Signs**:
- Perfect/near-perfect training accuracy
- Much worse test accuracy
- Large gap between training and validation curves

**Solutions**:
- Get more training data
- Use regularization (dropout, weight decay)
- Simplify model (fewer layers/neurons)
- Early stopping
- Data augmentation

#### Just Right (Good Generalization)
```
Training accuracy: 97%
Test accuracy: 96%

Small gap = good generalization!
```

### Visual Example

```
Training Loss vs Validation Loss:

Underfit:
Train ─────────  (high, plateau)
Val   ─────────  (high, plateau)

Good Fit:
Train ───────╲╲  (decreasing)
Val   ────────╲  (decreasing, close to train)

Overfit:
Train ─────────╲╲╲╲ (very low)
Val   ─────╱╱╱╱     (starts decreasing, then increases!)
```

---

## Regularization

### What Is Regularization?

Techniques to prevent overfitting by constraining the model's complexity.

### Common Regularization Techniques

#### 1. **Dropout**
```python
dropout = nn.Dropout(p=0.5)
```

**How it works**: Randomly "drop" (set to 0) some neurons during training

**Why it works**:
- Forces network to not rely on any single neuron
- Creates ensemble effect
- Prevents co-adaptation of neurons

**When to use**: Between layers in deep networks

**Typical values**: 0.2 to 0.5

```python
# Example:
x = self.fc1(x)
x = self.relu(x)
x = self.dropout(x)  # 50% of neurons set to 0
x = self.fc2(x)
```

#### 2. **Weight Decay (L2 Regularization)**
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

**How it works**: Penalizes large weights

**Math**: `Loss_total = Loss + λ × Σ(weights²)`

**Why it works**: Encourages smaller, more distributed weights

**Typical values**: 0.0001 to 0.01

#### 3. **Early Stopping**

**How it works**: Stop training when validation loss stops improving

```python
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in epochs:
    train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break
```

#### 4. **Data Augmentation**

**How it works**: Create variations of training data

For images:
- Random crops
- Flips
- Rotations
- Color jittering

```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])
```

**Why it works**: More diverse data = better generalization

#### 5. **Batch Normalization**
```python
bn = nn.BatchNorm1d(128)
```

**How it works**: Normalizes activations within each batch

**Benefits**:
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization (slight noise from batch statistics)
- Makes training more stable

---

## Normalization

### Why Normalize Data?

**Problem**: Raw data has different scales
```
Pixel values: 0-255
Age: 0-100
Income: 0-1,000,000
```

**Solution**: Scale to similar ranges

### Types of Normalization

#### 1. **Input Normalization** (Standardization)
```python
# For each feature: (value - mean) / std
transform = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
```

**Result**: Mean=0, Std=1

**Benefits**:
- Faster convergence
- More stable gradients
- All features contribute equally

#### 2. **Batch Normalization**
```python
bn = nn.BatchNorm1d(num_features=128)
```

**Where**: Between layers during training

**What it does**: Normalizes activations per batch

**Benefits**:
- Allows higher learning rates
- Reduces dependence on initialization
- Acts as regularization

#### 3. **Layer Normalization**

**Where**: Used in transformers (BERT, GPT)

**Difference from BatchNorm**: Normalizes across features (not batch)

**When to use**: When batch size is small or variable (NLP)

### MNIST Normalization Example

```python
# MNIST mean and std (computed from training set)
mean = 0.1307
std = 0.3081

# Original pixel: 127 (middle gray)
normalized = (127/255 - 0.1307) / 0.3081 = 1.03

# Original pixel: 255 (white)
normalized = (255/255 - 0.1307) / 0.3081 = 2.82

# Original pixel: 0 (black)
normalized = (0/255 - 0.1307) / 0.3081 = -0.42
```

Now values centered around 0 with similar scale!

---

## Learning Rate

### What Is Learning Rate?

The **learning rate** controls how big a step we take when updating weights.

```python
weight_new = weight_old - learning_rate × gradient
```

### The Goldilocks Problem

#### Too High (lr = 0.1)
```
Loss over time:
 |    *
 |  *   *
 | *  *  *
 |*  *  *
 |_________
 
Result: Bouncing around, never converges, might diverge
```

#### Too Low (lr = 0.00001)
```
Loss over time:
 |*
 |*
 |*
 |*___________
 
Result: Very slow training, might get stuck
```

#### Just Right (lr = 0.001)
```
Loss over time:
 |*
 | *
 |  *
 |   *___
 |_________
 
Result: Steady decrease, converges nicely
```

### Typical Values

| Optimizer | Typical LR | Range |
|-----------|------------|-------|
| SGD | 0.01-0.1 | 0.001-0.5 |
| Adam | 0.001 | 0.0001-0.01 |
| AdamW | 0.001 | 0.0001-0.01 |

### Learning Rate Schedules

Instead of fixed learning rate, change it during training:

#### 1. **Step Decay**
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# Every 10 epochs: lr = lr × 0.1
```

#### 2. **Exponential Decay**
```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# Each epoch: lr = lr × 0.95
```

#### 3. **Cosine Annealing**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# Smoothly decrease LR following cosine curve
```

#### 4. **ReduceLROnPlateau** ⭐ Popular
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
# Reduce LR when validation loss plateaus
```

### Learning Rate Warmup

Start with small LR, gradually increase:

```python
# Common in transformer training
# Prevents instability in early training
for epoch in range(warmup_epochs):
    lr = initial_lr * (epoch / warmup_epochs)
    set_learning_rate(optimizer, lr)
```

### Finding the Right Learning Rate

**Learning Rate Finder**:
1. Start with very small LR
2. Gradually increase
3. Plot loss vs LR
4. Choose LR where loss decreases fastest

```
Loss
 |     *
 |    * *
 |   *   *
 |  *     **
 | *        ***
 |*____________***
   ↑           ↑
   Too small   Too large
       ↑
    Choose this!
```

---

## Epochs, Batches & Iterations

### Terminology

#### **Epoch**
One complete pass through the entire training dataset.

```python
# If you have 60,000 samples and train for 10 epochs:
# The model sees all 60,000 samples 10 times
```

#### **Batch**
A subset of training samples processed together.

```python
# With 60,000 samples and batch_size=64:
# Each batch contains 64 samples
# Number of batches per epoch = 60,000 / 64 = 937 batches
```

#### **Iteration**
One forward + backward pass on one batch. Same as "step".

```python
# 10 epochs × 937 batches = 9,370 iterations total
```

### Visual Example

```
Dataset: 1000 samples
Batch size: 100
Epochs: 3

Epoch 1:
  Batch 1: samples 0-99     (iteration 1)
  Batch 2: samples 100-199  (iteration 2)
  ...
  Batch 10: samples 900-999 (iteration 10)

Epoch 2:
  Batch 1: samples 0-99     (iteration 11)
  Batch 2: samples 100-199  (iteration 12)
  ...
  Batch 10: samples 900-999 (iteration 20)

Epoch 3:
  Batch 1: samples 0-99     (iteration 21)
  ...
  Batch 10: samples 900-999 (iteration 30)

Total: 3 epochs, 30 iterations
```

### Why Batches?

#### **Why not process all data at once?**
- Too much memory (60,000 images won't fit in GPU)
- Slower learning (rare updates)
- No benefit from gradient noise

#### **Why not process one at a time?**
- Very slow (can't parallelize on GPU)
- Very noisy gradients
- Unstable training

#### **Batch size = 64 is Goldilocks**
- Fits in GPU memory
- Good parallelization
- Balanced gradient noise
- Stable training

### Batch Size Effects

| Batch Size | Memory | Speed | Generalization | When to Use |
|------------|--------|-------|----------------|-------------|
| 1 (SGD) | Very low | Slow | Good | Almost never |
| 32 | Low | Fast | Good | Small datasets, limited GPU |
| 64 | Medium | Fast | Good | **Default choice** ⭐ |
| 128 | Medium | Fast | Good | Large datasets |
| 256+ | High | Fastest | Can be worse | Distributed training |

### How Many Epochs?

**Too few**: Underfitting (haven't learned enough)
**Too many**: Overfitting (memorizing training data)

**How to decide**:
- Watch validation loss
- Stop when it stops improving (early stopping)
- Typical: 10-100 epochs for simple problems, 100+ for complex ones

---

## Forward & Backward Pass

### Forward Pass

**What happens**: Data flows forward through the network to produce output

```python
# Example with 2-layer network:
input = [1, 2, 3]  # Shape: (3,)

# Layer 1
hidden = relu(W1 @ input + b1)  # Shape: (128,)

# Layer 2
output = W2 @ hidden + b2  # Shape: (10,)

# Loss
loss = cross_entropy(output, target)
```

**Each layer**:
1. Multiply by weights
2. Add bias
3. Apply activation
4. Pass to next layer

**Purpose**: Make predictions

### Backward Pass

**What happens**: Error flows backward to compute gradients

```python
# Starts from loss
loss.backward()  # Triggers backpropagation

# Computes gradients for all parameters:
# ∂loss/∂W2, ∂loss/∂b2, ∂loss/∂W1, ∂loss/∂b1
```

**Each layer (going backward)**:
1. Compute gradient w.r.t. its output
2. Compute gradient w.r.t. its weights
3. Compute gradient w.r.t. its input
4. Pass gradient to previous layer

**Purpose**: Update weights to reduce loss

### Complete Training Step

```python
# 1. FORWARD PASS
optimizer.zero_grad()         # Clear old gradients
output = model(input)         # Forward pass
loss = criterion(output, target)

# 2. BACKWARD PASS
loss.backward()               # Compute gradients

# 3. UPDATE
optimizer.step()              # Update weights

# Repeat for all batches in all epochs
```

### Computational Graph

PyTorch builds a graph during forward pass:

```
Forward:
input → W1 → relu → W2 → output → loss
        ↓      ↓     ↓      ↓       ↓
      saves  saves saves saves   (for backward)

Backward:
input ← W1 ← relu ← W2 ← output ← loss
        ↑      ↑     ↑      ↑       ↑
      ∂L/∂W1 ∂L/∂a ∂L/∂W2  ∂L/∂o   1
```

This is why `torch.no_grad()` saves memory - it doesn't build the graph!

---

## Model Capacity

### What Is Model Capacity?

The **capacity** of a model is its ability to fit complex patterns. Think of it as the model's "learning power".

### What Determines Capacity?

1. **Number of parameters**
   - More parameters = more capacity
   - MNIST model: ~100K parameters
   - GPT-3: 175 billion parameters

2. **Number of layers (depth)**
   - Deeper = can learn hierarchical features
   - 2 layers: simple patterns
   - 100 layers: very complex patterns

3. **Width (neurons per layer)**
   - Wider = more capacity per layer
   - 128 neurons vs 1024 neurons

### Capacity vs Performance

```
           Performance
                ↑
                |        ________ (overfitting)
                |      /
                |    /  ← optimal
                |  /
                |/
                |________________→ Model Capacity
                
Too little: Underfits (can't learn patterns)
Just right: Good generalization
Too much: Overfits (memorizes training data)
```

### Examples

#### Low Capacity
```python
# Simple network
model = nn.Sequential(
    nn.Linear(784, 10)  # Just 7,850 parameters
)
# Might underfit - too simple for complex patterns
```

#### Medium Capacity (Good for MNIST)
```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
# ~101K parameters - just right for MNIST
```

#### High Capacity
```python
model = nn.Sequential(
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
)
# ~2M parameters - overkill for MNIST, likely overfits
```

### How to Choose?

1. **Start simple** - Baseline model
2. **Check for underfitting** - If training accuracy is low, increase capacity
3. **Check for overfitting** - If gap between train/test is large, decrease capacity or regularize
4. **Iterate** - Adjust based on results

---

## Generalization

### What Is Generalization?

**Generalization** is the model's ability to perform well on new, unseen data (not just memorizing training data).

### The Core Challenge of ML

```
Training: Model sees 60,000 examples
Testing: Model must work on new examples
Goal: Learn patterns, not memorize

Good: Learned "what makes a '7' a '7'"
Bad: Memorized "training sample 42 is a '7'"
```

### Measuring Generalization

**Training accuracy**: How well model memorized
**Test accuracy**: How well model generalizes

```
Scenario 1 - Good Generalization:
Train: 97%
Test:  96%
Gap:   1% ✓

Scenario 2 - Poor Generalization (Overfitting):
Train: 99.9%
Test:  85%
Gap:   14.9% ✗
```

### Why Models Fail to Generalize

1. **Not enough data**
   - Model hasn't seen enough examples
   - Solution: Get more data, data augmentation

2. **Too complex model**
   - Model has capacity to memorize
   - Solution: Simpler model, regularization

3. **Data distribution shift**
   - Test data different from training
   - Solution: Better data collection, domain adaptation

4. **Overfitting to noise**
   - Learning random fluctuations
   - Solution: Regularization, more data

### Improving Generalization

1. **More training data** - More examples of the pattern
2. **Data augmentation** - Create variations
3. **Regularization** - Dropout, weight decay
4. **Simpler model** - Reduce capacity
5. **Early stopping** - Stop before overfitting
6. **Cross-validation** - Ensure consistent performance
7. **Ensemble methods** - Combine multiple models

### The Bias-Variance Tradeoff

```
High Bias (Underfitting):
- Model too simple
- Misses patterns
- Both train and test error high

High Variance (Overfitting):
- Model too complex
- Captures noise
- Train error low, test error high

Sweet Spot:
- Balanced complexity
- Captures signal, ignores noise
- Similar train and test error
```

---

## Hyperparameters

### What Are Hyperparameters?

**Hyperparameters** are settings you choose before training (not learned from data).

**Parameters** (learned): Weights and biases
**Hyperparameters** (chosen): Learning rate, batch size, number of layers, etc.

### Critical Hyperparameters

#### 1. **Learning Rate**
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- Most important hyperparameter!
- Too high: Training unstable
- Too low: Training too slow
- Typical: 0.001 for Adam

#### 2. **Batch Size**
```python
DataLoader(dataset, batch_size=64)
```
- Affects memory and convergence
- Typical: 32, 64, 128, 256

#### 3. **Number of Epochs**
```python
for epoch in range(10):
```
- How long to train
- Use early stopping instead of fixed number

#### 4. **Architecture**
```python
nn.Linear(784, 128)  # Hidden size = 128
```
- Number of layers
- Neurons per layer
- Type of layers (Conv, LSTM, etc.)

#### 5. **Optimizer Choice**
```python
optim.Adam vs optim.SGD
```
- Adam: Good default
- SGD: Sometimes better generalization

#### 6. **Regularization**
```python
nn.Dropout(p=0.5)
optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
```
- Dropout probability
- Weight decay strength

### How to Tune Hyperparameters

#### **Manual Tuning**
1. Start with defaults
2. Change one at a time
3. Observe results
4. Iterate

#### **Grid Search**
```python
# Try all combinations
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        train_model(lr, bs)
        # Pick best
```

#### **Random Search**
- Randomly sample hyperparameter combinations
- Often better than grid search
- Can try more values

#### **Bayesian Optimization**
- Smart search using previous results
- Tools: Optuna, Ray Tune

### Priority Order (What to Tune First)

1. **Learning rate** - Biggest impact
2. **Architecture** - Capacity matters
3. **Batch size** - Memory/speed tradeoff
4. **Regularization** - If overfitting
5. **Optimizer** - Usually Adam is fine
6. **Other** - Usually less critical

### Good Defaults to Start With

```python
# Image classification (like MNIST)
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 64
epochs = 10-50 (with early stopping)
hidden_size = 128-512
dropout = 0.2-0.5

# Usually works well out of the box!
```

---

## Summary

This guide covers the essential concepts you need to understand deep learning:

- **Gradients** drive learning (how much to change weights)
- **Backpropagation** computes gradients efficiently
- **Loss functions** measure error (CrossEntropy, MSE)
- **Optimizers** update weights (Adam is usually best)
- **Activations** add non-linearity (ReLU is standard)
- **Overfitting** is memorization (regularize to fix)
- **Normalization** makes training stable
- **Learning rate** is critical to tune
- **Batches** balance memory and convergence
- **Generalization** is the ultimate goal

Master these concepts and you'll understand 90% of deep learning! The rest is practice and domain-specific knowledge.

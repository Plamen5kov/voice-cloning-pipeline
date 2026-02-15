# Hyperparameter Tuning Guide for Deep Neural Networks

A practical guide to choosing and tuning hyperparameters for your DNN models.

## Table of Contents
1. [Learning Rate](#learning-rate)
2. [Number of Iterations](#number-of-iterations)
3. [Network Architecture](#network-architecture)
4. [Other Important Hyperparameters](#other-important-hyperparameters)
5. [Systematic Tuning Strategy](#systematic-tuning-strategy)

---

## Learning Rate

### What It Does
The learning rate controls how big a step we take in the direction of the gradient during each update. It's arguably the most important hyperparameter to get right.

### Rule of Thumb

**Start with 0.01** (most common default)

**Typical ranges:**
- **0.1** - Very aggressive, only for simple/convex problems
- **0.01** - Good starting point for most problems ✓
- **0.001** - Conservative, use if 0.01 diverges
- **0.0001** - Very slow, rarely needed for small networks

### The Reasoning

**Why not always use a large learning rate?**
- Gradient descent can "overshoot" the minimum
- Cost might oscillate wildly or even diverge (increase)
- Like trying to find the bottom of a valley while taking giant leaps - you keep jumping over it

**Why not always use a tiny learning rate?**
- Training takes forever
- Might get stuck in local minima or saddle points
- Like taking baby steps - you'll get there eventually, but it's painfully slow

**The sweet spot:**
- Large enough to make meaningful progress each iteration
- Small enough to converge smoothly without overshooting

### Diagnostic Rules

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| Cost **increases** or **explodes** | Learning rate too high | Reduce by 10x (e.g., 0.01 → 0.001) |
| Cost **oscillates wildly** | Learning rate too high | Reduce by 5-10x |
| Cost **decreases very slowly** (< 0.01 per 100 iterations) | Learning rate too low | Increase by 2-5x (e.g., 0.001 → 0.005) |
| Cost **decreases smoothly** | Just right! ✓ | Keep it |
| Cost **decreases fast then stalls** | Could try slightly higher | Experiment with 1.5-2x |

### For Binary Classification (This Lab)

The default learning rate of **0.0075** is reasonable - slightly conservative.

**When to adjust:**
- If training is slow after kernel restart, try **0.01** or **0.015**
- If cost increases, drop to **0.001** or **0.005**
- Aim for cost to drop by ~0.1-0.2 in the first 100 iterations

### Pro Tips

**1. Grid Search (Best for finding optimal value)**
```python
learning_rates = [0.001, 0.005, 0.01, 0.05]
for lr in learning_rates:
    print(f"\nTesting learning_rate = {lr}")
    parameters, costs = L_layer_model(
        train_x, train_y, layers_dims, 
        learning_rate=lr, 
        num_iterations=200,  # Quick test
        print_cost=True
    )
```
Pick the one with fastest initial cost decrease that remains stable.

**2. Learning Rate Decay**
Start high, gradually reduce:
```python
# In L_layer_model, inside the loop:
current_lr = learning_rate / (1 + decay_rate * i / 1000)
```

**3. Adaptive Learning Rates**
Advanced optimizers (Adam, RMSprop) adjust learning rate automatically. We'll cover these in later modules!

---

## Number of Iterations

### What It Does
Iterations control how many times the model sees the entire training dataset (for batch gradient descent). More iterations = more learning, but with diminishing returns.

### Rule of Thumb

**Dataset-based starting points:**

| Dataset Size | Recommended Iterations | Reasoning |
|--------------|----------------------|-----------|
| **< 500 samples** | 500-1000 | Model sees all data quickly; converges fast; overfitting risk if too many |
| **500-5000 samples** | 1000-2000 | More patterns to learn; needs time to extract all useful information |
| **> 5000 samples** | 2000-5000+ | Many complex patterns; each update is small relative to dataset size |

**For this lab:** With 2,703 samples, start with **1500 iterations**.

### The Reasoning

**Small datasets (< 500 samples):**
- Model processes all examples in each iteration
- Fewer unique patterns to learn
- Learns quickly but can memorize (overfit) if trained too long
- Example: With 240 samples, might memorize in 500 iterations

**Medium datasets (500-5000):**
- More patterns to discover across more diverse examples
- Gradient updates more stable (less noisy)
- Takes longer to extract all useful patterns
- **Your 2,703-sample dataset falls here**

**Large datasets (> 5000):**
- Many complex patterns distributed across thousands of examples
- Each gradient update is relatively small
- Needs many passes to absorb all information
- Professional models often train for 10,000+ iterations

**Key insight**: More data = more information = more iterations needed to absorb it all.

### How to Decide (Practical Approaches)

**1. Visual Inspection (Simplest)**
After training, plot cost vs iterations:
```python
plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iterations (per hundreds)')
plt.title('Learning curve')
plt.show()
```

Stop when the curve **flattens** (diminishing returns):
- Still steep slope → train longer
- Nearly horizontal → you're done
- Increasing → overfitting or learning rate too high

**2. Monitor Print Statements**
Your notebook prints every 100 iterations. Watch for:
- Cost decreasing rapidly (e.g., 0.6 → 0.5 → 0.4) → keep training
- Cost barely changing (< 0.001 per 100 iterations) → converged
- Cost increasing → overfitting or learning rate issue

**3. Early Stopping (Best Practice)**
Stop automatically when improvement slows. Add to `L_layer_model`:

```python
# After the print statement in the iteration loop:
if i > 100 and i % 100 == 0:
    # Check if cost stopped improving
    if len(costs) >= 2 and costs[-1] >= costs[-2] * 0.999:
        print(f"Early stopping at iteration {i} (cost plateau)")
        break
```

**4. Validation Set Approach (Most Robust)**
Split your data into train/validation/test:
- Train on training set
- Evaluate on validation set each 100 iterations
- Stop when validation cost stops decreasing
- This prevents overfitting!

### Expected Behavior for This Lab

With 2,703 samples and learning_rate=0.0075:
- **Iterations 0-500**: Rapid cost decrease (0.693 → ~0.4)
- **Iterations 500-1200**: Gradual improvement (0.4 → ~0.3)
- **Iterations 1200-1500**: Slow convergence (0.3 → ~0.25-0.28)
- **Beyond 1500**: Diminishing returns or slight overfitting

**Recommendation**: Start with 1500. If cost still decreasing noticeably, try 2500. If it flattened at iteration 800, reduce to 1000 next time.

---

## Network Architecture

### Hidden Layer Sizes

**Current architecture:** `layers_dims = [n_x, 20, n_y]` (one hidden layer with 20 units)

**Rule of thumb:**
- Hidden units should be between input and output size
- Common choices: 10, 20, 50, 100
- Too few → underfitting (can't learn complex patterns)
- Too many → overfitting (memorizes training data)

**For this lab:**
- Input: 16,512 features (mel-spectrogram)
- Output: 1 (binary classification)
- 20 hidden units is reasonable for starting

**Experiment with:**
```python
layers_dims = [n_x, 50, n_y]    # More capacity
layers_dims = [n_x, 10, n_y]    # Simpler model
layers_dims = [n_x, 30, 15, n_y]  # Two hidden layers
```

### Number of Hidden Layers

**Current:** 1 hidden layer (2-layer network)

**Guidelines:**
- **1 hidden layer**: Good for simple patterns (linearly separable with transformation)
- **2-3 hidden layers**: Better for complex patterns
- **4+ layers**: "Deep" learning - for very complex problems (images, speech, etc.)

**Diminishing returns:** More layers ≠ always better
- Harder to train (vanishing gradients)
- Requires more data
- Takes longer

**For this lab:** Start with 1 layer. If accuracy plateaus < 80%, try 2 layers.

---

## Other Important Hyperparameters

### 1. Initialization Method

**Current:** Random initialization scaled by 0.01

**Why it matters:**
- Too large → activations explode
- Too small → gradients vanish
- Different scaling for different activation functions

**Better approaches:**
- **He initialization** (for ReLU): `np.random.randn(shape) * np.sqrt(2/n_prev)`
- **Xavier initialization** (for sigmoid/tanh): `np.random.randn(shape) * np.sqrt(1/n_prev)`

### 2. Regularization

**Not currently used** - helps prevent overfitting

**L2 Regularization (Ridge):**
```python
# Add to cost function:
L2_cost = (lambd / (2 * m)) * sum(np.sum(np.square(W)) for W in weights)
cost = cross_entropy_cost + L2_cost

# Modify gradients:
dW = dW + (lambd / m) * W
```

**Typical λ (lambda) values:** 0.01, 0.1, 0.5, 1.0

### 3. Dropout

Randomly "drop" units during training:
```python
# In forward propagation:
D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
A = A * D
A = A / keep_prob  # Inverted dropout
```

**Typical keep_prob:** 0.8 or 0.5 (drop 20% or 50%)

### 4. Batch Size

**Current:** Using entire dataset (batch gradient descent)

**Alternatives:**
- **Mini-batch gradient descent:** Use subset (32, 64, 128, 256 samples)
- Faster convergence
- More noise → can escape local minima
- Requires more iterations

---

## Systematic Tuning Strategy

### Step 1: Get a Baseline
```python
# Start with reasonable defaults:
layers_dims = [n_x, 20, n_y]
learning_rate = 0.01
num_iterations = 1500
```

### Step 2: Tune Learning Rate First
Most impactful parameter. Try: [0.001, 0.005, 0.01, 0.05, 0.1]

Pick the one that:
- Converges fastest
- Remains stable (no oscillation)

### Step 3: Tune Number of Iterations
- Plot the cost curve
- Find where it flattens
- Add 20% buffer for safety

### Step 4: Tune Architecture
- Try different hidden layer sizes: [10, 20, 50, 100]
- Try adding layers: [n_x, 50, 25, n_y]
- Pick simplest model that achieves target accuracy

### Step 5: Add Regularization (if overfitting)
If train accuracy >> test accuracy:
- Try L2 regularization: λ = [0.01, 0.1, 0.5]
- Or dropout: keep_prob = [0.8, 0.5]

### Step 6: Final Polish
- Fine-tune learning rate
- Try learning rate decay
- Experiment with better initialization

---

## Quick Reference Table

| Hyperparameter | Good Starting Value | Typical Range | Tune If... |
|----------------|-------------------|---------------|------------|
| **Learning rate** | 0.01 | 0.0001 - 0.1 | Cost increasing or too slow |
| **Iterations** | 1000-2000 | 500 - 5000 | Cost hasn't plateaued |
| **Hidden units** | 20-50 | 10 - 200 | Accuracy too low/high |
| **Hidden layers** | 1-2 | 1 - 4 | Accuracy plateaus low |
| **λ (L2 reg)** | 0 (off) | 0 - 1.0 | Train >> test accuracy |
| **Dropout** | 1.0 (off) | 0.5 - 0.9 | Overfitting persists |

---

## Expected Results for This Lab

With the baseline configuration after kernel restart:
- **Initial cost:** ~0.69 (random guessing for binary)
- **Final cost:** 0.25 - 0.35
- **Train accuracy:** 85-95%
- **Test accuracy:** 80-90%
- **Train-test gap:** < 10% (no severe overfitting)

If you're not hitting these targets, revisit the hyperparameters using this guide!

---

## Further Reading

- Andrew Ng's Deep Learning Specialization (Course 2: Improving Neural Networks)
- "A Recipe for Training Neural Networks" by Andrej Karpathy
- "Practical recommendations for gradient-based training of deep architectures" by Yoshua Bengio

---

**Happy tuning! 🎯**

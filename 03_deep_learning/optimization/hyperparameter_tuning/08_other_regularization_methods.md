# Other Regularization Methods

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 0:15 / 8:23

## Introduction

In addition to L2 regularization and dropout regularization, there are a few other techniques for reducing overfitting in your neural network.

## Data Augmentation

### The Problem: Getting More Data is Expensive

Let's say you're fitting a cat classifier and you're overfitting:

- **Getting more training data can help**
- But getting more data can be **expensive**
- Sometimes you just **can't get more data**

### Solution: Augment Your Existing Training Set

Instead of collecting new data, you can create variations of your existing data.

### Technique 1: Horizontal Flipping

```
Original Image:        Flipped Image:
   🐱                      🐱
  /o o\                   /o o\
 ( = ^ = )      →        ( = ^ = )
   )   (                  (   )
```

**What you do:**
- Take an existing image
- Flip it horizontally
- Add it to your training set

**Effect:**
- **Double the size** of your training set
- Almost free computationally
- Still recognizably a cat

### Technique 2: Random Crops and Distortions

**Operations you can apply:**
- Random rotation
- Random zoom/crop
- Slight translations
- Random distortions

```
Original:        Zoomed:         Rotated:        Cropped:
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  /\_/\  │     │         │     │    /    │     │  /\_    │
│ ( o.o ) │  →  │ ( o.o ) │  →  │  /\_/\  │  →  │ ( o.    │
│  > ^ <  │     │  > ^ <  │     │ ( o.o)  │     │  >      │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

**Still looks like a cat!**

### What You're Really Telling Your Algorithm

By synthesizing examples like this, you're communicating:

✓ **If something is a cat, then:**
- Flipping it horizontally is still a cat
- Randomly zooming in to part of the image is probably still a cat
- Slight rotations are still a cat

✗ **But be careful:**
- Don't flip vertically (we don't want upside-down cats)
- Don't distort so much it becomes unrecognizable

### Data Augmentation for Optical Character Recognition (OCR)

For digit recognition, you can augment by:

```
Original '4':    Rotated:     Distorted:    Warped:
    ┐                ┐             ┐            ╱┐
    │──             ──┘           ─┘           ╱ │──
    │               │             │           │  │
                               (subtle)     (subtle)
```

**Augmentation techniques:**
- Random rotations
- Random distortions
- Slight warping

**Note:** The examples shown here use **strong** distortion for illustration. In practice, use more **subtle distortions**.

### Limitations of Data Augmentation

**These extra fake training examples:**
- Don't add as much information as truly new independent examples
- Are somewhat redundant
- Still based on the same underlying images

**But the advantages:**
- Almost free (just computational cost)
- Inexpensive way to give your algorithm more data
- Acts as a regularization technique
- Reduces overfitting

### When to Use Data Augmentation

✓ **Use when:**
- You're overfitting
- Getting more real data is expensive/impossible
- You have computational resources for augmentation
- Your problem naturally allows transformations (images, audio, etc.)

✗ **Don't overdo it:**
- Keep transformations realistic
- Don't distort beyond recognition
- Maintain the true label after transformation

## Early Stopping

### How Early Stopping Works

As you run gradient descent, you plot two things:

#### 1. Training Error
- Could be 0/1 classification error on training set
- Or just the cost function J being optimized
- Should decrease **monotonically**

#### 2. Dev Set Error
- Could be classification error on dev set
- Or loss function (logistic loss, log loss) on dev set
- Will typically follow a U-shape

### The Typical Pattern

```
Error
  ↑
  │  Dev Set Error
  │      ╱
  │     ╱
  │    ╱  ←── Starts increasing
  │   ╱
  │  ╱____
  │ ╱      ╲___
  │╱            ╲___
  │                  ╲___
  │  Training Error
  │      (keeps decreasing)
  └────────────────────────────→ Iterations
         ↑
    Stop here!
```

### What Early Stopping Does

1. Monitor both training and dev set errors
2. **Dev set error goes down** for a while
3. **Dev set error starts increasing** (overfitting begins)
4. **Stop training** at the iteration where dev set error was lowest
5. Use the weights from that iteration

**You stop training your neural network halfway** - hence "early stopping".

### Why Does Early Stopping Work?

#### The Evolution of Weights During Training

```
Start of training:
  - Random initialization → w is small (close to zero)
  - Parameters are small values

Mid-training:
  - w gets bigger
  - Medium-sized parameters

End of training:
  - w gets very large
  - Large parameter values
```

**By stopping halfway:**
- You only have a **mid-size value** of w
- Similar to L2 regularization picking smaller ||w||²
- Smaller parameters → less overfitting

## The Downside of Early Stopping: Orthogonalization

### The Machine Learning Process: Two Separate Tasks

The machine learning process comprises different steps:

#### Task 1: Optimize the Cost Function J

**Goal:** Find w and b such that J(w,b) is as small as possible

**Tools:**
- Gradient descent
- Momentum
- RMSprop
- Adam
- Other optimization algorithms

**Focus:** Just reduce J. Don't think about anything else.

#### Task 2: Not Overfit (Reduce Variance)

**Goal:** Prevent overfitting after optimizing J

**Tools:**
- Regularization (L2, L1)
- Dropout
- Data augmentation
- Getting more data

**Focus:** Just reduce variance. Use separate tools.

### The Principle: Orthogonalization

**Orthogonalization means:**
- Being able to think about **one task at a time**
- Having **independent** sets of tools for different problems
- Not coupling multiple objectives together

```
Traditional Approach (Orthogonalized):
┌─────────────────┐     ┌──────────────────┐
│  Optimize J     │     │  Prevent         │
│                 │     │  Overfitting     │
│  Tools:         │     │                  │
│  • Gradient     │     │  Tools:          │
│    descent      │     │  • L2 reg        │
│  • Momentum     │     │  • Dropout       │
│  • Adam         │     │  • More data     │
└─────────────────┘     └──────────────────┘
     ↓                         ↓
  Independent              Independent
  optimization            regularization
```

### The Problem with Early Stopping

**Early stopping couples these two tasks:**

```
Early Stopping (NOT Orthogonalized):
┌─────────────────────────────────────┐
│  Early Stopping                     │
│                                     │
│  • Stops optimizing J early         │
│  • AND tries to prevent overfitting │
│                                     │
│  Mixes both objectives!             │
└─────────────────────────────────────┘
```

**Why this is problematic:**

❌ You're **not doing a great job** reducing cost function J
❌ You've sort of **broken** the optimization process
❌ You're **simultaneously** trying not to overfit
❌ Instead of using **different tools** for two problems, you're using **one tool** that mixes both
❌ Makes the set of things you could try **more complicated** to think about

### Early Stopping vs. L2 Regularization

#### Option 1: L2 Regularization (Preferred by Some)

```python
# Just train as long as possible
# Use L2 regularization with different λ values
lambda_values = [0.01, 0.1, 1.0, 10.0]

for lambda in lambda_values:
    train_with_L2_reg(lambda)
    evaluate_on_dev_set()
```

**Advantages:**
- ✓ Search space of hyperparameters easier to decompose
- ✓ Easier to search over
- ✓ Orthogonalized approach
- ✓ Optimize J fully, then regularize

**Disadvantages:**
- ✗ Might have to try many values of λ
- ✗ More computationally expensive
- ✗ Need to train multiple times

#### Option 2: Early Stopping

```python
# Run gradient descent just once
# Get small w, mid-size w, and large w in one run
train_and_monitor()
stop_when_dev_error_increases()
```

**Advantages:**
- ✓ Running gradient descent **just once**
- ✓ Try out values of small w, mid-size w, and large w
- ✓ **Without** needing to try lots of λ values
- ✓ Less computationally expensive
- ✓ Automatic stopping criterion

**Disadvantages:**
- ✗ Couples optimization and regularization
- ✗ Doesn't fully optimize J
- ✗ Less orthogonalized
- ✗ Harder to reason about

## Comparison Summary

| Aspect | L2 Regularization | Early Stopping |
|--------|-------------------|----------------|
| **Computational cost** | Higher (multiple training runs) | Lower (single training run) |
| **Orthogonalization** | Yes - separate tools | No - couples tasks |
| **Optimize J fully** | Yes | No - stops early |
| **Hyperparameters** | Need to tune λ | Automatic stopping |
| **Conceptual simplicity** | Clear separation | Mixed objectives |
| **Preferred by** | Some practitioners | Many practitioners |

## Personal Preferences (From the Instructor)

### Andrew Ng's Preference

**Prefers L2 regularization:**
- Try different values of λ
- Assuming you can afford the computation
- Clearer separation of concerns
- Easier to reason about

### Why Many People Use Early Stopping

**Despite its disadvantages:**
- Gets similar effect to L2 regularization
- Without explicitly trying lots of λ values
- Computationally efficient
- Widely used in practice

## When to Use Each Technique

### Data Augmentation
```
Use when:
  ✓ You're overfitting
  ✓ Can't get more real data
  ✓ Have computational resources
  ✓ Problem allows realistic transformations
```

### Early Stopping
```
Use when:
  ✓ Computational budget is tight
  ✓ Want automatic stopping
  ✓ Don't mind coupling optimization and regularization
  ✓ Need quick results
```

### L2 Regularization
```
Use when:
  ✓ Can afford computational cost
  ✓ Want orthogonalized approach
  ✓ Prefer clean separation of concerns
  ✓ Want to fully optimize J
```

## Summary of All Regularization Techniques

### Regularization Toolkit

| Technique | How It Works | Main Benefit | Main Cost |
|-----------|--------------|--------------|-----------|
| **L2 Regularization** | Add ||W||² penalty to J | Shrinks weights | Need to tune λ |
| **Dropout** | Randomly drop units | Can't rely on any feature | Harder to debug |
| **Data Augmentation** | Create fake examples | More training data | Computational cost |
| **Early Stopping** | Stop when dev error increases | Automatic, efficient | Couples objectives |

### Choosing Your Strategy

```
Start with:
  1. Check if you're overfitting (high variance)
  2. If not overfitting → Don't use regularization
  
If overfitting:
  1. Try data augmentation (almost free)
  2. Add L2 regularization or dropout
  3. Consider early stopping if computation is expensive
  4. Get more data if possible
```

## What's Next

You've now seen:
- ✓ Data augmentation techniques
- ✓ Early stopping and its trade-offs
- ✓ The concept of orthogonalization

**Next:** Let's talk about some techniques for **setting up your optimization problem** to make your training go quickly.

---

## Quick Reference: Data Augmentation Examples

### Computer Vision
```
Original → Horizontal flip
Original → Random crop
Original → Rotation (small angle)
Original → Zoom/scale
Original → Translation (shift)
Original → Color jittering
Original → Adding noise (subtle)
```

### Optical Character Recognition
```
Digit → Rotation
Digit → Elastic distortion
Digit → Scaling
Digit → Translation
```

### Audio (Not Covered But Common)
```
Audio → Time stretching
Audio → Pitch shifting
Audio → Adding background noise
Audio → Random cropping
```

### Natural Language Processing
```
Text → Synonym replacement
Text → Random insertion
Text → Random swap
Text → Random deletion
```

## Orthogonalization Principle

```
Task 1: Optimize J          Task 2: Reduce Variance
     ↓                            ↓
 Use tools for               Use tools for
 optimization only          regularization only
     ↓                            ↓
 Independent thinking       Independent thinking
     ↓                            ↓
 Easier to reason about     Easier to reason about
```

**Early stopping violates this principle** by mixing both tasks, but is still widely used due to computational efficiency.

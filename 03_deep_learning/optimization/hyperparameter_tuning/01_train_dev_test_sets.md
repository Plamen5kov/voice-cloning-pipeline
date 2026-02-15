# Train / Dev / Test Sets

**Source:** DeepLearning.AI - Practical Aspects of Deep Learning  
**Duration:** 0:46 / 12:04

## Introduction to Practical Deep Learning

Welcome to this course on the practical aspects of deep learning. Perhaps now you've learned how to implement a neural network. In this week, you'll learn the practical aspects of how to make your neural network work well. Ranging from things like hyperparameter tuning to how to set up your data, to how to make sure your optimization algorithm runs quickly so that you get your learning algorithm to learn in a reasonable amount of time.

In this first week, we'll first talk about how the cellular machine learning problem, then we'll talk about randomization, then we'll talk about some tricks for making sure your neural network implementation is correct.

## The Importance of Proper Data Setup

Making good choices in how you set up your training, development, and test sets can make a huge difference in helping you quickly find a good high-performance neural network.

### The Iterative Nature of Deep Learning

When training a neural network, you have to make a lot of decisions, such as:
- How many layers will your neural network have?
- How many hidden units do you want each layer to have?
- What's the learning rate?
- What are the activation functions you want to use for the different layers?

When you're starting on a new application, it's almost impossible to correctly guess the right values for all of these, and for other hyperparameter choices, on your first attempt.

**Applied machine learning is a highly iterative process:**
1. Start with an idea (network architecture, hyperparameters, etc.)
2. Code it up and try it
3. Run an experiment and evaluate results
4. Refine your ideas based on the outcome
5. Iterate to find better configurations

## Cross-Domain Challenges

Today, deep learning has found great success in many areas:
- Natural language processing (NLP)
- Computer vision
- Speech recognition
- Structured data applications:
  - Advertisements
  - Web search
  - Shopping websites
  - Computer security
  - Logistics

### Important Insight on Domain Transfer

**Intuitions from one domain or application area often do not transfer to other application areas.**

The best choices may depend on:
- Amount of data available
- Number of input features
- Computer configuration (GPUs vs CPUs)
- Specific hardware setup
- Many other factors

Even very experienced deep learning practitioners find it almost impossible to correctly guess the best choice of hyperparameters the very first time.

## Setting Up Train, Dev, and Test Sets

### Traditional Approach

If you have your training data, you traditionally would:
1. **Training Set** - Train algorithms on this data
2. **Development Set (Dev Set)** - Also called hold-out cross validation set; use to evaluate different models
3. **Test Set** - Use final model to get an unbiased estimate of algorithm performance

### Traditional Split Ratios (Older Era)

For smaller datasets (100 - 10,000 examples):
- **70/30 split:** 70% train, 30% test
- **60/20/20 split:** 60% train, 20% dev, 20% test

These ratios were perfectly reasonable rules of thumb for datasets of this size.

### Modern Big Data Era Splits

In the modern big data era (e.g., 1 million+ examples):
- Dev and test sets have become a much smaller percentage of the total
- The dev set just needs to be big enough to evaluate different algorithms and see which works better

#### Examples:

**For 1 million examples:**
- Dev set: 10,000 examples (1%)
- Test set: 10,000 examples (1%)
- Training set: 980,000 examples (98%)
- **Ratio: 98% / 1% / 1%**

**For even larger datasets:**
- 99.5% train / 0.25% dev / 0.25% test
- 99.5% train / 0.4% dev / 0.1% test

### Key Principles

**Development Set Purpose:**
- Test different algorithms
- Quickly decide which algorithm works better
- Doesn't need to be 20% of your data

**Test Set Purpose:**
- Give a confident estimate of final classifier performance
- 10,000 examples often sufficient for a million-example dataset

## Understanding Why We Separate Dev and Test Sets

### The Key Realization: Protecting Against Experimenter Bias

**The test set doesn't protect the model from its bias — it protects the evaluation from YOUR bias.**

There are actually two different types of "bias" at play:

#### 1. Model Bias (Underfitting)
- The model itself is too simple to capture patterns in the data
- Not enough capacity or complexity
- This is the "bias" we discuss in bias vs. variance trade-offs

#### 2. Experimenter Bias (Selection Bias)
- **YOU** cherry-pick the model that looks best on whatever data you evaluate
- **YOUR** decisions get optimized toward the data you're seeing
- **YOUR** choices inject bias into which model gets selected

**The test set protects against the second type of bias.**

### Why You Can't Use the Test Set for Model Comparison

Every time you evaluate multiple models and pick the best one, you're introducing selection bias:

```
Evaluate 20 models on test set:
  Model 1:  89.2% 
  Model 2:  88.7%
  Model 3:  90.1% ← You pick this one!
  ...
  Model 20: 88.9%
```

**What just happened?** You selected Model 3 because it performed best. But maybe:
- All models actually have the same true performance (85%)
- Model 3 just got "lucky" with this particular test set
- By picking the best performer, you've selected for random luck, not quality

**The reported 90.1% is now biased upward** — it's not a true estimate of real-world performance.

### The Exam Analogy

Think of studying for an exam:

- **Training Set** = Your textbook and homework problems you study from
- **Dev Set** = Practice exams you take while preparing
  - Take as many practice exams as you want
  - Learn from your mistakes
  - Adjust your study strategy based on results
  - Retake them to see improvement
- **Test Set** = The actual final exam
  - Take it exactly once
  - Represents your true performance
  - Someone else grades it (no bias)

**The Problem with Grading Your Own Final Exam:**

If you took the final exam 20 times, picked your best performance, and reported that score, you'd be lying to yourself! You'd have:
- Cherry-picked the attempt where you got lucky
- Optimized for that specific set of questions
- Given yourself an inflated, biased score

The same thing happens when you evaluate multiple models on the test set and pick the winner.

### Your Decisions Are a Form of Fitting

Even though you're not training on the dev set, you're still "fitting" to it through your decisions:

```
You observe dev set results and decide:
  "Model 3 is best" → You pick it
  "Learning rate 0.01 works better" → You choose it
  "Adding dropout helps" → You keep it
  "5 layers beats 3 layers" → You go with 5

After 100 of these decisions, you've tuned your entire pipeline 
to perform well on the dev set!
```

This is **indirect fitting** — using data to make decisions, even though you're not computing gradients.

### Why the Test Set Must Remain Untouched

The test set sits isolated from all your decision-making. After you've:
- Tried 20 different architectures on the dev set
- Tuned hyperparameters based on dev set performance
- Made hundreds of choices optimized for the dev set

The test set gives you an answer to: **"How well does this actually work, independent of all my biased choices?"**

It's your reality check that's uncorrupted by your experimentation process.

### The Workflow

```
✓ CORRECT:
  For each of 20 models:
    1. Train on Training Set
    2. Evaluate on Dev Set ← Use this for comparison
    3. Pick best on Dev Set
  
  Final step (once only):
    4. Evaluate on Test Set ← Unbiased final performance

✗ WRONG:
  For each of 20 models:
    1. Train on Training Set
    2. Evaluate on Test Set ← DON'T comparison shop here!
    3. Pick best on Test Set
  
  Final step:
    4. Report Test Set performance ← Now biased/meaningless
```

## "Why Not Just Try All Combinations?"

### Even With Infinite Compute, You'd Still Need Dev/Test Splits

A common question: "If I had unlimited computational resources, couldn't I just try every possible combination and pick the best?"

**The answer reveals a fundamental insight: The dev/test split isn't just about saving computation — it's about measuring generalization.**

### The Combinatorial Explosion

Even simple choices create astronomical combinations:

```
Network layers: {1, 2, 3, 4, 5, 10, 20, 50}               = 8 choices
Hidden units: {10, 50, 100, 200, 500, 1000}              = 6 choices  
Learning rate: {0.0001, 0.001, 0.01, 0.1, 1.0}           = 5 choices
Activation: {ReLU, sigmoid, tanh, leaky ReLU}            = 4 choices
Dropout: {0, 0.1, 0.2, 0.3, 0.5}                         = 5 choices
Batch size: {16, 32, 64, 128, 256}                       = 5 choices
Optimizer: {SGD, Adam, RMSprop, AdaGrad}                 = 4 choices

Total combinations: 8 × 6 × 5 × 4 × 5 × 5 × 4 = 96,000 combinations
```

And that's just 7 hyperparameters! Many hyperparameters are **continuous** (infinite possible values):
- Learning rate: 0.001, 0.00101, 0.00102... (infinite values)
- Dropout rate: any value between 0 and 1
- Weight decay: any positive real number

### The Fundamental Problem: Where Do You Evaluate?

Here's the key insight: **Even if you could try all combinations, where would you compare them?**

#### If You Compare on the Training Set:

```
Try all 96,000 models, evaluated on training set:
  Model 1:        92% training accuracy
  Model 2:        94% training accuracy
  ...
  Model 50,000:   99.99% training accuracy ← Best on training!
  ...
  Model 96,000:   85% training accuracy

You pick Model 50,000: Perfect training accuracy!
```

**Problem:** Model 50,000 might just be the most overfitted model! It memorized the training data perfectly but learned nothing generalizable.

**You've selected for overfitting, not for actual performance.**

#### You'd Still Need the Dev Set:

```
Try all 96,000 models (or infinite continuous variations):
  1. Train each on Training Set
  2. Evaluate each on Dev Set ← HERE is where you compare!
  3. Pick the best Dev Set performer
  
Then:
  4. Evaluate once on Test Set ← Final unbiased assessment
```

### What an Ideal World Would Actually Look Like

In a truly ideal world with infinite resources, you'd:

1. **Try infinite combinations** (all possible architectures, all continuous hyperparameter values)
2. **Evaluate all on a dev set** to see which generalizes best (can't use training set for this!)
3. **Still keep a separate test set** for final unbiased evaluation (can't reuse dev set after all that searching)

### The Real Constraint Isn't Just Computational

The need for dev/test splits comes from a **fundamental statistical constraint**:

- You **can't** measure generalization using the same data you trained on
- You **can't** get an unbiased estimate using data you've made decisions based on
- These are mathematical realities, not just practical limitations

### Real-World Approach

Since we can't try everything and wouldn't want to even if we could:

- Use **smart search strategies**: Grid search, random search, Bayesian optimization
- Apply **intuition and experience** to narrow the search space  
- **Iterate efficiently** using the dev set for comparison
- Accept we won't find the absolute optimal configuration
- Focus on finding something "good enough" efficiently

### Key Takeaway

The three-way split isn't a workaround for limited compute. It's a **fundamental requirement** for:
- **Training Set**: Learning patterns
- **Dev Set**: Measuring which models generalize (can't use training set)
- **Test Set**: Getting unbiased final estimate (can't reuse dev set)

Even with infinite compute, you still can't measure generalization without data the model (and you!) haven't already optimized for.

## Mismatched Train and Test Distributions

### Modern Trend

More people are training on mismatched train and test distributions.

**Example Scenario:**
- Building an app to find cat pictures
- **Training set:** High-resolution, professional cat pictures downloaded from the Internet
- **Dev/Test sets:** Blurrier, lower-resolution images from users' cell phone cameras

### Critical Rule of Thumb

**Make sure that the dev and test sets come from the same distribution.**

Why?
- You'll be using the dev set to evaluate many different models
- You'll be trying hard to improve performance on the dev set
- Having dev and test from the same distribution ensures progress translates to real-world performance

**Training Set Exception:**
- Training set data might not come from the same distribution as dev/test sets
- Deep learning's hunger for data often requires creative tactics (like web crawling)
- This trade-off is acceptable for faster progress

## When You Don't Need a Test Set

### Purpose Reminder
The goal of the test set is to give you an unbiased estimate of the performance of your final network.

### It's Okay to Skip the Test Set If:
- You don't need an unbiased estimate
- You only have a train and dev set

**Workflow without test set:**
1. Train on the training set
2. Try different model architectures
3. Evaluate on the dev set
4. Iterate to get a good model

**Important Note:** Fitting your model to the dev set means it no longer gives you an unbiased estimate of performance.

### Terminology Warning

In the machine learning world, when there's only a train and dev set:
- Most people call it a "training set" and "test set"
- What they call the "test set" is actually being used as a hold-out cross validation set
- This leads to overfitting to the "test set"
- **More correct terminology:** Train set and dev set

This practice is okay if you don't need a completely unbiased estimate of algorithm performance.

## Summary

Setting up train, dev, and test sets properly will:
- Allow you to iterate more quickly
- Enable more efficient measurement of bias and variance
- Help you more efficiently select ways to improve your algorithm

### Best Practices Recap:

1. **For small datasets (< 10K examples):** Traditional 70/30 or 60/20/20 splits are fine
2. **For large datasets (> 1M examples):** Much smaller dev/test percentages (e.g., 98/1/1)
3. **Always ensure dev and test sets come from the same distribution**
4. **Training set can come from a different distribution** if needed to get more data
5. **Test set is optional** if you don't need an unbiased performance estimate
6. **The goal is to iterate quickly** through the development cycle

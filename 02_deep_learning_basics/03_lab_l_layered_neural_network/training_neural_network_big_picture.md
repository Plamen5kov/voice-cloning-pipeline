# Training a Neural Network: The Big Picture

## Overview

Training a neural network is like teaching it through repetition. Each training iteration has 4 phases:

1. **Initialization** (once)
2. **Forward Propagation** (every iteration)
3. **Backward Propagation** (every iteration)
4. **Parameter Update** (every iteration)

---

## PHASE 1: INITIALIZATION (Once at the beginning)

### Exercise 1 & 2: `initialize_parameters` / `initialize_parameters_deep`

**Why?** You need starting values for weights (W) and biases (b) before training begins.

**Why random weights?** If all weights start at zero, every neuron in a layer learns the exact same thing (symmetry problem). Random small values break this symmetry.

**Why small values (×0.01)?** Large initial weights can cause exploding gradients and slow learning.

**When called?** Once, before any training iterations.

---

## PHASE 2: FORWARD PROPAGATION (Every iteration)

This is how the network makes predictions.

### Exercise 3: `linear_forward`

```
Z[l] = W[l] · A[l-1] + b[l]
```

**Why?** This is the fundamental computation in neural networks - taking inputs from the previous layer and computing a weighted sum.

**Why cache values?** You'll need `(A_prev, W, b)` later during backpropagation to compute gradients.

**When called?** Called by `linear_activation_forward` for every single layer.

---

### Exercise 4: `linear_activation_forward`

```
A[l] = activation(Z[l])
```

**Why?** The linear function alone can only learn linear patterns. Activations (ReLU, Sigmoid) introduce non-linearity, allowing the network to learn complex patterns like curves, XOR, image features, etc.

**Why ReLU vs Sigmoid?**
- **ReLU** (hidden layers): Faster training, avoids vanishing gradients
- **Sigmoid** (output layer): Outputs probabilities between 0 and 1 for binary classification

**When called?** Called by `L_model_forward` for each of the L layers.

---

### Exercise 5: `L_model_forward`

**Why?** Instead of manually coding forward pass for each layer, this function automates it for ANY number of layers.

**What it does:**
- Calls `linear_activation_forward` with ReLU for layers 1 to L-1
- Calls `linear_activation_forward` with Sigmoid for layer L
- Collects all caches from each layer

**When called?** Once per training iteration to get predictions (AL) from input data (X).

---

### Exercise 6: `compute_cost`

```
J = -1/m Σ [y·log(AL) + (1-y)·log(1-AL)]
```

**Why?** You need a single number that tells you "how wrong" your predictions are. This is what you're trying to minimize.

**Why this formula?** It's cross-entropy loss for binary classification - heavily penalizes confident wrong predictions, works well with sigmoid output.

**When called?** After forward propagation, to measure current performance.

---

## PHASE 3: BACKWARD PROPAGATION (Every iteration)

This is where the network learns - computing how to adjust weights to reduce the cost.

### Exercise 7: `linear_backward`

```
dW[l] = 1/m · dZ[l] · A[l-1]ᵀ
db[l] = 1/m · Σ dZ[l]
dA[l-1] = W[l]ᵀ · dZ[l]
```

**Why?** Computes the gradients (slopes) that tell you:
- How much to change each weight `W[l]`
- How much to change each bias `b[l]`
- How errors propagate backward to the previous layer

**Why these formulas?** They come from **calculus chain rule** - derivative of cost with respect to each parameter.

**When called?** Called by `linear_activation_backward` for every layer during backprop.

---

### Exercise 8: `linear_activation_backward`

**Why?** Adds the activation function's derivative to the gradient computation.

**What it does:**
1. Computes `dZ[l] = dA[l] * g'(Z[l])` where g' is activation derivative
2. Calls `linear_backward` to get `dW, db, dA_prev`

**Why activation derivatives?**
- **ReLU derivative**: `1 if Z > 0 else 0` - only passes gradient where neuron was active
- **Sigmoid derivative**: `σ(Z)·(1-σ(Z))` - gradient of sigmoid function

**When called?** Called by `L_model_backward` for each layer.

---

### Exercise 9: `L_model_backward`

**Why?** Automates backpropagation through all L layers, collecting all gradients.

**What it does:**
1. Starts with `dAL = ∂J/∂AL` (gradient of cost w.r.t final predictions)
2. Goes backward through layers L → 1
3. Calls `linear_activation_backward` for each layer
4. Collects all `dW[1..L]` and `db[1..L]` in a dictionary

**Why backward order?** Chain rule - you need gradients from layer `l+1` to compute gradients for layer `l`.

**When called?** Once per iteration, after computing cost.

---

## PHASE 4: PARAMETER UPDATE (Every iteration)

### Exercise 10: `update_parameters`

```
W[l] = W[l] - α · dW[l]
b[l] = b[l] - α · db[l]
```

**Why?** This is where learning happens! Adjust weights in the direction that reduces cost.

**Why minus sign?** Gradients point uphill (increasing cost). We want to go downhill, so we subtract.

**What is α (alpha)?** Learning rate - how big each step is. Too large → overshooting, too small → slow learning.

**When called?** After backpropagation, to apply the computed gradients.

---

## Why This Order?

```
1. INITIALIZE → 2. FORWARD → 3. COST → 4. BACKWARD → 5. UPDATE
      ↑                                                    ↓
      └────────────────── REPEAT STEPS 2-5 ───────────────┘
```

### The Learning Cycle:

1. **Forward**: Make predictions with current weights
2. **Cost**: Measure how bad predictions are
3. **Backward**: Calculate how to improve weights
4. **Update**: Actually improve the weights
5. **Repeat** until cost is low enough

**Why this works?** **Gradient descent** - repeatedly taking small steps downhill on the cost function surface until you reach a minimum (hopefully the global minimum).

---

## Function Call Hierarchy

### Forward Pass:
```
L_model_forward (Exercise 5)
    └─> linear_activation_forward (Exercise 4) [called L times]
            └─> linear_forward (Exercise 3)
            └─> activation (ReLU or Sigmoid)
```

### Backward Pass:
```
L_model_backward (Exercise 9)
    └─> linear_activation_backward (Exercise 8) [called L times]
            └─> activation_backward (ReLU or Sigmoid)
            └─> linear_backward (Exercise 7)
```

---

## The Cache System

Each forward function stores values in a "cache" that the corresponding backward function needs:

| Forward Function | Caches | Used By |
|-----------------|--------|---------|
| `linear_forward` | `(A_prev, W, b)` | `linear_backward` |
| `activation(Z)` | `Z` | `activation_backward` |
| `L_model_forward` | List of all layer caches | `L_model_backward` |

**Why?** During backpropagation, you need the values from the forward pass to compute gradients efficiently.

---

## Data Flow Example (3-layer network)

```
Forward Pass:
X → [LINEAR→RELU] → A[1] → [LINEAR→RELU] → A[2] → [LINEAR→SIGMOID] → A[3]=AL → Cost
         ↓cache              ↓cache                    ↓cache

Backward Pass:
    dA[0] ← [RELU←LINEAR] ← dA[1] ← [RELU←LINEAR] ← dA[2] ← [SIGMOID←LINEAR] ← dAL
    dW[1]                   dW[2]                    dW[3]
    db[1]                   db[2]                    db[3]
```

---

## Visualize It!

Run these commands to see the training process in action:

```bash
# Step-by-step execution with real data
python3 visualize_flow.py

# View computational graph diagrams
ls visual_representation/
```

The generated graphs show:
- **computation_graph.png**: Complete forward/backward flow with caches and gradients
- **layer_graph.png**: Simplified layer-by-layer architecture
- **pytorch_autograd.png**: PyTorch autograd comparison

---

## Key Takeaways

1. **Building Blocks**: Simple functions (Exercises 3, 4, 7, 8) do one thing well
2. **Composition**: Complex functions (Exercises 5, 9) combine building blocks
3. **Caching**: Forward pass saves values that backward pass needs
4. **Automation**: Single iteration handles networks of ANY depth (L layers)
5. **Gradient Descent**: Repeated small improvements lead to learning

The beauty of this design is that once you implement these 10 functions, you can train neural networks with 2, 5, 10, or 100 layers without changing any code!

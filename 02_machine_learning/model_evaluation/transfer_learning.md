# Transfer Learning

## Core Concept

Use knowledge learned from one task (with lots of data) to improve performance on a different but related task (with limited data).

## The Motivation

### The Problem

```
Scenario: Radiology diagnosis from X-rays
- Only 1,000 labeled X-ray images available
- Training from scratch gives poor results (not enough data)
- Need better performance

Traditional approach: ❌ Train from scratch, get ~70% accuracy
Transfer learning: ✅ Start with pre-trained model, get 90% accuracy
```

### Why It Works

**Key insight:** Low-level features are often transferable

```
ImageNet pre-training learns:
- Edge detectors
- Color blob detectors
- Curve detectors
- Texture patterns

These features are useful for X-rays too!
```

## How Transfer Learning Works

### Standard Process

```
Step 1: Pre-training (Source Task)
  Task: Image classification on ImageNet
  Data: 1,000,000 images, 1000 classes
  Result: Model with general visual features

Step 2: Transfer (Target Task)
  Task: X-ray diagnosis
  Data: 1,000 medical images
  Action: Replace output layer, retrain
  Result: Specialized model for X-rays
```

### Three Training Strategies

#### Strategy 1: Fine-tune Only Last Layer (Small Dataset)

```
Use when: Very little target data (100-1000 examples)

Procedure:
1. Take pre-trained model
2. Remove last layer (output layer)
3. Add new last layer for your task
4. FREEZE all earlier layers
5. Train ONLY the new last layer

Example:
  Pre-trained: ImageNet model (1000 classes)
  Your task: Cat breed classification (10 classes)
  Action: Replace 1000-output layer with 10-output layer
  Training: Only update weights in final layer
```

**When to use:**
- N < 10K examples
- Target task very similar to source task
- Fast training needed
- Limited compute resources

#### Strategy 2: Fine-tune Several Layers (Medium Dataset)

```
Use when: Moderate target data (1K-100K examples)

Procedure:
1. Take pre-trained model
2. Replace last layer
3. FREEZE early layers (keep low-level features)
4. Train later layers + new last layer

Example:
  Layers: Conv1-Conv2-Conv3-Conv4-Conv5-FC-Output
  Freeze: Conv1-Conv2-Conv3 (general features)
  Train: Conv4-Conv5-FC-Output (task-specific features)
```

**When to use:**
- 1K < N < 100K examples
- Target moderately different from source
- Have some compute budget
- Want better performance than strategy 1

#### Strategy 3: Fine-tune Entire Network (Large Dataset)

```
Use when: Lots of target data (100K+ examples)

Procedure:
1. Take pre-trained model
2. Replace last layer
3. Use pre-trained weights as INITIALIZATION
4. Train ALL layers with LOWER learning rate

Example:
  All layers trainable
  Learning rate: 0.0001 (vs 0.001 for training from scratch)
  Epochs: Fewer needed due to good initialization
```

**When to use:**
- N > 100K examples
- Have compute resources
- Target somewhat different from source
- Want maximum performance

### Visual Comparison

```
Training from scratch:
[Random] → [Training] → [Final Model]
  ↓           ↓              ↓
  Slow    Need lots     Okay result
          of data

Transfer learning (small data):
[Pre-trained] → [Freeze early] → [Train last] → [Final Model]
      ↓              ↓               ↓               ↓
   Feature        Fast           Little        Great result
   rich          setup           data

Transfer learning (large data):
[Pre-trained] → [Fine-tune all] → [Final Model]
      ↓              ↓                  ↓
   Great         Faster            Best result
   start        convergence
```

## When Transfer Learning Helps

### ✅ Transfer Learning Makes Sense When:

**1. Tasks are related**
```
✅ ImageNet → Medical imaging (both are images)
✅ English sentiment → Spanish sentiment (both are text sentiment)
✅ Speech recognition → Speaker ID (both use audio)
✅ Object detection → Semantic segmentation (both need object understanding)
```

**2. Source task has WAY more data**
```
✅ ImageNet (1M images) → X-rays (1K images)
✅ Wikipedia (billions of words) → Legal documents (100K words)
✅ General speech (10K hours) → Medical transcription (100 hours)
```

**3. Low-level features are transferable**
```
✅ Image features: edges, textures, shapes
✅ Text features: word patterns, grammar, context
✅ Audio features: frequency patterns, phonemes
```

### ❌ Transfer Learning Doesn't Help When:

**1. Tasks are unrelated**
```
❌ ImageNet → Predict stock prices (images ≠ financial data)
❌ English text → Audio waveform (different modalities)
❌ Face recognition → Board game AI (no feature overlap)
```

**2. Source task has less data than target**
```
❌ Medical scans (1K) → ImageNet (1M)
   Just train on ImageNet from scratch instead!
```

**3. You have tons of target data anyway**
```
Debatable: If you have 10M target examples, pre-training helps less
But: Still often improves results and speeds convergence
```

## Common Transfer Learning Scenarios

### Computer Vision

#### Scenario 1: Small Custom Dataset

```
Task: Classify your company's products (5 categories, 500 images)

Approach:
1. Use ResNet50 pre-trained on ImageNet
2. Remove final layer (1000 classes)
3. Add new layer (5 classes)
4. Freeze all ResNet layers
5. Train only final layer for 10 epochs

Result: ~85% accuracy vs ~60% training from scratch
```

#### Scenario 2: Medical Imaging

```
Task: Detect pneumonia from chest X-rays (10,000 images)

Approach:
1. Use DenseNet pre-trained on ImageNet
2. Replace final layer
3. Freeze first 50 layers
4. Fine-tune last 20 layers + new final layer
5. Use lower learning rate (1e-4)

Result: ~92% accuracy vs ~78% from scratch
```

#### Scenario 3: Face Recognition

```
Task: Identify employees (1000 people, 50 images each)

Approach:
1. Use VGGFace pre-trained on celebrity faces
2. Remove classification layer
3. Add new layer for 1000 employees
4. Fine-tune last 3 layers
5. Use triplet loss or softmax

Result: High accuracy with few images per person
```

### Natural Language Processing

#### Scenario 1: Sentiment Analysis

```
Task: Classify customer reviews (5,000 labeled reviews)

Approach:
1. Use BERT pre-trained on Wikipedia + Books
2. Add classification head
3. Fine-tune BERT (all layers, low learning rate)
4. Train for 3-5 epochs

Result: ~90% accuracy vs ~75% with LSTM from scratch
```

#### Scenario 2: Named Entity Recognition

```
Task: Extract medical entities (drugs, diseases, symptoms)

Approach:
1. Use BioBERT (BERT pre-trained on medical literature)
2. Add token classification layer
3. Fine-tune on medical NER dataset
4. Use domain-specific pre-training

Result: Better than general BERT or training from scratch
```

### Speech Recognition

#### Scenario 1: Accent-Specific Recognition

```
Task: Recognize Indian-accented English (1000 hours)

Approach:
1. Use wav2vec 2.0 pre-trained on Librispeech (10K hours)
2. Add CTC or attention decoder
3. Fine-tune on Indian English
4. Use lower learning rate

Result: Much better than training from scratch
```

## Advanced Transfer Learning Techniques

### Multi-Stage Transfer Learning

```
Stage 1: Pre-train on ImageNet (1M images, general objects)
         ↓
Stage 2: Fine-tune on iNaturalist (1M images, animals/plants)
         ↓
Stage 3: Fine-tune on bird dataset (50K images, bird species)
         ↓
Stage 4: Fine-tune on local birds (1K images, your region)

Each stage makes the model more specialized
```

### Few-Shot Learning with Transfer

```
Scenario: Only 5 examples per class

Approach:
1. Pre-train on large dataset
2. Use feature extractor (frozen backbone)
3. Train simple classifier (prototypical networks, matching networks)
4. Achieve good results with minimal data

Application: Quick adaptation to new categories
```

### Domain Adaptation

```
Problem: Source and target domains differ (distribution mismatch)

Approach:
1. Pre-train on source domain
2. Use adversarial training to make features domain-invariant
3. Fine-tune with both labeled source and unlabeled target data

Example: Real photos → Synthetic photos
```

## Practical Implementation

### PyTorch Example

```python
import torch
import torchvision.models as models
from torch import nn

# Strategy 1: Fine-tune only last layer
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = 10  # Your task has 10 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only final layer will be trained
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

```python
# Strategy 2: Fine-tune last few layers
model = models.resnet50(pretrained=True)

# Freeze early layers
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Train unfrozen layers
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)
```

```python
# Strategy 3: Fine-tune entire network
model = models.resnet50(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Train all layers with low learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```

### TensorFlow/Keras Example

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

# Strategy 1: Freeze all except last layer
base_model = ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False  # Freeze

# Add custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
output = layers.Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Only new layers will train
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

```python
# Strategy 2: Freeze partial
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze first 80% of layers
for layer in base_model.layers[:int(len(base_model.layers) * 0.8)]:
    layer.trainable = False

# Rest is trainable
# ... add custom layers and compile
```

## Common Pitfalls

### ❌ Mistake 1: Using Too High Learning Rate

```
Problem:
  Pre-trained weights are good
  High LR destroys them

Solution:
  Use 10-100x lower LR than training from scratch
  Example: 0.0001 instead of 0.001
```

### ❌ Mistake 2: Forgetting to Freeze Layers

```
Problem:
  Intended to train only last layer
  Forgot to freeze → trains all layers
  Overfits on small dataset

Solution:
  Always explicitly freeze layers
  Verify with param.requires_grad or layer.trainable
```

### ❌ Mistake 3: Wrong Input Preprocessing

```
Problem:
  Pre-trained on images normalized differently
  Your images normalized differently
  Model performs poorly

Solution:
  Use SAME preprocessing as pre-training
  Check mean/std normalization values
```

### ❌ Mistake 4: Catastrophic Forgetting

```
Problem:
  Fine-tuned too aggressively
  Model forgot pre-trained knowledge
  Performance drops

Solution:
  Use lower learning rate
  Fine-tune gradually (start with last layer)
  Use early stopping
```

## Guidelines for Choosing Strategy

```
Decision tree:

How much target data do you have?
├─ < 1K examples
│   └─> Strategy 1: Freeze all, train last layer only
│
├─ 1K - 100K examples
│   └─> Strategy 2: Freeze early, fine-tune later layers
│       
│       How similar is target to source?
│       ├─ Very similar → Freeze more layers
│       └─ Somewhat different → Freeze fewer layers
│
└─ > 100K examples
    └─> Strategy 3: Fine-tune entire network
        
        Have compute budget?
        ├─ Yes → Full fine-tuning (best performance)
        └─ No → Strategy 2 (good compromise)
```

## Key Takeaways

1. **Transfer learning is powerful when:**
   - Target task has limited data
   - Source task is related
   - Source has much more data

2. **Three main strategies:**
   - Freeze all, train last layer (small data)
   - Freeze early, fine-tune later (medium data)
   - Fine-tune everything (large data)

3. **Use lower learning rates** for fine-tuning (10-100x lower)

4. **Pre-training sources:**
   - Vision: ImageNet, COCO, OpenImages
   - NLP: BERT, GPT, RoBERTa
   - Speech: wav2vec, Whisper

5. **More similar tasks = freeze more layers**

6. **Always check preprocessing** matches pre-training

7. **Transfer learning usually beats training from scratch**
   - Faster convergence
   - Better performance
   - Needs less data

## When NOT to Use Transfer Learning

```
❌ Skip transfer learning if:

1. Tasks are completely unrelated
   - Features won't transfer
   - May hurt performance

2. You have massive target dataset
   - Can train from scratch effectively
   - Transfer gives diminishing returns
   - But often still helps convergence

3. Source domain is too different
   - Medical imaging → Natural images: ✅ OK
   - Medical imaging → Audio: ❌ Too different
```

---

**Related:** [Data Mismatch](data_mismatch.md), [Multi-Task Learning](multi_task_learning.md), [End-to-End Learning](end_to_end_learning.md)

# Deep Learning Basics - Module Overview

Welcome to the foundational module of the voice cloning learning path! This folder contains everything you need to master core deep learning concepts through hands-on PyTorch exercises.

**📖 For detailed concept explanations, see [LEARNING_GUIDE.md](LEARNING_GUIDE.md)**  
**📚 For deep-dive reference on 14 ML concepts, see [ML_CONCEPTS_EXPLAINED.md](ML_CONCEPTS_EXPLAINED.md)**

---

## 🚀 Quick Start

1. **Read this README** - Get the big picture (5 min)
2. **Read [LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - Understand concepts in depth (30 min)
3. **Skim [ML_CONCEPTS_EXPLAINED.md](ML_CONCEPTS_EXPLAINED.md)** - Reference when needed
4. **Run scripts in order**: 01 → 02 → 03 → 04 → 05

---

## 📂 What's in This Folder

### 📖 Documentation
- **README.md** (this file) - High-level overview
- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - Detailed explanations and concepts
- **[ML_CONCEPTS_EXPLAINED.md](ML_CONCEPTS_EXPLAINED.md)** - Comprehensive reference (14 core ML topics)

### 💻 Code
- **[dl_utils.py](dl_utils.py)** - Utility functions (GPU, data loading, evaluation)
- **[01_hello_pytorch.py](01_hello_pytorch.py)** - PyTorch basics
- **[02_load_mnist.py](02_load_mnist.py)** - Data exploration
- **[03_train_mnist.py](03_train_mnist.py)** - Training loop ⭐ CORE
- **[04_load_model.py](04_load_model.py)** - Model persistence
- **[05_experiment_architectures.py](05_experiment_architectures.py)** - Architecture experiments

---

## 🎯 Learning Objectives

By completing this module:

- ✅ Understand neural network training (gradients, backpropagation)
- ✅ Implement and debug training loops
- ✅ Recognize overfitting, underfitting, and good fit
- ✅ Choose hyperparameters effectively
- ✅ Build confidence with PyTorch

---

## 🎓 Recommended Path

### Phase 1: Foundation (Day 1)
1. Read [LEARNING_GUIDE.md](LEARNING_GUIDE.md)
2. Skim [ML_CONCEPTS_EXPLAINED.md](ML_CONCEPTS_EXPLAINED.md)
3. Run `01_hello_pytorch.py` and `02_load_mnist.py`

### Phase 2: Core Training (Day 2)
1. Run `03_train_mnist.py` - Study the training loop carefully
2. Understand training curves (see [LEARNING_GUIDE.md](LEARNING_GUIDE.md#understanding-training-results))
3. Run `04_load_model.py`

### Phase 3: Experimentation (Day 3)
1. Run `05_experiment_architectures.py`
2. Experiment with hyperparameters
3. Try to improve results

**See [LEARNING_GUIDE.md](LEARNING_GUIDE.md) for detailed explanations of each concept**

---

## 📊 What Each Script Does

| Script | Purpose | Key Output | Time |
|--------|---------|------------|------|
| 01_hello_pytorch.py | PyTorch basics | Console | 5 min |
| 02_load_mnist.py | Data exploration | mnist_samples.png | 5 min |
| 03_train_mnist.py | Training loop ⭐ | mnist_model.pth, training_history.png | 30 sec - 3 min |
| 04_load_model.py | Model persistence | Console | 10 sec |
| 05_experiment_architectures.py | Architecture comparison | architecture_comparison.png | 2-3 min |

**For detailed script explanations, see [LEARNING_GUIDE.md](LEARNING_GUIDE.md#script-details)**

---

## 🎯 Success Metrics

You've mastered this module when you can:

- [ ] Explain what gradients are
- [ ] Write a simple training loop
- [ ] Diagnose overfitting from plots
- [ ] Choose hyperparameters
- [ ] Explain model architecture requirements

**Expected Results:** Training ~97-98%, Validation ~95-97%, Gap 1-3%

---

## 🚀 Next Steps

After this module:
- **[03_tts_systems](../03_tts_systems/)** - Apply DL to text-to-speech
- **[04_speech_audio_processing](../04_speech_audio_processing/)** - Audio analysis
- **[05_nlp](../05_nlp/)** - Natural language processing

---

**Time Estimate**: 10-15 hours for complete mastery

**Remember**: Concepts in [LEARNING_GUIDE.md](LEARNING_GUIDE.md) are explained in detail. Use it as your main learning resource!


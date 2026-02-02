# Deep Learning Basics - Module Overview

Welcome to the foundational module of the voice cloning learning path! This folder contains everything you need to master core deep learning concepts through hands-on PyTorch exercises.

For detailed exercises and tasks, see **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)**.

For comprehensive concept explanations, see **[ML_CONCEPTS_EXPLAINED.md](ML_CONCEPTS_EXPLAINED.md)**.

---

## 🚀 Quick Start

1. **Read [LEARNING_GUIDE.md](LEARNING_GUIDE.md)** first (15 min)
2. **Skim [ML_CONCEPTS_EXPLAINED.md](ML_CONCEPTS_EXPLAINED.md)** (30 min)
3. **Run scripts in order**: 01 → 02 → 03 → 04 → 05
4. **Reference ML_CONCEPTS_EXPLAINED.md** when you encounter unfamiliar terms

---

## 📂 What's in This Folder

### 📖 Documentation
- **README.md** (this file) - Module overview
- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - Detailed learning path, exercises, and outcomes
- **[ML_CONCEPTS_EXPLAINED.md](ML_CONCEPTS_EXPLAINED.md)** - Comprehensive reference for 14 core ML concepts

### 💻 Code
- **[dl_utils.py](dl_utils.py)** - Reusable utility functions (GPU detection, data loading, evaluation)
- **[01_hello_pytorch.py](01_hello_pytorch.py)** - PyTorch basics and tensor operations
- **[02_load_mnist.py](02_load_mnist.py)** - Data exploration and visualization
- **[03_train_mnist.py](03_train_mnist.py)** - Training loop ⭐ MOST IMPORTANT
- **[04_load_model.py](04_load_model.py)** - Model persistence and loading
- **[05_experiment_architectures.py](05_experiment_architectures.py)** - Architecture comparisons

---

## 🎯 Learning Objectives

By completing this module, you will:

- ✅ Understand how neural networks learn (gradients, backpropagation)
- ✅ Implement training loops from scratch
- ✅ Evaluate and debug models effectively
- ✅ Recognize and handle overfitting
- ✅ Choose appropriate hyperparameters
- ✅ Build confidence with PyTorch

---

## 🎓 Recommended Path

### Phase 1: Foundation (Day 1)
1. Read [LEARNING_GUIDE.md](LEARNING_GUIDE.md)
2. Skim [ML_CONCEPTS_EXPLAINED.md](ML_CONCEPTS_EXPLAINED.md) - focus on Gradients, Loss Functions, Optimizers
3. Run `01_hello_pytorch.py`
4. Run `02_load_mnist.py`

### Phase 2: Core Training (Day 2)
1. Deep read ML_CONCEPTS_EXPLAINED.md sections: Gradients, Backpropagation, Learning Rate
2. Run `03_train_mnist.py` - READ EVERY COMMENT
3. Read `dl_utils.py` docstrings
4. Run `04_load_model.py`

### Phase 3: Experimentation (Day 3)
1. Run `05_experiment_architectures.py`
2. Modify `03_train_mnist.py` - try different learning rates, batch sizes, architectures
3. Reference ML_CONCEPTS_EXPLAINED.md as needed

### Phase 4: Mastery
- Can you achieve >98% accuracy?
- Can you do it with fewer parameters?
- Try adding data augmentation
- Move to next module: [03_tts_systems](../03_tts_systems/)

---

## 📊 What Each Script Does

| Script | Purpose | Output | Time |
|--------|---------|--------|------|
| 01_hello_pytorch.py | PyTorch basics | Console output | 5 min |
| 02_load_mnist.py | Data exploration | mnist_samples.png | 5 min |
| 03_train_mnist.py | Training loop ⭐ | mnist_model.pth, training_history.png | 30 sec - 3 min |
| 04_load_model.py | Model persistence | Console output | 10 sec |
| 05_experiment_architectures.py | Architecture comparison | architecture_comparison.png | 2-3 min |

---

## 🎯 Success Metrics

You've mastered this module when you can:

- [ ] Explain gradients to someone else
- [ ] Write a training loop from scratch
- [ ] Diagnose overfitting from plots
- [ ] Choose appropriate hyperparameters
- [ ] Explain ReLU vs Tanh
- [ ] Understand every line in 03_train_mnist.py

---

## 🔗 Quick Reference

**Encounter unfamiliar term?** → Look it up in [ML_CONCEPTS_EXPLAINED.md](ML_CONCEPTS_EXPLAINED.md)

**Need context?** → Check [LEARNING_GUIDE.md](LEARNING_GUIDE.md)

**Why this choice?** → Read docstrings in [dl_utils.py](dl_utils.py)

---

## 🚀 Next Steps

After mastering this module:
- **[03_tts_systems](../03_tts_systems/)** - Apply DL to text-to-speech
- **[04_speech_audio_processing](../04_speech_audio_processing/)** - Audio analysis
- **[05_nlp](../05_nlp/)** - Natural language processing

---

**Time Estimate**: 10-15 hours for complete mastery

**Golden Rule**: Don't just read code. Run it. Modify it. Break it. Fix it. Understand it.

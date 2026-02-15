# Environment Setup - Learning Guide

## 🎯 Module Overview

This foundational module ensures you have a properly configured development environment for AI and machine learning work. Setting up your environment correctly saves countless hours of debugging later.

## 📚 What You'll Learn

- Python version management and virtual environments
- Installing and managing ML libraries (PyTorch, TensorFlow, etc.)
- GPU/CUDA setup for accelerated training
- Jupyter notebook configuration
- Dependency management best practices
- Environment troubleshooting

## 🎓 Learning Objectives

By completing this module, you should be able to:
- [ ] Create and activate Python virtual environments
- [ ] Install PyTorch with CUDA support
- [ ] Verify GPU availability and proper configuration
- [ ] Set up Jupyter notebooks for experimentation
- [ ] Manage project dependencies with requirements.txt
- [ ] Troubleshoot common installation issues

## 📝 Key Concepts

### Virtual Environments
**Why**: Isolate project dependencies to avoid conflicts between projects
**Tools**: venv, conda, virtualenv

### GPU Acceleration
**Why**: 10-100x speedup for deep learning training
**Requirements**: NVIDIA GPU, CUDA toolkit, cuDNN

### Dependency Management
**Why**: Reproducible environments across machines
**Tools**: pip, conda, requirements.txt, environment.yml

## 🚀 Exercises & Tasks

### Task 1: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

**Learning Point**: Always use isolated environments for ML projects

### Task 2: Install Core Libraries
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jupyter numpy pandas matplotlib scikit-learn
```

**Learning Point**: Order matters - install CUDA-enabled PyTorch before other libraries

### Task 3: Verify GPU Setup
Create and run a verification script to check:
- Python version
- PyTorch version
- CUDA availability
- GPU name and memory

**Expected Output**: Should confirm GPU is accessible

### Task 4: Set Up Jupyter
```bash
pip install jupyter
jupyter notebook
```

**Learning Point**: Notebooks are essential for exploration and visualization

## 📊 Success Criteria

You've completed this module when:
- ✅ Virtual environment is activated
- ✅ PyTorch installed with CUDA support
- ✅ GPU is detected and accessible
- ✅ Jupyter notebooks launch successfully
- ✅ All core libraries import without errors

## 🔗 Next Steps

Once your environment is ready:
→ Move to **[01_python_programming](../01_python_programming/)** to practice Python basics for ML

## 📖 Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

## ⚠️ Common Issues & Solutions

**Issue**: CUDA out of memory
**Solution**: Reduce batch size, use smaller models, or clear GPU cache

**Issue**: PyTorch not detecting GPU
**Solution**: Reinstall with correct CUDA version matching your GPU drivers

**Issue**: Import errors
**Solution**: Verify virtual environment is activated and dependencies installed

---

**Time Estimate**: 1-2 hours (depending on download speeds and troubleshooting)

# Lab 4: Deep Neural Network for Voice Gender Classification - Application

## Overview

In this lab, you will apply everything you've learned from the previous assignments to build a voice gender classifier using deep neural networks. Building on Lab 1's binary speech vs music classification, you'll now classify voice recordings as male or female using both shallow and deep networks with real audio data.

## Learning Objectives

By the end of this assignment, you'll be able to:
- Build and train a deep L-layer neural network
- Apply the neural network to binary voice classification
- Compare the performance of shallow vs deep networks
- Work with real voice audio data from TTS systems
- Analyze model errors and understand common failure cases in voice classification

## Structure

The lab is divided into the following sections:

1. **Packages** - Import necessary libraries
2. **Load and Process Dataset** - Load voice recordings and preprocess spectrograms
3. **Model Architecture** - Understand 2-layer and L-layer network architectures
4. **Two-layer Neural Network** - Build and train a 2-layer network
5. **L-layer Neural Network** - Build and train a deep L-layer network
6. **Results Analysis** - Analyze model performance and prediction errors
7. **Test with Your Own Audio** - Optional exercise to test with custom audio files

## Files

### Core Lab Files
- `audio_genre_dnn.ipynb` - Main Jupyter notebook with exercises and model training
- `dnn_app_utils_v3.py` - Deep neural network helper functions (L-layer forward/backward propagation, cost computation)
- `audio_utils.py` - Audio loading, mel-spectrogram extraction, and dataset preprocessing utilities
- `public_tests.py` - Unit tests to validate your model implementation
- `test_setup.py` - Verify environment setup and dependencies
- `clean_notebook.py` - Utility to reset notebook outputs for clean commits
- `requirements.txt` - Python package dependencies (numpy, librosa, soundfile, matplotlib)

### Data Preparation Scripts

**Quick Setup (320 samples):**
- `download_samples.py` - Downloads LibriSpeech dev-clean (~350MB), extracts 120 train + 40 test samples per gender, creates proper directory structure automatically

**Full Setup (2,703 samples):**
- `sort_dev_clean_by_gender.py` - Organizes LibriSpeech by speaker gender (20 male + 20 female speakers)
- `convert_flac_to_wav.py` - Converts all FLAC audio files to WAV format (22.05kHz)
- `prepare_lab4_dataset.py` - Creates 75/25 train/test split and copies files to `data/` with clean naming

### Documentation
- `LOSS_FUNCTIONS_EXPLAINED.md` - Mathematical derivations of binary cross-entropy vs categorical cross-entropy, gradient calculations, why both simplify to (a-y)
- `HYPERPARAMETER_TUNING_GUIDE.md` - Practical guide to choosing learning rate, iterations, network architecture, regularization, and systematic tuning strategies
- `README.md` - This file (setup instructions and lab overview)

### Data Directory
- `data/` - Voice samples directory (created by setup scripts, gitignored)

## Dataset

This lab uses voice gender samples from the **LibriSpeech** dataset, a large corpus of read English speech.

**Dataset Options:**

**Basic Dataset** (via `download_samples.py`):
- **Training set:** 240 voice clips (120 male + 120 female)
- **Test set:** 80 voice clips (40 male + 40 female)
- **Total:** 320 samples
- **Best for:** Quick setup, faster training, learning the concepts

**Full Dataset** (via preparation scripts):
- **Training set:** ~2,026 voice clips (996 male + 1,030 female)
- **Test set:** ~677 voice clips (333 male + 344 female)
- **Total:** 2,703 samples from 40 speakers (20 male, 20 female)
- **Best for:** Better model performance, more realistic results

**Common Properties:**
- **Labels:** Binary (0=male, 1=female)
- **Format:** WAV files, 22.05 kHz sample rate, 3 seconds duration
- **Preprocessing:** Each audio clip is converted to a mel-spectrogram of shape (128, time_steps)
- **Features:** Mel-frequency spectral coefficients capture voice characteristics (pitch, formants, etc.)

**Directory Structure:**
```
data/
├── train/
│   ├── male/          # Male voice training samples
│   └── female/        # Female voice training samples
└── test/
    ├── male/          # Male voice test samples
    └── female/        # Female voice test samples
```

The balanced dataset (roughly equal male/female samples) ensures the model learns to distinguish gender-specific voice features rather than biases from class imbalance.

## Setup

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

### 2. Get Training Data

This lab uses male and female voice samples from the LibriSpeech dataset.

**Option A: Download Automatically** (Recommended - Self-Contained)

Run the included download script to get LibriSpeech samples (~350MB):

```bash
python download_samples.py
```

This script will:
- Download LibriSpeech dev-clean subset
- Extract and organize 120 male + 120 female training samples
- Extract and organize 40 male + 40 female test samples
- Create proper directory structure automatically

**Option B: Use Pre-Processed LibriSpeech Data**

If you've already run the data preparation scripts, use the larger dataset:

```bash
# From the Lab 4 directory
python sort_dev_clean_by_gender.py      # Organize by gender (if not done)
python convert_flac_to_wav.py          # Convert to WAV (if not done)
python prepare_lab4_dataset.py         # Split and copy to data/
```

This will give you the full dataset:
- **Training:** ~2,000 samples (996 male + 1,030 female)
- **Testing:** ~677 samples (333 male + 344 female)

### 3. Verify Data Setup

Check that your data is ready:

```bash
ls data/train/male/ | wc -l      # Should show number of male samples
ls data/train/female/ | wc -l    # Should show number of female samples
ls data/test/male/ | wc -l       # Should show number of test male samples
ls data/test/female/ | wc -l     # Should show number of test female samples
```

You can also listen to samples to verify:

```bash
aplay data/train/male/male_0000.wav      # Play a male voice
aplay data/train/female/female_0000.wav  # Play a female voice
```

## Usage

Open the Jupyter notebook and follow the instructions:

```bash
jupyter notebook audio_genre_dnn.ipynb
```

Complete each exercise by filling in the code sections marked with `# YOUR CODE STARTS HERE` and `# YOUR CODE ENDS HERE`.

## Expected Results

- **2-layer Network**: ~70-80% test accuracy (binary classification with real data)
- **4-layer Network**: ~80-90% test accuracy (binary classification with real data)
- **Baseline (Random)**: ~50% accuracy for binary classification

The deep network should outperform the shallow network by better capturing voice characteristics like pitch, formant frequencies, and spectral patterns that distinguish male from female voices.

## Connection to Previous Labs

This lab builds upon concepts from Lab 1 and Lab 2:

- **Lab 1**: Binary classification (speech vs music) with logistic regression
  - Introduced audio preprocessing with mel-spectrograms
  - Single-layer classifier (no hidden layers)
  
- **Lab 2**: Binary classification (male vs female voices) with single hidden layer
  - Same voice gender dataset used in this lab
  - 1 hidden layer neural network
  - Achieved ~85-90% accuracy with shallow network

- **Lab 4** (this lab): Binary classification (male vs female voices) with deep neural networks
  - **Same dataset as Lab 2** (240 train, 80 test samples)
  - Multiple hidden layers (2-layer and L-layer networks)
  - Compare shallow vs deep architectures
  - Learn how depth improves feature extraction

**Key Progression:**
```
Lab 1: No hidden layers → Lab 2: 1 hidden layer → Lab 4: Multiple hidden layers (L-layer)
```

The deeper networks in this lab should match or exceed Lab 2's performance while demonstrating the power of depth in learning complex voice patterns.

## Notes

This lab builds upon concepts from all previous labs. Make sure you understand:
- Initialization strategies for deep networks
- Forward and backward propagation through multiple layers
- The role of activation functions (ReLU vs sigmoid vs softmax)
- How network depth affects learning capacity

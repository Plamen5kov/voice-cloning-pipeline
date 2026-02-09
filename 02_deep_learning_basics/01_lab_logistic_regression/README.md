# Audio Classification Lab: Speech vs Music

A hands-on lab for learning logistic regression with audio data. This lab adapts the classic "cat vs non-cat" image classification exercise to work with audio spectrograms.

## Overview

This lab teaches you how to:
- Convert audio files to mel-spectrograms
- Build a logistic regression classifier from scratch
- Train a binary classifier to distinguish speech from music
- Understand forward/backward propagation and gradient descent
- Work with audio data using the same principles as image classification

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Audio Data

#### Quick Start (Recommended for Testing)

Run the download script and choose option 3:

```bash
python download_samples.py
# Choose option 3: Download speech + create test music (quick start)
```

This will:
1. Download 60 training + 20 test **real speech** samples from LibriSpeech
2. Create 60 training + 20 test **synthetic music** samples (modified speech for testing)

**Note:** The music samples are pitch-shifted/time-stretched speech - good enough to test the lab, but for real training you should replace them with actual music.

#### Verify Setup

Test that everything works:

```bash
python test_setup.py
```

#### Add Real Music (Optional but Recommended)

For better results, replace the synthetic music with real files:

1. Collect 60 training + 20 test music files (.wav, .mp3, etc.)
2. Place in `data/train/music/` and `data/test/music/`
3. Files will be auto-converted to 3-second .wav format

**Music sources:**
- Your personal music library
- Free Music Archive: https://freemusicarchive.org/
- YouTube Audio Library: https://youtube.com/audiolibrary
- Any royalty-free music sources

### 3. Run the Notebook

Open `audio_classification_logreg.ipynb` in Jupyter:

```bash
jupyter notebook audio_classification_logreg.ipynb
```

Or use VS Code's Jupyter extension.

## Lab Structure

The notebook guides you through 8 exercises:

1. **Exercise 1**: Understand the dataset dimensions
2. **Exercise 2**: Flatten spectrograms into vectors
3. **Exercise 3**: Implement the sigmoid activation function
4. **Exercise 4**: Initialize parameters (weights and bias)
5. **Exercise 5**: Implement forward and backward propagation
6. **Exercise 6**: Implement gradient descent optimization
7. **Exercise 7**: Implement prediction function
8. **Exercise 8**: Build the complete model

Each exercise includes:
- Clear instructions
- Code templates with hints
- Automated tests to verify your implementation

## Audio Processing Details

- **Sample Rate**: 22050 Hz
- **Duration**: 3 seconds per clip
- **Mel Bands**: 128
- **Representation**: Mel-spectrogram (time-frequency representation)
- **Normalization**: Values scaled to [0, 1]

The mel-spectrogram converts 1D audio signals into 2D image-like representations, making them suitable for the same machine learning techniques used for images.

## Files

- `audio_classification_logreg.ipynb` - Main notebook with exercises
- `audio_utils.py` - Audio loading and preprocessing utilities
- `public_tests.py` - Automated tests for your functions
- `requirements.txt` - Python package dependencies
- `data/` - Directory for audio files

## Learning Outcomes

After completing this lab, you will:
- Understand how to preprocess audio data for machine learning
- Know how to implement logistic regression from scratch
- Understand the connection between audio and image classification
- Be able to tune hyperparameters (learning rate, iterations)
- Recognize overfitting and understand model evaluation

## Tips

1. **Start with small datasets** - 10-20 samples per class to test quickly
2. **Use real audio** - Synthetic audio is just for testing the code
3. **Experiment with parameters** - Try different learning rates and iterations
4. **Visualize spectrograms** - Understand what the model "sees"
5. **Check for overfitting** - Compare training vs test accuracy

## Troubleshooting

**"No training data found" warning**:
- Make sure you've added `.wav` files to the `data/train/speech/` and `data/train/music/` folders

**Poor accuracy**:
- Ensure you have enough training samples (50+ per class)
- Use real audio, not synthetic samples
- Try adjusting the learning rate
- Increase the number of iterations

**Import errors**:
- Run `pip install -r requirements.txt`
- Make sure you're in the correct directory

## Next Steps

After completing this lab, you can:
- Try different audio classification tasks (male vs female voice, instruments, etc.)
- Experiment with different spectrogram parameters
- Move on to neural networks for better performance
- Explore deep learning frameworks (PyTorch, TensorFlow)

## Credits

This lab is adapted from the classic logistic regression image classification exercise, modified to work with audio data using mel-spectrograms.

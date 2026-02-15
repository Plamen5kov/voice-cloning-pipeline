# Audio Classification with Neural Networks

## Lab Overview

In this lab, you'll build your first **neural network with a hidden layer** to classify audio samples. You'll implement everything from scratch using NumPy, gaining deep understanding of:

- Neural network architecture with hidden layers
- Forward and backward propagation
- Tanh and sigmoid activation functions
- Gradient descent optimization
- Audio feature extraction using mel-spectrograms

## Task: Male vs Female Voice Classification

You'll build a binary classifier that can distinguish between male and female voices based on audio spectrograms. This is a real-world application of neural networks in audio processing!

## Prerequisites

- Python 3.8+
- Understanding of basic neural network concepts
- Familiarity with NumPy

## Setup Instructions

### 1. Install Dependencies

```bash
pip install numpy matplotlib librosa soundfile ipython
```

### 2. Download and Prepare Audio Samples

**Option A: Automated Setup (Easiest)**

Run the automated setup script:

```bash
python setup_lab2_data.py
```

This runs the complete pipeline automatically and verifies the setup.

**Option B: Quick Setup (Manual but simple)**

Run the all-in-one download script:

```bash
python download_samples.py
```

This will automatically:
- Download LibriSpeech dev-clean subset (~350MB)
- Extract 120 training + 40 test samples for each class  
- Save audio clips in `data/train/` and `data/test/`

**Note**: The download may take several minutes depending on your internet connection.

**Option C: Full Pipeline (Step-by-step control)**

For better control and reusability across labs:

```bash
# Step 1: Download LibriSpeech (if not already downloaded)
python download_samples.py  # Or download manually to data/LibriSpeech/

# Step 2: Organize by gender (creates male_samples/, female_samples/)
python sort_dev_clean_by_gender.py

# Step 3: Convert FLAC to WAV (creates male_samples_wav/, female_samples_wav/)
python convert_flac_to_wav.py

# Step 4: Create train/test split (120 train + 40 test per gender)
python prepare_lab2_dataset.py
```

This pipeline:
- Organizes all LibriSpeech samples by gender (20 speakers each, ~2,700 total files)
- Converts to WAV format (22.05 kHz)
- Creates a balanced 240 train + 80 test dataset
- Intermediate files can be reused by other labs (Lab 4 uses the full dataset)

### 3. Verify Setup

Run the test script to ensure everything is properly configured:

```bash
python test_setup.py
```

Expected output:
```
✓ All directories exist
✓ Training data: 240 samples (120 female + 120 male)
✓ Test data: 80 samples (40 female + 40 male)
✓ Sample duration: 3.0 seconds
✓ Audio features: (16640, 240)
✓ Setup complete! You're ready to start the lab.
```

### 4. Open the Notebook

```bash
jupyter notebook audio_classification_nn.ipynb
```

Or open it in VS Code with the Jupyter extension.

## Data Workflow

**Automated Setup (setup_lab2_data.py):**
- Runs the complete pipeline automatically in one command
- Verifies each step and provides clear progress updates
- Recommended for first-time users

**Quick Setup (download_samples.py):**
- All-in-one script that downloads, processes, and creates the 240 train + 80 test dataset directly
- Good for quick starts when you don't need the full organized dataset

**Full Pipeline (step-by-step):**

1. **Download**: Fetches LibriSpeech dev-clean (~350MB) → `data/LibriSpeech/`
2. **Organize by Gender** (`sort_dev_clean_by_gender.py`): Separates speakers by gender → `male_samples/`, `female_samples/` (FLAC files, ~2,700 files)
3. **Convert Format** (`convert_flac_to_wav.py`): FLAC → WAV (22.05kHz) → `male_samples_wav/`, `female_samples_wav/`
4. **Create Dataset Split** (`prepare_lab2_dataset.py`): 
   - Randomly selects 120 train + 40 test samples per gender
   - Saves to `data/train/male/`, `data/train/female/`, `data/test/male/`, `data/test/female/`

**The notebook uses only `data/train/` and `data/test/` directories.** The intermediate directories (male_samples, female_samples, etc.) contain the full organized LibriSpeech dataset (~2,700 samples) and can be reused by other labs - Lab 4 uses this full dataset for training.

## Lab Structure

The lab consists of **9 exercises**:

1. **Exercise 1**: Dataset exploration - understand the data shapes
2. **Exercise 2**: Define neural network structure (`layer_sizes`)
3. **Exercise 3**: Initialize parameters (`initialize_parameters`)
4. **Exercise 4**: Forward propagation (`forward_propagation`)
5. **Exercise 5**: Compute cost function (`compute_cost`)
6. **Exercise 6**: Backward propagation (`backward_propagation`)
7. **Exercise 7**: Update parameters (`update_parameters`)
8. **Exercise 8**: Complete neural network model (`nn_model`)
9. **Exercise 9**: Make predictions (`predict`)

## Files in This Lab

### Core Lab Files
- `audio_classification_nn.ipynb` - Main notebook with exercises
- `audio_utils.py` - Audio loading and preprocessing utilities
- `public_tests.py` - Automated test functions
- `testCases_v2.py` - Test case generators
- `test_setup.py` - Setup verification script
- `clean_notebook.py` - Removes outputs and solutions (for instructors)
- `README.md` - This file

### Data Preparation
- `setup_lab2_data.py` - Automated setup script (runs entire pipeline)
- `download_samples.py` - Downloads and prepares dataset from LibriSpeech (all-in-one, quick setup)
- `sort_dev_clean_by_gender.py` - Organizes LibriSpeech by speaker gender (20 male + 20 female speakers)
- `convert_flac_to_wav.py` - Converts FLAC files to WAV format (22.05kHz)
- `prepare_lab2_dataset.py` - Creates train/test split (120 train + 40 test per gender)

### Data Directories
- `data/train/` - Training samples (120 male + 120 female WAV files)
- `data/test/` - Test samples (40 male + 40 female WAV files)
- `data/LibriSpeech/` - Original LibriSpeech dev-clean subset (downloaded, can be deleted after setup)

### Intermediate Processing Directories (Created during data preparation)
- `male_samples/` - Organized male speaker FLAC files (1,329 files from 20 speakers)
- `female_samples/` - Organized female speaker FLAC files (1,374 files from 20 speakers)
- `male_samples_wav/` - Converted male WAV files (1,329 files)
- `female_samples_wav/` - Converted female WAV files (1,374 files)
- `male_speaker_ids.json` - Male speaker metadata
- `female_speaker_ids.json` - Female speaker metadata

**Note**: The intermediate directories contain the full organized LibriSpeech samples. The `data/` directory contains the final train/test split used by the notebook.

## Expected Results

After completing the lab, your neural network should achieve:
- **Training accuracy**: ~90-95%
- **Test accuracy**: ~85-90%

The model learns to distinguish male and female voices based on:
- Fundamental frequency (pitch) differences
- Formant patterns
- Spectral energy distribution

## Tips for Success

1. **Read the instructions carefully** - Each exercise has specific requirements
2. **Use the hints** - They point you to the right NumPy functions
3. **Check shapes** - Dimension mismatches are common errors
4. **Run tests** - Each exercise has an automated test function
5. **Experiment** - Try different hidden layer sizes in Section 5

## Common Issues

### Download fails
- Check your internet connection
- Ensure you have ~500MB free disk space
- LibriSpeech servers may be temporarily unavailable - try again later

### Import errors
- Verify all dependencies are installed: `pip install numpy matplotlib librosa soundfile`
- Make sure you're in the correct directory

### Shape mismatches
- Pay attention to whether you need row vectors (1, m) or column vectors (m, 1)
- Use `.reshape()` or slicing to fix dimensions
- Remember: matrix multiplication requires compatible dimensions

### Tests failing
- Review the mathematical formulas in the notebook
- Check that you're using the correct variables from cache/parameters
- Ensure you're not modifying input parameters directly (use `copy.deepcopy()`)

## Learning Outcomes

By the end of this lab, you will be able to:

✅ Implement a 2-class classification neural network with a hidden layer  
✅ Use non-linear activation functions (tanh, sigmoid)  
✅ Compute cross-entropy loss  
✅ Implement forward and backward propagation  
✅ Apply gradient descent to optimize parameters  
✅ Classify audio data using neural networks  
✅ Tune hyperparameters (hidden layer size, learning rate)  

## Next Steps

After completing this lab:
- Try the logistic regression lab (`../lab/`) to compare with no hidden layer
- Experiment with deeper networks (multiple hidden layers)
- Try different audio classification tasks
- Learn about regularization to prevent overfitting

## Need Help?

- Review the mathematical derivations in the notebook
- Check the test error messages - they often indicate what's wrong
- Compare your implementation with the expected shapes
- Make sure intermediate values (Z1, A1, Z2, A2) are correctly computed

## Acknowledgments

Dataset: LibriSpeech ASR corpus (http://www.openslr.org/12/)
- Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur
- "LibriSpeech: An ASR corpus based on public domain audio books", ICASSP 2015

---

**Good luck and enjoy building your neural network!** 🎵🤖

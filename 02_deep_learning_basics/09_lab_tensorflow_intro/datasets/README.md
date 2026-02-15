# Dataset Files

This lab requires the Hand Sign Digits dataset in HDF5 format.

## Required Files

You need to place the following files in this directory:

- `train_signs.h5` - Training dataset (1080 images of hand signs for digits 0-5)
- `test_signs.h5` - Test dataset (120 images of hand signs for digits 0-5)

## Dataset Information

- **Dataset**: Hand Sign Digits (subset for digits 0-5)
- **Image dimensions**: 64x64x3 (RGB images)
- **Number of classes**: 6 (digits 0 through 5)
- **Training samples**: 1080
- **Test samples**: 120

## Where to Get the Dataset

### Option 1: Coursera Deep Learning Specialization
If you're taking the Deep Learning Specialization on Coursera, the dataset files are provided in the assignment materials for "Introduction to TensorFlow" (Course 2, Week 3).

### Option 2: Download from Source
The dataset may be available from:
- DeepLearning.AI course materials
- Public repositories hosting Coursera assignment datasets (check licensing)

### Option 3: Create Your Own
If you cannot access the original dataset, you can create a similar dataset of hand sign images and save them in HDF5 format with the following structure:

**train_signs.h5**:
- `train_set_x`: (1080, 64, 64, 3) array of uint8 images
- `train_set_y`: (1080,) array of labels (0-5)

**test_signs.h5**:
- `test_set_x`: (120, 64, 64, 3) array of uint8 images
- `test_set_y`: (120,) array of labels (0-5)

## Verification

After placing the files here, you can verify they're correctly formatted by running this Python code:

```python
import h5py

# Check training data
train_dataset = h5py.File('datasets/train_signs.h5', "r")
print("Training set images shape:", train_dataset['train_set_x'].shape)
print("Training set labels shape:", train_dataset['train_set_y'].shape)

# Check test data
test_dataset = h5py.File('datasets/test_signs.h5', "r")
print("Test set images shape:", test_dataset['test_set_x'].shape)
print("Test set labels shape:", test_dataset['test_set_y'].shape)
```

Expected output:
```
Training set images shape: (1080, 64, 64, 3)
Training set labels shape: (1080,)
Test set images shape: (120, 64, 64, 3)
Test set labels shape: (120,)
```

## License and Attribution

Please ensure you have the appropriate rights to use the dataset. The original dataset is part of the Deep Learning Specialization course materials by DeepLearning.AI and should be used in accordance with Coursera's terms of service.

---

**Note**: The notebook will not run without these dataset files. Make sure to obtain and place them here before running the exercises.

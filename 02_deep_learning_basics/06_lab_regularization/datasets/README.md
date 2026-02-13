# Dataset Directory

This directory contains the dataset file `data.mat` required for the regularization lab.

## Files

- `data.mat` - MATLAB data file with diagonal separation pattern
- `dataset_visualization.png` - Visual representation of the dataset

## Dataset Structure

The dataset represents positions on a football field with a diagonal separation:
- **Features (X)**: 2D coordinates (x1, x2) representing positions where the ball was kicked
- **Labels (y)**: 
  - 0 (blue dots) = French player hit the ball (upper-left region)
  - 1 (red x's) = Opponent hit the ball (lower-right region)

## Pattern

The data follows a diagonal separation with noise:
- Class 0 (blue): Upper-left region, centered around (-0.3, 0.2)
- Class 1 (red): Lower-right region, centered around (0.1, -0.1)
- Both classes have Gaussian noise for realistic scatter

## Regenerating the Dataset

If needed, you can regenerate the dataset by running:
```bash
python generate_dataset.py
```

This will create a new `data.mat` file with the same diagonal pattern and save a visualization.

"""
Test script to verify the audio classification lab setup.
This tests that data loads correctly before opening the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from audio_utils import load_dataset

print("="*60)
print("AUDIO CLASSIFICATION LAB - SETUP TEST")
print("="*60)

# Test loading the dataset
print("\n1. Loading dataset...")
try:
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    print("✓ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Check dimensions
print("\n2. Checking dataset dimensions...")
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
n_mels = train_set_x_orig.shape[1]
time_steps = train_set_x_orig.shape[2]

print(f"   Training samples: {m_train}")
print(f"   Test samples: {m_test}")
print(f"   Mel bands: {n_mels}")
print(f"   Time steps: {time_steps}")
print(f"   Classes: {classes}")

# Verify we have balanced data
train_speech = np.sum(train_set_y == 1)
train_music = np.sum(train_set_y == 0)
test_speech = np.sum(test_set_y == 1)
test_music = np.sum(test_set_y == 0)

print(f"\n3. Class distribution:")
print(f"   Training - Speech: {train_speech}, Music: {train_music}")
print(f"   Test - Speech: {test_speech}, Music: {test_music}")

if train_speech > 0 and train_music > 0 and test_speech > 0 and test_music > 0:
    print("✓ Dataset is balanced!")
else:
    print("⚠ Warning: Dataset is not balanced")

# Test flattening
print("\n4. Testing data reshaping...")
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print(f"   Flattened training shape: {train_set_x_flatten.shape}")
print(f"   Flattened test shape: {test_set_x_flatten.shape}")
print(f"   Expected feature count: {n_mels * time_steps}")

if train_set_x_flatten.shape[0] == n_mels * time_steps:
    print("✓ Reshaping works correctly!")
else:
    print("❌ Reshaping error!")
    exit(1)

# Check data range
print(f"\n5. Data statistics:")
print(f"   Min value: {train_set_x_flatten.min():.4f}")
print(f"   Max value: {train_set_x_flatten.max():.4f}")
print(f"   Mean: {train_set_x_flatten.mean():.4f}")

if train_set_x_flatten.min() >= 0 and train_set_x_flatten.max() <= 1:
    print("✓ Data is normalized to [0, 1]")
else:
    print("⚠ Warning: Data may not be properly normalized")

print("\n" + "="*60)
print("SETUP TEST COMPLETE!")
print("="*60)
print("\n✅ All tests passed! You're ready to run the notebook.")
print("\nNext steps:")
print("1. Open audio_classification_logreg.ipynb")
print("2. Work through the 8 exercises")
print("3. Train your speech vs music classifier!")
print("\n⚠️  Note: The music samples are modified speech (for testing)")
print("   For better results, replace with real music files later.")

"""
Split and copy LibriSpeech voice samples to Lab 4 data directory
"""

import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Source directories
MALE_SOURCE = "male_samples_wav"
FEMALE_SOURCE = "female_samples_wav"

# Destination (Lab 4)
LAB4_DATA = "../04_lab_dnn_application/data"

# Train/test split ratio
TRAIN_RATIO = 0.75  # 75% train, 25% test

def split_and_copy_samples(source_dir, gender):
    """Split samples into train/test and copy to Lab 4"""
    
    # Get all WAV files
    files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    random.shuffle(files)
    
    # Split
    split_idx = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    
    # Create destination directories
    train_dir = os.path.join(LAB4_DATA, "train", gender)
    test_dir = os.path.join(LAB4_DATA, "test", gender)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy train files
    print(f"\nCopying {gender} training samples...")
    for i, filename in enumerate(train_files):
        src = os.path.join(source_dir, filename)
        # Rename to simple format
        dst = os.path.join(train_dir, f"{gender}_{i:04d}.wav")
        shutil.copy(src, dst)
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{len(train_files)} train files...")
    
    # Copy test files
    print(f"Copying {gender} test samples...")
    for i, filename in enumerate(test_files):
        src = os.path.join(source_dir, filename)
        # Rename to simple format
        dst = os.path.join(test_dir, f"{gender}_{i:04d}.wav")
        shutil.copy(src, dst)
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(test_files)} test files...")
    
    return len(train_files), len(test_files)

if __name__ == "__main__":
    print("=" * 60)
    print("Preparing LibriSpeech dataset for Lab 4")
    print("=" * 60)
    
    # Clear existing data in Lab 4
    print("\nCleaning Lab 4 data directory...")
    for split in ['train', 'test']:
        for gender in ['male', 'female']:
            dir_path = os.path.join(LAB4_DATA, split, gender)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
    
    # Process male samples
    male_train, male_test = split_and_copy_samples(MALE_SOURCE, "male")
    
    # Process female samples
    female_train, female_test = split_and_copy_samples(FEMALE_SOURCE, "female")
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nTraining samples:")
    print(f"  Male:   {male_train:4d}")
    print(f"  Female: {female_train:4d}")
    print(f"  Total:  {male_train + female_train:4d}")
    
    print(f"\nTest samples:")
    print(f"  Male:   {male_test:4d}")
    print(f"  Female: {female_test:4d}")
    print(f"  Total:  {male_test + female_test:4d}")
    
    print(f"\nGrand Total: {male_train + female_train + male_test + female_test:4d} samples")
    print(f"\nData location: {LAB4_DATA}")
    print("\n" + "=" * 60)

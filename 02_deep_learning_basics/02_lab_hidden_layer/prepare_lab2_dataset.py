"""
Split and copy LibriSpeech voice samples to Lab 2 data directory
Creates a smaller dataset: 120 train + 40 test per gender
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

# Destination (Lab 2)
LAB2_DATA = "data"

# Fixed sample counts for Lab 2
TRAIN_SAMPLES = 120  # 120 train samples per gender
TEST_SAMPLES = 40    # 40 test samples per gender

def split_and_copy_samples(source_dir, gender):
    """Split samples into train/test and copy to Lab 2"""
    
    # Get all WAV files
    files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    random.shuffle(files)
    
    # Take fixed number of samples
    train_files = files[:TRAIN_SAMPLES]
    test_files = files[TRAIN_SAMPLES:TRAIN_SAMPLES + TEST_SAMPLES]
    
    # Verify we have enough samples
    if len(train_files) < TRAIN_SAMPLES:
        print(f"⚠ Warning: Only {len(train_files)}/{TRAIN_SAMPLES} train samples available for {gender}")
    if len(test_files) < TEST_SAMPLES:
        print(f"⚠ Warning: Only {len(test_files)}/{TEST_SAMPLES} test samples available for {gender}")
    
    # Create destination directories
    train_dir = os.path.join(LAB2_DATA, "train", gender)
    test_dir = os.path.join(LAB2_DATA, "test", gender)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy train files
    print(f"\nCopying {gender} training samples...")
    for i, filename in enumerate(train_files):
        src = os.path.join(source_dir, filename)
        # Rename to simple format
        dst = os.path.join(train_dir, f"{gender}_{i:04d}.wav")
        shutil.copy(src, dst)
    print(f"  ✓ Copied {len(train_files)} train files")
    
    # Copy test files
    print(f"Copying {gender} test samples...")
    for i, filename in enumerate(test_files):
        src = os.path.join(source_dir, filename)
        # Rename to simple format
        dst = os.path.join(test_dir, f"{gender}_{i:04d}.wav")
        shutil.copy(src, dst)
    print(f"  ✓ Copied {len(test_files)} test files")
    
    return len(train_files), len(test_files)

if __name__ == "__main__":
    print("=" * 60)
    print("Preparing LibriSpeech dataset for Lab 2")
    print("=" * 60)
    
    # Check source directories exist
    if not os.path.exists(MALE_SOURCE):
        print(f"\n✗ Error: {MALE_SOURCE} not found!")
        print("Run sort_dev_clean_by_gender.py and convert_flac_to_wav.py first")
        exit(1)
    
    if not os.path.exists(FEMALE_SOURCE):
        print(f"\n✗ Error: {FEMALE_SOURCE} not found!")
        print("Run sort_dev_clean_by_gender.py and convert_flac_to_wav.py first")
        exit(1)
    
    # Clear existing data in Lab 2
    print("\nCleaning Lab 2 data directory...")
    for split in ['train', 'test']:
        for gender in ['male', 'female']:
            dir_path = os.path.join(LAB2_DATA, split, gender)
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
    print(f"\nData location: {LAB2_DATA}")
    print("\n" + "=" * 60)
    print("\nNext steps:")
    print("1. Run: python test_setup.py")
    print("2. Open: audio_classification_nn.ipynb")
    print("=" * 60)

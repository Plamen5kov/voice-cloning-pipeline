#!/usr/bin/env python3
"""
Complete setup script for Lab 2 - Male vs Female Voice Classification
Runs the full data preparation pipeline
"""

import os
import sys
import subprocess

def run_command(description, command):
    """Run a command and handle errors"""
    print("\n" + "=" * 70)
    print(f"Step: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {description}")
        print(f"Command failed with exit code {e.returncode}")
        return False

def check_librispeech_exists():
    """Check if LibriSpeech is already downloaded"""
    return os.path.exists("data/LibriSpeech/dev-clean")

def main():
    """Run the complete setup pipeline"""
    print("=" * 70)
    print("Lab 2: Voice Gender Classification - Data Setup")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Download LibriSpeech dev-clean (~350MB) if not present")
    print("  2. Organize samples by speaker gender")
    print("  3. Convert FLAC files to WAV format")
    print("  4. Create train/test split (120+40 per gender)")
    print("\nTotal time: ~10-15 minutes (first run)")
    
    response = input("\nContinue? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Setup cancelled.")
        return
    
    # Step 1: Download LibriSpeech (if needed)
    if check_librispeech_exists():
        print("\n✓ LibriSpeech already downloaded, skipping download step")
    else:
        if not run_command(
            "Downloading LibriSpeech dev-clean",
            "python download_samples.py"
        ):
            print("\n✗ Setup failed at download step")
            return 1
    
    # Step 2: Organize by gender
    if not run_command(
        "Organizing samples by gender",
        "python sort_dev_clean_by_gender.py"
    ):
        print("\n✗ Setup failed at organization step")
        return 1
    
    # Step 3: Convert to WAV
    if not run_command(
        "Converting FLAC to WAV format",
        "python convert_flac_to_wav.py"
    ):
        print("\n✗ Setup failed at conversion step")
        return 1
    
    # Step 4: Create dataset split
    if not run_command(
        "Creating train/test split",
        "python prepare_lab2_dataset.py"
    ):
        print("\n✗ Setup failed at dataset preparation step")
        return 1
    
    # Step 5: Verify setup
    if not run_command(
        "Verifying setup",
        "python test_setup.py"
    ):
        print("\n⚠ Warning: Setup verification had issues")
        print("You may still be able to run the notebook")
    
    # Success!
    print("\n" + "=" * 70)
    print("✓ SETUP COMPLETE!")
    print("=" * 70)
    print("\nYour dataset is ready:")
    print("  - Training samples: 240 (120 male + 120 female)")
    print("  - Test samples: 80 (40 male + 40 female)")
    print("  - Location: data/train/ and data/test/")
    print("\nNext steps:")
    print("  1. Open audio_classification_nn.ipynb")
    print("  2. Run the notebook cells to train your neural network")
    print("\nHappy learning! 🎵🤖")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
Verification script to ensure lab setup is correct.
Run this after downloading samples to verify everything is ready.
"""

import os
import sys
import numpy as np

def check_directories():
    """Check that all required directories exist."""
    required_dirs = [
        'data',
        'data/train',
        'data/train/female',
        'data/train/male',
        'data/test',
        'data/test/female',
        'data/test/male'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"✗ Missing directory: {dir_path}")
            return False
    
    print("✓ All directories exist")
    return True


def check_samples():
    """Check that audio samples are present and correctly distributed."""
    train_female = len([f for f in os.listdir('data/train/female') if f.endswith('.wav')])
    train_male = len([f for f in os.listdir('data/train/male') if f.endswith('.wav')])
    test_female = len([f for f in os.listdir('data/test/female') if f.endswith('.wav')])
    test_male = len([f for f in os.listdir('data/test/male') if f.endswith('.wav')])
    
    total_train = train_female + train_male
    total_test = test_female + test_male
    
    print(f"✓ Training data: {total_train} samples ({train_female} female + {train_male} male)")
    print(f"✓ Test data: {total_test} samples ({test_female} female + {test_male} male)")
    
    if total_train < 100 or total_test < 50:
        print("⚠ Warning: Sample counts seem low. Expected at least 100 train and 50 test.")
        return False
    
    return True


def check_audio_properties():
    """Check audio file properties."""
    try:
        import librosa
        import soundfile as sf
        
        # Check a sample file
        sample_file = None
        for root, dirs, files in os.walk('data/train/female'):
            for file in files:
                if file.endswith('.wav'):
                    sample_file = os.path.join(root, file)
                    break
            if sample_file:
                break
        
        if not sample_file:
            print("✗ Could not find any WAV files")
            return False
        
        # Load and check properties
        audio, sr = librosa.load(sample_file, sr=None)
        duration = len(audio) / sr
        
        print(f"✓ Sample rate: {sr} Hz")
        print(f"✓ Sample duration: {duration:.1f} seconds")
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing library: {e}")
        print("  Please install: pip install librosa soundfile")
        return False
    except Exception as e:
        print(f"✗ Error checking audio: {e}")
        return False


def check_data_loading():
    """Check that audio_utils can load the dataset."""
    try:
        from audio_utils import load_dataset
        
        print("\nLoading dataset...")
        train_x, train_y, test_x, test_y, classes = load_dataset()
        
        print(f"✓ Training features shape: {train_x.shape}")
        print(f"✓ Training labels shape: {train_y.shape}")
        print(f"✓ Test features shape: {test_x.shape}")
        print(f"✓ Test labels shape: {test_y.shape}")
        print(f"✓ Classes: {classes}")
        
        # Check that data is normalized
        if train_x.max() <= 1.0 and train_x.min() >= 0.0:
            print("✓ Features are normalized to [0, 1]")
        else:
            print(f"⚠ Warning: Features not in [0, 1] range: [{train_x.min():.3f}, {train_x.max():.3f}]")
        
        # Check label distribution
        train_female = np.sum(train_y == 0)
        train_male = np.sum(train_y == 1)
        print(f"✓ Training label distribution: {train_female} female (0), {train_male} male (1)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def main():
    """Run all setup checks."""
    print("=" * 60)
    print("Lab Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Directory structure", check_directories),
        ("Audio samples", check_samples),
        ("Audio properties", check_audio_properties),
        ("Data loading", check_data_loading)
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        print(f"\n[{name}]")
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"✗ Check failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ Setup complete! You're ready to start the lab.")
        print("   Open: audio_classification_nn.ipynb")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        print("   Run: python download_samples.py")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

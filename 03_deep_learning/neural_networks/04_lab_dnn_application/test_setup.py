"""
Test Setup for Lab 4: Deep Neural Network Application

This script tests that the environment is properly configured.
"""

import sys


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('h5py', None),
        ('scipy', None),
        ('PIL', None),
    ]
    
    failed = []
    
    for package_info in required_packages:
        if isinstance(package_info, tuple):
            package, alias = package_info
        else:
            package = package_info
            alias = None
            
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True


def test_python_version():
    """Check Python version"""
    print(f"\nPython version: {sys.version}")
    
    version_info = sys.version_info
    if version_info.major >= 3 and version_info.minor >= 7:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.7+ is required")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Lab 4: Audio Genre Classification DNN - Environment Test")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_python_version()
    all_passed &= test_imports()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! You're ready to start the lab.")
        print("\nNext steps:")
        print("1. Add audio files to data/train/ and data/test/ directories")
        print("2. Open audio_genre_dnn.ipynb and follow the instructions")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

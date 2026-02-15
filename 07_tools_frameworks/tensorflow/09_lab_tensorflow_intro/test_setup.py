#!/usr/bin/env python3
"""
Test setup script for TensorFlow Introduction Lab
Verifies that all required packages are installed and the environment is ready.
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        
        # Check TensorFlow version
        major, minor, _ = tf.__version__.split('.')
        if int(major) < 2 or (int(major) == 2 and int(minor) < 16):
            print(f"  ⚠ Warning: TensorFlow 2.16+ recommended, you have {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow: {e}")
        return False
    
    try:
        import h5py
        print(f"✓ h5py {h5py.__version__}")
    except ImportError as e:
        print(f"✗ h5py: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        from tensorflow.python.framework.ops import EagerTensor
        from tensorflow.python.ops.resource_variable_ops import ResourceVariable
        print("✓ TensorFlow components (EagerTensor, ResourceVariable)")
    except ImportError as e:
        print(f"✗ TensorFlow components: {e}")
        return False
    
    return True


def check_tensorflow_gpu():
    """Check if TensorFlow has GPU support."""
    print("\nChecking GPU support...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ {len(gpus)} GPU(s) available:")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("ℹ No GPU detected. TensorFlow will use CPU.")
            print("  (This is fine for learning, but training will be slower)")
    except Exception as e:
        print(f"⚠ Error checking GPU: {e}")


def check_dataset():
    """Check if dataset files are present."""
    print("\nChecking dataset files...")
    
    dataset_dir = "datasets"
    required_files = ["train_signs.h5", "test_signs.h5"]
    
    all_present = True
    for filename in required_files:
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {filename} - NOT FOUND")
            all_present = False
    
    if not all_present:
        print("\n⚠ Missing dataset files!")
        print(f"  Please download the required files and place them in the '{dataset_dir}/' directory.")
        print(f"  See '{dataset_dir}/README.md' for instructions.")
    
    return all_present


def test_basic_tensorflow_operations():
    """Test basic TensorFlow operations."""
    print("\nTesting basic TensorFlow operations...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Test constant creation
        x = tf.constant([1, 2, 3])
        assert x.shape == (3,), "Constant creation failed"
        print("✓ Constant creation")
        
        # Test variable creation
        v = tf.Variable([1.0, 2.0, 3.0])
        assert isinstance(v, tf.Variable), "Variable creation failed"
        print("✓ Variable creation")
        
        # Test matrix multiplication
        a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
        c = tf.matmul(a, b)
        assert c.shape == (2, 2), "Matrix multiplication failed"
        print("✓ Matrix multiplication")
        
        # Test gradient tape
        with tf.GradientTape() as tape:
            x = tf.Variable(3.0)
            y = x ** 2
        dy_dx = tape.gradient(y, x)
        assert abs(dy_dx.numpy() - 6.0) < 1e-5, "GradientTape failed"
        print("✓ GradientTape")
        
        return True
    except Exception as e:
        print(f"✗ TensorFlow operations failed: {e}")
        return False


def main():
    """Run all setup tests."""
    print("=" * 60)
    print("TensorFlow Introduction Lab - Setup Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Check GPU
    check_tensorflow_gpu()
    
    # Check dataset
    dataset_ok = check_dataset()
    
    # Test basic operations
    operations_ok = test_basic_tensorflow_operations()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if imports_ok and operations_ok:
        print("✓ Environment setup: READY")
    else:
        print("✗ Environment setup: INCOMPLETE")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
    
    if not dataset_ok:
        print("⚠ Dataset files: MISSING")
        print("  The lab will not run without the dataset files.")
        print("  See datasets/README.md for instructions.")
    else:
        print("✓ Dataset files: PRESENT")
    
    print("\nIf all tests passed, you're ready to start the lab!")
    print("Run: jupyter notebook tensorflow_intro.ipynb")
    print("=" * 60)
    
    return imports_ok and operations_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

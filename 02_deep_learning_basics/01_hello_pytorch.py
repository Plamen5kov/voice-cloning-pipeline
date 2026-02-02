"""
Deep Learning "Hello World" - Simple tensor operations with PyTorch
"""

import torch
import numpy as np
from dl_utils import get_device, print_header

def hello_pytorch():
    """Basic PyTorch operations"""
    print_header("PyTorch Hello World")
    
    # Check PyTorch version and CUDA availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create tensors
    print("\n--- Creating Tensors ---")
    x = torch.tensor([1, 2, 3, 4, 5])
    print(f"1D Tensor: {x}")
    
    y = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(f"2D Tensor:\n{y}")
    
    # Tensor operations
    print("\n--- Basic Operations ---")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a.mean() = {a.mean()}")
    print(f"a.std() = {a.std()}")
    
    # Matrix multiplication
    print("\n--- Matrix Operations ---")
    m1 = torch.tensor([[1, 2], [3, 4]])
    m2 = torch.tensor([[5, 6], [7, 8]])
    print(f"Matrix 1:\n{m1}")
    print(f"Matrix 2:\n{m2}")
    print(f"Matrix multiplication:\n{torch.matmul(m1, m2)}")
    
    # Random tensors
    print("\n--- Random Tensors ---")
    random_tensor = torch.randn(3, 3)
    print(f"Random tensor (3x3):\n{random_tensor}")
    
    # GPU operations (if available)
    device = get_device()
    if torch.cuda.is_available():
        print("\n--- GPU Operations ---")
        gpu_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        print(f"Tensor on GPU: {gpu_tensor}")
        print(f"Device: {gpu_tensor.device}")
        
        # Move back to CPU
        cpu_tensor = gpu_tensor.cpu()
        print(f"Moved to CPU: {cpu_tensor}")
    
    print_header("✓ PyTorch Hello World Complete!")

if __name__ == "__main__":
    hello_pytorch()

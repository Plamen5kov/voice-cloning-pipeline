"""
Load MNIST dataset and visualize samples

WHY DATA EXPLORATION MATTERS:
- Understanding your data is the first step in ML
- Helps identify problems (class imbalance, corrupted data, etc.)
- Informs architecture and preprocessing choices
- Visualization reveals patterns humans can see but models might miss
"""

import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from dl_utils import load_mnist_data, print_header, print_section, save_plot

def load_mnist():
    """
    Load MNIST dataset
    
    MNIST DATASET:
    - 70,000 handwritten digit images (0-9)
    - 60,000 training + 10,000 test samples
    - Each image: 28x28 grayscale pixels
    - One of the most famous ML datasets (the "Hello World" of computer vision)
    - Created in 1998, still used for education and benchmarking
    
    WHY IT'S GOOD FOR LEARNING:
    - Small enough to train quickly on CPU/GPU
    - Complex enough to require a neural network
    - Simple enough that most architectures work
    - Lets you focus on ML concepts, not data preprocessing
    """
    print("Loading MNIST dataset...")
    
    # Transform to convert images to tensors and normalize
    # Mean=0.1307, Std=0.3081 computed from the training set
    transform = transforms.Compose([
        transforms.ToTensor(),  # PIL Image or numpy -> Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Standardization
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def visualize_samples(dataset, num_samples=10):
    """
    Visualize random samples from the dataset
    
    WHY VISUALIZE:
    - Sanity check: are images loading correctly?
    - Understand difficulty: are digits clear or ambiguous?
    - Spot issues: corrupted images, wrong labels, etc.
    - Build intuition: what makes a '3' different from an '8'?
    
    WHAT TO LOOK FOR:
    - Image quality and clarity
    - Variation within each class (different handwriting styles)
    - Potential confusion between similar digits (1 vs 7, 3 vs 8)
    - Label accuracy
    """
    print(f"\nVisualizing {num_samples} random samples...")
    
    # Get random indices
    indices = np.random.randint(0, len(dataset), num_samples)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('MNIST Dataset Samples', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        # Get image and label
        image, label = dataset[indices[idx]]
        
        # Convert from tensor to numpy and remove normalization for visualization
        image = image.squeeze().numpy()
        
        # Display image
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    
    save_plot('mnist_samples.png')

def show_dataset_stats(train_dataset, test_dataset):
    """Show statistics about the dataset"""
    print_section("Dataset Statistics")
    
    # Count samples per class in training set
    train_labels = [label for _, label in train_dataset]
    unique, counts = np.unique(train_labels, return_counts=True)
    
    print("\nTraining set distribution:")
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit}: {count:5d} samples")
    
    # Image shape
    sample_image, _ = train_dataset[0]
    print(f"\nImage shape: {sample_image.shape}")
    print(f"  - Channels: {sample_image.shape[0]}")
    print(f"  - Height: {sample_image.shape[1]}")
    print(f"  - Width: {sample_image.shape[2]}")
    
    print("="*60)

def main():
    print_header("MNIST Dataset Loading and Visualization")
    
    # Load dataset
    print("\nLoading MNIST dataset...")
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(batch_size=64, data_dir='./data')
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Show statistics
    show_dataset_stats(train_dataset, test_dataset)
    
    # Visualize samples
    visualize_samples(train_dataset, num_samples=10)
    
    print("\n✓ Complete!")

if __name__ == "__main__":
    main()

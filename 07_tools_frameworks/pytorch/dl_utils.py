"""
Utility functions for deep learning exercises
Provides reusable code for device detection, data loading, model evaluation, and formatting
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_device():
    """
    Detect and return the available device (CUDA or CPU)
    
    WHY THIS MATTERS:
    - GPUs can train neural networks 10-100x faster than CPUs
    - Modern deep learning is only practical because of GPU acceleration
    - GPUs have thousands of cores that can process matrix operations in parallel
    - This is crucial for large models and datasets
    
    Returns:
        torch.device: The device to use for training/inference
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    return device

def load_mnist_data(batch_size=64, data_dir='./data'):
    """
    Load MNIST dataset with standard normalization
    
    WHY THESE CHOICES MATTER:
    
    1. NORMALIZATION (0.1307, 0.3081):
       - These are the mean and std of MNIST dataset
       - Normalizing helps the network learn faster and more stably
       - Without it, large pixel values (0-255) cause unstable gradients
       - Normalized data centers around 0, making optimization easier
    
    2. BATCH SIZE (default 64):
       - Training uses batches instead of individual samples for efficiency
       - Larger batches = more stable gradients but slower updates
       - Smaller batches = noisier gradients but faster learning
       - 64 is a good balance and fits well in GPU memory
    
    3. SHUFFLE=True for training:
       - Randomizes the order each epoch to prevent the model from
         learning spurious patterns based on data order
       - Essential for good generalization
    
    4. SHUFFLE=False for testing:
       - Test set doesn't need shuffling since we're just measuring performance
       - Keeps results reproducible
    
    Args:
        batch_size (int): Number of samples per batch (affects memory and convergence)
        data_dir (str): Directory to store/load MNIST data
    
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    # Convert images to tensors and normalize to mean=0, std=1
    # This makes the network train faster and more reliably
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image [0,255] to tensor [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Standardize using dataset statistics
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # DataLoader handles batching and shuffling automatically
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset

def evaluate_model(model, data_loader, device, criterion=None):
    """
    Evaluate model accuracy on a dataset
    
    WHY THIS MATTERS:
    
    1. model.eval():
       - Disables dropout and batch normalization training behavior
       - Essential for consistent evaluation results
       - Without this, results would vary randomly
    
    2. torch.no_grad():
       - Disables gradient calculation during evaluation
       - Saves memory (gradients not needed for inference)
       - Makes evaluation ~2x faster
       - Critical for large models
    
    3. Separate train/test evaluation:
       - Training accuracy measures memorization
       - Test accuracy measures generalization (what we really care about)
       - Gap between them indicates overfitting
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on
        criterion: Optional loss function to compute average loss
    
    Returns:
        tuple: (accuracy, average_loss) or just accuracy if criterion is None
    """
    model.eval()  # Switch to evaluation mode (affects dropout, batchnorm, etc.)
    correct = 0
    total = 0
    total_loss = 0.0
    
    # Don't compute gradients during evaluation - saves memory and time
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if criterion is not None:
                loss = criterion(output, target)
                total_loss += loss.item()
            
            # Get the class with highest probability
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    
    if criterion is not None:
        avg_loss = total_loss / len(data_loader)
        return accuracy, avg_loss
    
    return accuracy

def count_parameters(model):
    """
    Count total number of trainable parameters in a model
    
    WHY THIS MATTERS:
    - More parameters = more capacity to learn complex patterns
    - But also = more memory, slower training, risk of overfitting
    - Typical ranges:
      - Small: 100K-1M (like our MNIST model)
      - Medium: 10M-100M (ResNet, BERT-small)
      - Large: 1B-100B+ (GPT, LLaMA)
    - This helps you understand model size and computational requirements
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_header(text, width=60, char='='):
    """
    Print a formatted header
    
    Args:
        text (str): Header text
        width (int): Total width of the header
        char (str): Character to use for border
    """
    print(f"\n{char * width}")
    print(f"{text}")
    print(f"{char * width}")

def print_section(text, width=60, char='-'):
    """
    Print a formatted section separator
    
    Args:
        text (str): Section text
        width (int): Total width of the separator
        char (str): Character to use for border
    """
    print(f"\n{char * width}")
    print(f"{text}")
    print(f"{char * width}")

def save_plot(filename, dpi=150):
    """
    Save matplotlib plot with standard settings
    
    Args:
        filename (str): Output filename
        dpi (int): Resolution in dots per inch
    """
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"\n✓ Plot saved to '{filename}'")
    plt.close()

def print_model_info(model, model_name="Model"):
    """
    Print model information including architecture and parameter count
    
    Args:
        model: PyTorch model
        model_name (str): Name of the model
    """
    params = count_parameters(model)
    print(f"\n{model_name} Architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {params:,}")

def plot_training_history(train_accs, val_accs, filename='training_history.png'):
    """
    Plot training and validation accuracy over epochs
    
    WHY PLOTTING MATTERS:
    - Visual inspection reveals training problems:
      - Gap widening = overfitting (model memorizing training data)
      - Both plateauing = need more capacity or better architecture
      - Validation dropping while training rises = severe overfitting
      - Both improving = healthy training
    - Helps decide when to stop training (early stopping)
    - Essential for diagnosing model behavior
    
    Args:
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        filename (str): Output filename
    """
    epochs = range(1, len(train_accs) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-o', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    save_plot(filename)

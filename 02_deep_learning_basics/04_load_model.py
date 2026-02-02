"""
Save and load trained models - Model Persistence

WHY MODEL PERSISTENCE MATTERS:
- Training takes time and computational resources
- Need to save models to use them later without retraining
- Essential for deploying models to production
- Allows sharing models with others
- Enables transfer learning (using pre-trained models)

TWO WAYS TO SAVE IN PYTORCH:
1. Save state_dict (recommended) - Just the weights
   - Smaller file size
   - More flexible (can load into different code)
   - Requires model architecture definition

2. Save entire model
   - Includes architecture
   - Less flexible, can break with code changes
   - Not recommended for production
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dl_utils import get_device, load_mnist_data, evaluate_model, print_header, print_section

class SimpleNN(nn.Module):
    """
    Simple feedforward neural network for MNIST
    
    IMPORTANT: This must match the architecture used during training!
    If you change the architecture, you can't load old weights.
    """
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_model(model_path='mnist_model.pth', device='cuda'):
    """
    Load a saved model from disk
    
    LOADING PROCESS:
    1. Create model instance with same architecture
    2. Load saved weights (state_dict)
    3. Move to appropriate device (CPU/GPU)
    4. Set to eval mode (important!)
    
    COMMON ISSUES:
    - Architecture mismatch: model definition changed
    - Device mismatch: saved on GPU, loading on CPU
    - Path errors: file not found
    
    Args:
        model_path: Path to saved .pth file
        device: Device to load model onto
    
    Returns:
        Loaded model ready for inference
    """
    model = SimpleNN().to(device)
    
    # Load saved weights
    # map_location ensures model loads on correct device
    # weights_only=True is safer (prevents arbitrary code execution)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # CRITICAL: Set to evaluation mode
    # Disables dropout, changes batch norm behavior
    model.eval()
    
    print(f"✓ Model loaded from '{model_path}'")
    return model

def test_model(model, device):
    """
    Test the loaded model on the test set
    
    VALIDATION VS TESTING:
    - Validation: used during training to tune hyperparameters
    - Testing: final evaluation on completely unseen data
    - Test set should only be used once (otherwise you're "cheating")
    
    This confirms the model maintained its performance after saving/loading
    """
    # Load test data
    _, test_loader, _, _ = load_mnist_data(batch_size=64)
    
    # Evaluate
    accuracy = evaluate_model(model, test_loader, device)
    print(f"✓ Test Accuracy: {accuracy:.2f}%")
    
    return accuracy

def predict_sample(model, device, sample_idx=0):
    """
    Make prediction on a single sample
    
    SINGLE SAMPLE INFERENCE:
    - Shows how model would be used in production
    - Process: Load image -> Preprocess -> Forward pass -> Interpret output
    - Same preprocessing must be applied as during training!
    
    OUTPUT INTERPRETATION:
    - Model outputs 10 numbers (logits) - one per class
    - Higher number = more confident prediction
    - Apply softmax to convert to probabilities
    - argmax gives the predicted class
    """
    # Load a sample
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    image, true_label = test_dataset[sample_idx]
    
    # Make prediction
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted].item() * 100
    
    print(f"\nSample #{sample_idx}:")
    print(f"  True label: {true_label}")
    print(f"  Predicted: {predicted}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  Correct: {'✓' if predicted == true_label else '✗'}")

def main():
    print_header("Load and Test Saved Model")
    
    # Set device
    device = get_device()
    
    # Load model
    print("\nLoading model...")
    model = load_model(device=device)
    
    # Test model
    print("\nTesting model on test set...")
    test_model(model, device)
    
    # Make predictions on individual samples
    print_section("Making predictions on individual samples")
    for i in range(5):
        predict_sample(model, device, sample_idx=i)
    
    print_header("✓ Complete!")

if __name__ == "__main__":
    main()

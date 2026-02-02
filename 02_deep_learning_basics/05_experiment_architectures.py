"""
Experiment with different neural network architectures
Compare models with different layers, activations, and sizes

WHY ARCHITECTURE MATTERS:
- Different problems need different architectures
- More layers = can learn more complex patterns (but harder to train)
- Different activations have different properties
- This script helps you understand these tradeoffs empirically
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dl_utils import (
    get_device, load_mnist_data, evaluate_model, count_parameters,
    print_header, print_section, save_plot
)

class TwoLayerNN(nn.Module):
    """
    Simple 2-layer network (baseline)
    
    ARCHITECTURE:
    - Input (784) -> Hidden (128) -> Output (10)
    - Single hidden layer with ReLU activation
    - ~101K parameters
    
    PROS:
    - Fast to train
    - Less prone to overfitting
    - Good for simple problems
    
    CONS:
    - Limited capacity for complex patterns
    - Might underfit on harder problems
    """
    def __init__(self):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ThreeLayerNN(nn.Module):
    """
    Deeper 3-layer network
    
    ARCHITECTURE:
    - Input (784) -> Hidden1 (256) -> Hidden2 (128) -> Output (10)
    - Two hidden layers with ReLU activations
    - ~235K parameters (2.3x more than TwoLayerNN)
    
    WHY GO DEEPER:
    - Can learn hierarchical features (edges -> shapes -> objects)
    - More expressive (can represent more complex functions)
    - Universal approximation theorem: deeper = more efficient
    
    TRADEOFFS:
    - Slower to train (more parameters)
    - Needs more data to avoid overfitting
    - Harder to optimize (vanishing gradients in very deep nets)
    
    FOR MNIST:
    - Probably overkill (MNIST is simple)
    - But demonstrates the concept
    """
    def __init__(self):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class TanhNN(nn.Module):
    """
    Network with Tanh activation (instead of ReLU)
    
    ACTIVATION FUNCTION COMPARISON:
    
    Tanh (this model):
    - Output range: [-1, 1]
    - Smooth, differentiable everywhere
    - Outputs centered around 0 (good for some problems)
    - PROBLEM: Saturates (gradients near 0 for large inputs)
    - Used more in older networks and RNNs
    
    ReLU (other models):
    - Output range: [0, ∞)
    - f(x) = max(0, x)
    - Doesn't saturate for positive values
    - Sparse activation (many neurons output 0)
    - PROBLEM: "Dead ReLU" (neurons stuck at 0)
    - Currently the standard for most deep networks
    
    EXPECTATION:
    - ReLU should train faster and reach higher accuracy
    - Tanh might converge more smoothly but to a lower accuracy
    - Both should work, demonstrating robustness of the approach
    """
    def __init__(self):
        super(TanhNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.tanh = nn.Tanh()  # Compare to ReLU
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

def train_and_evaluate(model, model_name, device, train_loader, test_loader, epochs=5):
    """Train and evaluate a model"""
    print_section(f"Training: {model_name}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Print model info
    total_params = count_parameters(model)
    print(f"Parameters: {total_params:,}")
    
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        val_acc = evaluate_model(model, test_loader, device)
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
    
    return train_accs, val_accs

def plot_comparison(results):
    """Plot comparison of different models"""
    plt.figure(figsize=(10, 6))
    
    for model_name, (train_accs, val_accs) in results.items():
        epochs = range(1, len(val_accs) + 1)
        plt.plot(epochs, val_accs, marker='o', label=model_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Model Architecture Comparison')
    plt.legend()
    plt.grid(True)
    
    save_plot('architecture_comparison.png')

def main():
    print_header("Neural Network Architecture Experiments")
    
    # Set device
    device = get_device()
    
    # Load data
    print("\nLoading data...")
    train_loader, test_loader, _, _ = load_mnist_data(batch_size=64)
    
    # Define models to compare
    models = {
        "2-Layer (ReLU)": TwoLayerNN(),
        "3-Layer (ReLU)": ThreeLayerNN(),
        "2-Layer (Tanh)": TanhNN()
    }
    
    # Train and evaluate each model
    results = {}
    for model_name, model in models.items():
        train_accs, val_accs = train_and_evaluate(
            model, model_name, device, train_loader, test_loader, epochs=5
        )
        results[model_name] = (train_accs, val_accs)
    
    # Plot comparison
    plot_comparison(results)
    
    # Summary
    print_section("Summary")
    for model_name, (_, val_accs) in results.items():
        print(f"{model_name:20s}: {val_accs[-1]:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

"""
Build and train a simple neural network for MNIST digit recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from dl_utils import (
    get_device, load_mnist_data, evaluate_model, print_header,
    print_model_info, plot_training_history
)

class SimpleNN(nn.Module):
    """
    Simple feedforward neural network for MNIST digit classification
    
    ARCHITECTURE DECISIONS:
    
    1. Input size = 784 (28x28 pixels flattened)
       - MNIST images are 28x28 grayscale
       - Flattening loses spatial info but works for simple datasets
       - CNNs (Convolutional Neural Networks) preserve spatial structure better
    
    2. Hidden size = 128
       - The "neurons" in the hidden layer
       - More neurons = more capacity to learn patterns
       - But too many = overfitting and slower training
       - 128 is reasonable for MNIST complexity
    
    3. ReLU activation (Rectified Linear Unit):
       - f(x) = max(0, x)
       - Introduces non-linearity (crucial for learning complex patterns)
       - Without activation, network would just be linear regression
       - ReLU is fast, doesn't saturate, and works well in practice
       - Alternatives: Tanh (slower), Sigmoid (vanishing gradients)
    
    4. Output size = 10 (one for each digit 0-9)
       - Raw scores (logits) before softmax
       - CrossEntropyLoss applies softmax internally
    """
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        # Layer 1: 784 inputs -> 128 hidden neurons
        # Parameters: 784 * 128 + 128 bias = 100,480
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Activation: allows network to learn non-linear patterns
        self.relu = nn.ReLU()
        
        # Layer 2: 128 hidden -> 10 output classes
        # Parameters: 128 * 10 + 10 bias = 1,290
        # Total: ~101,770 trainable parameters
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass: input -> hidden layer -> activation -> output
        This is called automatically during training and inference
        """
        # Flatten 28x28 image to 784-dimensional vector
        x = x.view(x.size(0), -1)  # Keep batch size, flatten rest
        
        # First layer with activation
        x = self.fc1(x)
        x = self.relu(x)  # Non-linearity is essential here
        
        # Output layer (no activation - CrossEntropyLoss handles it)
        x = self.fc2(x)
        return x


def train_epoch(model, device, train_loader, optimizer, criterion):
    """
    Train the model for one complete pass through the training data
    
    THE TRAINING LOOP - Core of deep learning:
    
    1. optimizer.zero_grad():
       - Gradients accumulate by default in PyTorch
       - Must reset to zero before each batch
       - Forgetting this causes incorrect gradient calculations
    
    2. Forward pass (model(data)):
       - Input flows through layers: input -> hidden -> output
       - Network makes predictions based on current weights
    
    3. Compute loss (criterion):
       - Measures how wrong the predictions are
       - CrossEntropyLoss combines softmax + negative log likelihood
       - Lower loss = better predictions
    
    4. Backward pass (loss.backward()):
       - **THE MAGIC OF DEEP LEARNING**
       - Computes gradient of loss with respect to each parameter
       - Uses chain rule (calculus) to propagate error backwards
       - This tells us how to adjust each weight to reduce loss
    
    5. optimizer.step():
       - Actually updates the weights using gradients
       - Adam optimizer: adaptive learning rates per parameter
       - Moves weights in direction that reduces loss
    
    WHY AN EPOCH:
    - One epoch = one pass through entire training set
    - Multiple epochs needed because:
      * One pass isn't enough to learn patterns
      * Gradients are noisy (batch-based approximation)
      * Typical: 10-100+ epochs depending on problem
    """
    model.train()  # Enable training mode (affects dropout, batchnorm if used)
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to GPU/CPU
        data, target = data.to(device), target.to(device)
        
        # STEP 1: Clear gradients from previous batch
        optimizer.zero_grad()
        
        # STEP 2: Forward pass - compute predictions
        output = model(data)
        loss = criterion(output, target)
        
        # STEP 3: Backward pass - compute gradients
        # This is backpropagation - the heart of neural network training
        loss.backward()
        
        # STEP 4: Update weights based on gradients
        optimizer.step()
        
        # Track statistics for monitoring
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, device, test_loader, criterion):
    """Validate the model"""
    accuracy, avg_loss = evaluate_model(model, test_loader, device, criterion)
    return avg_loss, accuracy

def main():
    print_header("MNIST Neural Network Training")
    
    # Set device
    device = get_device()
    
    # HYPERPARAMETERS - These control the learning process
    # ================================================
    
    batch_size = 64
    # Why 64? Common choices: 32, 64, 128, 256
    # - Larger = more stable gradients, better GPU utilization
    # - Smaller = noisier gradients (can escape local minima), faster updates
    # - 64 is a good default for most problems
    
    learning_rate = 0.001
    # How big a step to take when updating weights
    # - Too large: training unstable, might not converge
    # - Too small: training very slow, might get stuck
    # - 0.001 is a good default for Adam optimizer
    # - Different optimizers need different learning rates
    
    num_epochs = 10
    # How many times to iterate through the entire dataset
    # - More epochs = more learning time, but risk overfitting
    # - Stop when validation accuracy plateaus
    # - MNIST is simple, so 10 epochs is enough
    # - Complex datasets might need 100+ epochs
    
    # Load data
    print("\nLoading data...")
    train_loader, test_loader, _, _ = load_mnist_data(batch_size)
    
    # Create model
    print("\nCreating model...")
    model = SimpleNN().to(device)
    
    # LOSS FUNCTION (Criterion)
    # CrossEntropyLoss is standard for classification:
    # - Combines LogSoftmax + NLLLoss
    # - Measures how far predictions are from true labels
    # - Output: lower is better, 0 = perfect prediction
    criterion = nn.CrossEntropyLoss()
    
    # OPTIMIZER - Algorithm for updating weights
    # Adam (Adaptive Moment Estimation):
    # - Adapts learning rate for each parameter
    # - Combines benefits of RMSprop + Momentum
    # - Works well out-of-the-box for most problems
    # - Alternatives: SGD (simpler, sometimes better), AdamW (better for transformers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print_model_info(model, "SimpleNN")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {num_epochs} epochs...")
    print(f"{'='*60}\n")
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_acc = validate(model, device, test_loader, criterion)
        
        # Record history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Plot results
    print("\n")
    plot_training_history(train_accs, val_accs, 'training_history.png')
    
    # Save model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("✓ Model saved to 'mnist_model.pth'")
    
    print_header(f"Training Complete! Final Validation Accuracy: {val_accs[-1]:.2f}%")

if __name__ == "__main__":
    main()

"""
Test cases for neural network with hidden layer exercises.
Provides sample data for testing each function.
"""

import numpy as np


def layer_sizes_test_case():
    """Test case for layer_sizes function."""
    np.random.seed(1)
    X = np.random.randn(5, 3)
    Y = np.random.randn(2, 3)
    return X, Y


def initialize_parameters_test_case():
    """Test case for initialize_parameters function."""
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y


def forward_propagation_test_case():
    """Test case for forward_propagation function."""
    np.random.seed(1)
    X = np.random.randn(2, 3)
    
    b1 = np.random.randn(4, 1)
    b2 = np.array([[1.62434536e-04]])
    
    W1 = np.random.randn(4, 2)
    W2 = np.random.randn(1, 4)
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return X, parameters


def compute_cost_test_case():
    """Test case for compute_cost function."""
    np.random.seed(1)
    Y = np.array([[1, 1, 0]])  # Binary labels for classification
    A2 = np.array([[0.5002307, 0.49985831, 0.50023963]])
    
    return A2, Y


def backward_propagation_test_case():
    """Test case for backward_propagation function."""
    np.random.seed(1)
    X = np.random.randn(2, 3)
    Y = np.random.randn(1, 3)
    
    parameters = {
        'W1': np.random.randn(4, 2),
        'b1': np.random.randn(4, 1),
        'W2': np.random.randn(1, 4),
        'b2': np.random.randn(1, 1)
    }
    
    cache = {
        'A1': np.random.randn(4, 3),
        'A2': np.random.randn(1, 3)
    }
    
    return parameters, cache, X, Y


def update_parameters_test_case():
    """Test case for update_parameters function."""
    np.random.seed(1)
    parameters = {
        'W1': np.random.randn(4, 2),
        'b1': np.random.randn(4, 1),
        'W2': np.random.randn(1, 4),
        'b2': np.random.randn(1, 1)
    }
    
    grads = {
        'dW1': np.random.randn(4, 2),
        'db1': np.random.randn(4, 1),
        'dW2': np.random.randn(1, 4),
        'db2': np.random.randn(1, 1)
    }
    
    return parameters, grads


def nn_model_test_case():
    """Test case for nn_model function."""
    np.random.seed(1)
    X = np.random.randn(2, 3)
    Y = np.random.randn(1, 3)
    
    return X, Y


def predict_test_case():
    """Test case for predict function."""
    np.random.seed(1)
    X = np.random.randn(2, 3)
    
    parameters = {
        'W1': np.random.randn(4, 2),
        'b1': np.random.randn(4, 1),
        'W2': np.random.randn(1, 4),
        'b2': np.random.randn(1, 1)
    }
    
    return parameters, X

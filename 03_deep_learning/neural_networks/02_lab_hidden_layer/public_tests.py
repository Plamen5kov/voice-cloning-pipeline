"""
Public test functions for neural network exercises.
"""

import numpy as np


def layer_sizes_test(target):
    """Test the layer_sizes function."""
    np.random.seed(1)
    X = np.array([[1.62434536, -0.61175641, -0.52817175],
                  [-1.07296862,  0.86540763, -2.3015387 ],
                  [1.74481176, -0.7612069 ,  0.3190391 ],
                  [-0.24937038,  1.46210794, -2.06014071],
                  [-0.3224172 , -0.38405435,  1.13376944]])
    Y = np.array([[1, 0, 1]])
    
    n_x, n_h, n_y = target(X, Y)
    
    assert n_x == 5, f"Expected n_x=5, got {n_x}"
    assert n_h == 4, f"Expected n_h=4, got {n_h}"
    assert n_y == 1, f"Expected n_y=1, got {n_y}"
    
    print("\033[92mAll tests passed!")


def initialize_parameters_test(target):
    """Test the initialize_parameters function."""
    np.random.seed(2)
    n_x, n_h, n_y = 2, 4, 1
    parameters = target(n_x, n_h, n_y)
    
    expected_W1 = np.array([[-0.00416758, -0.00056267],
                            [-0.02136196,  0.01640271],
                            [-0.01793436, -0.00841747],
                            [ 0.00502881, -0.01245288]])
    
    expected_b1 = np.zeros((4, 1))
    
    expected_W2 = np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]])
    
    expected_b2 = np.zeros((1, 1))
    
    assert np.allclose(parameters['W1'], expected_W1), "W1 is incorrect"
    assert np.allclose(parameters['b1'], expected_b1), "b1 is incorrect"
    assert np.allclose(parameters['W2'], expected_W2), "W2 is incorrect"
    assert np.allclose(parameters['b2'], expected_b2), "b2 is incorrect"
    
    print("\033[92mAll tests passed!")


def forward_propagation_test(target):
    """Test the forward_propagation function."""
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
    
    A2, cache = target(X, parameters)
    
    # Expected values computed with correct implementation
    expected_A2 = np.array([[0.26541412, 0.42229234, 0.55544138]])
    
    assert np.allclose(A2, expected_A2), f"A2 is incorrect. Expected {expected_A2}, got {A2}"
    assert 'Z1' in cache, "Cache should contain Z1"
    assert 'A1' in cache, "Cache should contain A1"
    assert 'Z2' in cache, "Cache should contain Z2"
    assert 'A2' in cache, "Cache should contain A2"
    
    print("\033[92mAll tests passed!")


def compute_cost_test(target):
    """Test the compute_cost function."""
    np.random.seed(1)
    A2 = np.array([[0.5002307,  0.49985831, 0.50023963]])
    Y = np.array([[1, 1, 0]])
    
    cost = target(A2, Y)
    
    expected_cost = 0.6932476810445216
    
    assert np.isclose(cost, expected_cost), f"Expected cost={expected_cost}, got {cost}"
    assert isinstance(cost, float), f"Cost should be a float, got {type(cost)}"
    
    print("\033[92mAll tests passed!")


def backward_propagation_test(target):
    """Test the backward_propagation function."""
    np.random.seed(1)
    X = np.random.randn(3, 7)
    Y = np.random.randn(1, 7) > 0
    
    parameters = {
        'W1': np.random.randn(9, 3),
        'b1': np.random.randn(9, 1),
        'W2': np.random.randn(1, 9),
        'b2': np.random.randn(1, 1)
    }
    
    cache = {
        'A1': np.random.randn(9, 7),
        'A2': np.random.rand(1, 7)
    }
    
    grads = target(parameters, cache, X, Y)
    
    # Check that all gradients exist
    assert 'dW1' in grads, "grads should contain dW1"
    assert 'db1' in grads, "grads should contain db1"
    assert 'dW2' in grads, "grads should contain dW2"
    assert 'db2' in grads, "grads should contain db2"
    
    # Check shapes
    assert grads['dW1'].shape == parameters['W1'].shape, f"dW1 shape mismatch"
    assert grads['db1'].shape == parameters['b1'].shape, f"db1 shape mismatch"
    assert grads['dW2'].shape == parameters['W2'].shape, f"dW2 shape mismatch"
    assert grads['db2'].shape == parameters['b2'].shape, f"db2 shape mismatch"
    
    print("\033[92mAll tests passed!")


def update_parameters_test(target):
    """Test the update_parameters function."""
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
    
    parameters_copy = {
        'W1': parameters['W1'].copy(),
        'b1': parameters['b1'].copy(),
        'W2': parameters['W2'].copy(),
        'b2': parameters['b2'].copy()
    }
    
    updated_parameters = target(parameters, grads, learning_rate=1.2)
    
    # Check that parameters were updated
    expected_W1 = parameters_copy['W1'] - 1.2 * grads['dW1']
    expected_b1 = parameters_copy['b1'] - 1.2 * grads['db1']
    expected_W2 = parameters_copy['W2'] - 1.2 * grads['dW2']
    expected_b2 = parameters_copy['b2'] - 1.2 * grads['db2']
    
    assert np.allclose(updated_parameters['W1'], expected_W1), "W1 update is incorrect"
    assert np.allclose(updated_parameters['b1'], expected_b1), "b1 update is incorrect"
    assert np.allclose(updated_parameters['W2'], expected_W2), "W2 update is incorrect"
    assert np.allclose(updated_parameters['b2'], expected_b2), "b2 update is incorrect"
    
    print("\033[92mAll tests passed!")


def nn_model_test(target):
    """Test the nn_model function."""
    np.random.seed(3)
    X = np.random.randn(5, 4)
    Y = np.random.randint(0, 2, (1, 4))
    
    n_h = 5
    parameters = target(X, Y, n_h, num_iterations=10000, print_cost=True)
    
    # Check that parameters exist
    assert 'W1' in parameters, "parameters should contain W1"
    assert 'b1' in parameters, "parameters should contain b1"
    assert 'W2' in parameters, "parameters should contain W2"
    assert 'b2' in parameters, "parameters should contain b2"
    
    # Check shapes
    assert parameters['W1'].shape == (n_h, X.shape[0]), f"W1 shape mismatch"
    assert parameters['b1'].shape == (n_h, 1), f"b1 shape mismatch"
    assert parameters['W2'].shape == (1, n_h), f"W2 shape mismatch"
    assert parameters['b2'].shape == (1, 1), f"b2 shape mismatch"
    
    print("\033[92mAll tests passed!")


def predict_test(target):
    """Test the predict function."""
    np.random.seed(1)
    X = np.random.randn(2, 3)
    
    parameters = {
        'W1': np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]),
        'b1': np.zeros((4, 1)),
        'W2': np.array([[1, 1, 1, 1]]),
        'b2': np.array([[0]])
    }
    
    predictions = target(parameters, X)
    
    # Check shape
    assert predictions.shape == (1, X.shape[1]), f"Predictions shape should be (1, {X.shape[1]}), got {predictions.shape}"
    
    # Check that predictions are binary
    assert np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary (0 or 1)"
    
    print("\033[92mAll tests passed!")

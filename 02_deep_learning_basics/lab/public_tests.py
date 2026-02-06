"""
Public tests for the audio logistic regression assignment.
"""

import numpy as np


def sigmoid_test(target):
    """Test the sigmoid function."""
    x = np.array([-1, 0, 1, 2])
    output = target(x)
    
    assert output.shape == x.shape, f"Wrong shape. Expected {x.shape}, got {output.shape}"
    expected_values = np.array([0.26894142, 0.5, 0.73105858, 0.88079708])
    assert np.allclose(output, expected_values), f"Wrong values. Expected {expected_values}, got {output}"
    
    print("\033[92mAll tests passed for sigmoid!\033[0m")


def initialize_with_zeros_test_1(target):
    """Test initialize_with_zeros with dimension 2."""
    dim = 2
    w, b = target(dim)
    
    expected_w = np.array([[0.], [0.]])
    expected_b = 0.0
    
    assert type(b) == float, f"Wrong type for b. Expected float, got {type(b)}"
    assert w.shape == (dim, 1), f"Wrong shape for w. Expected {(dim, 1)}, got {w.shape}"
    assert np.allclose(w, expected_w), f"Wrong values for w. Expected {expected_w}, got {w}"
    assert b == expected_b, f"Wrong value for b. Expected {expected_b}, got {b}"
    
    print("\033[92mTest 1 passed for initialize_with_zeros!\033[0m")


def initialize_with_zeros_test_2(target):
    """Test initialize_with_zeros with dimension 1."""
    dim = 1
    w, b = target(dim)
    
    expected_w = np.array([[0.]])
    expected_b = 0.0
    
    assert type(b) == float, f"Wrong type for b. Expected float, got {type(b)}"
    assert w.shape == (dim, 1), f"Wrong shape for w. Expected {(dim, 1)}, got {w.shape}"
    assert np.allclose(w, expected_w), f"Wrong values for w. Expected {expected_w}, got {w}"
    assert b == expected_b, f"Wrong value for b. Expected {expected_b}, got {b}"
    
    print("\033[92mTest 2 passed for initialize_with_zeros!\033[0m")


def propagate_test(target):
    """Test the propagate function."""
    w = np.array([[1.], [2.]])
    b = 1.5
    X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
    Y = np.array([[1, 1, 0]])
    
    expected_dw = np.array([[0.25071532], [-0.06604096]])
    expected_db = -0.1250040450043965
    expected_cost = 0.15900537707692405
    expected_grads = {'dw': expected_dw, 'db': expected_db}
    
    grads, cost = target(w, b, X, Y)
    
    assert type(grads['dw']) == np.ndarray, f"Wrong type for dw. Expected np.ndarray, got {type(grads['dw'])}"
    assert grads['dw'].shape == w.shape, f"Wrong shape for dw. Expected {w.shape}, got {grads['dw'].shape}"
    assert np.allclose(grads['dw'], expected_dw), f"Wrong values for dw. Expected {expected_dw}, got {grads['dw']}"
    
    assert type(grads['db']) == np.float64, f"Wrong type for db. Expected np.float64, got {type(grads['db'])}"
    assert np.allclose(grads['db'], expected_db), f"Wrong value for db. Expected {expected_db}, got {grads['db']}"
    
    assert np.allclose(cost, expected_cost), f"Wrong value for cost. Expected {expected_cost}, got {cost}"
    
    print("\033[92mAll tests passed for propagate!\033[0m")


def optimize_test(target):
    """Test the optimize function."""
    w = np.array([[1.], [2.]])
    b = 1.5
    X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
    Y = np.array([[1, 1, 0]])
    
    expected_w = np.array([[0.80956046], [2.0508202]])
    expected_b = 1.5948713189708588
    expected_dw = np.array([[0.17860505], [-0.04840656]])
    expected_db = -0.08888460336847771
    expected_costs = [0.15900537707692405]
    
    params, grads, costs = target(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
    
    assert type(costs) == list, f"Wrong type for costs. Expected list, got {type(costs)}"
    assert len(costs) == 1, f"Wrong length for costs. Expected 1, got {len(costs)}"
    assert np.allclose(costs, expected_costs), f"Wrong values for costs. Expected {expected_costs}, got {costs}"
    
    assert np.allclose(params['w'], expected_w), f"Wrong values for w. Expected {expected_w}, got {params['w']}"
    assert np.allclose(params['b'], expected_b), f"Wrong value for b. Expected {expected_b}, got {params['b']}"
    
    assert np.allclose(grads['dw'], expected_dw), f"Wrong values for dw. Expected {expected_dw}, got {grads['dw']}"
    assert np.allclose(grads['db'], expected_db), f"Wrong value for db. Expected {expected_db}, got {grads['db']}"
    
    print("\033[92mAll tests passed for optimize!\033[0m")


def predict_test(target):
    """Test the predict function."""
    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    
    expected_predictions = np.array([[1., 1., 0.]])
    
    predictions = target(w, b, X)
    
    assert predictions.shape == (1, X.shape[1]), f"Wrong shape for predictions. Expected {(1, X.shape[1])}, got {predictions.shape}"
    assert np.allclose(predictions, expected_predictions), f"Wrong values for predictions. Expected {expected_predictions}, got {predictions}"
    
    print("\033[92mAll tests passed for predict!\033[0m")


def model_test(target):
    """Test the full model function."""
    np.random.seed(1)
    
    # Generate small test dataset
    X_train = np.random.randn(100, 10)
    Y_train = np.random.randint(0, 2, (1, 10))
    X_test = np.random.randn(100, 5)
    Y_test = np.random.randint(0, 2, (1, 5))
    
    d = target(X_train, Y_train, X_test, Y_test, num_iterations=50, learning_rate=0.01, print_cost=False)
    
    assert 'costs' in d, "Dictionary must contain 'costs' key"
    assert 'w' in d, "Dictionary must contain 'w' key"
    assert 'b' in d, "Dictionary must contain 'b' key"
    assert 'Y_prediction_train' in d, "Dictionary must contain 'Y_prediction_train' key"
    assert 'Y_prediction_test' in d, "Dictionary must contain 'Y_prediction_test' key"
    
    assert d['Y_prediction_train'].shape == Y_train.shape, f"Wrong shape for Y_prediction_train"
    assert d['Y_prediction_test'].shape == Y_test.shape, f"Wrong shape for Y_prediction_test"
    assert d['w'].shape == (X_train.shape[0], 1), f"Wrong shape for w"
    
    print("\033[92mAll tests passed for model!\033[0m")

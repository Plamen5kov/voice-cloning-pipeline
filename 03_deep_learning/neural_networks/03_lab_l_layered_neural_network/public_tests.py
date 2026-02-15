"""
Public test functions for deep neural network exercises
Validates student implementations against expected outputs
"""

import numpy as np


def initialize_parameters_test_1(target):
    """
    Test case 1 for initialize_parameters function
    """
    parameters = target(3, 2, 1)
    
    assert parameters["W1"].shape == (2, 3)
    assert parameters["b1"].shape == (2, 1)
    assert parameters["W2"].shape == (1, 2)
    assert parameters["b2"].shape == (1, 1)
    
    assert np.allclose(parameters["W1"], 
                       np.array([[ 0.01624345, -0.00611756, -0.00528172],
                                 [-0.01072969,  0.00865408, -0.02301539]]))
    assert np.allclose(parameters["b1"], 
                       np.array([[0.], [0.]]))
    assert np.allclose(parameters["W2"], 
                       np.array([[ 0.01744812, -0.00761207]]))
    assert np.allclose(parameters["b2"], 
                       np.array([[0.]]))
    
    print(" All tests passed.")


def initialize_parameters_test_2(target):
    """
    Test case 2 for initialize_parameters function
    """
    parameters = target(4, 3, 2)
    
    assert parameters["W1"].shape == (3, 4)
    assert parameters["b1"].shape == (3, 1)
    assert parameters["W2"].shape == (2, 3)
    assert parameters["b2"].shape == (2, 1)
    
    print(" All tests passed.")


def initialize_parameters_deep_test_1(target):
    """
    Test case 1 for initialize_parameters_deep function
    """
    parameters = target([5, 4, 3])
    
    assert parameters["W1"].shape == (4, 5)
    assert parameters["b1"].shape == (4, 1)
    assert parameters["W2"].shape == (3, 4)
    assert parameters["b2"].shape == (3, 1)
    
    assert np.allclose(parameters["W1"][:, 0], 
                       np.array([0.01788628, -0.00354759, -0.01313865, -0.00404677]))
    assert np.allclose(parameters["b1"], 
                       np.zeros((4, 1)))
    
    print(" All tests passed.")


def initialize_parameters_deep_test_2(target):
    """
    Test case 2 for initialize_parameters_deep function
    """
    parameters = target([4, 3, 2])
    
    assert parameters["W1"].shape == (3, 4)
    assert parameters["b1"].shape == (3, 1)
    assert parameters["W2"].shape == (2, 3)
    assert parameters["b2"].shape == (2, 1)
    
    print(" All tests passed.")


def linear_forward_test(target):
    """
    Test for linear_forward function
    """
    from testCases import linear_forward_test_case
    t_A, t_W, t_b = linear_forward_test_case()
    t_Z, t_linear_cache = target(t_A, t_W, t_b)
    
    assert t_Z.shape == (1, 2)
    assert np.allclose(t_Z, np.array([[3.26295337, -1.23429987]]))
    
    assert t_linear_cache[0].shape == (3, 2)
    assert t_linear_cache[1].shape == (1, 3)
    assert t_linear_cache[2].shape == (1, 1)
    
    print(" All tests passed.")


def linear_activation_forward_test(target):
    """
    Test for linear_activation_forward function
    """
    from testCases import linear_activation_forward_test_case
    t_A_prev, t_W, t_b = linear_activation_forward_test_case()
    
    t_A, t_linear_activation_cache = target(t_A_prev, t_W, t_b, activation="sigmoid")
    assert t_A.shape == (1, 2)
    assert np.allclose(t_A, np.array([[0.96890023, 0.11013289]]))
    
    t_A, t_linear_activation_cache = target(t_A_prev, t_W, t_b, activation="relu")
    assert t_A.shape == (1, 2)
    assert np.allclose(t_A, np.array([[3.43896131, 0.0]]))
    
    print(" All tests passed.")


def L_model_forward_test(target):
    """
    Test for L_model_forward function
    """
    from testCases import L_model_forward_test_case_2hidden
    X, parameters = L_model_forward_test_case_2hidden()
    
    AL, caches = target(X, parameters)
    
    assert AL.shape == (1, 4)
    assert len(caches) == 3
    
    print(" All tests passed.")


def compute_cost_test(target):
    """
    Test for compute_cost function
    """
    from testCases import compute_cost_test_case
    Y, AL = compute_cost_test_case()
    
    cost = target(AL, Y)
    
    assert isinstance(cost, float) or (isinstance(cost, np.ndarray) and cost.shape == ())
    assert np.isclose(cost, 0.2797765635793422)
    
    print(" All tests passed.")


def linear_backward_test(target):
    """
    Test for linear_backward function
    """
    from testCases import linear_backward_test_case
    dZ, linear_cache = linear_backward_test_case()
    
    dA_prev, dW, db = target(dZ, linear_cache)
    
    assert dA_prev.shape == (5, 4)
    assert dW.shape == (3, 5)
    assert db.shape == (3, 1)
    
    print(" All tests passed.")


def linear_activation_backward_test(target):
    """
    Test for linear_activation_backward function
    """
    from testCases import linear_activation_backward_test_case
    dAL, linear_activation_cache = linear_activation_backward_test_case()
    
    dA_prev, dW, db = target(dAL, linear_activation_cache, activation="sigmoid")
    assert dA_prev.shape == (3, 2)
    assert dW.shape == (1, 3)
    assert db.shape == (1, 1)
    
    dA_prev, dW, db = target(dAL, linear_activation_cache, activation="relu")
    assert dA_prev.shape == (3, 2)
    assert dW.shape == (1, 3)
    assert db.shape == (1, 1)
    
    print(" All tests passed.")


def L_model_backward_test(target):
    """
    Test for L_model_backward function
    """
    from testCases import L_model_backward_test_case
    AL, Y, caches = L_model_backward_test_case()
    
    grads = target(AL, Y, caches)
    
    assert "dW1" in grads
    assert "db1" in grads
    assert "dW2" in grads
    assert "db2" in grads
    assert "dA0" in grads or "dA1" in grads
    
    print(" All tests passed.")


def update_parameters_test(target):
    """
    Test for update_parameters function
    """
    from testCases import update_parameters_test_case
    parameters, grads = update_parameters_test_case()
    
    updated_params = target(parameters, grads, 0.1)
    
    assert "W1" in updated_params
    assert "b1" in updated_params
    assert "W2" in updated_params
    assert "b2" in updated_params
    
    assert updated_params["W1"].shape == (3, 4)
    assert updated_params["b1"].shape == (3, 1)
    assert updated_params["W2"].shape == (1, 3)
    assert updated_params["b2"].shape == (1, 1)
    
    # Check that parameters were actually updated
    assert not np.allclose(updated_params["W1"], parameters["W1"])
    assert not np.allclose(updated_params["b1"], parameters["b1"])
    
    print(" All tests passed.")

import numpy as np


def initialize_parameters_zeros_test(target):
    """
    Test function for initialize_parameters_zeros
    """
    print("\033[92mTest case for initialize_parameters_zeros:\033[0m")
    
    # Test case 1
    layers_dims = [3, 2, 1]
    parameters = target(layers_dims)
    
    assert parameters["W1"].shape == (2, 3), f"Wrong shape for W1. Expected (2, 3), got {parameters['W1'].shape}"
    assert parameters["b1"].shape == (2, 1), f"Wrong shape for b1. Expected (2, 1), got {parameters['b1'].shape}"
    assert parameters["W2"].shape == (1, 2), f"Wrong shape for W2. Expected (1, 2), got {parameters['W2'].shape}"
    assert parameters["b2"].shape == (1, 1), f"Wrong shape for b2. Expected (1, 1), got {parameters['b2'].shape}"
    
    assert np.allclose(parameters["W1"], np.zeros((2, 3))), "W1 should be all zeros"
    assert np.allclose(parameters["b1"], np.zeros((2, 1))), "b1 should be all zeros"
    assert np.allclose(parameters["W2"], np.zeros((1, 2))), "W2 should be all zeros"
    assert np.allclose(parameters["b2"], np.zeros((1, 1))), "b2 should be all zeros"
    
    print("\033[92m All tests passed!\033[0m")


def initialize_parameters_random_test(target):
    """
    Test function for initialize_parameters_random
    """
    print("\033[92mTest case for initialize_parameters_random:\033[0m")
    
    # Test case 1
    layers_dims = [3, 2, 1]
    parameters = target(layers_dims)
    
    assert parameters["W1"].shape == (2, 3), f"Wrong shape for W1. Expected (2, 3), got {parameters['W1'].shape}"
    assert parameters["b1"].shape == (2, 1), f"Wrong shape for b1. Expected (2, 1), got {parameters['b1'].shape}"
    assert parameters["W2"].shape == (1, 2), f"Wrong shape for W2. Expected (1, 2), got {parameters['W2'].shape}"
    assert parameters["b2"].shape == (1, 1), f"Wrong shape for b2. Expected (1, 1), got {parameters['b2'].shape}"
    
    # Check that weights are not all zeros (random initialization)
    assert not np.allclose(parameters["W1"], np.zeros((2, 3))), "W1 should not be all zeros"
    assert not np.allclose(parameters["W2"], np.zeros((1, 2))), "W2 should not be all zeros"
    
    # Check that biases are zeros
    assert np.allclose(parameters["b1"], np.zeros((2, 1))), "b1 should be all zeros"
    assert np.allclose(parameters["b2"], np.zeros((1, 1))), "b2 should be all zeros"
    
    # Check expected values with seed=3
    expected_W1 = np.array([[ 17.88628473,   4.36509851,   0.96497759],
                            [-18.63492703,  -2.77388203,  -3.54758979]])
    expected_W2 = np.array([[-0.82741481, -6.27000677]])
    
    assert np.allclose(parameters["W1"], expected_W1), f"W1 values don't match expected. Got:\n{parameters['W1']}"
    assert np.allclose(parameters["W2"], expected_W2), f"W2 values don't match expected. Got:\n{parameters['W2']}"
    
    print("\033[92m All tests passed!\033[0m")


def initialize_parameters_he_test(target):
    """
    Test function for initialize_parameters_he
    """
    print("\033[92mTest case for initialize_parameters_he:\033[0m")
    
    # Test case 1
    layers_dims = [2, 4, 1]
    parameters = target(layers_dims)
    
    assert parameters["W1"].shape == (4, 2), f"Wrong shape for W1. Expected (4, 2), got {parameters['W1'].shape}"
    assert parameters["b1"].shape == (4, 1), f"Wrong shape for b1. Expected (4, 1), got {parameters['b1'].shape}"
    assert parameters["W2"].shape == (1, 4), f"Wrong shape for W2. Expected (1, 4), got {parameters['W2'].shape}"
    assert parameters["b2"].shape == (1, 1), f"Wrong shape for b2. Expected (1, 1), got {parameters['b2'].shape}"
    
    # Check that weights are not all zeros (random initialization)
    assert not np.allclose(parameters["W1"], np.zeros((4, 2))), "W1 should not be all zeros"
    assert not np.allclose(parameters["W2"], np.zeros((1, 4))), "W2 should not be all zeros"
    
    # Check that biases are zeros
    assert np.allclose(parameters["b1"], np.zeros((4, 1))), "b1 should be all zeros"
    assert np.allclose(parameters["b2"], np.zeros((1, 1))), "b2 should be all zeros"
    
    # Check expected values with seed=3 and He initialization
    expected_W1 = np.array([[ 1.78862847,  0.43650985],
                            [ 0.09649747, -1.8634927 ],
                            [-0.2773882,  -0.35475898],
                            [-0.08274148, -0.62700068]])
    expected_W2 = np.array([[-0.03098412, -0.33744411, -0.92904268,  0.62552248]])
    
    assert np.allclose(parameters["W1"], expected_W1), f"W1 values don't match expected. Got:\n{parameters['W1']}"
    assert np.allclose(parameters["W2"], expected_W2), f"W2 values don't match expected. Got:\n{parameters['W2']}"
    
    print("\033[92m All tests passed!\033[0m")

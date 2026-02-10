"""
Public test cases for Deep Neural Network Application
"""

import numpy as np


def two_layer_model_test(target):
    """
    Test the two_layer_model function
    """
    np.random.seed(1)
    
    # Create simple test data for binary classification (male/female)
    X = np.random.randn(12288, 10)
    Y = np.random.randint(0, 2, (1, 10))  # Binary: 0=male, 1=female
    layers_dims = (12288, 7, 1)  # Output 1 neuron for binary
    
    print("Testing two_layer_model...")
    
    try:
        parameters, costs = target(X, Y, layers_dims, num_iterations=2, print_cost=False)
        
        # Check that parameters were returned
        assert 'W1' in parameters, "W1 not found in parameters"
        assert 'b1' in parameters, "b1 not found in parameters"
        assert 'W2' in parameters, "W2 not found in parameters"
        assert 'b2' in parameters, "b2 not found in parameters"
        
        # Check shapes
        assert parameters['W1'].shape == (7, 12288), f"W1 shape is {parameters['W1'].shape}, expected (7, 12288)"
        assert parameters['b1'].shape == (7, 1), f"b1 shape is {parameters['b1'].shape}, expected (7, 1)"
        assert parameters['W2'].shape == (1, 7), f"W2 shape is {parameters['W2'].shape}, expected (1, 7)"
        assert parameters['b2'].shape == (1, 1), f"b2 shape is {parameters['b2'].shape}, expected (1, 1)"
        
        # Check that costs were tracked
        assert len(costs) > 0, "Costs list is empty"
        
        # Check that cost is in reasonable range (around 0.69 for binary random initialization)
        first_cost = costs[0]
        assert 0.5 < first_cost < 0.9, f"First cost is {first_cost}, expected around 0.69 (≈-log(0.5))"
        
        print("✓ All tests passed for two_layer_model!")
        print(f"  First iteration cost: {first_cost:.4f} (expected ~0.69 for binary)")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        raise


def L_layer_model_test(target):
    """
    Test the L_layer_model function
    """
    np.random.seed(1)
    
    # Create simple test data for binary classification
    X = np.random.randn(12288, 10)
    Y = np.random.randint(0, 2, (1, 10))  # Binary: 0=male, 1=female
    layers_dims = [12288, 20, 7, 5, 1]  # 4-layer network with 1 output (binary)
    
    print("Testing L_layer_model...")
    
    try:
        parameters, costs = target(X, Y, layers_dims, num_iterations=1, print_cost=False)
        
        # Check that all parameters were returned
        for l in range(1, len(layers_dims)):
            assert f'W{l}' in parameters, f"W{l} not found in parameters"
            assert f'b{l}' in parameters, f"b{l} not found in parameters"
            
            # Check shapes
            expected_W_shape = (layers_dims[l], layers_dims[l-1])
            expected_b_shape = (layers_dims[l], 1)
            
            assert parameters[f'W{l}'].shape == expected_W_shape, \
                f"W{l} shape is {parameters[f'W{l}'].shape}, expected {expected_W_shape}"
            assert parameters[f'b{l}'].shape == expected_b_shape, \
                f"b{l} shape is {parameters[f'b{l}'].shape}, expected {expected_b_shape}"
        
        # Check that costs were tracked
        assert len(costs) > 0, "Costs list is empty"
        
        # Check that cost is in reasonable range (around 0.69 for binary random initialization)
        first_cost = costs[0]
        assert 0.5 < first_cost < 0.9, f"First cost is {first_cost}, expected around 0.69 (≈-log(0.5))"
        
        print("✓ All tests passed for L_layer_model!")
        print(f"  First iteration cost: {first_cost:.4f} (expected ~0.69 for binary)")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    print("Public test cases loaded successfully!")
    print("Import these functions to test your implementations:")
    print("  - two_layer_model_test()")
    print("  - L_layer_model_test()")

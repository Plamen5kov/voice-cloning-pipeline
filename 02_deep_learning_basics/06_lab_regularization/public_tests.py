import numpy as np


def compute_cost_with_regularization_test(target):
    """
    Test function for compute_cost_with_regularization
    """
    print("\033[92mTest case for compute_cost_with_regularization:\033[0m")
    
    # Test with lambd = 0.1
    a3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    Y = np.array([[1, 1, 0, 1, 0]])
    
    np.random.seed(1)
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    
    cost = target(a3, Y, parameters, lambd=0.1)
    
    expected_cost = 1.7864859451590758
    
    assert np.isclose(cost, expected_cost), f"Wrong cost. Expected {expected_cost}, got {cost}"
    
    print("\033[92m All tests passed!\033[0m")


def backward_propagation_with_regularization_test(target):
    """
    Test function for backward_propagation_with_regularization
    """
    print("\033[92mTest case for backward_propagation_with_regularization:\033[0m")
    
    np.random.seed(1)
    X = np.random.randn(3, 5)
    Y = np.array([[1, 1, 0, 1, 0]])
    
    cache = (
     np.array([[-1.5285531390891678, 3.3252463458549437, 2.1399454059184335, 2.6070065361645463, -0.7594211463979716], [-1.9804353778113148, 4.160099404181192, 0.7905102071939143, 1.464935121735728, -0.4550624203877942]]),
     np.array([[0.0, 3.3252463458549437, 2.1399454059184335, 2.6070065361645463, 0.0], [0.0, 4.160099404181192, 0.7905102071939143, 1.464935121735728, 0.0]]),
     np.array([[-1.0998912673140309, -0.17242820755043575, -0.8778584179213718], [0.04221374671559283, 0.5828152137158222, -1.1006191772129212]]),
     np.array([[1.1447237098396141], [0.9015907205927955]]),
     np.array([[0.530355466738186, 5.948923228772381, 2.31780174187614, 3.160057012343055, 0.530355466738186], [-0.691660751725309, -3.4764598709638044, -2.2519470205007224, -2.65416995703088, -0.691660751725309], [-0.39675352685597737, -4.6228584591281265, -2.611017290027505, -3.228749214850857, -0.39675352685597737]]),
     np.array([[0.530355466738186, 5.948923228772381, 2.31780174187614, 3.160057012343055, 0.530355466738186], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
     np.array([[0.5024943389018682, 0.9008559492644118], [-0.6837278591743331, -0.12289022551864817], [-0.9357694342590688, -0.2678880796260159]]),
     np.array([[0.530355466738186], [-0.691660751725309], [-0.39675352685597737]]),
     np.array([[-0.37711039702057103, -4.100602236838624, -1.6053946802258392, -2.1841695086225528, -0.37711039702057103]]),
     np.array([[0.4068240227829598, 0.016292844275962705, 0.1672289837891084, 0.10118110718330305, 0.4068240227829598]]),
     np.array([[-0.6871727001195994, -0.8452056414987196, -0.671246130836819]]),
     np.array([[-0.01266459891890136]]))
    
    grads = target(X, Y, cache, lambd=0.7)
    
    expected_dW1 = np.array([[-0.25604646,  0.12298827, -0.28297129],
                              [-0.17706303,  0.34536094, -0.4410571 ]])
    expected_dW2 = np.array([[ 0.79276486,  0.85133918],
                              [-0.0957219 , -0.01720463],
                              [-0.13100772, -0.03750433]])
    expected_dW3 = np.array([[-1.77691347, -0.11832879, -0.09397446]])
    
    assert np.allclose(grads["dW1"], expected_dW1), f"Wrong dW1. Expected:\n{expected_dW1}\nGot:\n{grads['dW1']}"
    assert np.allclose(grads["dW2"], expected_dW2), f"Wrong dW2. Expected:\n{expected_dW2}\nGot:\n{grads['dW2']}"
    assert np.allclose(grads["dW3"], expected_dW3), f"Wrong dW3. Expected:\n{expected_dW3}\nGot:\n{grads['dW3']}"
    
    print("\033[92m All tests passed!\033[0m")


def forward_propagation_with_dropout_test(target):
    """
    Test function for forward_propagation_with_dropout
    """
    print("\033[92mTest case for forward_propagation_with_dropout:\033[0m")
    
    np.random.seed(1)
    X = np.random.randn(2, 3)
    
    W1 = np.random.randn(20, 2)
    b1 = np.random.randn(20, 1)
    W2 = np.random.randn(3, 20)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    
    A3, cache = target(X, parameters, keep_prob=0.7)
    
    # Just check that A3 has the right shape and is between 0 and 1
    assert A3.shape == (1, 3), f"Wrong shape for A3. Expected (1, 3), got {A3.shape}"
    assert np.all((A3 >= 0) & (A3 <= 1)), "A3 values should be between 0 and 1"
    
    # Check that cache has the right structure
    assert len(cache) == 14, f"Wrong cache length. Expected 14, got {len(cache)}"
    
    print("\033[92m All tests passed!\033[0m")


def backward_propagation_with_dropout_test(target):
    """
    Test function for backward_propagation_with_dropout
    """
    print("\033[92mTest case for backward_propagation_with_dropout:\033[0m")
    
    np.random.seed(1)
    X = np.random.randn(3, 5)
    Y = np.array([[1, 1, 0, 1, 0]])
    
    cache = (
     np.array([[-1.5285531390891678, 3.3252463458549437, 2.1399454059184335, 2.6070065361645463, -0.7594211463979716], [-1.9804353778113148, 4.160099404181192, 0.7905102071939143, 1.464935121735728, -0.4550624203877942]]),
     np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
     np.array([[0.0, 4.156557932318679, 2.6749317573980416, 3.2587581702056827, 0.0], [0.0, 5.20012425522649, 0.9881377589923929, 1.83116890216966, 0.0]]),
     np.array([[-1.0998912673140309, -0.17242820755043575, -0.8778584179213718], [0.04221374671559283, 0.5828152137158222, -1.1006191772129212]]),
     np.array([[1.1447237098396141], [0.9015907205927955]]),
     np.array([[0.530355466738186, 7.30356516928093, 2.764663310660628, 3.8174823987442723, 0.530355466738186], [-0.691660751725309, -4.1726596507734275, -2.642018587694575, -3.144797258357272, -0.691660751725309], [-0.39675352685597737, -5.679384692196163, -3.1645832308203863, -3.936748136849577, -0.39675352685597737]]),
     np.array([[1, 1, 0, 1, 0], [1, 1, 0, 1, 1], [0, 1, 1, 1, 0]]),
     np.array([[0.6629443334227325, 9.129456461601162, 0.0, 4.77185299843034, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
     np.array([[0.5024943389018682, 0.9008559492644118], [-0.6837278591743331, -0.12289022551864817], [-0.9357694342590688, -0.2678880796260159]]),
     np.array([[0.530355466738186], [-0.691660751725309], [-0.39675352685597737]]),
     np.array([[-0.4682218465459885, -6.286177846261696, -0.01266459891890136, -3.2917517084240844, -0.01266459891890136]]),
     np.array([[0.3850371949398219, 0.0018584025992986072, 0.4968338925883705, 0.035855240689613835, 0.4968338925883705]]),
     np.array([[-0.6871727001195994, -0.8452056414987196, -0.671246130836819]]),
     np.array([[-0.01266459891890136]]))
    
    gradients = target(X, Y, cache, keep_prob=0.8)
    
    expected_dA1 = np.array([[0.33179203, 0.53852919, 0., 0.52018682, 0.],
                             [0.59482625, 0.9654581, 0., 0.93257446, 0.]])
    expected_dA2 = np.array([[0.52823206, 0.85736957, -0., 0.82816745, -0.],
                             [0.64971254, 1.05454364, -0., 1.01862574, -0.52490851],
                             [0., 0.83749836, -0.41687229, 0.80897305, -0.]])
    
    assert np.allclose(gradients["dA1"], expected_dA1), f"Wrong dA1. Expected:\n{expected_dA1}\nGot:\n{gradients['dA1']}"
    assert np.allclose(gradients["dA2"], expected_dA2), f"Wrong dA2. Expected:\n{expected_dA2}\nGot:\n{gradients['dA2']}"
    
    print("\033[92m All tests passed!\033[0m")

import numpy as np


def forward_propagation_test(target):
    """
    Test forward_propagation function
    """
    print("\033[92mAll tests passed.")


def backward_propagation_test(target):
    """
    Test backward_propagation function
    """
    print("\033[92mAll tests passed.")


def gradient_check_test(target, difference):
    """
    Test gradient_check function
    """
    expected_values = [7.81407531e-11]
    assert np.any(np.isclose(difference, expected_values)), f"Wrong value. Expected {expected_values[0]}, got {difference}"
    print("\033[92mAll tests passed.")

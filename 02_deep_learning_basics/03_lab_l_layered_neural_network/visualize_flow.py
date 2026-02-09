#!/usr/bin/env python3
"""
Visual step-by-step demonstration of L-layer neural network
Shows exactly what happens at each function call with real data
"""

import numpy as np
import sys

# Add color support for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title):
    """Print a bold section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}\n")

def print_step(step_num, title):
    """Print a step header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}Step {step_num}: {title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*60}{Colors.ENDC}")

def print_data(name, data, color=Colors.GREEN):
    """Print array/value with formatting"""
    if isinstance(data, np.ndarray):
        shape_str = f"shape {data.shape}"
        if data.size <= 10:
            print(f"{color}{name}: {data.ravel()}{Colors.ENDC} ({shape_str})")
        else:
            print(f"{color}{name}: [...]{Colors.ENDC} ({shape_str})")
    else:
        print(f"{color}{name}: {data}{Colors.ENDC}")

def print_function_call(func_name, inputs, outputs):
    """Print function call details"""
    print(f"  {Colors.YELLOW}→ {func_name}({', '.join(inputs)}){Colors.ENDC}")
    print(f"  {Colors.GREEN}  Returns: {', '.join(outputs)}{Colors.ENDC}")


def demonstrate_l_layer_nn():
    """
    Demonstrate a complete forward and backward pass through a simple network
    """
    print_section("L-LAYER NEURAL NETWORK - VISUAL WALKTHROUGH")
    
    print(f"{Colors.HEADER}This demonstration shows exactly what happens at each step")
    print(f"of training an L-layer neural network.{Colors.ENDC}\n")
    
    # ============================================================================
    # SETUP: Create a tiny dataset
    # ============================================================================
    print_section("SETUP: DATASET AND NETWORK ARCHITECTURE")
    
    np.random.seed(1)
    X = np.random.randn(3, 2)  # 3 features, 2 examples
    Y = np.array([[1, 0]])      # 1 output, 2 examples
    
    print_data("Input X", X)
    print_data("Labels Y", Y)
    
    print(f"\n{Colors.YELLOW}Network Architecture:{Colors.ENDC}")
    layer_dims = [3, 4, 1]  # 3 inputs → 4 hidden → 1 output
    print(f"  Layer 0 (input):  {layer_dims[0]} units")
    print(f"  Layer 1 (hidden): {layer_dims[1]} units (ReLU)")
    print(f"  Layer 2 (output): {layer_dims[2]} units (Sigmoid)")
    print(f"\n  Parameters: W[1]: (4,3), b[1]: (4,1)")
    print(f"             W[2]: (1,4), b[2]: (1,1)")
    
    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    print_section("PHASE 1: INITIALIZATION (Exercises 1-2)")
    
    print_step(1, "initialize_parameters_deep(layer_dims)")
    
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        print_data(f"  W[{l}]", parameters['W' + str(l)])
        print_data(f"  b[{l}]", parameters['b' + str(l)])
    
    print(f"\n{Colors.GREEN}✓ Initialized {L-1} layers of parameters{Colors.ENDC}")
    
    # ============================================================================
    # FORWARD PROPAGATION
    # ============================================================================
    print_section("PHASE 2: FORWARD PROPAGATION (Exercises 3-6)")
    
    caches = []
    A = X
    L = len(parameters) // 2
    
    print_step(2, f"Starting forward pass with A[0] = X")
    print_data("  A[0]", A)
    
    # Layer 1: LINEAR -> RELU
    print_step(3, "Layer 1: linear_activation_forward (ReLU)")
    print_function_call("linear_forward", ["A[0]", "W[1]", "b[1]"], ["Z[1]", "linear_cache"])
    
    Z1 = np.dot(parameters['W1'], A) + parameters['b1']
    print_data("    Z[1]", Z1)
    
    print_function_call("relu", ["Z[1]"], ["A[1]"])
    A1 = np.maximum(0, Z1)
    print_data("    A[1]", A1)
    
    linear_cache1 = (A, parameters['W1'], parameters['b1'])
    activation_cache1 = Z1
    cache1 = (linear_cache1, activation_cache1)
    caches.append(cache1)
    print(f"    {Colors.YELLOW}Cache stored: (A[0], W[1], b[1], Z[1]){Colors.ENDC}")
    
    # Layer 2: LINEAR -> SIGMOID
    print_step(4, "Layer 2: linear_activation_forward (Sigmoid)")
    print_function_call("linear_forward", ["A[1]", "W[2]", "b[2]"], ["Z[2]", "linear_cache"])
    
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    print_data("    Z[2]", Z2)
    
    print_function_call("sigmoid", ["Z[2]"], ["A[2]"])
    AL = 1 / (1 + np.exp(-Z2))
    print_data("    A[2] (AL)", AL)
    
    linear_cache2 = (A1, parameters['W2'], parameters['b2'])
    activation_cache2 = Z2
    cache2 = (linear_cache2, activation_cache2)
    caches.append(cache2)
    print(f"    {Colors.YELLOW}Cache stored: (A[1], W[2], b[2], Z[2]){Colors.ENDC}")
    
    print(f"\n{Colors.GREEN}✓ Forward propagation complete!{Colors.ENDC}")
    print(f"  {Colors.BOLD}Predictions (AL): {AL.ravel()}{Colors.ENDC}")
    print(f"  {Colors.BOLD}True labels (Y):  {Y.ravel()}{Colors.ENDC}")
    
    # Compute cost
    print_step(5, "compute_cost(AL, Y)")
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    print_data("    Cost J", cost, Colors.RED)
    
    # ============================================================================
    # BACKWARD PROPAGATION
    # ============================================================================
    print_section("PHASE 3: BACKWARD PROPAGATION (Exercises 7-9)")
    
    print_step(6, "Initialize backprop: compute dAL")
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    print_data("  dAL (∂L/∂AL)", dAL)
    
    grads = {}
    
    # Backward through layer 2
    print_step(7, "Layer 2: linear_activation_backward (Sigmoid)")
    print(f"  {Colors.YELLOW}Using cache from forward pass: cache[1]{Colors.ENDC}")
    
    linear_cache, activation_cache = cache2
    A_prev, W, b = linear_cache
    
    print_function_call("sigmoid_backward", ["dAL", "Z[2]"], ["dZ[2]"])
    s = 1 / (1 + np.exp(-activation_cache))
    dZ2 = dAL * s * (1 - s)
    print_data("    dZ[2]", dZ2)
    
    print_function_call("linear_backward", ["dZ[2]", "cache"], ["dA[1]", "dW[2]", "db[2]"])
    m = A_prev.shape[1]
    dW2 = 1/m * np.dot(dZ2, A_prev.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W.T, dZ2)
    
    print_data("    dW[2]", dW2)
    print_data("    db[2]", db2)
    print_data("    dA[1]", dA1)
    
    grads['dW2'] = dW2
    grads['db2'] = db2
    
    # Backward through layer 1
    print_step(8, "Layer 1: linear_activation_backward (ReLU)")
    print(f"  {Colors.YELLOW}Using cache from forward pass: cache[0]{Colors.ENDC}")
    
    linear_cache, activation_cache = cache1
    A_prev, W, b = linear_cache
    
    print_function_call("relu_backward", ["dA[1]", "Z[1]"], ["dZ[1]"])
    dZ1 = np.array(dA1, copy=True)
    dZ1[activation_cache <= 0] = 0
    print_data("    dZ[1]", dZ1)
    
    print_function_call("linear_backward", ["dZ[1]", "cache"], ["dA[0]", "dW[1]", "db[1]"])
    m = A_prev.shape[1]
    dW1 = 1/m * np.dot(dZ1, A_prev.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    dA0 = np.dot(W.T, dZ1)
    
    print_data("    dW[1]", dW1)
    print_data("    db[1]", db1)
    
    grads['dW1'] = dW1
    grads['db1'] = db1
    
    print(f"\n{Colors.GREEN}✓ Backward propagation complete!{Colors.ENDC}")
    print(f"  Computed gradients for all parameters")
    
    # ============================================================================
    # PARAMETER UPDATE
    # ============================================================================
    print_section("PHASE 4: UPDATE PARAMETERS (Exercise 10)")
    
    learning_rate = 0.01
    print_step(9, f"update_parameters (learning_rate = {learning_rate})")
    
    L = len(parameters) // 2
    for l in range(L):
        layer = str(l + 1)
        print(f"\n  {Colors.YELLOW}Layer {layer}:{Colors.ENDC}")
        
        old_W = parameters['W' + layer].copy()
        old_b = parameters['b' + layer].copy()
        
        parameters['W' + layer] -= learning_rate * grads['dW' + layer]
        parameters['b' + layer] -= learning_rate * grads['db' + layer]
        
        print(f"    W[{layer}]: shape {old_W.shape}, change: {np.sum(np.abs(old_W - parameters['W' + layer])):.6f}")
        print(f"    b[{layer}]: shape {old_b.shape}, change: {np.sum(np.abs(old_b - parameters['b' + layer])):.6f}")
    
    print(f"\n{Colors.GREEN}✓ Parameters updated!{Colors.ENDC}")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print_section("SUMMARY")
    
    print(f"{Colors.BOLD}Complete training iteration executed:{Colors.ENDC}\n")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} Exercise 2: Initialized parameters for L=2 layer network")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} Exercise 3-5: Forward propagation → computed AL (predictions)")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} Exercise 6: Computed cost J = {cost:.6f}")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} Exercise 7-9: Backward propagation → computed all gradients")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} Exercise 10: Updated parameters using gradient descent")
    
    print(f"\n{Colors.YELLOW}In a real training loop:{Colors.ENDC}")
    print(f"  • Repeat steps 2-9 for many iterations (epochs)")
    print(f"  • Cost should decrease with each iteration")
    print(f"  • Parameters converge to optimal values")
    print(f"  • Model learns to map inputs to correct outputs")
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"\n{Colors.HEADER}🎉 Now you understand how all 10 functions work together!{Colors.ENDC}\n")


if __name__ == "__main__":
    demonstrate_l_layer_nn()

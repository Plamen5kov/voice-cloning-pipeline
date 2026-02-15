#!/usr/bin/env python3
"""
Computational Graph Visualizer for L-Layer Neural Network
Creates visual diagrams showing the forward/backward computation flow
"""

import numpy as np
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("⚠️  Graphviz not installed. Install with: pip install graphviz")
    print("   Also ensure graphviz system package is installed: sudo apt install graphviz")


def create_computation_graph(layer_dims, save_path='computation_graph'):
    """
    Create a detailed computational graph visualization
    
    Args:
        layer_dims: list of layer dimensions [n_x, n_h1, ..., n_y]
        save_path: path to save the graph (without extension)
    """
    if not GRAPHVIZ_AVAILABLE:
        return None
    
    # Create directed graph
    dot = Digraph(comment='L-Layer Neural Network Computation Graph')
    dot.attr(rankdir='LR', size='12,8')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    
    # Color scheme
    colors = {
        'input': '#E8F4F8',
        'linear': '#B3D9FF',
        'activation': '#FFE5B3',
        'cost': '#FFB3B3',
        'gradient': '#D4EDDA',
        'cache': '#F0F0F0'
    }
    
    L = len(layer_dims) - 1
    
    # Title
    dot.attr(label=f'\\n\\nL-Layer Neural Network: {layer_dims}\\n' + 
                   f'Architecture: [LINEAR→RELU]×{L-1} → LINEAR→SIGMOID\\n',
             fontsize='16', fontname='Arial Bold')
    
    # ============================================================================
    # FORWARD PASS
    # ============================================================================
    
    with dot.subgraph(name='cluster_forward') as c:
        c.attr(label='FORWARD PROPAGATION', fontsize='14', style='dashed')
        c.attr(color='blue')
        
        # Input layer
        c.node('X', f'X\\n(Input)\\n{layer_dims[0]}×m', fillcolor=colors['input'])
        c.node('A0', f'A[0] = X\\n{layer_dims[0]}×m', fillcolor=colors['input'])
        c.edge('X', 'A0', style='dashed')
        
        # Hidden and output layers
        for l in range(1, L + 1):
            # Linear transformation
            linear_label = f'Z[{l}] = W[{l}]·A[{l-1}] + b[{l}]\\n({layer_dims[l]}×m)'
            c.node(f'Z{l}', linear_label, fillcolor=colors['linear'])
            
            # Parameters
            param_label = f'W[{l}]: {layer_dims[l]}×{layer_dims[l-1]}\\nb[{l}]: {layer_dims[l]}×1'
            c.node(f'W{l}', param_label, shape='cylinder', fillcolor='#FFE0E0')
            
            # Connections
            c.edge(f'A{l-1}', f'Z{l}', label='matmul')
            c.edge(f'W{l}', f'Z{l}', label='params', style='dashed')
            
            # Activation
            if l < L:
                activation = 'ReLU'
                act_formula = 'max(0, Z)'
            else:
                activation = 'Sigmoid'
                act_formula = '1/(1+e^(-Z))'
            
            act_label = f'A[{l}] = {activation}(Z[{l}])\\n{act_formula}\\n({layer_dims[l]}×m)'
            c.node(f'A{l}', act_label, fillcolor=colors['activation'])
            c.edge(f'Z{l}', f'A{l}', label=activation)
            
            # Cache nodes
            cache_label = f'Cache[{l}]\\n(A[{l-1}], W[{l}], b[{l}], Z[{l}])'
            c.node(f'cache{l}', cache_label, fillcolor=colors['cache'], 
                   shape='note', style='filled')
            c.edge(f'Z{l}', f'cache{l}', style='dotted', color='gray')
            c.edge(f'A{l}', f'cache{l}', style='dotted', color='gray')
        
        # Cost function
        c.node('Y', f'Y\\n(Labels)\\n1×m', fillcolor=colors['input'])
        cost_formula = '-1/m Σ[y·log(a) + (1-y)·log(1-a)]'
        c.node('J', f'Cost J\\n{cost_formula}\\nscalar', fillcolor=colors['cost'])
        c.edge(f'A{L}', 'J', label='compare')
        c.edge('Y', 'J', label='labels')
    
    # ============================================================================
    # BACKWARD PASS
    # ============================================================================
    
    with dot.subgraph(name='cluster_backward') as c:
        c.attr(label='BACKWARD PROPAGATION', fontsize='14', style='dashed')
        c.attr(color='red')
        
        # Initialize gradients
        c.node('dAL', f'dA[{L}] = ∂J/∂A[{L}]\\n({layer_dims[L]}×m)', 
               fillcolor=colors['gradient'])
        
        # Backward through each layer
        for l in range(L, 0, -1):
            # Activation backward
            if l < L:
                activation = 'ReLU'
                grad_formula = 'dZ = dA * (Z>0)'
            else:
                activation = 'Sigmoid' 
                grad_formula = 'dZ = dA * σ(Z) * (1-σ(Z))'
            
            dz_label = f'dZ[{l}]\\n{grad_formula}\\n({layer_dims[l]}×m)'
            c.node(f'dZ{l}', dz_label, fillcolor=colors['gradient'])
            
            if l == L:
                c.edge('dAL', f'dZ{l}', label=f'{activation}_backward')
            else:
                c.edge(f'dA{l}', f'dZ{l}', label=f'{activation}_backward')
            
            # Use cache
            c.edge(f'cache{l}', f'dZ{l}', style='dotted', color='gray', 
                   label='use cache')
            
            # Linear backward
            grad_label = f'dW[{l}] = 1/m·dZ[{l}]·A[{l-1}]ᵀ\\ndb[{l}] = 1/m·Σ dZ[{l}]\\ndA[{l-1}] = W[{l}]ᵀ·dZ[{l}]'
            c.node(f'grads{l}', grad_label, fillcolor='#C8E6C9', shape='box3d')
            c.edge(f'dZ{l}', f'grads{l}', label='linear_backward')
            
            if l > 1:
                c.node(f'dA{l-1}', f'dA[{l-1}]\\n({layer_dims[l-1]}×m)', 
                       fillcolor=colors['gradient'])
                c.edge(f'grads{l}', f'dA{l-1}', label='propagate')
        
        # Connect cost to backward pass
        c.edge('J', 'dAL', label='∂J/∂A[L]', color='red', style='bold')
    
    # ============================================================================
    # PARAMETER UPDATE
    # ============================================================================
    
    with dot.subgraph(name='cluster_update') as c:
        c.attr(label='PARAMETER UPDATE (Gradient Descent)', fontsize='14', 
               style='dashed')
        c.attr(color='green')
        
        for l in range(1, L + 1):
            update_label = f'W[{l}] = W[{l}] - α·dW[{l}]\\nb[{l}] = b[{l}] - α·db[{l}]'
            c.node(f'update{l}', update_label, fillcolor='#80CBC4', shape='box')
            c.edge(f'grads{l}', f'update{l}', label='α (learning rate)')
            c.edge(f'update{l}', f'W{l}', label='update', color='green', 
                   style='bold', constraint='false')
    
    # Save
    try:
        dot.render(save_path, format='png', cleanup=True)
        print(f"✓ Computation graph saved to {save_path}.png")
        dot.render(save_path + '_svg', format='svg', cleanup=True) 
        print(f"✓ SVG version saved to {save_path}_svg.svg")
        return dot
    except Exception as e:
        print(f"✗ Error rendering graph: {e}")
        return None


def create_layer_by_layer_graph(layer_dims, save_path='layer_graph'):
    """
    Create a simplified layer-by-layer view
    """
    if not GRAPHVIZ_AVAILABLE:
        return None
        
    dot = Digraph(comment='Layer-by-Layer View')
    dot.attr(rankdir='TB', size='10,12')
    dot.attr('node', shape='record', style='filled', fontname='Arial')
    
    L = len(layer_dims) - 1
    
    dot.attr(label=f'\\n\\nLayer-by-Layer Computation Flow\\n', 
             fontsize='16', fontname='Arial Bold')
    
    # Create layers
    for l in range(L + 1):
        if l == 0:
            label = f'{{Layer {l}|INPUT|{layer_dims[l]} units|A[{l}] = X}}'
            color = '#E8F4F8'
        elif l < L:
            label = f'{{Layer {l}|HIDDEN (ReLU)|{layer_dims[l]} units|Z[{l}] = W[{l}]·A[{l-1}] + b[{l}]|A[{l}] = max(0, Z[{l}])}}'
            color = '#FFE5B3'
        else:
            label = f'{{Layer {l}|OUTPUT (Sigmoid)|{layer_dims[l]} units|Z[{l}] = W[{l}]·A[{l-1}] + b[{l}]|A[{l}] = σ(Z[{l}])}}'
            color = '#B3D9FF'
        
        dot.node(f'layer{l}', label, fillcolor=color)
        
        if l > 0:
            dot.edge(f'layer{l-1}', f'layer{l}', 
                    label=f'W[{l}]({layer_dims[l]}×{layer_dims[l-1]})\\n+\\nb[{l}]({layer_dims[l]}×1)')
    
    # Add cost computation
    dot.node('cost', '{Cost Function|J = -1/m Σ[y·log(a)+...]}', fillcolor='#FFB3B3')
    dot.edge(f'layer{L}', 'cost', label='predictions vs labels')
    
    try:
        dot.render(save_path, format='png', cleanup=True)
        print(f"✓ Layer graph saved to {save_path}.png")
        return dot
    except Exception as e:
        print(f"✗ Error rendering graph: {e}")
        return None


def create_pytorch_comparison(layer_dims, save_path='pytorch_comparison'):
    """
    Create a PyTorch equivalent and visualize with torchviz
    """
    try:
        import torch
        import torch.nn as nn
        from torchviz import make_dot
        
        print("\n" + "="*80)
        print("PyTorch Autograd Graph Visualization")
        print("="*80)
        
        # Create PyTorch model
        class LLayerNN(nn.Module):
            def __init__(self, layer_dims):
                super().__init__()
                self.layers = nn.ModuleList()
                L = len(layer_dims) - 1
                
                for l in range(L):
                    self.layers.append(nn.Linear(layer_dims[l], layer_dims[l+1]))
                
            def forward(self, x):
                L = len(self.layers)
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if i < L - 1:  # ReLU for hidden layers
                        x = torch.relu(x)
                    else:  # Sigmoid for output layer
                        x = torch.sigmoid(x)
                return x
        
        # Create model and dummy input
        model = LLayerNN(layer_dims)
        batch_size = 2
        x = torch.randn(batch_size, layer_dims[0], requires_grad=True)
        y = torch.randint(0, 2, (batch_size, layer_dims[-1])).float()
        
        # Forward pass
        output = model(x)
        loss = nn.BCELoss()(output, y)
        
        # Create visualization
        dot = make_dot(loss, params=dict(model.named_parameters()), 
                      show_attrs=True, show_saved=True)
        dot.attr(label=f'\\n\\nPyTorch Autograd Graph: {layer_dims}\\n',
                fontsize='16')
        
        # Save
        dot.render(save_path, format='png', cleanup=True)
        print(f"✓ PyTorch autograd graph saved to {save_path}.png")
        
        # Print model summary
        print(f"\nPyTorch Model Architecture:")
        print(model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
        
        return dot
        
    except ImportError as e:
        print(f"\n⚠️  PyTorch visualization not available: {e}")
        print("Install with: pip install torch torchviz")
        return None
    except Exception as e:
        print(f"✗ Error creating PyTorch graph: {e}")
        return None


def main():
    """Generate all visualizations"""
    print("="*80)
    print("L-Layer Neural Network - Advanced Visualizations")
    print("="*80)
    
    # Create output directory
    import os
    output_dir = "visual_representation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example architecture
    layer_dims = [3, 4, 1]  # 3 inputs -> 4 hidden -> 1 output
    
    print(f"\nNetwork Architecture: {layer_dims}")
    print(f"  Input:  {layer_dims[0]} features")
    print(f"  Hidden: {layer_dims[1]} units (ReLU)")
    print(f"  Output: {layer_dims[-1]} units (Sigmoid)")
    print()
    
    if not GRAPHVIZ_AVAILABLE:
        print("\n⚠️  Graphviz not available. Install it to generate visualizations:")
        print("    pip install graphviz")
        print("    sudo apt install graphviz  # On Ubuntu/Debian")
        print("    brew install graphviz      # On macOS")
        return
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    # 1. Detailed computation graph
    create_computation_graph(layer_dims, os.path.join(output_dir, 'computation_graph'))
    
    # 2. Layer-by-layer view
    create_layer_by_layer_graph(layer_dims, os.path.join(output_dir, 'layer_graph'))
    
    # 3. PyTorch comparison (optional)
    create_pytorch_comparison(layer_dims, os.path.join(output_dir, 'pytorch_autograd'))
    
    print("\n" + "="*80)
    print("✓ All visualizations generated!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. visual_representation/computation_graph.png     - Complete forward/backward graph")
    print("  2. visual_representation/computation_graph_svg.svg - SVG version (scalable)")
    print("  3. visual_representation/layer_graph.png          - Simplified layer view")
    print("  4. visual_representation/pytorch_autograd.png     - PyTorch comparison (if available)")
    print("\nOpen these images to see the computational flow!")
    

if __name__ == "__main__":
    main()

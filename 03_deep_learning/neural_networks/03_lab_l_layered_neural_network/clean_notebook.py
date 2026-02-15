#!/usr/bin/env python3
"""
Clean the notebook by removing outputs, execution counts, and student solutions.
Prepares the notebook for fresh student use.
"""

import json
import sys

NOTEBOOK_PATH = "l_layer_neural_network.ipynb"


def clean_notebook(filepath):
    """
    Remove outputs, execution counts, and student solutions from notebook.
    
    Args:
        filepath: Path to the notebook file
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Track statistics
        cells_cleared = 0
        solutions_removed = 0
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                # Clear outputs
                if cell.get('outputs'):
                    cell['outputs'] = []
                    cells_cleared += 1
                
                # Clear execution count
                cell['execution_count'] = None
                
                # Check if this cell has student code to remove
                source = cell.get('source', [])
                if isinstance(source, list):
                    source_text = ''.join(source)
                else:
                    source_text = source
                
                if '# YOUR CODE STARTS HERE' in source_text and '# YOUR CODE ENDS HERE' in source_text:
                    # Find the pattern and remove student solution
                    lines = source if isinstance(source, list) else source.split('\n')
                    new_lines = []
                    in_solution = False
                    
                    for line in lines:
                        line_str = line if isinstance(line, str) else line
                        if '# YOUR CODE STARTS HERE' in line_str:
                            new_lines.append(line)
                            new_lines.append('\n')  # Add blank line
                            in_solution = True
                        elif '# YOUR CODE ENDS HERE' in line_str:
                            new_lines.append(line)
                            in_solution = False
                            solutions_removed += 1
                        elif not in_solution:
                            new_lines.append(line)
                    
                    cell['source'] = new_lines
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
            f.write('\n')  # Add trailing newline
        
        # Print summary
        print("✓ Notebook cleaned successfully!")
        print(f"  - Cleared outputs from {cells_cleared} code cells")
        print(f"  - Reset all execution counts")
        print(f"  - Removed {solutions_removed} student solutions")
        print(f"  - Notebook is ready for a fresh start")
        
        return True
        
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return False
    except Exception as e:
        print(f"✗ Error cleaning notebook: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("Cleaning L-Layer Neural Network Notebook")
    print("=" * 60)
    print(f"File: {NOTEBOOK_PATH}\n")
    
    success = clean_notebook(NOTEBOOK_PATH)
    
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

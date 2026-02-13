"""
Clean notebook utility

This script removes output from Jupyter notebooks and clears solution code
between YOUR CODE STARTS HERE / YOUR CODE ENDS HERE markers.
"""

import json
import sys


def clean_notebook(notebook_path):
    """
    Remove all outputs and solution code from a Jupyter notebook
    
    Args:
        notebook_path: Path to the notebook file
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Track statistics
    cells_cleared = 0
    solutions_removed = 0
    
    # Clear outputs and solution code from all cells
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
                        new_lines.append('\n')  # Add single blank line
                        in_solution = True
                    elif '# YOUR CODE ENDS HERE' in line_str:
                        new_lines.append(line)
                        in_solution = False
                        solutions_removed += 1
                    elif not in_solution:
                        new_lines.append(line)
                
                cell['source'] = new_lines
    
    # Write cleaned notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write('\n')  # Add trailing newline
    
    print(f"✓ Cleaned notebook: {notebook_path}")
    print(f"  - Cleared outputs from {cells_cleared} code cells")
    print(f"  - Removed {solutions_removed} student solutions")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_notebook.py <notebook_path>")
        sys.exit(1)
    
    clean_notebook(sys.argv[1])

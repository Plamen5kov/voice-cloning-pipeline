#!/usr/bin/env python3
"""
Clean the TensorFlow Introduction notebook by removing all outputs, execution counts, and student solutions.
This prepares the notebook for someone else to complete the exercises.
"""

import json
import sys
import re

def clean_notebook(notebook_path):
    """Remove all outputs, execution counts, and solutions from a Jupyter notebook."""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Clear outputs and execution counts from all code cells
    cells_cleaned = 0
    solutions_removed = 0
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            cells_cleaned += 1
            
            # Check if cell contains solution code (YOUR CODE markers)
            source = ''.join(cell['source'])
            if 'YOUR CODE STARTS HERE' in source and 'YOUR CODE ENDS HERE' in source:
                # Extract lines before, solution area, and after
                lines = cell['source']
                new_lines = []
                in_solution = False
                indent = ''
                
                for i, line in enumerate(lines):
                    if 'YOUR CODE STARTS HERE' in line:
                        in_solution = True
                        new_lines.append(line)
                        # Add blank lines in solution area
                        new_lines.append('\n')
                        new_lines.append('\n')
                    elif 'YOUR CODE ENDS HERE' in line:
                        in_solution = False
                        new_lines.append(line)
                        solutions_removed += 1
                    elif not in_solution:
                        new_lines.append(line)
                
                cell['source'] = new_lines
    
    # Clear kernel metadata if present
    if 'metadata' in nb:
        if 'language_info' in nb['metadata']:
            # Keep language info but remove version-specific details
            nb['metadata']['language_info'].pop('version', None)
        
        # Clear kernel spec version if present
        if 'kernelspec' in nb['metadata']:
            nb['metadata']['kernelspec'].pop('display_name', None)
    
    # Write back the cleaned notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Notebook cleaned successfully!")
    print(f"  - Cleared outputs from {cells_cleaned} code cells")
    print(f"  - Reset all execution counts")
    print(f"  - Removed {solutions_removed} student solutions")
    print(f"  - Notebook is ready for a fresh start")

if __name__ == '__main__':
    notebook_path = 'tensorflow_intro.ipynb'
    clean_notebook(notebook_path)

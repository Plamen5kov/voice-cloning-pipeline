#!/usr/bin/env python3
"""
Clean notebook script for gradient checking lab.
This script removes all outputs and student solutions from the notebook,
creating a clean version ready for distribution.
"""

import json
import sys
from pathlib import Path


def clean_notebook(notebook_path, output_path=None):
    """
    Clean a Jupyter notebook by removing outputs and student solutions.
    
    Args:
        notebook_path: Path to the notebook to clean
        output_path: Path for the cleaned notebook (default: same as input)
    """
    if output_path is None:
        output_path = notebook_path
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Process each cell
    for cell in nb['cells']:
        # Clear outputs from all cells
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
        
        # Clear student solutions in code cells
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            if isinstance(source, str):
                source = source.split('\n')
            
            new_source = []
            in_solution = False
            
            for line in source:
                # Detect start of student code section
                if '# YOUR CODE STARTS HERE' in line:
                    new_source.append(line if line.endswith('\n') else line + '\n')
                    in_solution = True
                    continue
                
                # Detect end of student code section
                if '# YOUR CODE ENDS HERE' in line:
                    new_source.append('\n')
                    new_source.append(line if line.endswith('\n') else line + '\n')
                    in_solution = False
                    continue
                
                # Skip lines within student solution
                if in_solution:
                    continue
                
                # Keep all other lines
                new_source.append(line if line.endswith('\n') else line + '\n')
            
            cell['source'] = new_source
    
    # Write the cleaned notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')  # Add final newline
    
    print(f"✓ Cleaned notebook saved to: {output_path}")


def main():
    """Main entry point for the script."""
    script_dir = Path(__file__).parent
    notebook_path = script_dir / "gradient_checking.ipynb"
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}", file=sys.stderr)
        sys.exit(1)
    
    # Allow output path to be specified as command-line argument
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = notebook_path
    
    clean_notebook(notebook_path, output_path)


if __name__ == "__main__":
    main()

# Visualizations Directory

This directory is used to save plots and visualizations generated during training and analysis.

## Generated Files

The notebook may save the following visualizations here:

- **Cost curves** - Training cost over iterations
- **Confusion matrices** - Genre classification confusion
- **Spectrograms** - Example mel-spectrograms
- **Mislabeled examples** - Audio clips the model got wrong
- **Performance comparisons** - 2-layer vs L-layer results

## Usage

The directory is initially empty. Visualizations are created when you run the notebook cells that generate plots.

You can save figures manually using:
```python
plt.savefig('visualizations/my_plot.png')
```

## Note

This directory is included in .gitignore to avoid committing large visualization files to the repository.

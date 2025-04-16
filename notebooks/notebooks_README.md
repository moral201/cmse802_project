# Notebooks (`notebooks/`)

This folder contains Jupyter notebooks used to visualize and debug the simulation during development.

## Files

- `main_simulation_visualization.ipynb`  
  Jupyter notebook used to run and animate the full-body simulation.  
  It integrates the stepping controller, arm dynamics, and fluid-body interaction for visualization and testing.

## Notes

- This notebook is primarily used for generating visual output, not for modular code development.
- All core logic is implemented in the `/src` folder; this notebook calls and visualizes those modules.

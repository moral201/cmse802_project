# Source Code (`src/`)

This folder contains the full implementation of the reactive stepping body simulation with fluid-body interaction.

## Files

- `main_simulation.py`  
  Entry point of the simulation. Contains the main loop for stepping body dynamics, arm control, and fluid updates.

- `physics_utils.py`  
  Holds reusable functions such as torque computation, body dynamics updates, particle initialization, and fluid-body collision logic.

- `global_constants.py`  
  Centralized file for all physical parameters, control gains, simulation settings, and fluid constants.

## Note

- All components are modular and designed to work together in the main simulation script.
- Only essential, final files are included here. Previous versions from earlier homework stages have been removed for clarity.

# Important NOTE: 
- Some import warnings (e.g., "could not be resolved") may appear in VSCode due to project root issues.
- These do not affect runtime and all modules function as expected when executed.
- The line limit was change to 120 characters.

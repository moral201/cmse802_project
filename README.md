# cmse802_project

## Final Project – Reactive Stepping Body with Environmental Fluid Interaction

This project simulates a 2D human-like body maintaining balance using a reactive stepping controller. The simulation also includes arm dynamics and interaction with environmental fluid particles (e.g., falling rain), demonstrating how the body reacts to continuous disturbances.

## Project Goals

- Model a physics-based, multi-joint body that actively balances under gravity.
- Implement arm control using joint torques to maintain posture.
- Simulate external fluid particles interacting with body segments.
- Visualize motion and particle-body collisions over time.

## Project Structure

This repository is organized as follows:

- **`src/`** – Contains all source code for physics and fluid simulations.  
- **`notebooks/`** – Jupyter notebooks for testing and visualizing simulations.  
- **`results/`** – Stores simulation results (generated videos).  
- **`tests/`** – Unit test scripts for verifying correctness of models.  
- **`.gitignore`** – Specifies files to exclude from version control.  
- **`README.md`** – This file, providing an overview of the project.  
- **`requirements.txt`** – Lists dependencies required for the project.

## Setting Up

### Clone the Repository
```sh
git clone https://github.com/your-username/cmse802_project.git
cd cmse802_project
pip install -r requirements.txt

## How to Run Simulation
cd notebooks
jupyter notebook main_simulation_visualization.ipynb

## Dependencies
Python 3.x
NumPy - Numerical computations
Matplotlib - Visualization



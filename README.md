# cmse802_project

# Physics-Based Human Body & Fluid Simulation  

## Project Overview  
This project simulates a **human-like body** reacting to external forces and fluid interactions, such as rain. The goal is to apply **physics-based modeling techniques** to maintain body stability, simulate external forces, and implement realistic fluid interactions.  

## Objectives  
- Develop a **multi-joint** physics-based body capable of maintaining balance.  
- Implement **external force simulations** and analyze response behavior.  
- Apply **fluid dynamics models** to simulate realistic environmental interactions.  
- Transition from **2D to 3D simulations**, increasing complexity gradually.  

## Project Structure  
This repository is organized as follows:  

- **`src/`** - Contains all source code for physics and fluid simulations.  
- **`notebooks/`** - Jupyter notebooks for testing simulations.  
- **`data/`** - Directory for sample datasets (if required).  
- **`docs/`** - Documentation and project reports.  
- **`results/`** - Stores simulation results and generated images.  
- **`tests/`** - Unit test scripts for verifying correctness of models.  
- **`.gitignore`** - Specifies files to exclude from version control.  
- **`README.md`** - This file, providing an overview of the project.  
- **`requirements.txt`** - Lists dependencies required for the project.   

##  Setting Up  
### ** Clone the Repository**
```sh
git clone https://github.com/your-username/cmse802_project.git
cd cmse802_project
pip install -r requirements.txt
python src/main.py

## Dependencies
Python 3.x
NumPy - Numerical computations
Matplotlib - Visualization
PyBullet - Physics engine for rigid-body simulation
PySPH - Particle-based fluid simulation



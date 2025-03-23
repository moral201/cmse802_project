# body_constants.py
"""
Module: body_constants.py

This module defines the body constants and related parameters for the physics-based 2D human body model using different approaches with inverted pendulum
to simulate biomechanical stability under gravitational forces. 

Author: Gerardo Morales
Date: March 2025
"""   

import numpy as np

# Physical constants
g = 9.81  # gravity (m/s^2)
l_leg = 1.0  # leg length (m)
l_torso = 1.2  # torso length (m)
l_arm = 1.0  # arm length (m)
head_radius = 0.3  # head size (m)
m = 1.0  # mass of the torso (kg)
b = 0.5  # damping coefficient

# Control parameters (default PD)
Kp = 100
Kd = 10

# Time-related parameters
dt = 0.02
total_time = 5  #seconds
num_steps = int(total_time / dt)

# Reactive step threshold
step_threshold = 0.4 # If center of mass moves this far from foot center, trigger a step

# Time vector (used in some models)
time = np.linspace(0, total_time, num_steps)

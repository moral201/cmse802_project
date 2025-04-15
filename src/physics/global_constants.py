# global_constants.py
"""
Module: global_constants.py

This module defines all global physical, control, simulation, and fluid parameters used in the simulations.
Constants are grouped into logical categories to facilitate readability and reusability.

Author: Gerardo Morales
Date: April 2025
"""

import numpy as np

# ------------------------------------------------------------------------------
# Physical Constants (Units in SI: meters, seconds, kilograms)
# ------------------------------------------------------------------------------
g = 9.81               # Gravity acceleration (m/s^2)
m = 1.0                # Mass of the body segments (kg)
b = 0.5                # Damping coefficient (kg*m^2/s)

# ------------------------------------------------------------------------------
# Body Dimensions (Lengths in meters)
# ------------------------------------------------------------------------------
l_leg = 1.0            # Length of each leg segment (m)
l_torso = 1.2          # Length of the torso (m)
l_arm = 1.0            # Total length of an arm (m)
l_upper = 0.4          # Length of upper arm (m)
l_forearm = 0.4        # Length of forearm (m)
head_radius = 0.3      # Radius of the circular head (m)

# ------------------------------------------------------------------------------
# Control Parameters for PD Controllers
# ------------------------------------------------------------------------------
Kp = 100               # Proportional gain for body balance controller
Kd = 10                # Derivative gain for body balance controller

# Arm-specific PD gains (used in the torque calculations)
Kp_arm_shoulder = 20
Kd_arm_shoulder = 3
Kp_arm_elbow = 15
Kd_arm_elbow = 2

# Target joint angles (radians) for default arm posture
# These are used for stabilizing the arms into a fixed pose.
target_theta1_L = 4 * np.pi / 3  # Left shoulder
target_theta2_L = 0             # Left elbow
target_theta1_R = 0             # Right shoulder
target_theta2_R = np.pi / 6     # Right elbow

# ------------------------------------------------------------------------------
# Time Parameters for Simulation
# ------------------------------------------------------------------------------
dt = 0.02                       # Time step (s)
total_time = 5                 # Total simulation time (s)
num_steps = int(total_time / dt)  # Total number of time steps
time = np.linspace(0, total_time, num_steps)  # Time vector

# ------------------------------------------------------------------------------
# Stepping Logic Parameters
# ------------------------------------------------------------------------------
step_threshold = 0.4           # COM distance from foot center to trigger a step (m)

# ------------------------------------------------------------------------------
# Arm Position Offset
# ------------------------------------------------------------------------------
shoulder_offset = 0.2          # Distance from torso top to shoulder joint (m)

# ------------------------------------------------------------------------------
# Fluid Simulation Parameters
# ------------------------------------------------------------------------------
fluid_x_range = (-2, 2)        # Horizontal range for rain spawn (m)
fluid_y_range = (2.5, 3.0)     # Initial vertical range of fluid particles (m)
fluid_y_reset = 3.0            # Y position to respawn particles (m)
fluid_elasticity = 0.2         # Coefficient of restitution for particle collisions
fluid_thickness = 0.07         # Effective collision radius for body segments (m)
fluid_seed = 42                # Random seed for particle generation

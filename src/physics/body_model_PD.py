# body_model_PD.py
"""
Module: body_model_PD.py

This module implements a physics-based 2D human body model using an inverted pendulum approach 
to simulate biomechanical stability under gravitational forces. The model applies a PD controller 
to adjust balance.

Author: Gerardo Morales
Date: March 2025
"""

import numpy as np
from body_constants import g, l_leg, l_torso, l_arm, head_radius, m, b, Kp, Kd, dt, total_time, num_steps

def compute_torque(theta, omega, Kp, Kd):
    """Compute PD controller torque."""
    return -Kp * theta - Kd * omega

def update_dynamics(theta, omega, torque, m, g, l_leg, b, dt):
    """Update angular dynamics."""
    domega_dt = (-m * g * l_leg * np.sin(theta) - b * omega + torque) / (m * l_leg**2)
    omega_new = omega + domega_dt * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new

def simulate_balance():
    """
    Simulates a two-legged human-like figure maintaining balance
    under gravitational forces using a PD-controlled inverted pendulum (the torso).
    
    Returns:
        time (numpy array): Time steps
        theta_vals (list): Angle values over time
        x_torso_vals (list): X-coordinates of the torso
        y_torso_vals (list): Y-coordinates of the torso
        x_arm_vals (list): X-coordinates of the arms
        y_arm_vals (list): Y-coordinates of the arms
        x_head_vals (list): X-coordinates of the head
        y_head_vals (list): Y-coordinates of the head
    """

    # Initial conditions
    theta = np.pi / 8  # Initial tilt
    omega = 0  # Initial angular velocity
    time = np.linspace(0, total_time, num_steps)

    # Arrays to store results
    theta_vals = []
    x_torso_vals = []
    y_torso_vals = []
    x_arm_vals = []
    y_arm_vals = []
    x_head_vals = []
    y_head_vals = []

    # Simulate balance dynamics
    for t in time:
        torque = compute_torque(theta, omega, Kp, Kd)
        theta, omega = update_dynamics(theta, omega, torque, m, g, l_leg, b, dt)
        
        # Convert to Cartesian coordinates
        x_torso = l_leg * np.sin(theta)
        y_torso = l_leg * np.cos(theta) + l_leg  # Torso position

        # Shoulder position 
        x_shoulder = x_torso
        y_shoulder = y_torso - 0.2 

        # Arms Position
        x_arm = [x_shoulder - l_arm / 2, x_shoulder + l_arm / 2]
        y_arm = [y_shoulder , y_shoulder ]

        # Head Position
        x_head = x_shoulder
        y_head = y_shoulder + head_radius + 0.2

        # Store values
        theta_vals.append(theta)
        x_torso_vals.append(x_torso)
        y_torso_vals.append(y_torso)
        x_arm_vals.append(x_arm)
        y_arm_vals.append(y_arm)
        x_head_vals.append(x_head)
        y_head_vals.append(y_head)

    return time, theta_vals, x_torso_vals, y_torso_vals, x_arm_vals, y_arm_vals, x_head_vals, y_head_vals

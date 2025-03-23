# reactive_stepping_body.py
"""
Module: reactive_stepping_body.py

Simulates a two-legged stick figure with knee joints. The figure maintains balance while standing
and takes a reactive step if pushed beyond a stability threshold.

Author: Gerardo
Date: March 2025
"""

import numpy as np
from body_constants import g, l_leg, l_torso, l_arm, head_radius, m, b, Kp, Kd, dt, total_time, num_steps, step_threshold

def compute_torque(theta, omega, Kp, Kd):
    """Compute PD controller torque with damping."""
    return -Kp * theta - Kd * omega

def update_stepping_dynamics(theta, omega, torque, m, l_leg, dt):
    """Update stepping body angular dynamics."""
    domega_dt = (torque - 0.2 * omega) / (m * l_leg**2)
    omega_new = omega + domega_dt * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new

def simulate_reactive_stepping_body():
    """
    Simulates a 2D human-like body standing on two legs with knees. If pushed too far, it triggers a step to regain balance.

    Returns:
        time (np.array): Time steps
        torso_x (list): X position of torso top
        torso_y (list): Y position of torso top
        foot_positions (list): [(left_foot_x, right_foot_x)]
        com_positions (list): X positions of center of mass over time
        step_events (list): Frame index when a step occurs
    """

    time = np.linspace(0, total_time, num_steps)

    # Initial conditions
    theta = np.pi / 10
    omega = 0
    x_com = 0.0
    v_com = 0.0

    # Fixed initial foot positions
    left_foot_x = -0.3
    right_foot_x = 0.3
    support_foot = "both"
    step_events = []

    # Storage
    torso_x = []
    torso_y = []
    com_positions = []
    foot_positions = []

    for i, t in enumerate(time):
        torque = compute_torque(theta, omega, Kp, Kd)
        theta, omega = update_stepping_dynamics(theta, omega, torque, m, l_leg, dt)

        # External push
        if 1.5 < t < 1.52:
            v_com += 3.0

        foot_center = (left_foot_x + right_foot_x) / 2
        if abs(x_com - foot_center) > step_threshold and support_foot == "both":
            right_foot_x = x_com + 0.3
            support_foot = "left"
            step_events.append(i)

        if support_foot == "left":
            correction_strength = 1
            v_com -= correction_strength * (x_com - foot_center)
            v_com *= 0.98

        x_com += v_com * dt

        # Torso position
        x_torso = x_com + l_leg * np.sin(theta)
        y_torso = l_leg * np.cos(theta) + l_leg

        # Store values
        torso_x.append(x_torso)
        torso_y.append(y_torso)
        com_positions.append(x_com)
        foot_positions.append((left_foot_x, right_foot_x))

    return time, torso_x, torso_y, foot_positions, com_positions, step_events
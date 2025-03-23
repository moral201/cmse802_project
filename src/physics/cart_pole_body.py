# cart_pole_body.py
"""
Module: cart_pole_body.py

Simulates a full-body human-like figure using a cart-pole approach, where the base (legs) moves horizontally
to balance the upper body (torso, arms, and head). This implements a PD-controlled inverted pendulum on a cart (legs).

Author: Gerardo Morales
Date: March 2025
"""

import numpy as np
from body_constants import g, l_leg, l_torso, l_arm, head_radius, m, b, Kp, Kd, dt, total_time, num_steps

def compute_force(theta, omega, x_base, Kp, Kd, Kx=10):
    """Compute horizontal PD control force for cart."""
    return -Kp * theta - Kd * omega - Kx * x_base

def update_cart_dynamics(theta, omega, x_base, v_base, force, m, g, l_leg, b, dt):
    """Update cart-pole dynamics and state."""
    # Horizontal base dynamics
    a_base = force / m
    v_base_new = v_base + a_base * dt
    x_base_new = x_base + v_base_new * dt

    # Angular pole dynamics
    torque = -m * g * l_leg * np.sin(theta) + force * l_leg * np.cos(theta)
    domega_dt = (torque - b * omega) / (m * l_leg**2)
    omega_new = omega + domega_dt * dt
    theta_new = theta + omega_new * dt

    return theta_new, omega_new, x_base_new, v_base_new

def simulate_cart_pole_body():
    """
    Simulates a cart-pole-like human body maintaining balance using horizontal base movement (legs).

    Returns:
        time (numpy array): Time steps
        x_base_vals (list): Horizontal base positions (legs)
        x_torso_vals (list): X-coordinates of the torso
        y_torso_vals (list): Y-coordinates of the torso
        x_arm_vals (list): X-coordinates of arms [x1, x2]
        y_arm_vals (list): Y-coordinates of arms [y1, y2]
        x_head_vals (list): X-coordinates of the head
        y_head_vals (list): Y-coordinates of the head
    """

    # Initial conditions
    theta = np.pi / 8  # initial tilt
    omega = 0          # angular velocity
    x_base = 0         # horizontal base (leg joint position)
    v_base = 0         # base velocity

    time = np.linspace(0, total_time, num_steps)

    # Output storage
    x_base_vals = []
    x_torso_vals = []
    y_torso_vals = []
    x_arm_vals = []
    y_arm_vals = []
    x_head_vals = []
    y_head_vals = []

    for t in time:
        # Compute PD controller force
        force = compute_force(theta, omega, x_base, Kp, Kd)

        # Update dynamics
        theta, omega, x_base, v_base = update_cart_dynamics(theta, omega, x_base, v_base,
                                                            force, m, g, l_leg, b, dt)

        # Torso position
        x_torso = x_base + l_leg * np.sin(theta)
        y_torso = l_leg * np.cos(theta) + l_leg

        # Shoulder (top of torso)
        x_shoulder = x_torso
        y_shoulder = y_torso - 0.2

        # Arms (straight horizontal from shoulder)
        x_arm = [x_shoulder - l_arm / 2, x_shoulder + l_arm / 2]
        y_arm = [y_shoulder, y_shoulder]

        # Head (above shoulders)
        x_head = x_shoulder
        y_head = y_shoulder + head_radius + 0.2

        # Store results
        x_base_vals.append(x_base)
        x_torso_vals.append(x_torso)
        y_torso_vals.append(y_torso)
        x_arm_vals.append(x_arm)
        y_arm_vals.append(y_arm)
        x_head_vals.append(x_head)
        y_head_vals.append(y_head)

    return time, x_base_vals, x_torso_vals, y_torso_vals, x_arm_vals, y_arm_vals, x_head_vals, y_head_vals
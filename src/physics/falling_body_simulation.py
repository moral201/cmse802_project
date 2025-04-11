"""
Module: falling_body_simulation.py

Simulates a full-body figure where gravity acts on the center of mass, and legs stay at fixed length.
No balancing control unless explicitly added. The body will fall naturally under gravity.

Author: Gerardo
Date: April 2025
"""

import numpy as np
from body_constants import g, l_leg, l_torso, m, b, Kp, Kd, dt, total_time, num_steps


def simulate_falling_body():
    """
    Simulates a falling body where gravity affects the center of mass.
    Returns:
        time, torso_x, torso_y, com_positions, foot_positions
    """

    time = np.linspace(0, total_time, num_steps)

    # Initial conditions
    theta = np.pi / 10  # Initial tilt
    omega = 0

    x_com = 0.0
    y_com = l_leg + l_leg  # Initial height
    v_com_x = 0.0
    v_com_y = 0.0

    left_foot_x = -0.3
    right_foot_x = 0.3

    torso_x, torso_y = [], []
    com_positions, foot_positions = [], []

    for _ in time:
        # Gravity torque causes fall
        gravity_torque = -m * g * l_leg * np.sin(theta)
        torque = gravity_torque  # No control torque

        domega_dt = (torque - b * omega) / (m * l_leg ** 2)
        omega += domega_dt * dt
        theta += omega * dt

        # COM dynamics
        v_com_y -= g * dt
        x_com += v_com_x * dt
        y_com += v_com_y * dt

        # Ground collision
        if y_com <= 0:
            y_com = 0
            v_com_y = 0

        # Torso top position
        x_torso = x_com + l_leg * np.sin(theta)
        y_torso = y_com + l_leg * np.cos(theta) + l_leg

        torso_x.append(x_torso)
        torso_y.append(y_torso)
        com_positions.append((x_com, y_com))
        foot_positions.append((left_foot_x, right_foot_x))

    return time, torso_x, torso_y, foot_positions, com_positions

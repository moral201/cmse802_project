"""
Module: physics_utils.py

Contains reusable functions for the reactive stepping body simulation
including torque computation, body dynamics update, fluid initialization,
and fluid-body collision handling.

Author: Gerardo Morales
Date: April 2025
"""

import numpy as np

from global_constants import head_radius


def compute_torque(theta, omega, Kp, Kd):
    """
    Computes the control torque for a given joint using a PD controller.

    Parameters:
        theta (float): Current angle (rad)
        omega (float): Current angular velocity (rad/s)
        Kp (float): Proportional gain
        Kd (float): Derivative gain

    Returns:
        float: Computed torque (N*m)
    """
    return -Kp * theta - Kd * omega


def update_stepping_dynamics(theta, omega, torque, m, len, dt):
    """
    Updates the angular position and velocity of a body segment.

    Parameters:
        theta (float): Current angle (rad)
        omega (float): Current angular velocity (rad/s)
        torque (float): Net applied torque (N*m)
        m (float): Segment mass (kg)
        len (float): Segment length (m)
        dt (float): Time step (s)

    Returns:
        tuple: Updated (theta, omega)
    """
    # Angular acceleration from torque
    domega_dt = (torque - 0.2 * omega) / (m * len**2)
    omega_new = omega + domega_dt * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new


def initialize_particles(num_particles, x_range=(-2, 2), y_range=(2.5, 3.0), seed=42):
    """
    Initializes fluid particles with random positions and zero velocity.

    Parameters:
        num_particles (int): Number of particles
        x_range (tuple): Min and max X positions
        y_range (tuple): Min and max Y positions
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (x_positions, y_positions, x_velocities, y_velocities)
    """
    np.random.seed(seed)
    x_positions = np.random.uniform(*x_range, num_particles)
    y_positions = np.random.uniform(*y_range, num_particles)
    y_velocities = np.zeros(num_particles)
    x_velocities = np.zeros(num_particles)
    return x_positions, y_positions, x_velocities, y_velocities


def reflect_particles_from_segment(
    xp, yp, xv, yv, x1, y1, x2, y2, thickness, elasticity
):
    """
    Reflects particles from a line segment using a normal-based reflection.

    Parameters:
        xp, yp (arrays): Particle positions
        xv, yv (arrays): Particle velocities
        x1, y1 (float): Segment start point
        x2, y2 (float): Segment end point
        thickness (float): Collision threshold (m)
        elasticity (float): Coefficient of restitution [0,1]

    Returns:
        tuple: (xv, yv) Updated velocities
    """
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length == 0:
        return xv, yv  # Avoid division by zero

    nx = -dy / length  # Normal vector components
    ny = dx / length

    dist = np.abs((xp - x1) * nx + (yp - y1) * ny)

    # Closest point along segment
    t = ((xp - x1) * dx + (yp - y1) * dy) / length**2
    t = np.clip(t, 0, 1)

    x_closest = x1 + t * dx
    y_closest = y1 + t * dy

    colliding = dist < thickness
    close_enough = (np.abs(xp - x_closest) < 0.5) & (np.abs(yp - y_closest) < 0.5)

    active = colliding & close_enough

    if active.any():
        v_dot_n = xv[active] * nx + yv[active] * ny
        xv[active] -= (1 + elasticity) * v_dot_n * nx
        yv[active] -= (1 + elasticity) * v_dot_n * ny

    return xv, yv


def update_fluid_dynamics(
    xp,
    yp,
    xv,
    yv,
    segments,
    x_head,
    y_head,
    shoulder_x,
    shoulder_y,
    elbow_x_L,
    elbow_y_L,
    g,
    elasticity,
    dt,
    x_range=(-2, 2),
    y_reset=3.0,
):
    """
    Updates the positions and velocities of fluid particles
    accounting for gravity, collisions, and body reflection.

    Parameters:
        xp, yp (arrays): Particle positions
        xv, yv (arrays): Particle velocities
        segments (list): List of segments for collision
        x_head, y_head (float): Head position
        shoulder_x, shoulder_y, elbow_x_L, elbow_y_L (float): Arm segment points
        g (float): Gravitational constant (m/s^2)
        elasticity (float): Collision restitution factor [0,1]
        dt (float): Time step (s)
        x_range (tuple): Respawn X range
        y_reset (float): Respawn height

    Returns:
        tuple: (yp, xv, yv) Updated positions and velocities
    """
    # Gravity
    yv -= g * dt
    xp += xv * dt
    yp += yv * dt

    # Ground reset
    ground_collision = yp <= 0
    yp[ground_collision] = y_reset
    xp[ground_collision] = np.random.uniform(*x_range, ground_collision.sum())
    yv[ground_collision] = 0
    xv[ground_collision] = 0

    # Head collision (offset for accuracy)
    dx = xp - x_head
    dy = yp - (y_head - 0.1)
    dist = np.sqrt(dx**2 + dy**2)
    head_collision = dist < head_radius

    xp[head_collision] = (
        x_head + dx[head_collision] * head_radius / dist[head_collision]
    )
    yp[head_collision] = (
        y_head + dy[head_collision] * head_radius / dist[head_collision]
    )

    v_dot_n = (
        xv[head_collision] * dx[head_collision] / dist[head_collision]
        + yv[head_collision] * dy[head_collision] / dist[head_collision]
    )

    xv[head_collision] -= (
        (1 + elasticity) * v_dot_n * dx[head_collision] / dist[head_collision]
    )
    yv[head_collision] -= (
        (1 + elasticity) * v_dot_n * dy[head_collision] / dist[head_collision]
    )

    # Reflect from all segments
    for x1, y1, x2, y2 in segments:
        xv, yv = reflect_particles_from_segment(
            xp, yp, xv, yv, x1, y1, x2, y2, thickness=0.07, elasticity=elasticity
        )

    return yp, xv, yv

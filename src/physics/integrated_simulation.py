"""
Module: integrated_simulation.py

Simulates a physics-based 2D human body model with reactive stepping control
and fluid-body collision dynamics (realistic horizontal particle movement).
All physical, control, and simulation constants are imported from global_constants.py.

Author: Gerardo Morales
Date: April 2025
"""

import numpy as np
from global_constants import (
    g, l_leg, l_torso, l_arm, head_radius, m, b, Kp, Kd,
    dt, total_time, num_steps, step_threshold
)

# ----- Torque Controller -----
def compute_torque(theta, omega, Kp, Kd):
    """PD controller torque computation."""
    return -Kp * theta - Kd * omega


# ----- Body Dynamics Update -----
def update_stepping_dynamics(theta, omega, torque, m, l, dt):
    """Updates angle and angular velocity for a single body segment."""
    domega_dt = (torque - 0.2 * omega) / (m * l**2)
    omega_new = omega + domega_dt * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new


# ----- Fluid Initialization -----
def initialize_particles(num_particles, x_range=(-2, 2), y_range=(2.5, 3.0), seed=42):
    """Initialize particle positions and velocities."""
    np.random.seed(seed)
    x_positions = np.random.uniform(*x_range, num_particles)
    y_positions = np.random.uniform(*y_range, num_particles)
    y_velocities = np.zeros(num_particles)
    x_velocities = np.zeros(num_particles)
    return x_positions, y_positions, x_velocities, y_velocities


# ----- Fluid Collision Reflection -----
def reflect_particles_from_segment(xp, yp, xv, yv, x1, y1, x2, y2, thickness, elasticity):
    """Reflects fluid particles from a body segment."""
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return xv, yv
    nx = -dy / length
    ny = dx / length
    dist = np.abs((xp - x1) * nx + (yp - y1) * ny)
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


# ----- Fluid Dynamics Update -----
def update_fluid_dynamics(xp, yp, xv, yv, segments, x_head, y_head,
                           shoulder_x, shoulder_y, elbow_x_L, elbow_y_L,
                           g, elasticity, dt, x_range=(-2, 2), y_reset=3.0):
    """Update fluid particles with gravity, collisions, and reflection."""
    # Gravity update
    yv -= g * dt
    xp += xv * dt
    yp += yv * dt

    # Ground reset
    ground_collision = yp <= 0
    yp[ground_collision] = y_reset
    xp[ground_collision] = np.random.uniform(*x_range, ground_collision.sum())
    yv[ground_collision] = 0
    xv[ground_collision] = 0

    # Head collision (offset y-axis slightly for better accuracy)
    dx = xp - x_head
    dy = yp - (y_head - 0.1)
    dist = np.sqrt(dx**2 + dy**2)
    head_collision = dist < head_radius
    xp[head_collision] = x_head + dx[head_collision] * head_radius / dist[head_collision]
    yp[head_collision] = y_head + dy[head_collision] * head_radius / dist[head_collision]
    v_dot_n = xv[head_collision] * dx[head_collision] / dist[head_collision] + \
              yv[head_collision] * dy[head_collision] / dist[head_collision]
    xv[head_collision] -= (1 + elasticity) * v_dot_n * dx[head_collision] / dist[head_collision]
    yv[head_collision] -= (1 + elasticity) * v_dot_n * dy[head_collision] / dist[head_collision]

    # Reflect from body segments
    for (x1, y1, x2, y2) in segments:
        xv, yv = reflect_particles_from_segment(xp, yp, xv, yv, x1, y1, x2, y2, thickness=0.07, elasticity=elasticity)

    return yp, xv, yv


# ----- Main Simulation -----
def simulate_reactive_stepping_body(num_particles=200, elasticity=0.2):
    """Simulates the full body dynamics with fluid-body interactions."""
    time = np.linspace(0, total_time, num_steps)

    # Initial states
    theta, omega, x_com, v_com = np.pi / 3, 0, 0.0, 0.0
    left_foot_x, right_foot_x = -0.3, 0.3
    support_foot, step_events = "both", []

    l_upper, l_forearm = 0.4, 0.4
    shoulder_offset = 0.2

    target_theta1_L, target_theta2_L = 4 * np.pi / 3, 0
    target_theta1_R, target_theta2_R = 0, np.pi / 6

    theta1_L, omega1_L, theta2_L, omega2_L = np.pi, 0.0, 0.0, 0.0
    theta1_R, omega1_R, theta2_R, omega2_R = 0.0, 0.0, 0.0, 0.0

    torso_x, torso_y, com_positions, foot_positions = [], [], [], []
    left_arm_joints, right_arm_joints = [], []

    x_particles, y_particles, x_velocities, y_velocities = initialize_particles(num_particles)

    y_trajectory = []
    x_trajectory = []

    # ----- Simulation Loop -----
    for i, t in enumerate(time):

        # Torso balance update
        gravity_torque = m * g * l_leg * np.sin(theta)
        control_torque = compute_torque(theta, omega, Kp, Kd)
        theta, omega = update_stepping_dynamics(theta, omega, gravity_torque + control_torque, m, l_leg, dt)

        # External force event
        if 1.5 < t < 1.52:
            v_com += 3.0

        # Step triggering
        foot_center = (left_foot_x + right_foot_x) / 2
        if abs(x_com - foot_center) > step_threshold and support_foot == "both":
            right_foot_x = x_com + 0.3
            support_foot = "left"
            step_events.append(i)

        if support_foot == "left":
            if abs(x_com - foot_center) > 0.05:
                v_com -= 1 * (x_com - foot_center)
            v_com *= 0.95

        # COM Update
        x_com += v_com * dt
        x_torso = x_com + l_leg * np.sin(theta)
        y_torso = l_leg * np.cos(theta) + l_leg

        total_body_length = l_leg + l_torso
        shoulder_ratio = (total_body_length - shoulder_offset) / total_body_length
        shoulder_x = x_com * (1 - shoulder_ratio) + x_torso * shoulder_ratio
        shoulder_y = l_leg * (1 - shoulder_ratio) + y_torso * shoulder_ratio

        # Arms control and positions
        target_theta2_R = np.pi / 6 + 0.3 * np.sin(2 * np.pi * 1.5 * t)

        torque1_L = compute_torque(theta1_L - target_theta1_L, omega1_L, 20, 3)
        torque2_L = compute_torque(theta2_L - target_theta2_L, omega2_L, 15, 2)
        theta1_L, omega1_L = update_stepping_dynamics(theta1_L, omega1_L, torque1_L, m, l_upper, dt)
        theta2_L, omega2_L = update_stepping_dynamics(theta2_L, omega2_L, torque2_L, m, l_forearm, dt)

        torque1_R = compute_torque(theta1_R - target_theta1_R, omega1_R, 20, 3)
        torque2_R = compute_torque(theta2_R - target_theta2_R, omega2_R, 15, 2)
        theta1_R, omega1_R = update_stepping_dynamics(theta1_R, omega1_R, torque1_R, m, l_upper, dt)
        theta2_R, omega2_R = update_stepping_dynamics(theta2_R, omega2_R, torque2_R, m, l_forearm, dt)

        elbow_x_L = shoulder_x + l_upper * np.cos(theta1_L)
        elbow_y_L = shoulder_y + l_upper * np.sin(theta1_L)
        hand_x_L = elbow_x_L + l_forearm * np.cos(theta1_L + theta2_L)
        hand_y_L = elbow_y_L + l_forearm * np.sin(theta1_L + theta2_L)

        elbow_x_R = shoulder_x + l_upper * np.cos(theta1_R)
        elbow_y_R = shoulder_y + l_upper * np.sin(theta1_R)
        hand_x_R = elbow_x_R + l_forearm * np.cos(theta1_R + theta2_R)
        hand_y_R = elbow_y_R + l_forearm * np.sin(theta1_R + theta2_R)

        # Store body state
        torso_x.append(x_torso)
        torso_y.append(y_torso)
        com_positions.append(x_com)
        foot_positions.append((left_foot_x, right_foot_x))
        left_arm_joints.append([(shoulder_x, shoulder_y), (elbow_x_L, elbow_y_L), (hand_x_L, hand_y_L)])
        right_arm_joints.append([(shoulder_x, shoulder_y), (elbow_x_R, elbow_y_R), (hand_x_R, hand_y_R)])

        # Segments for collisions
        segments = [
            (x_com, l_leg, x_torso, y_torso),
            (shoulder_x, shoulder_y, elbow_x_L, elbow_y_L),
            (elbow_x_L, elbow_y_L, hand_x_L, hand_y_L),
            (shoulder_x+0.25, shoulder_y, elbow_x_R, elbow_y_R),
            (elbow_x_R, elbow_y_R, hand_x_R, hand_y_R),
            (left_foot_x, 0, x_com, l_leg),
            (right_foot_x, 0, x_com, l_leg),
        ]

        # Head position
        x_head = shoulder_x
        y_head = shoulder_y + head_radius + 0.2

        # Fluid Update
        y_particles, x_velocities, y_velocities = update_fluid_dynamics(
            x_particles, y_particles, x_velocities, y_velocities,
            segments, x_head, y_head,
            shoulder_x, shoulder_y, elbow_x_L, elbow_y_L,
            g, elasticity, dt
        )

        y_trajectory.append(y_particles.copy())
        x_trajectory.append(x_particles.copy())

    print("Running integrated_simulation.py with BODY + FLUID + HORIZONTAL FLUID COLLISION")

    return time, torso_x, torso_y, foot_positions, com_positions, step_events, left_arm_joints, right_arm_joints, x_trajectory, y_trajectory

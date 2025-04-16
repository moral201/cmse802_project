"""
Module: main_simulation.py

Main simulation loop for a physics-based 2D human body model with reactive stepping
and fluid-body interaction using vertical and horizontal particle collisions. Constants are
imported from global_constants.py for clarity and maintainability.

Author: Gerardo Morales
Date: April 2025
"""

import numpy as np

from global_constants import (
    Kd,
    Kd_arm_elbow,
    Kd_arm_shoulder,
    Kp,
    Kp_arm_elbow,
    Kp_arm_shoulder,
    dt,
    fluid_elasticity,
    fluid_seed,
    fluid_x_range,
    fluid_y_range,
    g,
    head_radius,
    l_forearm,
    l_leg,
    l_torso,
    l_upper,
    m,
    num_steps,
    shoulder_offset,
    step_threshold,
    target_theta1_L,
    target_theta1_R,
    target_theta2_L,
    target_theta2_R,
    total_time,
)
from physics_utils import (
    compute_torque,
    initialize_particles,
    update_fluid_dynamics,
    update_stepping_dynamics,
)


def simulate_reactive_stepping_body(num_particles=200, elasticity=fluid_elasticity):
    """
    Simulates the full body dynamics with reactive stepping and fluid-body interactions.

    Args:
        num_particles (int): Number of fluid particles to simulate.
        elasticity (float): Elasticity coefficient for collisions.

    Returns:
        tuple: Simulation time series data including body and fluid states.
    """
    time = np.linspace(0, total_time, num_steps)

    # Initial state variables
    theta, omega, x_com, v_com = np.pi / 3, 0, 0.0, 0.0
    left_foot_x, right_foot_x = -0.3, 0.3
    support_foot, step_events = "both", []

    # Arm angles (initial angles and angular velocities)
    theta1_L, omega1_L, theta2_L, omega2_L = np.pi, 0.0, 0.0, 0.0
    theta1_R, omega1_R, theta2_R, omega2_R = 0.0, 0.0, 0.0, 0.0

    # Trajectories for output
    torso_x, torso_y = [], []
    com_positions, foot_positions = [], []
    left_arm_joints, right_arm_joints = [], []
    x_particles, y_particles, x_velocities, y_velocities = initialize_particles(
        num_particles, x_range=fluid_x_range, y_range=fluid_y_range, seed=fluid_seed
    )
    x_trajectory, y_trajectory = [], []

    for i, t in enumerate(time):
        # --- TORSO BALANCE CONTROL ---
        gravity_torque = m * g * l_leg * np.sin(theta)
        control_torque = compute_torque(theta, omega, Kp, Kd)
        theta, omega = update_stepping_dynamics(
            theta, omega, gravity_torque + control_torque, m, l_leg, dt
        )

        # --- EXTERNAL IMPULSE EVENT ---
        if 1.5 < t < 1.52:
            v_com += 3.0

        # --- REACTIVE STEP LOGIC ---
        foot_center = (left_foot_x + right_foot_x) / 2
        if abs(x_com - foot_center) > step_threshold and support_foot == "both":
            right_foot_x = x_com + 0.3
            support_foot = "left"
            step_events.append(i)

        if support_foot == "left":
            if abs(x_com - foot_center) > 0.05:
                v_com -= 1 * (x_com - foot_center)
            v_com *= 0.95

        # --- UPDATE BODY POSITIONS ---
        x_com += v_com * dt
        x_torso = x_com + l_leg * np.sin(theta)
        y_torso = l_leg * np.cos(theta) + l_leg

        total_body_length = l_leg + l_torso
        shoulder_ratio = (total_body_length - shoulder_offset) / total_body_length
        shoulder_x = x_com * (1 - shoulder_ratio) + x_torso * shoulder_ratio
        shoulder_y = l_leg * (1 - shoulder_ratio) + y_torso * shoulder_ratio

        # --- ARM CONTROL ---
        # Add dynamic oscillation to target right elbow angle
        target_theta2_R_osc = target_theta2_R + 0.3 * np.sin(2 * np.pi * 1.5 * t)

        torque1_L = compute_torque(
            theta1_L - target_theta1_L, omega1_L, Kp_arm_shoulder, Kd_arm_shoulder
        )
        torque2_L = compute_torque(
            theta2_L - target_theta2_L, omega2_L, Kp_arm_elbow, Kd_arm_elbow
        )
        theta1_L, omega1_L = update_stepping_dynamics(
            theta1_L, omega1_L, torque1_L, m, l_upper, dt
        )
        theta2_L, omega2_L = update_stepping_dynamics(
            theta2_L, omega2_L, torque2_L, m, l_forearm, dt
        )

        torque1_R = compute_torque(
            theta1_R - target_theta1_R, omega1_R, Kp_arm_shoulder, Kd_arm_shoulder
        )
        torque2_R = compute_torque(
            theta2_R - target_theta2_R_osc, omega2_R, Kp_arm_elbow, Kd_arm_elbow
        )
        theta1_R, omega1_R = update_stepping_dynamics(
            theta1_R, omega1_R, torque1_R, m, l_upper, dt
        )
        theta2_R, omega2_R = update_stepping_dynamics(
            theta2_R, omega2_R, torque2_R, m, l_forearm, dt
        )

        # --- ARM SEGMENT POSITIONS ---
        elbow_x_L = shoulder_x + l_upper * np.cos(theta1_L)
        elbow_y_L = shoulder_y + l_upper * np.sin(theta1_L)
        hand_x_L = elbow_x_L + l_forearm * np.cos(theta1_L + theta2_L)
        hand_y_L = elbow_y_L + l_forearm * np.sin(theta1_L + theta2_L)

        elbow_x_R = shoulder_x + l_upper * np.cos(theta1_R)
        elbow_y_R = shoulder_y + l_upper * np.sin(theta1_R)
        hand_x_R = elbow_x_R + l_forearm * np.cos(theta1_R + theta2_R)
        hand_y_R = elbow_y_R + l_forearm * np.sin(theta1_R + theta2_R)

        # --- SAVE BODY STATE ---
        torso_x.append(x_torso)
        torso_y.append(y_torso)
        com_positions.append(x_com)
        foot_positions.append((left_foot_x, right_foot_x))
        left_arm_joints.append(
            [(shoulder_x, shoulder_y), (elbow_x_L, elbow_y_L), (hand_x_L, hand_y_L)]
        )
        right_arm_joints.append(
            [(shoulder_x, shoulder_y), (elbow_x_R, elbow_y_R), (hand_x_R, hand_y_R)]
        )

        # --- COLLISION SEGMENTS ---
        segments = [
            (x_com, l_leg, x_torso, y_torso),
            (shoulder_x, shoulder_y, elbow_x_L, elbow_y_L),
            (elbow_x_L, elbow_y_L, hand_x_L, hand_y_L),
            (
                shoulder_x + 0.25,
                shoulder_y,
                elbow_x_R,
                elbow_y_R,
            ),  # moved right to avoid visual bug
            (elbow_x_R, elbow_y_R, hand_x_R, hand_y_R),
            (left_foot_x, 0, x_com, l_leg),
            (right_foot_x, 0, x_com, l_leg),
        ]

        # --- FLUID SIMULATION ---
        x_head = shoulder_x
        y_head = shoulder_y + head_radius + 0.2

        y_particles, x_velocities, y_velocities = update_fluid_dynamics(
            x_particles,
            y_particles,
            x_velocities,
            y_velocities,
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
        )

        # --- SAVE FLUID STATE ---
        y_trajectory.append(y_particles.copy())
        x_trajectory.append(x_particles.copy())

    return (
        time,
        torso_x,
        torso_y,
        foot_positions,
        com_positions,
        step_events,
        left_arm_joints,
        right_arm_joints,
        x_trajectory,
        y_trajectory,
    )

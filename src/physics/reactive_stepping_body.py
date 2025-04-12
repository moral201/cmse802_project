"""
Module: reactive_stepping_body.py

Simulates a two-legged stick figure with knee joints and arms.
The figure maintains balance while standing and takes a reactive step if pushed beyond a stability threshold.

Author: Gerardo
Date: April 2025
"""

import numpy as np
from body_constants import g, l_leg, l_torso, l_arm, head_radius, m, b, Kp, Kd, dt, total_time, num_steps, step_threshold

def compute_torque(theta, omega, Kp, Kd):
    return -Kp * theta - Kd * omega

def update_stepping_dynamics(theta, omega, torque, m, l, dt):
    domega_dt = (torque - 0.2 * omega) / (m * l**2)
    omega_new = omega + domega_dt * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new

def simulate_reactive_stepping_body():
    time = np.linspace(0, total_time, num_steps)

    # Initial conditions
    theta = np.pi / 3
    omega = 0
    x_com = 0.0
    v_com = 0.0

    left_foot_x = -0.3
    right_foot_x = 0.3
    support_foot = "both"
    step_events = []

    # Arm parameters
    l_upper = 0.4
    l_forearm = 0.4
    shoulder_offset = 0.2  # Vertical offset from torso top

    # Target angles for arms
    target_theta1_L = 4*np.pi/3    # Left shoulder bends left
    target_theta2_L = 0    # Left elbow bends

    target_theta1_R = 0    # Right shoulder bends right
    target_theta2_R = np.pi/6     # Right elbow bends

    # Arm states
    theta1_L, omega1_L = np.pi, 0.0
    theta2_L, omega2_L = 0.0, 0.0
    theta1_R, omega1_R = 0.0, 0.0
    theta2_R, omega2_R = 0.0, 0.0

    torso_x, torso_y, com_positions, foot_positions = [], [], [], []
    left_arm_joints, right_arm_joints = [], []

    for i, t in enumerate(time):
        gravity_torque = m * g * l_leg * np.sin(theta)
        control_torque = compute_torque(theta, omega, Kp, Kd)
        torque = gravity_torque + control_torque

        theta, omega = update_stepping_dynamics(theta, omega, torque, m, l_leg, dt)

        if 1.5 < t < 1.52:
            v_com += 3.0

        foot_center = (left_foot_x + right_foot_x) / 2
        if abs(x_com - foot_center) > step_threshold and support_foot == "both":
            right_foot_x = x_com + 0.3
            support_foot = "left"
            step_events.append(i)

        if support_foot == "left":
            correction_strength = 1
            dead_zone = 0.05
            if abs(x_com - foot_center) > dead_zone:
                v_com -= correction_strength * (x_com - foot_center)
            v_com *= 0.95

        x_com += v_com * dt

        # Torso position
        x_torso = x_com + l_leg * np.sin(theta)
        y_torso = l_leg * np.cos(theta) + l_leg

        # Compute shoulder position along torso line
        total_body_length = l_leg + l_torso
        shoulder_ratio = (total_body_length - shoulder_offset) / total_body_length
        shoulder_x = x_com * (1 - shoulder_ratio) + x_torso * shoulder_ratio
        shoulder_y = (l_leg) * (1 - shoulder_ratio) + y_torso * shoulder_ratio  # hip y is always l_leg

        # Waving animation for right arm shoulder
        target_theta2_R = np.pi/6  + 0.3 * np.sin(2 * np.pi * 1.5 * t)

        # Arm control - Left
        torque1_L = compute_torque(theta1_L - target_theta1_L, omega1_L, 20, 3)
        torque2_L = compute_torque(theta2_L - target_theta2_L, omega2_L, 15, 2)

        theta1_L, omega1_L = update_stepping_dynamics(theta1_L, omega1_L, torque1_L, m, l_upper, dt)
        theta2_L, omega2_L = update_stepping_dynamics(theta2_L, omega2_L, torque2_L, m, l_forearm, dt)

        # Arm control - Right
        torque1_R = compute_torque(theta1_R - target_theta1_R, omega1_R, 20, 3)
        torque2_R = compute_torque(theta2_R - target_theta2_R, omega2_R, 15, 2)

        theta1_R, omega1_R = update_stepping_dynamics(theta1_R, omega1_R, torque1_R, m, l_upper, dt)
        theta2_R, omega2_R = update_stepping_dynamics(theta2_R, omega2_R, torque2_R, m, l_forearm, dt)

        # Arm positions (left)
        shoulder_x_L = shoulder_x
        elbow_x_L = shoulder_x_L + l_upper * np.cos(theta1_L)
        elbow_y_L = shoulder_y + l_upper * np.sin(theta1_L)
        hand_x_L = elbow_x_L + l_forearm * np.cos(theta1_L + theta2_L)
        hand_y_L = elbow_y_L + l_forearm * np.sin(theta1_L + theta2_L)

        # Arm positions (right)
        shoulder_x_R = shoulder_x
        elbow_x_R = shoulder_x_R + l_upper * np.cos(theta1_R)
        elbow_y_R = shoulder_y + l_upper * np.sin(theta1_R)
        hand_x_R = elbow_x_R + l_forearm * np.cos(theta1_R + theta2_R)
        hand_y_R = elbow_y_R + l_forearm * np.sin(theta1_R + theta2_R)

        torso_x.append(x_torso)
        torso_y.append(y_torso)
        com_positions.append(x_com)
        foot_positions.append((left_foot_x, right_foot_x))
        left_arm_joints.append([(shoulder_x_L, shoulder_y), (elbow_x_L, elbow_y_L), (hand_x_L, hand_y_L)])
        right_arm_joints.append([(shoulder_x_R, shoulder_y), (elbow_x_R, elbow_y_R), (hand_x_R, hand_y_R)])

    return time, torso_x, torso_y, foot_positions, com_positions, step_events, left_arm_joints, right_arm_joints

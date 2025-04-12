import numpy as np
from body_constants import g, l_leg, l_torso, l_arm, head_radius, m, b, Kp, Kd, dt, total_time, num_steps, step_threshold

def compute_torque(theta, omega, Kp, Kd):
    return -Kp * theta - Kd * omega

def update_stepping_dynamics(theta, omega, torque, m, l, dt):
    domega_dt = (torque - 0.2 * omega) / (m * l**2)
    omega_new = omega + domega_dt * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new

# FLUID: Initialize particle positions
def initialize_particles(num_particles, x_range=(-2, 2), y_range=(2.5, 3.0), seed=42):
    np.random.seed(seed)
    x_positions = np.random.uniform(*x_range, num_particles)
    y_positions = np.random.uniform(*y_range, num_particles)
    y_velocities = np.zeros(num_particles)
    return x_positions, y_positions, y_velocities

# FLUID: Update fluid positions with gravity and bounce
def update_fluid_dynamics(y_positions, y_velocities, g, elasticity, dt):
    y_velocities -= g * dt
    y_positions += y_velocities * dt
    collisions = y_positions <= 0
    y_positions[collisions] = 0
    y_velocities[collisions] *= -elasticity
    return y_positions, y_velocities


def simulate_reactive_stepping_body(num_particles=200, elasticity=0.2):
    time = np.linspace(0, total_time, num_steps)

    # Body initial conditions
    theta, omega, x_com, v_com = np.pi / 3, 0, 0.0, 0.0
    left_foot_x, right_foot_x = -0.3, 0.3
    support_foot, step_events = "both", []

    # Arm parameters
    l_upper, l_forearm = 0.4, 0.4
    shoulder_offset = 0.2

    target_theta1_L, target_theta2_L = 4 * np.pi / 3, 0
    target_theta1_R, target_theta2_R = 0, np.pi / 6

    theta1_L, omega1_L, theta2_L, omega2_L = np.pi, 0.0, 0.0, 0.0
    theta1_R, omega1_R, theta2_R, omega2_R = 0.0, 0.0, 0.0, 0.0

    torso_x, torso_y, com_positions, foot_positions = [], [], [], []
    left_arm_joints, right_arm_joints = [], []

    # FLUID: Initialize particles
    x_particles, y_particles, y_velocities = initialize_particles(num_particles)
    y_trajectory = []

    for i, t in enumerate(time):
        # Body control
        gravity_torque = m * g * l_leg * np.sin(theta)
        control_torque = compute_torque(theta, omega, Kp, Kd)
        theta, omega = update_stepping_dynamics(theta, omega, gravity_torque + control_torque, m, l_leg, dt)

        if 1.5 < t < 1.52:
            v_com += 3.0

        foot_center = (left_foot_x + right_foot_x) / 2
        if abs(x_com - foot_center) > step_threshold and support_foot == "both":
            right_foot_x = x_com + 0.3
            support_foot = "left"
            step_events.append(i)

        if support_foot == "left":
            if abs(x_com - foot_center) > 0.05:
                v_com -= 1 * (x_com - foot_center)
            v_com *= 0.95

        x_com += v_com * dt
        x_torso = x_com + l_leg * np.sin(theta)
        y_torso = l_leg * np.cos(theta) + l_leg

        # Shoulder location
        total_body_length = l_leg + l_torso
        shoulder_ratio = (total_body_length - shoulder_offset) / total_body_length
        shoulder_x = x_com * (1 - shoulder_ratio) + x_torso * shoulder_ratio
        shoulder_y = l_leg * (1 - shoulder_ratio) + y_torso * shoulder_ratio

        target_theta2_R = np.pi / 6 + 0.3 * np.sin(2 * np.pi * 1.5 * t)

        # Arms control
        torque1_L = compute_torque(theta1_L - target_theta1_L, omega1_L, 20, 3)
        torque2_L = compute_torque(theta2_L - target_theta2_L, omega2_L, 15, 2)
        theta1_L, omega1_L = update_stepping_dynamics(theta1_L, omega1_L, torque1_L, m, l_upper, dt)
        theta2_L, omega2_L = update_stepping_dynamics(theta2_L, omega2_L, torque2_L, m, l_forearm, dt)

        torque1_R = compute_torque(theta1_R - target_theta1_R, omega1_R, 20, 3)
        torque2_R = compute_torque(theta2_R - target_theta2_R, omega2_R, 15, 2)
        theta1_R, omega1_R = update_stepping_dynamics(theta1_R, omega1_R, torque1_R, m, l_upper, dt)
        theta2_R, omega2_R = update_stepping_dynamics(theta2_R, omega2_R, torque2_R, m, l_forearm, dt)

        # Arm positions
        elbow_x_L = shoulder_x + l_upper * np.cos(theta1_L)
        elbow_y_L = shoulder_y + l_upper * np.sin(theta1_L)
        hand_x_L = elbow_x_L + l_forearm * np.cos(theta1_L + theta2_L)
        hand_y_L = elbow_y_L + l_forearm * np.sin(theta1_L + theta2_L)

        elbow_x_R = shoulder_x + l_upper * np.cos(theta1_R)
        elbow_y_R = shoulder_y + l_upper * np.sin(theta1_R)
        hand_x_R = elbow_x_R + l_forearm * np.cos(theta1_R + theta2_R)
        hand_y_R = elbow_y_R + l_forearm * np.sin(theta1_R + theta2_R)

        torso_x.append(x_torso)
        torso_y.append(y_torso)
        com_positions.append(x_com)
        foot_positions.append((left_foot_x, right_foot_x))
        left_arm_joints.append([(shoulder_x, shoulder_y), (elbow_x_L, elbow_y_L), (hand_x_L, hand_y_L)])
        right_arm_joints.append([(shoulder_x, shoulder_y), (elbow_x_R, elbow_y_R), (hand_x_R, hand_y_R)])

        # FLUID update
        y_particles, y_velocities = update_fluid_dynamics(y_particles, y_velocities, g, elasticity, dt)
        y_trajectory.append(y_particles.copy())

    print("Running integrated_simulation.py with BODY + FLUIDS")

    return time, torso_x, torso_y, foot_positions, com_positions, step_events, left_arm_joints, right_arm_joints, x_particles, y_trajectory

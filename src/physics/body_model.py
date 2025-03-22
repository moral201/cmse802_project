import numpy as np

def simulate_balance():
    """
    Simulates a two-legged human-like figure maintaining balance
    under gravitational forces using a PID-controlled inverted pendulum.
    
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
    
    # Constants
    g = 9.81  # Gravity (m/s^2)
    l_leg = 1.0  # Leg length (m)
    l_torso = 1.2  # Torso length (m)
    l_arm = 1.0  # Arm length (m)
    head_radius = 0.3  # Head size (m)

    m = 1.0  # Mass of the torso (kg)
    b = 0.5  # Damping coefficient (joint friction)
    Kp = 50  # Proportional gain (PID)
    Kd = 5  # Derivative gain (PID)
    dt = 0.02  # Time step
    total_time = 5  # Total simulation time (s)
    num_steps = int(total_time / dt)

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
        torque = -Kp * theta - Kd * omega
        domega_dt = (-m * g * l_leg * np.sin(theta) - b * omega + torque) / (m * l_leg**2)
        omega += domega_dt * dt
        theta += omega * dt
        
        # Convert to Cartesian coordinates
        x_torso = l_leg * np.sin(theta)
        y_torso = l_leg * np.cos(theta) + l_leg  # Torso position

        # Shoulder position (top of torso)
        x_shoulder = x_torso
        y_shoulder = y_torso - 0.2 #+ (l_torso / 2)

        # Arms (from shoulders outward)
        x_arm = [x_shoulder - l_arm / 2, x_shoulder + l_arm / 2]
        y_arm = [y_shoulder , y_shoulder ]

        # Head (above shoulders)
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

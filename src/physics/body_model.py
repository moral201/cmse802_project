"""
body_model.py

This module implements a physics-based 2D human body model using an inverted pendulum approach 
to simulate biomechanical stability under gravitational forces. The model applies a PID controller 
to adjust balance in response to external forces.

Author: [Your Name]
Date: [March 2025]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def simulate_balance(Kp=50, Kd=5, total_time=5, dt=0.02):
    """
    Simulates the balance of a two-legged standing figure using an inverted pendulum model.
    
    Parameters:
        Kp (float): Proportional gain for the PID controller.
        Kd (float): Derivative gain for the PID controller.
        total_time (float): Total simulation time in seconds.
        dt (float): Time step for the simulation.
    
    Returns:
        tuple: (time, theta_vals, x_torso_vals, y_torso_vals)
    """
    g = 9.81  # gravity (m/s^2)
    l_leg = 1.0  # length of each leg (m)
    m = 1.0  # mass of the torso (kg)
    b = 0.5  # damping coefficient (joint friction)
    
    # Initial conditions
    theta = np.pi / 8  # Small initial tilt (radians)
    omega = 0  # Initial angular velocity
    num_steps = int(total_time / dt)
    time = np.linspace(0, total_time, num_steps)

    # Arrays to store results
    theta_vals = []
    x_torso_vals = []
    y_torso_vals = []

    # Simulate the balance dynamics
    for t in time:
        torque = -Kp * theta - Kd * omega  # Control force (stabilizing torque)
        domega_dt = (-m * g * l_leg * np.sin(theta) - b * omega + torque) / (m * l_leg**2)
        omega += domega_dt * dt
        theta += omega * dt
        
        # Convert to Cartesian coordinates for the torso position
        x_torso = l_leg * np.sin(theta)
        y_torso = l_leg * np.cos(theta) + l_leg  # Raise torso above legs

        # Store values
        theta_vals.append(theta)
        x_torso_vals.append(x_torso)
        y_torso_vals.append(y_torso)

    return time, theta_vals, x_torso_vals, y_torso_vals

if __name__ == "__main__":
    # Run the simulation
    time, theta_vals, x_torso_vals, y_torso_vals = simulate_balance()
    
    # Print a sample output to check correctness
    print(f"Final angle: {theta_vals[-1]:.4f} radians")


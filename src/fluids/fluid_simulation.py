# fluid_simulation.py
"""
Module: fluid_simulation.py

A basic 2D particle-based fluid simulation using gravity and simple bounce collisions
with the ground. This serves as a first step toward Smoothed Particle Hydrodynamics (SPH).

Author: Gerardo Morales
Date: March 2025
"""

import numpy as np

def simulate_fluid(num_particles=100, dt=0.02, total_time=2.0, g=9.81, elasticity=0.2):
    """
    Simulates falling particles under gravity with bouncing ground collision.

    Args:
        num_particles (int): Number of fluid particles.
        dt (float): Time step.
        total_time (float): Total simulation duration in seconds.
        g (float): Gravitational acceleration.
        elasticity (float): Bounciness factor on collision with ground.

    Returns:
        x_positions (np.ndarray): Constant x-coordinates of particles.
        y_trajectory (list of np.ndarray): Y-coordinates of all particles over time.
    """
    steps = int(total_time / dt)
    
    # Initialize particle positions and velocities
    np.random.seed(42)
    x_positions = np.random.uniform(-0.5, 0.5, num_particles)
    y_positions = np.random.uniform(1.5, 2.0, num_particles)
    y_velocities = np.zeros(num_particles)

    # Storage for animation frames
    y_trajectory = []

    for _ in range(steps):
        y_velocities -= g * dt
        y_positions += y_velocities * dt

        # Bounce on ground
        for i in range(num_particles):
            if y_positions[i] <= 0:
                y_positions[i] = 0
                y_velocities[i] = -y_velocities[i] * elasticity

        y_trajectory.append(y_positions.copy())

    return x_positions, y_trajectory, dt

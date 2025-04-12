# fluid_simulation.py
"""
Module: fluid_simulation.py

A basic 2D particle-based fluid simulation using gravity and simple bounce collisions
with the ground. This serves as a first step toward Smoothed Particle Hydrodynamics (SPH).

Author: Gerardo Morales
Date: March 2025
"""

import numpy as np

def initialize_particles(num_particles, x_range=(-0.5, 0.5), y_range=(1.5, 2.0), seed=42):
    """Initialize positions and velocities of particles."""
    np.random.seed(seed)
    x_positions = np.random.uniform(*x_range, num_particles)
    y_positions = np.random.uniform(*y_range, num_particles)
    y_velocities = np.zeros(num_particles)
    return x_positions, y_positions, y_velocities

def update_fluid_dynamics(y_positions, y_velocities, g, elasticity, dt):
    """Update fluid dynamics for a single timestep."""
    y_velocities -= g * dt
    y_positions += y_velocities * dt

    # Ground collision and bounce
    collisions = y_positions <= 0
    y_positions[collisions] = 0
    y_velocities[collisions] *= -elasticity

    return y_positions, y_velocities

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
        dt (float): Simulation time step.
    """
    steps = int(total_time / dt)
    x_positions, y_positions, y_velocities = initialize_particles(num_particles)

    # Storage for animation frames
    y_trajectory = []

    for _ in range(steps):
        y_positions, y_velocities = update_fluid_dynamics(y_positions, y_velocities, g, elasticity, dt)
        y_trajectory.append(y_positions.copy())

    return x_positions, y_trajectory, dt
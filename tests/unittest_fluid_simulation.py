# unittest_fluid_simulation.py
"""
Module: unittest_fluid_simulation.py

This module test the implemented model fro fluid simulation. 

Author: Gerardo Morales
Date: March 2025
""" 

import sys
import os

# Adjust the path to match your system:
sys.path.append(os.path.abspath("C:\\Users\\Admin\\Documents\\Michigan MSU\\Spring 2025\\CMSE 802\\Project\\cmse802_project\\src\\fluids"))

import unittest
import numpy as np
from fluid_simulation import initialize_particles, update_fluid_dynamics

class TestFluidSimulation(unittest.TestCase):

    def setUp(self):
        self.num_particles = 10
        self.dt = 0.02
        self.g = 9.81
        self.elasticity = 0.2
        self.x_positions, self.y_positions, self.y_velocities = initialize_particles(self.num_particles)

    def test_initialize_particles(self):
        self.assertEqual(len(self.x_positions), self.num_particles)
        self.assertEqual(len(self.y_positions), self.num_particles)
        self.assertTrue(np.all(self.y_velocities == 0))

    def test_update_fluid_dynamics_single_step(self):
        initial_y_positions = self.y_positions.copy()
        initial_y_velocities = self.y_velocities.copy()

        y_positions_updated, y_velocities_updated = update_fluid_dynamics(
            self.y_positions, self.y_velocities, self.g, self.elasticity, self.dt
        )

        expected_y_velocities = initial_y_velocities - self.g * self.dt
        expected_y_positions = initial_y_positions + expected_y_velocities * self.dt

        collisions = expected_y_positions <= 0
        expected_y_positions[collisions] = 0
        expected_y_velocities[collisions] *= -self.elasticity

        np.testing.assert_array_almost_equal(y_positions_updated, expected_y_positions)
        np.testing.assert_array_almost_equal(y_velocities_updated, expected_y_velocities)

    def test_ground_collision(self):
        # Manually force collision with ground
        self.y_positions.fill(-0.1)
        self.y_velocities.fill(-1.0)

        # Update dynamics
        y_positions_updated, y_velocities_updated = update_fluid_dynamics(
            self.y_positions, self.y_velocities, self.g, self.elasticity, self.dt
        )

        # Gravity affects velocities before collision:
        expected_velocities = (-1.0 - self.g * self.dt) * -self.elasticity

        np.testing.assert_array_almost_equal(y_positions_updated, np.zeros(self.num_particles))
        np.testing.assert_array_almost_equal(
            y_velocities_updated,
            np.full(self.num_particles, expected_velocities),
            decimal=5
        )


if __name__ == '__main__':
    unittest.main()

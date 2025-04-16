# unittest_physics_utils.py
"""
Module: unittest_physics_utils.py

This module tests the utility functions implemented in physics_utils.py.
It verifies correct torque computation, checks angle/angular velocity updates,
ensures correct particle initialization, and validates basic particle reflection.

Author: Gerardo Morales
Date: April 2025
"""

import os
import sys
import unittest

import numpy as np

# Adjust path to import from src/physics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from physics_utils import (
    compute_torque,
    initialize_particles,
    reflect_particles_from_segment,
    update_stepping_dynamics,
)


class TestPhysicsUtils(unittest.TestCase):
    """
    Unit test class for physics utility functions.
    """

    def setUp(self):
        """
        Initialize default test parameters.
        """
        self.Kp = 50
        self.Kd = 5
        self.theta = np.pi / 8
        self.omega = 0.0
        self.m = 1.0
        self.l_leg = 1.0
        self.dt = 0.01

    def test_compute_torque(self):
        """
        Test if compute_torque returns correct PD controller output.
        """
        expected_torque = -self.Kp * self.theta - self.Kd * self.omega
        actual_torque = compute_torque(self.theta, self.omega, self.Kp, self.Kd)
        self.assertAlmostEqual(expected_torque, actual_torque, places=5)

    def test_update_stepping_dynamics(self):
        """
        Test if update_stepping_dynamics correctly updates theta and omega.
        """
        torque = compute_torque(self.theta, self.omega, self.Kp, self.Kd)
        theta_new, omega_new = update_stepping_dynamics(
            self.theta, self.omega, torque, self.m, self.l_leg, self.dt
        )

        domega_dt_expected = (torque - 0.2 * self.omega) / (self.m * self.l_leg**2)
        omega_expected = self.omega + domega_dt_expected * self.dt
        theta_expected = self.theta + omega_expected * self.dt

        self.assertAlmostEqual(omega_new, omega_expected, places=5)
        self.assertAlmostEqual(theta_new, theta_expected, places=5)

    def test_initialize_particles(self):
        """
        Test if initialize_particles generates correct particle array sizes.
        """
        num_particles = 100
        x, y, xv, yv = initialize_particles(num_particles)
        self.assertEqual(len(x), num_particles)
        self.assertEqual(len(y), num_particles)
        self.assertEqual(len(xv), num_particles)
        self.assertEqual(len(yv), num_particles)

    def test_reflect_particles_from_segment_no_collision(self):
        """
        Test reflect_particles_from_segment with no collision (particles far away).
        Expect unchanged velocities.
        """
        xp, yp = np.array([10.0]), np.array([10.0])  # Far from segment
        xv, yv = np.array([1.0]), np.array([1.0])
        x1, y1, x2, y2 = 0, 0, 1, 0  # Segment along x-axis

        xv_new, yv_new = reflect_particles_from_segment(
            xp, yp, xv, yv, x1, y1, x2, y2, thickness=0.1, elasticity=0.2
        )

        np.testing.assert_array_equal(xv, xv_new)
        np.testing.assert_array_equal(yv, yv_new)

    def test_vertical_reflection_from_horizontal_segment(self):
        # Setup: one particle right above a horizontal segment
        xp = np.array([1.0])
        yp = np.array([1.05])
        xv = np.array([0.0])
        yv = np.array([-1.0])  # Falling straight down

        # Segment is horizontal from (0,1) to (2,1)
        x1, y1 = 0.0, 1.0
        x2, y2 = 2.0, 1.0
        thickness = 0.1
        elasticity = 1.0  # Perfectly elastic bounce

        xv_new, yv_new = reflect_particles_from_segment(
            xp, yp, xv, yv, x1, y1, x2, y2, thickness, elasticity
        )

        self.assertAlmostEqual(xv_new[0], 0.0, places=5)
        self.assertAlmostEqual(yv_new[0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

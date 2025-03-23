# unittest_cart_model_PD.py
"""
Module: unittest_cart_model_PD.py

This module test the implemented models. It verifies the computed force is correct, confirms that angles update correctly after one simulation step and check angular velocities.

Author: Gerardo Morales
Date: March 2025
"""   

import sys
import os

# Adjust the path to match your system:
sys.path.append(os.path.abspath("C:\\Users\\Admin\\Documents\\Michigan MSU\\Spring 2025\\CMSE 802\\Project\\cmse802_project\\src\\physics"))

import unittest
import numpy as np
from cart_pole_body import compute_force, update_cart_dynamics

class TestCartModel(unittest.TestCase):

    def setUp(self):
        # Common parameters for testing
        self.Kp = 50
        self.Kd = 5
        self.Kx = 10
        self.theta = np.pi / 6  # 30 degrees initial angle
        self.omega = 0.0
        self.x_base = 0.1  # Small initial displacement
        self.m = 1.0
        self.g = 9.81
        self.l_leg = 1.0
        self.b = 0.1
        self.dt = 0.01

    def test_compute_force(self):
        expected_force = -self.Kp * self.theta - self.Kd * self.omega - self.Kx * self.x_base
        actual_force = compute_force(self.theta, self.omega, self.x_base, self.Kp, self.Kd, self.Kx)
        self.assertAlmostEqual(expected_force, actual_force, places=5)

    def test_update_cart_dynamics(self):
        v_base = 0.0  # Initial base velocity
        force = compute_force(self.theta, self.omega, self.x_base, self.Kp, self.Kd, self.Kx)
        theta_new, omega_new, x_base_new, v_base_new = update_cart_dynamics(
            self.theta, self.omega, self.x_base, v_base,
            force, self.m, self.g, self.l_leg, self.b, self.dt
        )

        # Manually calculate expected values
        a_base_expected = force / self.m
        v_base_expected = v_base + a_base_expected * self.dt
        x_base_expected = self.x_base + v_base_expected * self.dt

        torque_expected = -self.m * self.g * self.l_leg * np.sin(self.theta) + force * self.l_leg * np.cos(self.theta)
        domega_dt_expected = (torque_expected - self.b * self.omega) / (self.m * self.l_leg**2)
        omega_expected = self.omega + domega_dt_expected * self.dt
        theta_expected = self.theta + omega_expected * self.dt

        # Assertions
        self.assertAlmostEqual(theta_new, theta_expected, places=5)
        self.assertAlmostEqual(omega_new, omega_expected, places=5)
        self.assertAlmostEqual(x_base_new, x_base_expected, places=5)
        self.assertAlmostEqual(v_base_new, v_base_expected, places=5)

    def test_edge_case_zero_initial_conditions(self):
        # Edge case: zero angle, zero angular velocity, zero displacement
        theta_zero = 0.0
        omega_zero = 0.0
        x_base_zero = 0.0

        force = compute_force(theta_zero, omega_zero, x_base_zero, self.Kp, self.Kd, self.Kx)
        self.assertEqual(force, 0.0, "Force should be zero for zero initial conditions.")

if __name__ == '__main__':
    unittest.main()

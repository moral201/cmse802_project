# unittest_body_model_PD.py
"""
Module: unittest_body_model_PD.py

This module test the implemented models. It verifies the computed torque is correct, confirms that angles update correctly after one simulation step and check angular velocities.

Author: Gerardo Morales
Date: March 2025
"""   

import sys
import os
sys.path.append(os.path.abspath("C:\\Users\\Admin\\Documents\\Michigan MSU\\Spring 2025\\CMSE 802\\Project\\cmse802_project\\src\\physics"))  # Add parent directory to path

import unittest
import numpy as np
from body_model_PD import compute_torque, update_dynamics

class TestBodyModelPD(unittest.TestCase):

    def setUp(self):
        # Define common parameters for testing
        self.Kp = 50
        self.Kd = 5
        self.theta = np.pi / 6  # 30 degrees
        self.omega = 0.0
        self.m = 1.0
        self.g = 9.81
        self.l_leg = 1.0
        self.b = 0.1
        self.dt = 0.01

    def test_compute_torque(self):
        # Expected torque based on PD control
        expected_torque = -self.Kp * self.theta - self.Kd * self.omega
        actual_torque = compute_torque(self.theta, self.omega, self.Kp, self.Kd)
        self.assertAlmostEqual(expected_torque, actual_torque, places=5)

    def test_update_dynamics(self):
        # Testing a single-step update
        torque = compute_torque(self.theta, self.omega, self.Kp, self.Kd)
        theta_new, omega_new = update_dynamics(self.theta, self.omega, torque,
                                               self.m, self.g, self.l_leg, self.b, self.dt)
        
        # Manually compute expected values
        domega_dt = (-self.m * self.g * self.l_leg * np.sin(self.theta) - self.b * self.omega + torque) / (self.m * self.l_leg**2)
        omega_expected = self.omega + domega_dt * self.dt
        theta_expected = self.theta + omega_expected * self.dt

        self.assertAlmostEqual(omega_new, omega_expected, places=5)
        self.assertAlmostEqual(theta_new, theta_expected, places=5)

    def test_edge_case_zero_angle_velocity(self):
        # Edge case: theta and omega initially zero
        theta_zero = 0.0
        omega_zero = 0.0
        torque = compute_torque(theta_zero, omega_zero, self.Kp, self.Kd)
        
        self.assertEqual(torque, 0.0, "Torque should be zero when theta and omega are zero.")

if __name__ == '__main__':
    unittest.main()

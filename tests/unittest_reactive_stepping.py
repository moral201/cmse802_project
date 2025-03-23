# unittest_reactive_stepping.py
"""
Module: unittest_reactive_stepping.py

This module test the implemented models. It verifies the computed torque is correct, confirms that angles update correctly after one simulation step and check angular velocities.

Author: Gerardo Morales
Date: March 2025
"""   
import sys
import os

# Adjust the path correctly for your system
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "physics")))

import unittest
import numpy as np
from reactive_stepping_body import compute_torque, update_stepping_dynamics

class TestReactiveSteppingModel(unittest.TestCase):

    def setUp(self):
        self.Kp = 50
        self.Kd = 5
        self.theta = np.pi / 8
        self.omega = 0.0
        self.m = 1.0
        self.l_leg = 1.0
        self.dt = 0.01

    def test_compute_torque(self):
        expected_torque = -self.Kp * self.theta - self.Kd * self.omega
        actual_torque = compute_torque(self.theta, self.omega, self.Kp, self.Kd)
        self.assertAlmostEqual(expected_torque, actual_torque, places=5)

    def test_update_stepping_dynamics(self):
        torque = compute_torque(self.theta, self.omega, self.Kp, self.Kd)
        theta_new, omega_new = update_stepping_dynamics(self.theta, self.omega, torque,
                                                        self.m, self.l_leg, self.dt)

        domega_dt_expected = (torque - 0.2 * self.omega) / (self.m * self.l_leg**2)
        omega_expected = self.omega + domega_dt_expected * self.dt
        theta_expected = self.theta + omega_expected * self.dt

        self.assertAlmostEqual(omega_new, omega_expected, places=5)
        self.assertAlmostEqual(theta_new, theta_expected, places=5)

    def test_zero_conditions(self):
        torque = compute_torque(0, 0, self.Kp, self.Kd)
        self.assertEqual(torque, 0.0, "Torque should be zero for zero initial conditions.")

if __name__ == '__main__':
    unittest.main()

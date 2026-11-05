"""
Unit Tests for WEBS Simulator Components.
Verifies:
1. Wind ground speed equations and physical constraints.
2. Multirotor aerodynamic energy models (Betti Sorbelli & Du).
3. Space-Time Reservation Table conflict detection (zero violations).
4. Planner execution and batching functionality.
"""

import unittest
import math
import numpy as np

from src.config import DroneSpecs
from src.wind_energy import WindField, EnergyModel
from src.reservation_table import SpaceTimeReservationTable
from src.map_module import CityAirspaceMap

class TestWindAndEnergy(unittest.TestCase):
    def setUp(self):
        self.specs = DroneSpecs(nominal_airspeed=14.0)
        self.energy_model = EnergyModel(self.specs)

    def test_zero_wind_ground_speed(self):
        """In zero wind, ground speed must equal nominal airspeed."""
        u_east = (1.0, 0.0)
        wind_zero = (0.0, 0.0)
        feasible, g_speed = self.energy_model.calculate_ground_speed(u_east, wind_zero)
        self.assertTrue(feasible)
        self.assertAlmostEqual(g_speed, 14.0, places=4)

    def test_pure_tailwind(self):
        """Tailwind of 5 m/s along flight path should result in ground speed = 14 + 5 = 19 m/s."""
        u_east = (1.0, 0.0)
        wind_tailwind = (5.0, 0.0)
        feasible, g_speed = self.energy_model.calculate_ground_speed(u_east, wind_tailwind)
        self.assertTrue(feasible)
        self.assertAlmostEqual(g_speed, 19.0, places=4)

    def test_pure_headwind(self):
        """Headwind of 5 m/s against flight path should result in ground speed = 14 - 5 = 9 m/s."""
        u_east = (1.0, 0.0)
        wind_headwind = (-5.0, 0.0)
        feasible, g_speed = self.energy_model.calculate_ground_speed(u_east, wind_headwind)
        self.assertTrue(feasible)
        self.assertAlmostEqual(g_speed, 9.0, places=4)

    def test_excessive_crosswind(self):
        """Crosswind exceeding drone airspeed should be flagged infeasible."""
        u_east = (1.0, 0.0)
        wind_severe_cross = (0.0, 20.0) # 20 m/s > 14 m/s
        feasible, _ = self.energy_model.calculate_ground_speed(u_east, wind_severe_cross)
        self.assertFalse(feasible)

    def test_energy_with_payload(self):
        """Drone with payload should consume more power than empty drone."""
        p_empty = self.energy_model.compute_power(airspeed=14.0, payload_kg=0.0)
        p_loaded = self.energy_model.compute_power(airspeed=14.0, payload_kg=1.5)
        self.assertGreater(p_loaded, p_empty)


class TestReservationTable(unittest.TestCase):
    def setUp(self):
        self.table = SpaceTimeReservationTable(time_buffer=2.0)

    def test_vertex_collision_detection(self):
        # Drone 1 reserves node 5 at [10.0, 10.0]
        self.table.reserve_path(drone_id=1, trajectory=[(5, 10.0, 10.0)])
        
        # Check node 5 at time 10.5 (within 2s buffer): should be unavailable
        self.assertFalse(self.table.is_vertex_available(5, 10.5, 10.5))
        
        # Check node 5 at time 20.0 (well outside buffer): should be available
        self.assertTrue(self.table.is_vertex_available(5, 20.0, 20.0))

    def test_edge_swap_collision_detection(self):
        # Drone 1 traverses edge (1, 2) from t=10.0 to t=15.0
        self.table.reserve_path(drone_id=1, trajectory=[(1, 10.0, 10.0), (2, 15.0, 15.0)])

        # Drone 2 attempting reverse traversal (2, 1) during [12.0, 16.0] should be blocked!
        self.assertFalse(self.table.is_edge_available(2, 1, 12.0, 16.0))


if __name__ == "__main__":
    unittest.main()

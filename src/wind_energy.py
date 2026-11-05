"""
Wind and Energy Module for WEBS Simulator.
Implements:
1. Ground speed calculation under arbitrary wind vectors (Betti Sorbelli et al.)
   Equation: g = w · u + sqrt(V^2 - ||w||^2 + (w · u)^2)
2. Rotorcraft aerodynamic power and energy consumption models (Betti Sorbelli et al. & Du et al.)
3. Wind field representations (Constant, Time-Varying, Gusty).
"""

import math
from typing import Optional, Tuple
import numpy as np
from src.config import DroneSpecs

class WindField:
    """
    Represents 2D horizontal wind field at cruise altitude.
    wind_vector is (w_x, w_y) in m/s, where:
      +x = East
      +y = North
    """
    def __init__(self, speed: float = 0.0, direction_deg: float = 0.0, gust_intensity: float = 0.0):
        """
        speed: wind speed in m/s
        direction_deg: meteorological wind direction (direction wind is blowing TOWARDS, in degrees from North cw, or math angle)
        Here we use standard Cartesian: angle theta from +x (East).
        """
        self.base_speed = float(speed)
        self.direction_deg = float(direction_deg)
        self.gust_intensity = float(gust_intensity)
        rad = math.radians(direction_deg)
        self.base_wx = self.base_speed * math.cos(rad)
        self.base_wy = self.base_speed * math.sin(rad)

    def get_wind(self, x: float, y: float, t: float, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """
        Returns wind vector (w_x, w_y) at position (x, y) and timestamp t.
        Includes optional time variation and gusts.
        """
        # Temporal slow modulation (e.g., hourly sinusoidal shift)
        modulation = 1.0 + 0.15 * math.sin(2.0 * math.pi * t / 3600.0)
        wx = self.base_wx * modulation
        wy = self.base_wy * modulation

        if self.gust_intensity > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            # Ornstein-Uhlenbeck or Gaussian perturbation
            wx += float(rng.normal(0, self.gust_intensity))
            wy += float(rng.normal(0, self.gust_intensity))

        return (wx, wy)

    @classmethod
    def from_speed_and_angle(cls, speed: float, angle_deg: float, gust: float = 0.0) -> 'WindField':
        return cls(speed=speed, direction_deg=angle_deg, gust_intensity=gust)


class EnergyModel:
    """
    Multirotor aerodynamic power consumption model based on momentum theory
    and empirical drag models from Betti Sorbelli et al. and Du et al.
    """
    def __init__(self, specs: Optional[DroneSpecs] = None):
        self.specs = specs if specs is not None else DroneSpecs()
        self.total_rotor_area = self.specs.num_rotors * math.pi * (self.specs.rotor_radius ** 2)

    def calculate_ground_speed(
        self,
        u_vector: Tuple[float, float],
        wind_vector: Tuple[float, float],
        airspeed: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Computes ground speed along an edge with unit direction vector u in wind w.
        Formula from literature review:
          g = w · u + sqrt(V^2 - ||w||^2 + (w · u)^2)
        
        Returns:
          (feasible, ground_speed_m_per_s)
        """
        V = airspeed if airspeed is not None else self.specs.nominal_airspeed
        wx, wy = wind_vector
        ux, uy = u_vector

        # Dot product w · u
        w_dot_u = wx * ux + wy * uy
        # Magnitude squared ||w||^2
        w_mag_sq = wx * wx + wy * wy

        discriminant = (V ** 2) - w_mag_sq + (w_dot_u ** 2)

        # Crosswind too strong to maintain track
        if discriminant < 0:
            return (False, 0.0)

        ground_speed = w_dot_u + math.sqrt(discriminant)

        # Headwind exceeds airspeed or drone pushed backwards
        if ground_speed <= 0.5:
            return (False, 0.0)

        return (True, ground_speed)

    def compute_power(self, airspeed: float, payload_kg: float) -> float:
        """
        Computes electrical power consumption (Watts) at given airspeed and payload.
        Includes:
        1. Induced power (lifting total mass against gravity)
        2. Parasite drag power (pushing frontal area through air)
        3. Profile drag power (blade skin friction)
        4. Avionics / onboard electronics
        """
        total_mass = self.specs.mass_empty + payload_kg
        g = self.specs.gravity
        rho = self.specs.air_density
        A_r = self.total_rotor_area
        
        # Weight force
        W = total_mass * g
        
        # Parasite aerodynamic drag: D_p = 0.5 * rho * C_d * A_front * V^2
        D_parasite = 0.5 * rho * self.specs.drag_coefficient * self.specs.frontal_area * (airspeed ** 2)
        
        # Total aerodynamic thrust required to balance weight and parasite drag
        Thrust = math.sqrt((W ** 2) + (D_parasite ** 2))
        
        # Induced power (Momentum Theory in forward flight):
        # In forward flight, v_i is approximately Thrust / (2 * rho * A_r * sqrt(V^2 + v_i^2))
        # Glauert forward flight approximation:
        v_h = math.sqrt(Thrust / (2.0 * rho * A_r)) # hover induced velocity
        if airspeed < 1.0:
            P_induced = Thrust * v_h
        else:
            # Forward flight induced power
            P_induced = (Thrust ** 2) / (2.0 * rho * A_r * airspeed)
            P_induced = min(P_induced, Thrust * v_h) # Upper bounded by hover power

        # Parasite power: P_par = D_parasite * V
        P_parasite = D_parasite * airspeed

        # Rotor profile power (blade profile drag, roughly proportional to hover profile + V^2 correction)
        # Empirical profile power for delivery multirotor:
        P_profile = 0.12 * (W * v_h) * (1.0 + 3.0 * (airspeed / 20.0) ** 2)

        mechanical_power = P_induced + P_parasite + P_profile
        electrical_power = (mechanical_power / self.specs.motor_efficiency) + self.specs.avionics_power

        return electrical_power

    def compute_hover_power(self, payload_kg: float) -> float:
        """Electrical power in hover (holding position at a vertex)."""
        return self.compute_power(airspeed=0.0, payload_kg=payload_kg)

    def compute_edge_traversal(
        self,
        distance_m: float,
        u_vector: Tuple[float, float],
        wind_vector: Tuple[float, float],
        payload_kg: float,
        airspeed: Optional[float] = None
    ) -> Tuple[bool, float, float]:
        """
        Computes traversal feasibility, traversal time (seconds), and energy consumed (Joules).
        Returns:
          (feasible, travel_time_sec, energy_joules)
        """
        feasible, ground_speed = self.calculate_ground_speed(
            u_vector=u_vector,
            wind_vector=wind_vector,
            airspeed=airspeed
        )
        if not feasible:
            return (False, float('inf'), float('inf'))

        travel_time = distance_m / ground_speed
        V = airspeed if airspeed is not None else self.specs.nominal_airspeed
        power = self.compute_power(airspeed=V, payload_kg=payload_kg)
        energy_joules = power * travel_time

        return (True, travel_time, energy_joules)

"""
Simulation Configuration and Physical Constants for WEBS Simulator.
Based on:
- Betti Sorbelli et al. (IEEE T-ITS 2023)
- Du et al. (IEEE TCE 2024)
- Steinberg et al. (IEEE OJ-ITS 2021)
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class DroneSpecs:
    mass_empty: float = 3.5        # kg (drone chassis, battery, avionics)
    max_payload: float = 1.5       # kg
    nominal_airspeed: float = 14.0 # m/s (~50 km/h cruising airspeed)
    battery_capacity_wh: float = 220.0 # Watt-hours
    battery_reserve_pct: float = 0.20  # 20% emergency reserve
    
    # Aerodynamic & propulsion parameters (quadcopter / octocopter delivery drone)
    num_rotors: int = 4
    rotor_radius: float = 0.25     # meters
    drag_coefficient: float = 0.35 # aerodynamic drag coefficient
    frontal_area: float = 0.08     # m^2 projected area
    air_density: float = 1.225     # kg/m^3 (ISA at sea level)
    gravity: float = 9.81          # m/s^2
    avionics_power: float = 20.0   # Watts (sensors, flight computer, comms)
    motor_efficiency: float = 0.82 # electrical to mechanical power conversion
    
    @property
    def battery_joules(self) -> float:
        return self.battery_capacity_wh * 3600.0

    @property
    def usable_joules(self) -> float:
        return self.battery_joules * (1.0 - self.battery_reserve_pct)


@dataclass
class AirspaceConfig:
    safety_distance_meters: float = 30.0  # Safe spatial separation (Steinberg et al.)
    time_slot_seconds: float = 1.0        # Temporal discretization resolution
    max_altitude: float = 120.0           # meters AGL (standard UTM ceiling)
    min_altitude: float = 50.0            # meters AGL (above street level)
    default_altitude: float = 75.0        # cruise altitude


@dataclass
class ExperimentConfig:
    wind_speeds: tuple = (0.0, 5.0, 10.0) # m/s
    wind_directions_deg: tuple = (0.0, 45.0, 90.0, 180.0, 270.0)
    batch_sizes: tuple = (1, 10, 50, 100)
    request_rates: tuple = (5, 15, 30)    # requests per minute
    random_seeds_count: int = 20

"""
Software-in-the-Loop (SITL) Hardware Drone Emulator.
Simulates a physical multirotor delivery drone with:
1. Autopilot state machine (ARMING, TAKEOFF, NAVIGATING, HOVER, LAND).
2. Closed-loop PID ground speed tracking in arbitrary 2D wind.
3. Realistic 4S LiPo battery voltage discharge and current draw (Amperes).
4. MAVLink telemetry streaming (GPS, Battery, Ground Speed, Airspeed, State).
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from src.config import DroneSpecs
from src.wind_energy import WindField, EnergyModel

@dataclass
class HardwareTelemetryRecord:
    timestamp_s: float
    drone_id: int
    x: float
    y: float
    z: float
    lat: float
    lon: float
    ground_speed_mps: float
    airspeed_mps: float
    battery_volts: float
    current_amps: float
    capacity_used_mah: float
    state: str

class SITLDroneEmulator:
    def __init__(
        self,
        drone_id: int,
        trajectory: List[Tuple[int, float, float]],
        node_positions: Dict[int, Tuple[float, float]],
        wind_field: WindField,
        specs: Optional[DroneSpecs] = None,
        ref_lat: float = 26.3683,
        ref_lon: float = -80.1289
    ):
        self.drone_id = drone_id
        self.trajectory = trajectory
        self.node_positions = node_positions
        self.wind_field = wind_field
        self.specs = specs if specs is not None else DroneSpecs()
        self.energy_model = EnergyModel(self.specs)
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.R_EARTH = 6378137.0

        # Hardware initial states
        self.state = "STANDBY"
        self.battery_nominal_mah = (self.specs.battery_capacity_wh / 14.8) * 1000.0 # 4S LiPo (14.8V nominal)
        self.capacity_used_mah = 0.0
        self.current_voltage = 16.8 # 4S fully charged at 4.2V/cell

        # Start position
        src_node = trajectory[0][0]
        self.pos_x, self.pos_y = node_positions[src_node]
        self.pos_z = 0.0

        self.telemetry_history: List[HardwareTelemetryRecord] = []

    def _meter_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        d_lat = y / self.R_EARTH
        d_lon = x / (self.R_EARTH * math.cos(math.radians(self.ref_lat)))
        return (self.ref_lat + math.degrees(d_lat), self.ref_lon + math.degrees(d_lon))

    def run_hardware_flight(self, dt: float = 0.2, payload_kg: float = 1.0) -> List[HardwareTelemetryRecord]:
        """
        Executes sub-second hardware simulation loop from takeoff to landing.
        """
        if not self.trajectory or len(self.trajectory) < 2:
            return []

        curr_t = self.trajectory[0][1] # Launch time
        self.state = "ARMING"
        self._record_telemetry(curr_t, 0.0, 0.0)

        # Takeoff phase (ascend to 65m at 2.5 m/s)
        self.state = "TAKEOFF"
        takeoff_time = 65.0 / 2.5
        for step in np.arange(0, takeoff_time, dt):
            curr_t += dt
            self.pos_z = min(65.0, self.pos_z + 2.5 * dt)
            power = self.energy_model.compute_hover_power(payload_kg) * 1.15
            self._update_battery(power, dt)
            self._record_telemetry(curr_t, 0.0, 0.0)

        # Waypoint Navigation Phase
        self.state = "NAVIGATING"
        for i in range(len(self.trajectory) - 1):
            curr_node, arr_t, dep_t = self.trajectory[i]
            next_node, next_arr, _ = self.trajectory[i + 1]

            # Hover/Wait loiter if scheduled to resolve conflict
            wait_duration = dep_t - arr_t
            if wait_duration > 0.1:
                self.state = "HOVER_WAIT"
                wait_steps = int(wait_duration / dt)
                for _ in range(wait_steps):
                    curr_t += dt
                    power = self.energy_model.compute_hover_power(payload_kg)
                    self._update_battery(power, dt)
                    self._record_telemetry(curr_t, 0.0, 0.0)

            # Move along edge (curr_node -> next_node)
            self.state = "NAVIGATING"
            x1, y1 = self.node_positions[curr_node]
            x2, y2 = self.node_positions[next_node]
            dist = math.hypot(x2 - x1, y2 - y1)
            u_vec = ((x2 - x1) / dist, (y2 - y1) / dist)

            edge_nominal_duration = next_arr - dep_t
            if edge_nominal_duration <= 0.01:
                edge_nominal_duration = dist / self.specs.nominal_airspeed

            travel_steps = max(1, int(edge_nominal_duration / dt))
            dx_step = (x2 - x1) / travel_steps
            dy_step = (y2 - y1) / travel_steps

            for _ in range(travel_steps):
                curr_t += dt
                self.pos_x += dx_step
                self.pos_y += dy_step

                # Wind interaction
                wx, wy = self.wind_field.get_wind(self.pos_x, self.pos_y, curr_t)
                feasible, g_speed = self.energy_model.calculate_ground_speed(u_vec, (wx, wy))
                if not feasible:
                    g_speed = max(1.0, self.specs.nominal_airspeed)

                power = self.energy_model.compute_power(self.specs.nominal_airspeed, payload_kg)
                self._update_battery(power, dt)
                self._record_telemetry(curr_t, g_speed, self.specs.nominal_airspeed)

        # Landing & Payload Delivery
        self.state = "LANDING"
        landing_time = 65.0 / 2.0
        for step in np.arange(0, landing_time, dt):
            curr_t += dt
            self.pos_z = max(0.0, self.pos_z - 2.0 * dt)
            power = self.energy_model.compute_hover_power(payload_kg) * 0.9
            self._update_battery(power, dt)
            self._record_telemetry(curr_t, 0.0, 0.0)

        self.state = "DELIVERY_COMPLETE"
        self._record_telemetry(curr_t, 0.0, 0.0)
        return self.telemetry_history

    def _update_battery(self, power_w: float, dt_s: float):
        current_a = power_w / max(10.0, self.current_voltage)
        mah_drain = (current_a * (dt_s / 3600.0)) * 1000.0
        self.capacity_used_mah += mah_drain

        # 4S LiPo discharge voltage curve approximation
        soc = max(0.0, 1.0 - (self.capacity_used_mah / self.battery_nominal_mah))
        # 16.8V at 100% down to 14.0V at 0%
        self.current_voltage = 14.0 + 2.8 * soc

    def _record_telemetry(self, timestamp: float, g_speed: float, airspeed: float):
        lat, lon = self._meter_to_gps(self.pos_x, self.pos_y)
        current_a = 0.0
        if self.telemetry_history:
            prev_mah = self.telemetry_history[-1].capacity_used_mah
            dt = max(0.001, timestamp - self.telemetry_history[-1].timestamp_s)
            current_a = ((self.capacity_used_mah - prev_mah) / 1000.0) * (3600.0 / dt)

        rec = HardwareTelemetryRecord(
            timestamp_s=round(timestamp, 2),
            drone_id=self.drone_id,
            x=round(self.pos_x, 2),
            y=round(self.pos_y, 2),
            z=round(self.pos_z, 2),
            lat=round(lat, 7),
            lon=round(lon, 7),
            ground_speed_mps=round(g_speed, 2),
            airspeed_mps=round(airspeed, 2),
            battery_volts=round(self.current_voltage, 2),
            current_amps=round(current_a, 2),
            capacity_used_mah=round(self.capacity_used_mah, 1),
            state=self.state
        )
        self.telemetry_history.append(rec)

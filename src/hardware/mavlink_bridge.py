"""
MAVLink Protocol & Ground Control Station (GCS) Hardware Bridge.
Converts WEBS 4D planned trajectories into:
1. QGroundControl (.plan) JSON files.
2. Standard MAVLink Waypoint Mission files (.waypoints).
3. Real-world WGS84 GPS coordinates (Latitude, Longitude, Altitude AGL).
Directly compatible with Pixhawk 4/6C, PX4 Autopilot, and ArduPilot SITL.
"""

import os
import json
import math
from typing import List, Tuple, Dict, Any
from src.planners.base_planner import PlanResult

class MAVLinkBridge:
    def __init__(
        self,
        ref_lat: float = 26.3683, # Boca Raton / Default Reference Datum Latitude
        ref_lon: float = -80.1289, # Reference Datum Longitude
        cruise_alt_m: float = 65.0 # Cruise altitude AGL (meters)
    ):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.cruise_alt_m = cruise_alt_m
        # Earth radius in meters
        self.R_EARTH = 6378137.0

    def meter_to_gps(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """
        Converts local Cartesian meters (East=x, North=y) to WGS84 GPS (Lat, Lon).
        """
        d_lat = y_m / self.R_EARTH
        d_lon = x_m / (self.R_EARTH * math.cos(math.radians(self.ref_lat)))
        lat = self.ref_lat + math.degrees(d_lat)
        lon = self.ref_lon + math.degrees(d_lon)
        return (lat, lon)

    def generate_qgroundcontrol_plan(
        self,
        plan: PlanResult,
        node_positions: Dict[int, Tuple[float, float]],
        output_file: str
    ) -> Dict[str, Any]:
        """
        Generates a standard QGroundControl (.plan) JSON mission file.
        This file can be opened directly in QGroundControl to upload to physical Pixhawk drones!
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        items = []
        seq = 0

        # Item 0: Home / Takeoff coordinate
        src_node, t0_arr, t0_dep = plan.trajectory[0]
        src_x, src_y = node_positions[src_node]
        src_lat, src_lon = self.meter_to_gps(src_x, src_y)

        # 1. Takeoff Command (MAV_CMD_NAV_TAKEOFF = 22)
        items.append({
            "autoContinue": True,
            "command": 22,
            "coordinate": [src_lat, src_lon, self.cruise_alt_m],
            "doJumpId": seq + 1,
            "frame": 3, # MAV_FRAME_GLOBAL_RELATIVE_ALT
            "params": [0, 0, 0, 0, src_lat, src_lon, self.cruise_alt_m],
            "type": "SimpleItem"
        })
        seq += 1

        # 2. Sequential 4D Waypoints with speed and hover reservations
        for i in range(len(plan.trajectory) - 1):
            curr_node, curr_arr, curr_dep = plan.trajectory[i]
            next_node, next_arr, _ = plan.trajectory[i + 1]

            # Hover/Wait loiter command if drone waited to maintain safety separation
            wait_time = curr_dep - curr_arr
            if wait_time > 0.5:
                cx, cy = node_positions[curr_node]
                clat, clon = self.meter_to_gps(cx, cy)
                items.append({
                    "autoContinue": True,
                    "command": 19, # MAV_CMD_NAV_LOITER_TIME
                    "coordinate": [clat, clon, self.cruise_alt_m],
                    "doJumpId": seq + 1,
                    "frame": 3,
                    "params": [wait_time, 0, 0, 0, clat, clon, self.cruise_alt_m],
                    "type": "SimpleItem"
                })
                seq += 1

            # Waypoint Navigation (MAV_CMD_NAV_WAYPOINT = 16)
            nx, ny = node_positions[next_node]
            nlat, nlon = self.meter_to_gps(nx, ny)
            items.append({
                "autoContinue": True,
                "command": 16,
                "coordinate": [nlat, nlon, self.cruise_alt_m],
                "doJumpId": seq + 1,
                "frame": 3,
                "params": [0, 0, 0, 0, nlat, nlon, self.cruise_alt_m],
                "type": "SimpleItem"
            })
            seq += 1

        # 3. Payload Drop / Landing (MAV_CMD_NAV_LAND = 21)
        dst_node = plan.trajectory[-1][0]
        dst_x, dst_y = node_positions[dst_node]
        dst_lat, dst_lon = self.meter_to_gps(dst_x, dst_y)
        items.append({
            "autoContinue": True,
            "command": 21,
            "coordinate": [dst_lat, dst_lon, 0.0],
            "doJumpId": seq + 1,
            "frame": 3,
            "params": [0, 0, 0, 0, dst_lat, dst_lon, 0.0],
            "type": "SimpleItem"
        })

        qgc_plan = {
            "fileType": "Plan",
            "version": 1,
            "groundStation": "WEBS_UTM",
            "mission": {
                "cruiseSpeed": 14.0,
                "hoverSpeed": 5.0,
                "items": items,
                "plannedHomePosition": [src_lat, src_lon, 10.0],
                "vehicleType": 2 # Multirotor
            }
        }

        with open(output_file, "w") as f:
            json.dump(qgc_plan, f, indent=2)

        return qgc_plan

    def generate_mavlink_waypoint_file(
        self,
        plan: PlanResult,
        node_positions: Dict[int, Tuple[float, float]],
        output_file: str
    ):
        """
        Generates standard MAVLink text waypoint file (.waypoints format)
        readable by Mission Planner and ArduPilot SITL.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        lines = ["QGC WPL 110\n"]
        seq = 0

        # Home
        src_node = plan.trajectory[0][0]
        src_x, src_y = node_positions[src_node]
        src_lat, src_lon = self.meter_to_gps(src_x, src_y)
        lines.append(f"{seq}\t1\t0\t16\t0\t0\t0\t0\t{src_lat:.7f}\t{src_lon:.7f}\t0.000000\t1\n")
        seq += 1

        # Takeoff
        lines.append(f"{seq}\t0\t3\t22\t0\t0\t0\t0\t{src_lat:.7f}\t{src_lon:.7f}\t{self.cruise_alt_m:.1f}\t1\n")
        seq += 1

        # Intermediate 4D Waypoints
        for _, (node, _, _) in enumerate(plan.trajectory[1:]):
            x, y = node_positions[node]
            lat, lon = self.meter_to_gps(x, y)
            lines.append(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat:.7f}\t{lon:.7f}\t{self.cruise_alt_m:.1f}\t1\n")
            seq += 1

        with open(output_file, "w") as f:
            f.writelines(lines)

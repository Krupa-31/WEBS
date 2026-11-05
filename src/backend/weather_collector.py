"""
WEBS v2 Module 2: Hyper-Local Weather Harvester & Anemometer Mesh.

Collects, simulates, and distributes multi-source micro-climate observations
across the Bangalore Central airspace. Fuses:
1. Rooftop Ultrasonic IoT Anemometer Stations (MG Road, Trinity, UB City, BRV Hospital)
2. Live Public Weather API (OpenWeather / IMD)
3. In-Flight Drones as Anemometric Probes
"""

import math
import time
import random
from typing import Dict, List, Any, Optional, Tuple

from src.backend.database import update_weather_station, get_all_weather_stations


class HyperLocalWeatherCollector:
    """
    Manages live weather observation ingestion and spatial interpolation.
    """

    def __init__(self, base_speed_mps: float = 6.0, base_dir_deg: float = 45.0):
        self.base_speed_mps = base_speed_mps
        self.base_dir_deg = base_dir_deg

        # Topographic wind channeling factors for Bangalore central corridors
        # UB City tower: high elevation, higher wind speed (+25%)
        # MG Road: East-West wind corridor (+10%)
        # Brigade Road: Narrow commercial street, reduced wind (-15%) but higher turbulence
        self.station_modifiers = {
            "WS-BLR-01": {"speed_mult": 1.05, "dir_shift": 0.0, "gust_factor": 1.35},  # MG Road Metro
            "WS-BLR-02": {"speed_mult": 1.15, "dir_shift": 5.0, "gust_factor": 1.40},  # Trinity Circle
            "WS-BLR-03": {"speed_mult": 1.30, "dir_shift": -5.0, "gust_factor": 1.55}, # UB City Tower Mast
            "WS-BLR-04": {"speed_mult": 0.90, "dir_shift": 0.0, "gust_factor": 1.25},  # BRV Hospital Drop
            "WS-BLR-05": {"speed_mult": 0.95, "dir_shift": 10.0, "gust_factor": 1.30}, # Brigade Road
        }

    def sync_weather_mesh(self, current_sim_time: float = 0.0) -> List[Dict[str, Any]]:
        """
        Polls or simulates live updates for all rooftop weather stations and persists to DB.
        """
        stations = get_all_weather_stations()
        updated_stations = []

        # Time-based oscillation (simulates realistic 10-minute urban thermal gust cycles)
        cycle_rad = (current_sim_time / 60.0) * (2.0 * math.pi / 10.0)
        global_gust = math.sin(cycle_rad) * 1.5

        for st in stations:
            sid = st["station_id"]
            mods = self.station_modifiers.get(sid, {"speed_mult": 1.0, "dir_shift": 0.0, "gust_factor": 1.3})

            # Micro-turbulence jitter (+- 0.4 m/s)
            jitter = (random.random() - 0.5) * 0.8
            speed = max(0.5, (self.base_speed_mps + global_gust + jitter) * mods["speed_mult"])
            direction = (self.base_dir_deg + mods["dir_shift"] + random.uniform(-4.0, 4.0)) % 360.0
            gust = speed * mods["gust_factor"]

            update_weather_station(
                station_id=sid,
                wind_speed_mps=round(speed, 2),
                wind_dir_deg=round(direction, 1),
                gust_mps=round(gust, 2)
            )

            updated_stations.append({
                "station_id": sid,
                "landmark_name": st["landmark_name"],
                "lat": st["lat"],
                "lon": st["lon"],
                "wind_speed_mps": round(speed, 2),
                "wind_dir_deg": round(direction, 1),
                "gust_mps": round(gust, 2),
                "air_density_kg_m3": st["air_density_kg_m3"]
            })

        return updated_stations

    def get_interpolated_wind_at_point(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Inverse-Distance-Weighted (IDW) interpolation from nearest weather stations.
        Returns (wind_speed_mps, wind_dir_deg).
        """
        stations = get_all_weather_stations()
        if not stations:
            return self.base_speed_mps, self.base_dir_deg

        weights = []
        u_comps = []
        v_comps = []

        for st in stations:
            d = math.hypot(st["lat"] - lat, st["lon"] - lon)
            w = 1.0 / max(d, 1e-5)
            weights.append(w)

            rad = math.radians(st["wind_dir_deg"])
            u = st["wind_speed_mps"] * math.cos(rad)
            v = st["wind_speed_mps"] * math.sin(rad)
            u_comps.append(u * w)
            v_comps.append(v * w)

        total_w = sum(weights)
        avg_u = sum(u_comps) / total_w
        avg_v = sum(v_comps) / total_w

        interp_speed = math.hypot(avg_u, avg_v)
        interp_deg = math.degrees(math.atan2(avg_v, avg_u)) % 360.0

        return round(interp_speed, 2), round(interp_deg, 1)

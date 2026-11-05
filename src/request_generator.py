"""
Delivery Request Generator for Public Autonomous Drone Delivery.
Generates realistic customer parcel delivery requests across the urban airspace graph
with release times, deadlines, and payload weights.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from src.map_module import CityAirspaceMap

@dataclass
class DeliveryRequest:
    request_id: int
    source: int
    destination: int
    release_time: float   # seconds
    deadline: float       # seconds (hard deadline)
    payload_kg: float     # kg
    slack_time: float = 0.0 # deadline - (release_time + shortest_path_time)
    
    @property
    def max_allowed_duration(self) -> float:
        return self.deadline - self.release_time


class RequestGenerator:
    """
    Synthesizes delivery requests according to Poisson or burst arrivals.
    """
    def __init__(
        self,
        city_map: CityAirspaceMap,
        seed: Optional[int] = 42
    ):
        self.city_map = city_map
        self.rng = np.random.default_rng(seed)
        self.nodes = list(city_map.graph.nodes())

    def generate_requests(
        self,
        num_requests: int = 50,
        sim_duration: float = 3600.0, # 1 hour
        min_distance_m: float = 300.0,
        max_distance_m: float = 3500.0,
        min_payload: float = 0.5,
        max_payload: float = 1.5,
        nominal_speed: float = 14.0
    ) -> List[DeliveryRequest]:
        """
        Generates requests distributed uniformly or in Poisson bursts over sim_duration.
        Deadlines are set with realistic slack factors (e.g. 1.4x - 2.5x of nominal shortest path time).
        """
        requests = []
        release_times = np.sort(self.rng.uniform(0.0, sim_duration * 0.8, num_requests))

        req_id = 0
        attempts = 0
        max_attempts = num_requests * 50

        while len(requests) < num_requests and attempts < max_attempts:
            attempts += 1
            src = int(self.rng.choice(self.nodes))
            dst = int(self.rng.choice(self.nodes))
            if src == dst:
                continue

            x1, y1 = self.city_map.get_pos(src)
            x2, y2 = self.city_map.get_pos(dst)
            dist = np.hypot(x2 - x1, y2 - y1)

            if not (min_distance_m <= dist <= max_distance_m):
                continue

            release_t = float(release_times[len(requests)])
            payload = float(self.rng.uniform(min_payload, max_payload))

            # Nominal straight-line time estimate
            nom_time = dist / nominal_speed
            # Slack factor between 1.35 and 2.2
            slack_factor = float(self.rng.uniform(1.35, 2.2))
            allowed_duration = max(nom_time * slack_factor, nom_time + 120.0)
            deadline = release_t + allowed_duration
            slack = allowed_duration - nom_time

            req = DeliveryRequest(
                request_id=req_id,
                source=src,
                destination=dst,
                release_time=release_t,
                deadline=deadline,
                payload_kg=payload,
                slack_time=slack
            )
            requests.append(req)
            req_id += 1

        return requests

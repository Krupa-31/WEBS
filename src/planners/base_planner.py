"""
Base Planner Interface and Trajectory Data Structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import time

@dataclass
class TrajectorySegment:
    u: int
    v: int
    t_start: float
    t_end: float
    distance: float
    energy_joules: float

@dataclass
class PlanResult:
    request_id: int
    drone_id: int
    success: bool
    status_reason: str # 'ACCEPTED', 'DEADLINE_EXCEEDED', 'BATTERY_EXHAUSTED', 'NO_CONFLICT_FREE_PATH', 'UNFEASIBLE_WIND'
    trajectory: List[Tuple[int, float, float]] = field(default_factory=list) # (node, t_arr, t_dep)
    flight_time_s: float = 0.0
    energy_joules: float = 0.0
    planning_time_ms: float = 0.0
    arrival_time: float = 0.0
    deadline: float = 0.0
    battery_depleted_en_route: bool = False
    late_delivery: bool = False

class BasePlanner(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def plan_batch(
        self,
        requests: list,
        current_time: float
    ) -> List[PlanResult]:
        """Plans trajectories for a batch of requests."""
        pass

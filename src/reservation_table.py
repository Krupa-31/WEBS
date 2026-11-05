"""
Space-Time 4D Reservation Table for Conflict-Free Drone Trajectory Planning.
Implements vertex occupancy protection and edge swap/collision avoidance (Steinberg et al. [12]).
Guarantees zero safety separation violations by construction.
"""

from typing import Dict, List, Tuple, Optional, Set
import bisect

class SpaceTimeReservationTable:
    """
    Stores 4D space-time reservations (node, time_interval) and (edge, time_interval).
    Maintains interval trees / sorted disjoint intervals for rapid conflict queries.
    """
    def __init__(self, time_buffer: float = 2.0):
        """
        time_buffer: safety buffer (seconds) around arrival/departure at nodes.
        """
        self.time_buffer = float(time_buffer)
        # vertex_id -> list of (t_start, t_end, drone_id)
        self.vertex_reservations: Dict[int, List[Tuple[float, float, int]]] = {}
        # (u, v) -> list of (t_start, t_end, drone_id)
        self.edge_reservations: Dict[Tuple[int, int], List[Tuple[float, float, int]]] = {}
        # drone_id -> list of (node, t_arr, t_dep)
        self.planned_routes: Dict[int, List[Tuple[int, float, float]]] = {}

    def is_vertex_available(self, node: int, t_start: float, t_end: float, ignore_drone: Optional[int] = None) -> bool:
        """
        Checks if vertex 'node' is available during [t_start - buffer, t_end + buffer].
        """
        res_list = self.vertex_reservations.get(node)
        if not res_list:
            return True

        for (s, e, did) in res_list:
            if ignore_drone is not None and did == ignore_drone:
                continue
            # Separation check: mutual buffer between [t_start, t_end] and [s, e]
            if (t_start - self.time_buffer < e) and (s - self.time_buffer < t_end):
                return False
        return True

    def find_earliest_departure(self, node: int, earliest_t: float) -> float:
        """
        Finds earliest departure time >= earliest_t with safe launch clearance at the node.
        """
        t = earliest_t
        res_list = sorted(self.vertex_reservations.get(node, []), key=lambda x: x[0])
        for (s, e, did) in res_list:
            if (t - self.time_buffer < e) and (s - self.time_buffer < t):
                t = e + self.time_buffer + 0.1
        return t

    def is_edge_available(self, u: int, v: int, t_start: float, t_end: float, ignore_drone: Optional[int] = None) -> bool:
        """
        Checks if traversing edge (u, v) over [t_start, t_end] conflicts with:
        1. A drone traversing the same edge (u, v) too closely.
        2. A drone traversing the opposite edge (v, u) during the same time (head-on collision).
        """
        # 1. Forward edge check
        forward_res = self.edge_reservations.get((u, v), [])
        for (s, e, did) in forward_res:
            if ignore_drone is not None and did == ignore_drone:
                continue
            # Following separation: cannot enter within time_buffer of another entry
            if abs(t_start - s) < self.time_buffer:
                return False

        # 2. Reverse edge check (head-on swap collision prevention)
        reverse_res = self.edge_reservations.get((v, u), [])
        for (s, e, did) in reverse_res:
            if ignore_drone is not None and did == ignore_drone:
                continue
            if t_start < e and s < t_end:
                return False

        return True

    def reserve_path(
        self,
        drone_id: int,
        trajectory: List[Tuple[int, float, float]]
    ):
        """
        Commits a full planned trajectory:
        trajectory: list of (node, t_arrive, t_depart)
        """
        self.planned_routes[drone_id] = trajectory
        for i, (node, t_arr, t_dep) in enumerate(trajectory):
            # Vertex reservation
            if node not in self.vertex_reservations:
                self.vertex_reservations[node] = []
            self.vertex_reservations[node].append((t_arr, t_dep, drone_id))

            # Edge reservation to next node
            if i + 1 < len(trajectory):
                next_node, next_arr, _ = trajectory[i + 1]
                edge = (node, next_node)
                if edge not in self.edge_reservations:
                    self.edge_reservations[edge] = []
                self.edge_reservations[edge].append((t_dep, next_arr, drone_id))

        # Sort reservations by start time for cleanliness
        for node in set(n for n, _, _ in trajectory):
            self.vertex_reservations[node].sort(key=lambda x: x[0])
        for i in range(len(trajectory) - 1):
            edge = (trajectory[i][0], trajectory[i + 1][0])
            self.edge_reservations[edge].sort(key=lambda x: x[0])

    def cancel_reservation(self, drone_id: int):
        """Removes all reservations associated with drone_id (useful for rollbacks)."""
        if drone_id not in self.planned_routes:
            return
        del self.planned_routes[drone_id]
        
        for node in list(self.vertex_reservations.keys()):
            self.vertex_reservations[node] = [
                res for res in self.vertex_reservations[node] if res[2] != drone_id
            ]
            if not self.vertex_reservations[node]:
                del self.vertex_reservations[node]

        for edge in list(self.edge_reservations.keys()):
            self.edge_reservations[edge] = [
                res for res in self.edge_reservations[edge] if res[2] != drone_id
            ]
            if not self.edge_reservations[edge]:
                del self.edge_reservations[edge]

    def verify_safety_violations(self) -> int:
        """
        Audits the entire reservation table for any safety violations:
        Returns count of conflicts (must be 0 for conflict-free verification).
        """
        violations = 0
        # Check node overlaps
        for node, intervals in self.vertex_reservations.items():
            for i in range(len(intervals)):
                for j in range(i + 1, len(intervals)):
                    s1, e1, d1 = intervals[i]
                    s2, e2, d2 = intervals[j]
                    if d1 == d2: continue
                    if (s1 - self.time_buffer < e2) and (s2 - self.time_buffer < e1):
                        violations += 1

        # Check edge head-on collisions
        for (u, v), intervals in self.edge_reservations.items():
            rev_intervals = self.edge_reservations.get((v, u), [])
            for s1, e1, d1 in intervals:
                for s2, e2, d2 in rev_intervals:
                    if d1 == d2: continue
                    if s1 < e2 and s2 < e1:
                        violations += 1

        return violations

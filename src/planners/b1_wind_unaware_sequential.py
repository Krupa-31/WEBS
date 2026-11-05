"""
Baseline 1 (B1): Wind-Unaware Sequential Space-Time Planner.
Plans requests one-by-one in FIFO arrival order without batching.
Assumes nominal airspeed (zero wind) during pathfinding and reservation table stamping.
"""

import heapq
import time
import math
from typing import List, Tuple, Dict, Optional
from src.config import DroneSpecs
from src.map_module import CityAirspaceMap
from src.reservation_table import SpaceTimeReservationTable
from src.request_generator import DeliveryRequest
from src.planners.base_planner import BasePlanner, PlanResult

class B1WindUnawareSequentialPlanner(BasePlanner):
    def __init__(
        self,
        city_map: CityAirspaceMap,
        reservation_table: SpaceTimeReservationTable,
        specs: Optional[DroneSpecs] = None,
        wait_step_s: float = 2.0,
        max_wait_steps: int = 4
    ):
        super().__init__(name="B1_WindUnaware_Seq")
        self.city_map = city_map
        self.reservation_table = reservation_table
        self.specs = specs if specs is not None else DroneSpecs()
        self.nominal_speed = self.specs.nominal_airspeed
        self.wait_step_s = wait_step_s
        self.max_wait_steps = max_wait_steps

    def plan_batch(
        self,
        requests: List[DeliveryRequest],
        current_time: float
    ) -> List[PlanResult]:
        """Plans requests sequentially in raw arrival order (FIFO)."""
        results: List[PlanResult] = []
        # Sort strictly by arrival time (no slack batching)
        fifo_reqs = sorted(requests, key=lambda r: r.release_time)

        for req in fifo_reqs:
            t0 = time.perf_counter()
            plan_res = self._plan_single_request(req)
            plan_res.planning_time_ms = (time.perf_counter() - t0) * 1000.0
            results.append(plan_res)

        return results

    def _plan_single_request(self, req: DeliveryRequest) -> PlanResult:
        src = req.source
        dst = req.destination
        deadline = req.deadline

        start_t = self.reservation_table.find_earliest_departure(src, req.release_time)
        if start_t > deadline:
            return PlanResult(
                request_id=req.request_id,
                drone_id=req.request_id,
                success=False,
                status_reason="DEADLINE_EXCEEDED",
                arrival_time=0.0,
                deadline=deadline
            )

        dst_x, dst_y = self.city_map.get_pos(dst)
        h_start = math.hypot(dst_x - self.city_map.get_pos(src)[0], dst_y - self.city_map.get_pos(src)[1]) / self.nominal_speed
        
        # open_set: (f, g, u, t, history, wait_count)
        open_set = [(h_start, 0.0, src, start_t, [(src, start_t, start_t)], 0)]
        closed_set: Dict[Tuple[int, int], float] = {}

        nodes_explored = 0
        max_explored = 3000

        while open_set and nodes_explored < max_explored:
            nodes_explored += 1
            f_score, g_cost, u, curr_t, history, waits = heapq.heappop(open_set)

            if u == dst:
                self.reservation_table.reserve_path(req.request_id, history)
                return PlanResult(
                    request_id=req.request_id,
                    drone_id=req.request_id,
                    success=True,
                    status_reason="ACCEPTED",
                    trajectory=history,
                    flight_time_s=curr_t - start_t,
                    arrival_time=curr_t,
                    deadline=deadline
                )

            t_bucket = int(curr_t / 1.0)
            state_key = (u, t_bucket)
            if state_key in closed_set and closed_set[state_key] <= g_cost:
                continue
            closed_set[state_key] = g_cost

            u_x, u_y = self.city_map.get_pos(u)

            # Spatial expansion (assuming nominal speed)
            for _, v, edge_data in self.city_map.graph.out_edges(u, data=True):
                dist = edge_data['length']
                trav_time = dist / self.nominal_speed
                arr_t = curr_t + trav_time

                if arr_t > deadline:
                    continue

                if not self.reservation_table.is_edge_available(u, v, curr_t, arr_t):
                    continue
                if not self.reservation_table.is_vertex_available(v, arr_t, arr_t):
                    continue

                new_history = list(history)
                new_history[-1] = (new_history[-1][0], new_history[-1][1], curr_t)
                new_history.append((v, arr_t, arr_t))

                v_x, v_y = self.city_map.get_pos(v)
                h_dist = math.hypot(dst_x - v_x, dst_y - v_y)
                new_g = arr_t - start_t
                new_f = new_g + (h_dist / self.nominal_speed)

                heapq.heappush(open_set, (new_f, new_g, v, arr_t, new_history, 0))

            # Temporal wait expansion
            if waits < self.max_wait_steps:
                wait_t = curr_t + self.wait_step_s
                if wait_t <= deadline and self.reservation_table.is_vertex_available(u, curr_t, wait_t):
                    new_history = list(history)
                    new_history[-1] = (new_history[-1][0], new_history[-1][1], wait_t)
                    h_dist = math.hypot(dst_x - u_x, dst_y - u_y)
                    new_g = wait_t - start_t
                    new_f = new_g + (h_dist / self.nominal_speed)
                    heapq.heappush(open_set, (new_f, new_g, u, wait_t, new_history, waits + 1))

        return PlanResult(
            request_id=req.request_id,
            drone_id=req.request_id,
            success=False,
            status_reason="NO_CONFLICT_FREE_PATH",
            arrival_time=0.0,
            deadline=deadline
        )

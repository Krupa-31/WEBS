"""
Proposed WEBS Planner: Wind- and Energy-Aware Batch Space-Time Planner.
Combines:
1. Dynamic slack-based batch prioritization (accounting for wind-adjusted travel time).
2. Space-Time A* search on the unit-segmented airspace graph.
3. Rigorous physical wind vector (w · u) ground speed and Betti Sorbelli / Du aerodynamic energy models.
4. Strict 4D reservation table conflict avoidance (vertex and edge swap collisions).
5. Early pruning of paths violating battery capacity or delivery time windows.
"""

import heapq
import time
import math
from typing import List, Tuple, Optional, Dict
from src.config import DroneSpecs
from src.map_module import CityAirspaceMap
from src.wind_energy import WindField, EnergyModel
from src.reservation_table import SpaceTimeReservationTable
from src.request_generator import DeliveryRequest
from src.planners.base_planner import BasePlanner, PlanResult

class WEBSPlanner(BasePlanner):
    def __init__(
        self,
        city_map: CityAirspaceMap,
        reservation_table: SpaceTimeReservationTable,
        wind_field: WindField,
        energy_model: Optional[EnergyModel] = None,
        specs: Optional[DroneSpecs] = None,
        energy_weight: float = 0.0001, # composite cost: time_sec + energy_weight * energy_joules
        wait_step_s: float = 2.0,
        max_wait_steps: int = 4
    ):
        super().__init__(name="WEBS")
        self.city_map = city_map
        self.reservation_table = reservation_table
        self.wind_field = wind_field
        self.specs = specs if specs is not None else DroneSpecs()
        self.energy_model = energy_model if energy_model is not None else EnergyModel(self.specs)
        self.energy_weight = energy_weight
        self.wait_step_s = wait_step_s
        self.max_wait_steps = max_wait_steps

    def _estimate_wind_aware_min_time(self, src: int, dst: int, release_time: float) -> float:
        """
        Calculates an optimistic wind-aware travel time lower bound from src to dst.
        Uses Euclidean distance and max possible ground speed along the wind direction.
        """
        x1, y1 = self.city_map.get_pos(src)
        x2, y2 = self.city_map.get_pos(dst)
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-3:
            return 0.0

        ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
        wx, wy = self.wind_field.get_wind((x1 + x2) / 2.0, (y1 + y2) / 2.0, release_time)
        feasible, g_speed = self.energy_model.calculate_ground_speed((ux, uy), (wx, wy))
        if not feasible or g_speed < 1.0:
            g_speed = max(1.0, self.specs.nominal_airspeed - math.hypot(wx, wy))

        return dist / max(g_speed, 1.0)

    def plan_batch(
        self,
        requests: List[DeliveryRequest],
        current_time: float
    ) -> List[PlanResult]:
        """
        Batches requests, calculates wind-aware slack time, sorts requests by slack (ascending),
        and plans collision-free space-time trajectories.
        """
        start_wall_time = time.perf_counter()
        results: List[PlanResult] = []

        # 1. Wind-aware slack calculation:
        # Requests with tighter deadlines and headwind penalties get highest priority!
        prioritized_reqs = []
        for req in requests:
            min_t = self._estimate_wind_aware_min_time(req.source, req.destination, req.release_time)
            # Slack: margin between latest acceptable arrival and fastest possible arrival
            wind_aware_slack = req.deadline - (req.release_time + min_t)
            prioritized_reqs.append((wind_aware_slack, req))

        # Sort by slack (smallest slack first -> most constrained first, Steinberg et al. [12])
        prioritized_reqs.sort(key=lambda item: item[0])

        # 2. Sequential Space-Time Planning for prioritized batch
        for slack, req in prioritized_reqs:
            t0 = time.perf_counter()
            plan_res = self._plan_single_request(req)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            plan_res.planning_time_ms = elapsed_ms
            results.append(plan_res)

        return results

    def _plan_single_request(self, req: DeliveryRequest) -> PlanResult:
        """
        Performs Wind- and Energy-Aware Space-Time A* search.
        """
        src = req.source
        dst = req.destination
        deadline = req.deadline
        payload = req.payload_kg
        battery_limit = self.specs.usable_joules

        # Ensure safe departure spacing from origin node
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
        max_airspeed = self.specs.nominal_airspeed + self.wind_field.base_speed

        # Priority queue stores tuples:
        # (f_cost, g_cost, current_node, current_time, cumulative_energy, path_history, wait_count)
        # where path_history is list of (node, t_arr, t_dep)
        h_start = math.hypot(dst_x - self.city_map.get_pos(src)[0], dst_y - self.city_map.get_pos(src)[1]) / max_airspeed
        start_history = [(src, start_t, start_t)]
        
        # open_set: (f_score, g_cost, node, t, energy, history, wait_count)
        open_set = [(h_start, 0.0, src, start_t, 0.0, start_history, 0)]
        
        # Best known cost to reach (node, discretized_time)
        time_slot = 1.0 # 1s resolution for state pruning
        closed_set: Dict[Tuple[int, int], float] = {}

        hover_power = self.energy_model.compute_hover_power(payload_kg=payload)

        nodes_explored = 0
        max_explored = 3000

        while open_set and nodes_explored < max_explored:
            nodes_explored += 1
            f_score, g_cost, u, curr_t, curr_energy, history, waits = heapq.heappop(open_set)

            # Reached destination?
            if u == dst:
                # Commit to reservation table!
                self.reservation_table.reserve_path(req.request_id, history)
                return PlanResult(
                    request_id=req.request_id,
                    drone_id=req.request_id,
                    success=True,
                    status_reason="ACCEPTED",
                    trajectory=history,
                    flight_time_s=curr_t - start_t,
                    energy_joules=curr_energy,
                    arrival_time=curr_t,
                    deadline=deadline
                )

            t_bucket = int(curr_t / time_slot)
            state_key = (u, t_bucket)
            if state_key in closed_set and closed_set[state_key] <= g_cost:
                continue
            closed_set[state_key] = g_cost

            u_x, u_y = self.city_map.get_pos(u)

            # --- 1. SPATIAL EXPANSIONS (Traverse outgoing edges) ---
            for _, v, edge_data in self.city_map.graph.out_edges(u, data=True):
                dist = edge_data['length']
                u_vec = edge_data['u_vec']

                # Get wind vector at edge midpoint and current time
                mid_x = (u_x + self.city_map.get_pos(v)[0]) / 2.0
                mid_y = (u_y + self.city_map.get_pos(v)[1]) / 2.0
                wind_vec = self.wind_field.get_wind(mid_x, mid_y, curr_t)

                # Compute ground speed & energy consumption
                feasible, trav_time, edge_energy = self.energy_model.compute_edge_traversal(
                    distance_m=dist,
                    u_vector=u_vec,
                    wind_vector=wind_vec,
                    payload_kg=payload
                )

                if not feasible:
                    continue # Edge impossible due to severe headwind / crosswind

                arr_t = curr_t + trav_time
                new_energy = curr_energy + edge_energy

                # Prune if exceeding battery capacity or deadline
                if arr_t > deadline or new_energy > battery_limit:
                    continue

                # Check conflict-free safety reservation for edge and vertex
                if not self.reservation_table.is_edge_available(u, v, curr_t, arr_t):
                    continue
                if not self.reservation_table.is_vertex_available(v, arr_t, arr_t):
                    continue

                # Update trajectory history
                # Previous node departure time is curr_t, new arrival is arr_t
                new_history = list(history)
                # update departure of u
                new_history[-1] = (new_history[-1][0], new_history[-1][1], curr_t)
                new_history.append((v, arr_t, arr_t))

                # Heuristic to destination
                v_x, v_y = self.city_map.get_pos(v)
                h_dist = math.hypot(dst_x - v_x, dst_y - v_y)
                h_cost = (h_dist / max_airspeed)

                new_g = (arr_t - start_t) + self.energy_weight * new_energy
                new_f = new_g + h_cost

                heapq.heappush(open_set, (new_f, new_g, v, arr_t, new_energy, new_history, 0))

            # --- 2. TEMPORAL EXPANSION (VTOL Hover / Wait at vertex) ---
            if waits < self.max_wait_steps:
                wait_t = curr_t + self.wait_step_s
                wait_energy = curr_energy + (hover_power * self.wait_step_s)

                if wait_t <= deadline and wait_energy <= battery_limit:
                    if self.reservation_table.is_vertex_available(u, curr_t, wait_t):
                        new_history = list(history)
                        new_history[-1] = (new_history[-1][0], new_history[-1][1], wait_t)
                        
                        h_dist = math.hypot(dst_x - u_x, dst_y - u_y)
                        h_cost = (h_dist / max_airspeed)
                        new_g = (wait_t - start_t) + self.energy_weight * wait_energy
                        new_f = new_g + h_cost

                        heapq.heappush(open_set, (new_f, new_g, u, wait_t, wait_energy, new_history, waits + 1))

        # If search exhausted without reaching destination
        return PlanResult(
            request_id=req.request_id,
            drone_id=req.request_id,
            success=False,
            status_reason="NO_CONFLICT_FREE_PATH",
            arrival_time=0.0,
            deadline=deadline
        )

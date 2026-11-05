"""
Baseline 3 (B3): Post-Hoc Energy and Deadline Rejection Planner.
Plans using Baseline 2 (wind-unaware batch space-time planner), then performs a
post-hoc physical feasibility check in the real wind field.
Rejects any trajectory where actual wind-induced flight time causes deadline violation
or battery exhaustion, rolling back its reservations.
"""

import time
import math
from typing import List, Tuple, Optional
from src.config import DroneSpecs
from src.map_module import CityAirspaceMap
from src.wind_energy import WindField, EnergyModel
from src.reservation_table import SpaceTimeReservationTable
from src.request_generator import DeliveryRequest
from src.planners.base_planner import BasePlanner, PlanResult
from src.planners.b2_wind_unaware_batch import B2WindUnawareBatchPlanner

class B3PostHocRejectionPlanner(BasePlanner):
    def __init__(
        self,
        city_map: CityAirspaceMap,
        reservation_table: SpaceTimeReservationTable,
        wind_field: WindField,
        energy_model: Optional[EnergyModel] = None,
        specs: Optional[DroneSpecs] = None
    ):
        super().__init__(name="B3_PostHoc_Rejection")
        self.city_map = city_map
        self.reservation_table = reservation_table
        self.wind_field = wind_field
        self.specs = specs if specs is not None else DroneSpecs()
        self.energy_model = energy_model if energy_model is not None else EnergyModel(self.specs)
        # Internal B2 planner instance
        self.b2_planner = B2WindUnawareBatchPlanner(
            city_map=city_map,
            reservation_table=reservation_table,
            specs=specs
        )

    def plan_batch(
        self,
        requests: List[DeliveryRequest],
        current_time: float
    ) -> List[PlanResult]:
        t0 = time.perf_counter()
        # 1. Run B2 optimistic planning
        preliminary_results = self.b2_planner.plan_batch(requests, current_time)
        results: List[PlanResult] = []

        # Map request id to req object
        req_dict = {r.request_id: r for r in requests}

        # 2. Post-hoc Physical Feasibility Validation
        for plan in preliminary_results:
            req = req_dict[plan.request_id]
            if not plan.success:
                results.append(plan)
                continue

            # Simulate physical flight along trajectory with actual wind
            feasible, actual_arr_t, actual_energy, reason = self._simulate_real_flight(
                plan.trajectory,
                req.payload_kg,
                req.release_time
            )

            if feasible and actual_arr_t <= req.deadline and actual_energy <= self.specs.usable_joules:
                # Feasible! Update with real metrics
                plan.flight_time_s = actual_arr_t - req.release_time
                plan.arrival_time = actual_arr_t
                plan.energy_joules = actual_energy
                results.append(plan)
            else:
                # Infeasible under real wind! Rollback reservation table and mark REJECTED
                self.reservation_table.cancel_reservation(plan.drone_id)
                rejection_reason = "BATTERY_EXHAUSTED" if actual_energy > self.specs.usable_joules else "DEADLINE_EXCEEDED"
                if not feasible:
                    rejection_reason = "UNFEASIBLE_WIND"
                
                results.append(PlanResult(
                    request_id=plan.request_id,
                    drone_id=plan.drone_id,
                    success=False,
                    status_reason=rejection_reason,
                    flight_time_s=actual_arr_t - req.release_time,
                    energy_joules=actual_energy,
                    arrival_time=actual_arr_t,
                    deadline=req.deadline,
                    battery_depleted_en_route=(actual_energy > self.specs.usable_joules),
                    late_delivery=(actual_arr_t > req.deadline)
                ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        for r in results:
            r.planning_time_ms = elapsed_ms / max(1, len(results))

        return results

    def _simulate_real_flight(
        self,
        trajectory: List[Tuple[int, float, float]],
        payload_kg: float,
        start_time: float
    ) -> Tuple[bool, float, float, str]:
        """
        Simulates the physical aerodynamics along the planned waypoint sequence
        under true wind vectors.
        """
        curr_t = start_time
        curr_energy = 0.0

        for i in range(len(trajectory) - 1):
            u = trajectory[i][0]
            v = trajectory[i + 1][0]
            
            # Hover/Wait time spent at u if any
            nominal_wait = trajectory[i][2] - trajectory[i][1]
            if nominal_wait > 0:
                curr_t += nominal_wait
                curr_energy += self.energy_model.compute_hover_power(payload_kg) * nominal_wait

            # Traversal along edge (u, v)
            edge_data = self.city_map.graph.get_edge_data(u, v)
            if not edge_data:
                # Re-check straight line if direct graph edge missing
                x1, y1 = self.city_map.get_pos(u)
                x2, y2 = self.city_map.get_pos(v)
                dist = math.hypot(x2 - x1, y2 - y1)
                u_vec = ((x2 - x1) / dist, (y2 - y1) / dist)
            else:
                dist = edge_data['length']
                u_vec = edge_data['u_vec']

            x1, y1 = self.city_map.get_pos(u)
            x2, y2 = self.city_map.get_pos(v)
            mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            wind_vec = self.wind_field.get_wind(mid_x, mid_y, curr_t)

            feasible, trav_time, edge_e = self.energy_model.compute_edge_traversal(
                distance_m=dist,
                u_vector=u_vec,
                wind_vector=wind_vec,
                payload_kg=payload_kg
            )
            if not feasible:
                return (False, curr_t, curr_energy, "WIND_TOO_STRONG")

            curr_t += trav_time
            curr_energy += edge_e

            if curr_energy > self.specs.usable_joules:
                return (False, curr_t, curr_energy, "BATTERY_EXHAUSTED")

        return (True, curr_t, curr_energy, "OK")

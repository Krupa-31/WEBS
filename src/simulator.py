"""
Execution Simulator and Physical State Evaluator.
Executes batch planning, simulates physical execution in time-varying wind fields,
and collects comprehensive comparative metrics:
1. Admission Ratio
2. On-Time Ratio
3. Successful Delivery Ratio (On-Time & Battery Feasible)
4. Energy per Delivery (kJ)
5. Safety Separation Violations (Audited dynamically)
6. Planning Time per Request (ms)
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.config import DroneSpecs, AirspaceConfig
from src.map_module import CityAirspaceMap
from src.wind_energy import WindField, EnergyModel
from src.reservation_table import SpaceTimeReservationTable
from src.backend.database import log_telemetry_batch
from src.request_generator import DeliveryRequest
from src.planners.base_planner import BasePlanner, PlanResult


@dataclass
class SimulationMetrics:
    planner_name: str
    wind_speed: float
    total_requests: int
    admitted_requests: int
    completed_deliveries: int
    late_deliveries: int
    battery_exhaustions: int
    safety_violations: int
    admission_ratio: float
    on_time_ratio: float
    success_ratio: float
    avg_energy_kj: float
    avg_flight_time_s: float
    avg_planning_time_ms: float
    detailed_results: List[PlanResult] = field(default_factory=list)


class DeliverySimulator:
    def __init__(
        self,
        city_map: CityAirspaceMap,
        wind_field: WindField,
        specs: Optional[DroneSpecs] = None,
        airspace_cfg: Optional[AirspaceConfig] = None
    ):
        self.city_map = city_map
        self.wind_field = wind_field
        self.specs = specs if specs is not None else DroneSpecs()
        self.airspace_cfg = airspace_cfg if airspace_cfg is not None else AirspaceConfig()
        self.energy_model = EnergyModel(self.specs)
        # List to collect telemetry records for batch logging
        self.telemetry_records: List[Dict[str, Any]] = []

    def run_experiment(
        self,
        planner_cls,
        requests: List[DeliveryRequest],
        batch_size: int = 10,
        planner_kwargs: Optional[dict] = None
    ) -> SimulationMetrics:
        """
        Runs an experiment with given planner class and batch size.
        """
        reservation_table = SpaceTimeReservationTable(time_buffer=2.0)
        kwargs = planner_kwargs or {}

        # Instantiate planner with fresh reservation table
        if "wind_field" in planner_cls.__init__.__code__.co_varnames:
            planner = planner_cls(
                city_map=self.city_map,
                reservation_table=reservation_table,
                wind_field=self.wind_field,
                energy_model=self.energy_model,
                specs=self.specs,
                **kwargs
            )
        else:
            planner = planner_cls(
                city_map=self.city_map,
                reservation_table=reservation_table,
                specs=self.specs,
                **kwargs
            )

        all_results: List[PlanResult] = []
        req_dict = {r.request_id: r for r in requests}

        # Divide requests into batches
        # Sort incoming requests by release time first
        sorted_requests = sorted(requests, key=lambda r: r.release_time)
        num_batches = int(math.ceil(len(sorted_requests) / batch_size))

        for b_idx in range(num_batches):
            batch = sorted_requests[b_idx * batch_size : (b_idx + 1) * batch_size]
            current_sim_time = batch[0].release_time
            batch_results = planner.plan_batch(batch, current_sim_time)
            all_results.extend(batch_results)

        # Physical execution check for all accepted flights
        completed = 0
        late = 0
        exhausted = 0
        energies_kj = []
        flight_times = []
        planning_times = []

        # For wind-unaware planners (B1, B2), the plan was accepted optimistically.
        # Now we evaluate what actually happens when the drone flies in the real wind!
        is_wind_aware = (planner.name == "WEBS" or planner.name == "B3_PostHoc_Rejection")

        for plan in all_results:
            planning_times.append(plan.planning_time_ms)
            if not plan.success:
                continue

            req = req_dict[plan.request_id]

            if is_wind_aware:
                # WEBS and B3 already computed exact physical wind ground speeds and energies
                actual_arr_t = plan.arrival_time
                actual_energy_j = plan.energy_joules
                is_feasible = True
            else:
                # B1 or B2: Fly through real wind field!
                is_feasible, actual_arr_t, actual_energy_j = self._evaluate_actual_flight(
                    plan.trajectory, req.payload_kg, req.release_time
                )

            # Check outcome
            if not is_feasible or actual_energy_j > self.specs.usable_joules:
                # Battery ran out (or edge infeasible in real wind) before arrival
                exhausted += 1
                plan.battery_depleted_en_route = True
            else:
                # Drone physically reached the destination
                completed += 1
                energies_kj.append(actual_energy_j / 1000.0)
                flight_times.append(actual_arr_t - req.release_time)
                if actual_arr_t > req.deadline:
                    late += 1
                    plan.late_delivery = True

            # Record telemetry for this delivery (simplified)
            record = {
                "timestamp_s": actual_arr_t,
                "drone_id": f"UAV-{plan.drone_id:02d}" if hasattr(plan, 'drone_id') else "UNKNOWN",
                "request_id": plan.request_id,
                "lat": self.city_map.get_pos(req.destination)[0],
                "lon": self.city_map.get_pos(req.destination)[1],
                "alt_agl_m": 0.0,
                "battery_volts": 0.0,
                "battery_soc_pct": max(0.0, 100.0 - (actual_energy_j / self.specs.usable_joules) * 100.0),
                "esc_current_amps": 0.0,
                "ground_speed_mps": 0.0,
                "wind_speed_mps": self.wind_field.base_speed,
                "wind_dir_deg": self.wind_field.direction_deg,  # FIXED: was base_dir_deg
            }
            self.telemetry_records.append(record)

        # Audit safety violations
        safety_violations = reservation_table.verify_safety_violations()

        total_req = len(requests)
        admitted = sum(1 for p in all_results if p.success)
        admission_ratio = admitted / total_req if total_req > 0 else 0.0
        on_time_ratio = (completed - late) / admitted if admitted > 0 else 0.0
        success_ratio = (completed - late) / total_req if total_req > 0 else 0.0
        avg_energy = float(np.mean(energies_kj)) if energies_kj else 0.0
        avg_f_time = float(np.mean(flight_times)) if flight_times else 0.0
        avg_p_time = float(np.mean(planning_times)) if planning_times else 0.0

        return SimulationMetrics(
            planner_name=planner.name,
            wind_speed=self.wind_field.base_speed,
            total_requests=total_req,
            admitted_requests=admitted,
            completed_deliveries=completed,
            late_deliveries=late,
            battery_exhaustions=exhausted,
            safety_violations=safety_violations,
            admission_ratio=admission_ratio,
            on_time_ratio=on_time_ratio,
            success_ratio=success_ratio,
            avg_energy_kj=avg_energy,
            avg_flight_time_s=avg_f_time,
            avg_planning_time_ms=avg_p_time,
            detailed_results=all_results
        )

    def _evaluate_actual_flight(
        self,
        trajectory: List[Tuple[int, float, float]],
        payload_kg: float,
        start_time: float
    ) -> Tuple[bool, float, float]:
        """
        Calculates the real flight duration and battery energy consumed
        by flying along trajectory nodes through the active wind field.
        """
        curr_t = start_time
        curr_energy = 0.0

        for i in range(len(trajectory) - 1):
            u = trajectory[i][0]
            v = trajectory[i + 1][0]

            nominal_wait = max(0.0, trajectory[i][2] - trajectory[i][1])
            if nominal_wait > 0:
                curr_t += nominal_wait
                curr_energy += self.energy_model.compute_hover_power(payload_kg) * nominal_wait

            edge_data = self.city_map.graph.get_edge_data(u, v)
            if not edge_data:
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
                return (False, curr_t, curr_energy)

            curr_t += trav_time
            curr_energy += edge_e

        return (True, curr_t, curr_energy)
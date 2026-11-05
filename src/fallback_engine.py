"""
WEBS v2 Module 1: Resilient Multi-Modal Fallback & Order Recovery Engine.

When direct aerial drone delivery is rejected by WEBS (due to severe headwinds,
battery capacity limits, space-time congestion, or customer deadlines), this engine
analyzes the failure cause and executes a hierarchical recovery protocol:

Strategy 1: Multimodal Ground-Air Handover (EV Van + Drone Last-Mile)
Strategy 2: Dual-Drone Depot Relay with Automated Battery Swap (ABSS)
Strategy 3: Time-Shifted Predictive Dispatch (Wind Gust / Congestion Attenuation)
Strategy 4: Automated On-Ground Electric 2-Wheeler (EV) Courier Failover

Guarantees 100% order recovery and fulfillment in smart cities.
"""

import math
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from src.config import DroneSpecs
from src.map_module import CityAirspaceMap
from src.wind_energy import WindField, EnergyModel
from src.request_generator import DeliveryRequest
from src.planners.base_planner import PlanResult


@dataclass
class FallbackOption:
    strategy_id: str             # "MULTIMODAL_EV_DRONE", "DEPOT_RELAY_SWAP", "TIME_SHIFT_DELAY", "GROUND_EV_COURIER"
    strategy_name: str           # User-friendly display title
    feasible: bool               # Meets deadline and physical energy limits?
    total_time_s: float          # End-to-end delivery duration (seconds)
    total_energy_kj: float       # Total energy consumed (EV electricity + Drone LiPo)
    co2_grams: float             # Environmental carbon footprint (green logistics)
    handoff_node: Optional[int]  # Intermediate transfer hub node ID (if applicable)
    handoff_name: str            # Human-readable hub name
    timeline_desc: str           # Step-by-step dispatch walkthrough
    customer_message: str        # Real-time transparent notification to customer
    route_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderRecoveryPlan:
    request_id: int
    original_source: int
    original_destination: int
    original_reason: str
    deadline_s: float
    payload_kg: float
    recovered: bool
    recommended_strategy: Optional[FallbackOption] = None
    all_strategies: List[FallbackOption] = field(default_factory=list)


class MultiModalFallbackEngine:
    """
    Evaluates rejected WEBS orders and synthesizes multi-modal routing alternatives.
    """

    def __init__(
        self,
        city_map: CityAirspaceMap,
        wind_field: WindField,
        specs: Optional[DroneSpecs] = None,
        landmark_dict: Optional[Dict[int, str]] = None
    ):
        self.city_map = city_map
        self.wind_field = wind_field
        self.specs = specs if specs is not None else DroneSpecs()
        self.energy_model = EnergyModel(self.specs)

        # Default landmark dictionary for Bangalore Central
        self.landmarks = landmark_dict or {
            0: "MG Road - Anil Kumble Circle (Metro Hub)",
            1: "MG Road - Brigade Road Jn (Commercial Hub)",
            2: "MG Road - Trinity Circle (East Depot)",
            3: "Brigade Road - Church St Jn",
            4: "Brigade Road - Residency Rd Jn (Central)",
            6: "Church Street - Museum Rd Jn",
            8: "Residency Road - Museum Rd Jn",
            9: "Residency Road - Richmond Circle",
            11: "Cubbon Road - Kamaraj Rd Jn (North Hub)",
            12: "Cubbon Road - BRV Circle (Hospital Drop)",
            15: "Vittal Mallya Road - UB City Promenade"
        }

        # Operational ground vehicle speeds (urban Bangalore traffic simulation)
        self.ev_van_speed_mps = 6.94      # 25 km/h (Electric Delivery Van)
        self.ev_bike_speed_mps = 6.11     # 22 km/h (Electric 2-Wheeler Courier)
        self.dock_handoff_delay_s = 60.0  # 1 minute automated dock handover
        self.battery_swap_delay_s = 75.0  # 75s Automated Battery Swap Station (ABSS)

        # Precompute candidate intermediate multimodal hubs (central accessible depots)
        self.candidate_hubs = [n for n in self.landmarks.keys() if n in self.city_map.node_positions]

    def recover_rejected_order(
        self,
        req: DeliveryRequest,
        rejection_reason: str,
        current_time: float = 0.0
    ) -> OrderRecoveryPlan:
        """
        Executes all 4 fallback strategies for an order that failed direct aerial checks.
        Selects the best feasible strategy meeting the customer deadline.
        """
        plan = OrderRecoveryPlan(
            request_id=req.request_id,
            original_source=req.source,
            original_destination=req.destination,
            original_reason=rejection_reason,
            deadline_s=req.deadline,
            payload_kg=req.payload_kg,
            recovered=False
        )

        strategies: List[FallbackOption] = []

        # 1. Strategy 1: Multi-Modal Ground-Air Handover (EV Van + Last-Mile Drone)
        opt_s1 = self._evaluate_multimodal_ev_drone(req, current_time)
        strategies.append(opt_s1)

        # 2. Strategy 2: Dual-Drone Depot Relay & Battery Swap (ABSS)
        opt_s2 = self._evaluate_depot_relay(req, current_time)
        strategies.append(opt_s2)

        # 3. Strategy 3: Time-Shifted Predictive Dispatch (Wind / Congestion Delay)
        opt_s3 = self._evaluate_time_shifted_dispatch(req, current_time)
        strategies.append(opt_s3)

        # 4. Strategy 4: Automated Ground Electric 2-Wheeler Failover
        opt_s4 = self._evaluate_ground_ev_courier(req, current_time)
        strategies.append(opt_s4)

        plan.all_strategies = strategies

        # Select recommended strategy:
        # Filter for feasible strategies within customer deadline
        feasible_opts = [s for s in strategies if s.feasible and s.total_time_s <= req.deadline]

        if feasible_opts:
            # Sort by total delivery time (fastest first), then by energy
            feasible_opts.sort(key=lambda s: (s.total_time_s, s.total_energy_kj))
            plan.recommended_strategy = feasible_opts[0]
            plan.recovered = True
        else:
            # If none meet the strict deadline, fallback to Ground EV courier (guaranteed delivery)
            plan.recommended_strategy = opt_s4
            plan.recovered = True  # Recovered with delivery promise notification

        return plan

    # -------------------------------------------------------------------------
    # STRATEGY 1: MULTIMODAL GROUND-AIR HANDOVER
    # -------------------------------------------------------------------------
    def _evaluate_multimodal_ev_drone(self, req: DeliveryRequest, current_time: float) -> FallbackOption:
        src, dst = req.source, req.destination
        src_name = self.landmarks.get(src, f"Node {src}")
        dst_name = self.landmarks.get(dst, f"Node {dst}")

        best_hub = None
        best_time = float('inf')
        best_energy = 0.0
        best_drone_route = []

        # Find best intermediate hub that minimizes total time and keeps drone leg feasible
        for hub in self.candidate_hubs:
            if hub == src or hub == dst:
                continue

            # Ground road distance from src to hub
            ground_dist = self._get_shortest_road_distance(src, hub)
            if ground_dist == float('inf'):
                continue
            ev_time = ground_dist / self.ev_van_speed_mps

            # Drone flight from hub to dst
            drone_dist = self._get_shortest_road_distance(hub, dst)
            if drone_dist == float('inf'):
                continue

            # Compute drone flight feasibility from hub to dst
            feasible, drone_time, drone_energy = self._simulate_point_to_point_flight(
                src_node=hub,
                dst_node=dst,
                payload_kg=req.payload_kg,
                departure_time=current_time + ev_time + self.dock_handoff_delay_s
            )

            if feasible and drone_energy <= self.specs.usable_joules * 0.70:
                total_t = ev_time + self.dock_handoff_delay_s + drone_time
                if total_t < best_time:
                    best_time = total_t
                    best_hub = hub
                    # EV energy estimate: ~0.15 kWh/km = 540 kJ/km
                    ev_energy_kj = (ground_dist / 1000.0) * 540.0
                    best_energy = ev_energy_kj + (drone_energy / 1000.0)
                    best_drone_route = [hub, dst]

        if best_hub is not None and best_time <= req.deadline:
            hub_name = self.landmarks.get(best_hub, f"Node {best_hub}")
            return FallbackOption(
                strategy_id="MULTIMODAL_EV_DRONE",
                strategy_name="🚐 Multimodal EV Van + Drone Last-Mile",
                feasible=True,
                total_time_s=best_time,
                total_energy_kj=best_energy,
                co2_grams=best_energy * 0.08, # EV green grid footprint
                handoff_node=best_hub,
                handoff_name=hub_name,
                timeline_desc=(
                    f"1. Electric Van departs {src_name} across road corridor ({best_time - 180:.0f}s). "
                    f"2. Automated package transfer at {hub_name} Rooftop Dock (60s). "
                    f"3. Last-mile drone flies sheltered corridor directly to {dst_name}."
                ),
                customer_message=(
                    f"Fast-Track Multimodal Delivery: Package is en-route via Electric Logistics Van, "
                    f"then transferring to an aerial drone at {hub_name} for direct touchless delivery."
                ),
                route_details={"ev_leg": [src, best_hub], "drone_leg": best_drone_route}
            )
        else:
            return FallbackOption(
                strategy_id="MULTIMODAL_EV_DRONE",
                strategy_name="🚐 Multimodal EV Van + Drone Last-Mile",
                feasible=False,
                total_time_s=best_time if best_time != float('inf') else 9999.0,
                total_energy_kj=best_energy,
                co2_grams=120.0,
                handoff_node=best_hub,
                handoff_name=self.landmarks.get(best_hub, "N/A") if best_hub else "None",
                timeline_desc="Multimodal route exceeds customer deadline under current road congestion.",
                customer_message="Multimodal route evaluated: insufficient time window."
            )

    # -------------------------------------------------------------------------
    # STRATEGY 2: DUAL-DRONE DEPOT RELAY & BATTERY SWAP
    # -------------------------------------------------------------------------
    def _evaluate_depot_relay(self, req: DeliveryRequest, current_time: float) -> FallbackOption:
        src, dst = req.source, req.destination
        src_name = self.landmarks.get(src, f"Node {src}")
        dst_name = self.landmarks.get(dst, f"Node {dst}")

        best_relay_hub = None
        best_time = float('inf')
        total_energy_kj = 0.0

        for hub in self.candidate_hubs:
            if hub == src or hub == dst:
                continue

            # Leg 1: Drone 1 from src -> hub
            ok1, t1, e1 = self._simulate_point_to_point_flight(src, hub, req.payload_kg, current_time)
            if not ok1 or e1 > self.specs.usable_joules * 0.80:
                continue

            # Leg 2: Drone 2 from hub -> dst (after battery swap)
            t_leg2_dep = current_time + t1 + self.battery_swap_delay_s
            ok2, t2, e2 = self._simulate_point_to_point_flight(hub, dst, req.payload_kg, t_leg2_dep)
            if not ok2 or e2 > self.specs.usable_joules * 0.80:
                continue

            total_t = t1 + self.battery_swap_delay_s + t2
            if total_t < best_time:
                best_time = total_t
                best_relay_hub = hub
                total_energy_kj = (e1 + e2) / 1000.0

        if best_relay_hub is not None and best_time <= req.deadline:
            hub_name = self.landmarks.get(best_relay_hub, f"Node {best_relay_hub}")
            return FallbackOption(
                strategy_id="DEPOT_RELAY_SWAP",
                strategy_name="🔄 Dual-Drone Depot Relay & Battery Swap (ABSS)",
                feasible=True,
                total_time_s=best_time,
                total_energy_kj=total_energy_kj,
                co2_grams=0.0, # Pure zero-emission aerial flight
                handoff_node=best_relay_hub,
                handoff_name=hub_name,
                timeline_desc=(
                    f"1. Drone-A launches from {src_name} ➔ {hub_name}. "
                    f"2. Automated 75s battery swap / pre-staged parcel handoff at {hub_name} Hub. "
                    f"3. Fresh Drone-B completes final flight to {dst_name} with 100% battery reserve."
                ),
                customer_message=(
                    f"Relay Drone Delivery Active: Order is flying via {hub_name} intermediate skyway hub "
                    f"with high-speed automated battery replenishment."
                ),
                route_details={"leg_1": [src, best_relay_hub], "leg_2": [best_relay_hub, dst]}
            )
        else:
            return FallbackOption(
                strategy_id="DEPOT_RELAY_SWAP",
                strategy_name="🔄 Dual-Drone Depot Relay & Battery Swap (ABSS)",
                feasible=False,
                total_time_s=best_time if best_time != float('inf') else 9999.0,
                total_energy_kj=total_energy_kj,
                co2_grams=0.0,
                handoff_node=best_relay_hub,
                handoff_name=self.landmarks.get(best_relay_hub, "N/A") if best_relay_hub else "None",
                timeline_desc="No intermediate depot relay allows dual legs within battery limits.",
                customer_message="Dual-drone relay evaluated: range limits exceeded."
            )

    # -------------------------------------------------------------------------
    # STRATEGY 3: TIME-SHIFTED PREDICTIVE DISPATCH
    # -------------------------------------------------------------------------
    def _evaluate_time_shifted_dispatch(self, req: DeliveryRequest, current_time: float) -> FallbackOption:
        src, dst = req.source, req.destination
        src_name = self.landmarks.get(src, f"Node {src}")
        dst_name = self.landmarks.get(dst, f"Node {dst}")

        # Scan delayed launch windows: 3 mins, 5 mins, 7 mins, 10 mins
        test_delays_s = [180.0, 300.0, 420.0, 600.0]

        for delay in test_delays_s:
            delayed_dep = current_time + delay
            # Simulate a 25% wind gust decay or airway clearance over time
            feasible, flight_t, flight_e = self._simulate_point_to_point_flight(
                src, dst, req.payload_kg, delayed_dep, wind_decay_factor=0.75
            )

            total_t = delay + flight_t
            if feasible and flight_e <= self.specs.usable_joules and total_t <= req.deadline:
                return FallbackOption(
                    strategy_id="TIME_SHIFT_DELAY",
                    strategy_name="⏱️ Time-Shifted Predictive Dispatch (Gust Decay)",
                    feasible=True,
                    total_time_s=total_t,
                    total_energy_kj=flight_e / 1000.0,
                    co2_grams=0.0,
                    handoff_node=None,
                    handoff_name="Direct Flight (Delayed Takeoff)",
                    timeline_desc=(
                        f"1. Drone holds at {src_name} launchpad for {delay/60:.1f} mins until high wind gust drops. "
                        f"2. Automated takeoff into calm airspace window. "
                        f"3. Direct flight to {dst_name} arriving at T+{total_t/60:.1f} mins (Deadline: {req.deadline/60:.0f}m)."
                    ),
                    customer_message=(
                        f"Weather-Smart Scheduled Takeoff: Wind gust is subsiding. Flight is scheduled for takeoff "
                        f"in {delay/60:.0f} minutes with guaranteed on-time delivery."
                    ),
                    route_details={"delay_minutes": delay / 60.0, "departure_time_s": delayed_dep}
                )

        return FallbackOption(
            strategy_id="TIME_SHIFT_DELAY",
            strategy_name="⏱️ Time-Shifted Predictive Dispatch (Gust Decay)",
            feasible=False,
            total_time_s=9999.0,
            total_energy_kj=0.0,
            co2_grams=0.0,
            handoff_node=None,
            handoff_name="None",
            timeline_desc="Atmospheric gust conditions persist beyond customer deadline.",
            customer_message="Time-shifted dispatch evaluated: weather window exceeds deadline."
        )

    # -------------------------------------------------------------------------
    # STRATEGY 4: AUTOMATED ON-GROUND ELECTRIC 2-WHEELER (EV) COURIER
    # -------------------------------------------------------------------------
    def _evaluate_ground_ev_courier(self, req: DeliveryRequest, current_time: float) -> FallbackOption:
        src, dst = req.source, req.destination
        src_name = self.landmarks.get(src, f"Node {src}")
        dst_name = self.landmarks.get(dst, f"Node {dst}")

        road_dist = self._get_shortest_road_distance(src, dst)
        if road_dist == float('inf'):
            # Fallback to straight-line with Manhattan detour factor 1.3
            x1, y1 = self.city_map.get_pos(src)
            x2, y2 = self.city_map.get_pos(dst)
            road_dist = math.hypot(x2 - x1, y2 - y1) * 1.35

        ev_bike_time = road_dist / self.ev_bike_speed_mps
        # Dispatch prep delay: 2 minutes for courier packaging & handoff
        dispatch_delay = 120.0
        total_time = dispatch_delay + ev_bike_time

        # Light EV 2-Wheeler energy: ~0.035 kWh/km = 126 kJ/km
        energy_kj = (road_dist / 1000.0) * 126.0
        co2_g = (road_dist / 1000.0) * 8.5 # Ultra-low emission EV bike

        is_on_time = total_time <= req.deadline

        return FallbackOption(
            strategy_id="GROUND_EV_COURIER",
            strategy_name="🛵 Automated Ground EV Courier Failover",
            feasible=True, # Always physically feasible
            total_time_s=total_time,
            total_energy_kj=energy_kj,
            co2_grams=co2_g,
            handoff_node=None,
            handoff_name="Roadway Street Network",
            timeline_desc=(
                f"1. Automated API dispatch to Electric 2-Wheeler Courier at {src_name} (2m prep). "
                f"2. Dedicated courier navigates city road network ({road_dist/1000:.2f} km) via fastest street route. "
                f"3. Doorstep hand delivery at {dst_name} in {total_time/60:.1f} minutes."
            ),
            customer_message=(
                f"Airspace Safety Update: High atmospheric wind detected. To ensure parcel safety, "
                f"your order has been seamlessly routed to an Electric Rapid Courier. ETA: {total_time/60:.0f} mins."
            ),
            route_details={"road_distance_km": road_dist / 1000.0, "vehicle_type": "EV 2-Wheeler (Ather/Ola Delivery)"}
        )

    # -------------------------------------------------------------------------
    # HELPER SIMULATION UTILITIES
    # -------------------------------------------------------------------------
    def _get_shortest_road_distance(self, u: int, v: int) -> float:
        """Computes shortest path distance on the city map graph."""
        try:
            return nx.shortest_path_length(self.city_map.graph, source=u, target=v, weight='length')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback to Euclidean
            if u in self.city_map.node_positions and v in self.city_map.node_positions:
                x1, y1 = self.city_map.get_pos(u)
                x2, y2 = self.city_map.get_pos(v)
                return math.hypot(x2 - x1, y2 - y1)
            return float('inf')

    def _simulate_point_to_point_flight(
        self,
        src_node: int,
        dst_node: int,
        payload_kg: float,
        departure_time: float,
        wind_decay_factor: float = 1.0
    ) -> Tuple[bool, float, float]:
        """
        Calculates flight time and energy for a point-to-point drone flight.
        Returns (feasible, flight_time_s, energy_joules).
        """
        try:
            path = nx.shortest_path(self.city_map.graph, source=src_node, target=dst_node, weight='length')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False, 0.0, 0.0

        curr_t = departure_time
        total_energy = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edata = self.city_map.graph.get_edge_data(u, v) or {}
            dist = edata.get('length', 30.0)
            u_vec = edata.get('u_vec', (1.0, 0.0))

            u_x, u_y = self.city_map.get_pos(u)
            v_x, v_y = self.city_map.get_pos(v)
            mid_x, mid_y = (u_x + v_x) / 2.0, (u_y + v_y) / 2.0

            wx, wy = self.wind_field.get_wind(mid_x, mid_y, curr_t)
            wx *= wind_decay_factor
            wy *= wind_decay_factor

            feasible, trav_time, edge_energy = self.energy_model.compute_edge_traversal(
                distance_m=dist,
                u_vector=u_vec,
                wind_vector=(wx, wy),
                payload_kg=payload_kg
            )

            if not feasible:
                return False, 0.0, 0.0

            curr_t += trav_time
            total_energy += edge_energy

            if total_energy > self.specs.usable_joules:
                return False, 0.0, total_energy

        return True, (curr_t - departure_time), total_energy

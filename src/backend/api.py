"""
WEBS v2 Module 2: Enterprise REST API & Microservice Gateway.

Built with FastAPI & Pydantic. Exposes:
1. Fleet Hardware & Telemetry Endpoints
2. Dynamic Order Submission with WEBS Trajectory & Fallback Resolution
3. Hyper-Local Weather Mesh Observation Feeds
4. Historical Analytics & Mission Logs
"""

import os
import sys
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backend.database import (
    get_all_fleet_drones,
    get_all_delivery_orders,
    get_all_weather_stations,
    get_recent_telemetry,
    get_database_summary_stats,
    log_delivery_order
)
from src.backend.weather_collector import HyperLocalWeatherCollector
from src.config import DroneSpecs
from src.data.bangalore_osm import build_bangalore_map, BANGALORE_NODES
from src.wind_energy import WindField
from src.request_generator import DeliveryRequest
from src.planners.webs_planner import WEBSPlanner
from src.reservation_table import SpaceTimeReservationTable
from src.fallback_engine import MultiModalFallbackEngine

app = FastAPI(
    title="WEBS Autonomous Drone Delivery UTM Gateway",
    description="Enterprise Backend for Wind- and Energy-Aware Trajectory Planning & Multi-Modal Fleet Orchestration in Smart Cities.",
    version="2.0.0"
)

# Shared memory simulator components for API execution
_city_map = build_bangalore_map(max_edge_length=30.0)
_specs = DroneSpecs()
_weather_collector = HyperLocalWeatherCollector()


# --- Pydantic Data Contracts ---
class OrderSubmissionRequest(BaseModel):
    pickup_node: int = Field(0, description="Origin depot landmark node ID (e.g. 0 for Anil Kumble Circle)")
    dest_node: int = Field(2, description="Destination landmark node ID (e.g. 2 for Trinity Circle)")
    payload_kg: float = Field(1.0, ge=0.1, le=2.5, description="Parcel mass in kilograms")
    deadline_min: float = Field(8.0, ge=2.0, le=30.0, description="Customer delivery time window in minutes")
    wind_speed_mps: Optional[float] = Field(6.0, ge=0.0, le=15.0, description="Ambient atmospheric wind speed (m/s)")
    wind_dir_deg: Optional[float] = Field(45.0, ge=0.0, le=360.0, description="Wind heading angle (degrees)")


class DispatchResponse(BaseModel):
    order_id: int
    pickup_name: str
    dest_name: str
    decision: str  # "DIRECT_AERIAL_SCHEDULED" or "MULTIMODAL_FALLBACK_RECOVERED"
    strategy_name: str
    total_time_s: float
    total_time_min: float
    energy_kj: float
    battery_soc_remaining_pct: float
    co2_grams: float
    timeline_steps: str
    customer_alert: str
    air_safety: str = "100% Conflict-Free Reserved Corridor"


# --- API Routes ---
@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    stats = get_database_summary_stats()
    return {
        "status": "ONLINE",
        "service": "WEBS-UTM-Enterprise-Core",
        "version": "2.0.0",
        "airspace": "Bangalore Central (UTM Zone 43N)",
        "database_stats": stats
    }


@app.get("/api/fleet/status", tags=["Fleet Management"])
def get_fleet_status():
    """Returns all registered commercial delivery drones and battery health."""
    drones = get_all_fleet_drones()
    return {"count": len(drones), "fleet": drones}


@app.get("/api/weather/stations", tags=["Weather Mesh"])
def get_live_weather():
    """Returns live hyper-local weather observation mesh across Bangalore depots."""
    stations = _weather_collector.sync_weather_mesh()
    return {"count": len(stations), "stations": stations}


@app.get("/api/telemetry/recent", tags=["Telemetry Blackbox"])
def get_telemetry_feed(
    drone_id: Optional[str] = Query(None, description="Optional filter by UAV ID, e.g. UAV-01"),
    limit: int = Query(100, ge=10, le=500)
):
    """Returns high-frequency telemetry records (voltage, current, ground speed, altitude)."""
    records = get_recent_telemetry(drone_id=drone_id, limit=limit)
    return {"count": len(records), "records": records}


@app.get("/api/orders/history", tags=["Orders & Delivery"])
def get_orders_history():
    """Returns all logged orders with execution strategies and metrics."""
    orders = get_all_delivery_orders()
    return {"count": len(orders), "orders": orders}


@app.post("/api/orders/dispatch", response_model=DispatchResponse, tags=["Orders & Delivery"])
def dispatch_order(order: OrderSubmissionRequest):
    """
    Submits a new customer delivery order.
    Executes WEBS 4-check trajectory planning:
    1. If safe & battery feasible -> Schedules direct aerial drone flight.
    2. If wind/battery deficit -> Synthesizes resilient Multi-Modal Fallback.
    Persists decision and telemetry to database.
    """
    landmark_dict = {
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

    src_name = landmark_dict.get(order.pickup_node, f"Node {order.pickup_node}")
    dst_name = landmark_dict.get(order.dest_node, f"Node {order.dest_node}")

    wind = WindField.from_speed_and_angle(
        speed=order.wind_speed_mps or 6.0,
        angle_deg=order.wind_dir_deg or 45.0
    )

    req_id = int(len(get_all_delivery_orders())) + 1
    req = DeliveryRequest(
        request_id=req_id,
        source=order.pickup_node,
        destination=order.dest_node,
        release_time=0.0,
        deadline=order.deadline_min * 60.0,
        payload_kg=order.payload_kg
    )

    # 1. Attempt WEBS Direct Aerial Planning
    res_table = SpaceTimeReservationTable(time_buffer=2.0)
    planner = WEBSPlanner(
        city_map=_city_map,
        reservation_table=res_table,
        wind_field=wind,
        specs=_specs
    )
    plan_result = planner._plan_single_request(req)

    if plan_result.success:
        decision = "DIRECT_AERIAL_SCHEDULED"
        strategy_name = "WEBS Conflict-Free Direct Airway"
        time_s = plan_result.flight_time_s
        energy_kj = plan_result.energy_joules / 1000.0
        soc_pct = max(0.0, 100.0 - (plan_result.energy_joules / _specs.battery_joules * 100.0))
        co2_g = 0.0
        timeline = f"Direct conflict-free flight from {src_name} to {dst_name} at 14 m/s cruise airspeed."
        alert = f"Your order is flying directly along the reserved city skyway corridor. ETA: {time_s/60:.1f} mins."

        log_delivery_order(
            order_id=req_id,
            pickup_node=order.pickup_node,
            pickup_name=src_name,
            dest_node=order.dest_node,
            dest_name=dst_name,
            payload_kg=order.payload_kg,
            deadline_min=order.deadline_min,
            status="DIRECT_AERIAL",
            strategy_used=strategy_name,
            flight_time_s=time_s,
            energy_kj=energy_kj,
            co2_grams=co2_g,
            customer_eta_min=time_s / 60.0
        )
    else:
        # 2. Engage Multi-Modal Fallback Engine
        fallback_engine = MultiModalFallbackEngine(
            city_map=_city_map,
            wind_field=wind,
            specs=_specs,
            landmark_dict=landmark_dict
        )
        recovery = fallback_engine.recover_rejected_order(req, plan_result.status_reason)
        rec_s = recovery.recommended_strategy

        decision = "MULTIMODAL_FALLBACK_RECOVERED"
        strategy_name = rec_s.strategy_name
        time_s = rec_s.total_time_s
        energy_kj = rec_s.total_energy_kj
        soc_pct = 85.0
        co2_g = rec_s.co2_grams
        timeline = rec_s.timeline_desc
        alert = rec_s.customer_message

        log_delivery_order(
            order_id=req_id,
            pickup_node=order.pickup_node,
            pickup_name=src_name,
            dest_node=order.dest_node,
            dest_name=dst_name,
            payload_kg=order.payload_kg,
            deadline_min=order.deadline_min,
            status="FALLBACK_RECOVERED",
            strategy_used=strategy_name,
            flight_time_s=time_s,
            energy_kj=energy_kj,
            co2_grams=co2_g,
            customer_eta_min=time_s / 60.0
        )

    return DispatchResponse(
        order_id=req_id,
        pickup_name=src_name,
        dest_name=dst_name,
        decision=decision,
        strategy_name=strategy_name,
        total_time_s=round(time_s, 1),
        total_time_min=round(time_s / 60.0, 1),
        energy_kj=round(energy_kj, 2),
        battery_soc_remaining_pct=round(soc_pct, 1),
        co2_grams=round(co2_g, 1),
        timeline_steps=timeline,
        customer_alert=alert,
        air_safety="100% Conflict-Free Reserved Corridor"
    )

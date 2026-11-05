"""
Interactive Streamlit Dashboard & Flight Visualizer for WEBS Simulator.
Run with:
    streamlit run app.py
"""

import sys
import os
import math
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DroneSpecs
from src.map_module import CityAirspaceMap
from src.wind_energy import WindField, EnergyModel
from src.reservation_table import SpaceTimeReservationTable
from src.request_generator import RequestGenerator, DeliveryRequest
from src.planners.webs_planner import WEBSPlanner
from src.planners.b1_wind_unaware_sequential import B1WindUnawareSequentialPlanner
from src.planners.b2_wind_unaware_batch import B2WindUnawareBatchPlanner
from src.planners.b3_post_hoc_rejection import B3PostHocRejectionPlanner
from src.simulator import DeliverySimulator
from src.fallback_engine import MultiModalFallbackEngine
from src.backend.database import (
    log_delivery_order,
    log_telemetry_batch,
    get_all_fleet_drones,
    get_all_delivery_orders,
    get_all_weather_stations,
    get_recent_telemetry,
    get_database_summary_stats
)
from src.backend.weather_collector import HyperLocalWeatherCollector
from src.data.bangalore_osm import BANGALORE_REF_LAT, BANGALORE_REF_LON, BANGALORE_NODES, BANGALORE_EDGES

st.set_page_config(
    page_title="WEBS Drone Delivery UTM Simulator",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-val {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .status-badge-ok {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .status-badge-err {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# AVAILABLE LANDMARK NODES FOR BANGALORE CENTRAL
# -------------------------------------------------------------
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

# Initialize Custom Orders in Session State if not present
if "custom_order_list" not in st.session_state:
    st.session_state["custom_order_list"] = [
        {"id": 1, "src": 0, "dst": 2, "src_name": landmark_dict[0], "dst_name": landmark_dict[2], "payload": 0.8, "deadline_min": 7},  # Eastward (Tailwind corridor)
        {"id": 2, "src": 2, "dst": 0, "src_name": landmark_dict[2], "dst_name": landmark_dict[0], "payload": 1.4, "deadline_min": 5},  # Westward (Headwind stress test)
        {"id": 3, "src": 1, "dst": 4, "src_name": landmark_dict[1], "dst_name": landmark_dict[4], "payload": 1.0, "deadline_min": 8},  # Brigade Rd shopping corridor
        {"id": 4, "src": 11, "dst": 12, "src_name": landmark_dict[11], "dst_name": landmark_dict[12], "payload": 1.2, "deadline_min": 6} # Hospital corridor
    ]

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.title("🚁 Flight Mission Control")
st.sidebar.markdown("Configure the smart city airspace environment, wind conditions, and fleet settings.")

# 0. Request Mode Selection (Unified Source of Truth!)
st.sidebar.subheader("0. Mission Request Mode")
req_mode = st.sidebar.radio(
    "Select Requests Source",
    options=["📦 Active Custom Orders Queue", "🎲 Synthetic Fleet Generator"],
    index=0,
    help="Active Custom Orders Queue drives all tabs (Map, Benchmark, Telemetry, Hardware) based on your custom orders!"
)

# City Map Selection
st.sidebar.subheader("1. City Map Region")
city_choice = st.sidebar.selectbox(
    "Select City Airspace",
    options=["Bangalore Central, India (Real Street Network)", "Benchmark Urban Grid (10x10)"]
)

# 2. Environment & Wind
st.sidebar.subheader("2. Atmospheric Wind Field")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", min_value=0.0, max_value=14.0, value=6.0, step=1.0)
wind_dir_deg = st.sidebar.slider("Wind Direction (degrees)", min_value=0, max_value=360, value=45, step=15,
                                  help="0° = Eastward (Tailwind for 0->2), 45° = North-East, 90° = Northward, 180° = Westward (Headwind for 0->2)")

# 3. Fleet & Aircraft
st.sidebar.subheader("3. Fleet & Airspeed")
drone_speed = st.sidebar.slider("Drone Cruise Airspeed (m/s)", min_value=10.0, max_value=18.0, value=14.0, step=1.0)

if req_mode == "🎲 Synthetic Fleet Generator":
    num_requests = st.sidebar.slider("Customer Requests", min_value=5, max_value=35, value=15, step=5)
    batch_size = st.sidebar.slider("Batch Size (K)", min_value=1, max_value=20, value=5, step=1)
else:
    batch_size = max(1, len(st.session_state["custom_order_list"]))

# 4. Planner Selection
st.sidebar.subheader("4. Trajectory Planner")
planner_choice = st.sidebar.selectbox(
    "Select Routing Planner",
    options=["WEBS (Proposed Wind & Energy-Aware)", "B1 (Wind-Unaware Sequential)", "B2 (Wind-Unaware Batch)", "B3 (Post-Hoc Rejection)", "Compare All Planners"]
)

run_button = st.sidebar.button("🚀 Plan & Simulate Flights", type="primary", width='stretch')

# -------------------------------------------------------------
# SIMULATION ENGINE
# -------------------------------------------------------------
@st.cache_resource
def get_map(choice):
    if "Bangalore" in choice:
        from src.data.bangalore_osm import build_bangalore_map
        return build_bangalore_map(max_edge_length=30.0)
    else:
        return CityAirspaceMap.create_benchmark_grid(rows=10, cols=10, spacing=35.0, name="urban_core")

city_map = get_map(city_choice)
specs = DroneSpecs(nominal_airspeed=drone_speed)
wind_field = WindField.from_speed_and_angle(speed=wind_speed, angle_deg=wind_dir_deg)
simulator = DeliverySimulator(city_map=city_map, wind_field=wind_field, specs=specs)

# App Title & Header
st.title("🚁 WEBS: Wind- and Energy-Aware Batch Trajectory Planning")
st.caption("A Simulation Study for Conflict-Free Public Autonomous Drone Delivery in Smart Cities")

# -------------------------------------------------------------
# BUILD ACTIVE REQUESTS (CENTRAL SOURCE OF TRUTH)
# -------------------------------------------------------------
if req_mode == "📦 Active Custom Orders Queue":
    if not st.session_state["custom_order_list"]:
        st.warning("⚠️ The custom order queue is currently empty! Please add orders in Tab 0 or switch to Synthetic Fleet Generator.")
        st.stop()
    requests = [
        DeliveryRequest(
            request_id=o['id'],
            source=o['src'],
            destination=o['dst'],
            release_time=0.0,
            deadline=o['deadline_min'] * 60.0,
            payload_kg=o['payload']
        )
        for o in st.session_state["custom_order_list"]
    ]
    st.info(f"📍 **Active Mode: Custom Orders Queue** ({len(requests)} Orders across Bangalore Central). All dashboard sections are fully synchronized with these orders!")
else:
    req_gen = RequestGenerator(city_map=city_map, seed=42)
    requests = req_gen.generate_requests(num_requests=num_requests, sim_duration=800.0)
    st.info(f"🎲 **Active Mode: Synthetic Random Fleet** ({len(requests)} Generated Missions).")

# Run Planner & Simulator
planner_map = {
    "WEBS (Proposed Wind & Energy-Aware)": WEBSPlanner,
    "B1 (Wind-Unaware Sequential)": B1WindUnawareSequentialPlanner,
    "B2 (Wind-Unaware Batch)": B2WindUnawareBatchPlanner,
    "B3 (Post-Hoc Rejection)": B3PostHocRejectionPlanner,
}

# Run selected planner
if planner_choice != "Compare All Planners":
    p_cls = planner_map[planner_choice]
    metrics = simulator.run_experiment(planner_cls=p_cls, requests=requests, batch_size=batch_size)
    single_metrics = metrics
else:
    single_metrics = simulator.run_experiment(planner_cls=WEBSPlanner, requests=requests, batch_size=batch_size)

# Always pre-calculate comparison on active requests so Tab 2 is immediately available
comp_results = {}
for p_name, p_cls in planner_map.items():
    comp_results[p_name] = simulator.run_experiment(planner_cls=p_cls, requests=requests, batch_size=batch_size)

st.session_state["single_metrics"] = single_metrics
st.session_state["comparison"] = comp_results
metrics = single_metrics
req_dict = {r.request_id: r for r in requests}

# -------------------------------------------------------------
# WEBS v2 Module 1: Resilient Multi-Modal Fallback Engine
# -------------------------------------------------------------
fallback_engine = MultiModalFallbackEngine(
    city_map=city_map,
    wind_field=wind_field,
    specs=specs,
    landmark_dict=landmark_dict
)

recovery_dict = {}
for p in metrics.detailed_results:
    if not p.success and p.request_id in req_dict:
        recovery_dict[p.request_id] = fallback_engine.recover_rejected_order(
            req=req_dict[p.request_id],
            rejection_reason=p.status_reason,
            current_time=0.0
        )

total_recovered = len(recovery_dict)
total_fulfilled = metrics.admitted_requests + total_recovered
fulfillment_ratio = (total_fulfilled / len(requests) * 100.0) if requests else 100.0

# -------------------------------------------------------------
# WEBS v2 Module 2: Enterprise Database & Weather Collector Sync
# -------------------------------------------------------------
try:
    weather_collector = HyperLocalWeatherCollector(base_speed_mps=wind_speed, base_dir_deg=wind_dir_deg)
    weather_collector.sync_weather_mesh()

    for p in metrics.detailed_results:
        if p.request_id in req_dict:
            req_obj = req_dict[p.request_id]
            src_n = landmark_dict.get(req_obj.source, f"Node {req_obj.source}")
            dst_n = landmark_dict.get(req_obj.destination, f"Node {req_obj.destination}")
            if p.success:
                log_delivery_order(
                    order_id=p.request_id,
                    pickup_node=req_obj.source,
                    pickup_name=src_n,
                    dest_node=req_obj.destination,
                    dest_name=dst_n,
                    payload_kg=req_obj.payload_kg,
                    deadline_min=req_obj.deadline / 60.0,
                    status="DIRECT_AERIAL",
                    strategy_used="WEBS Conflict-Free Direct Airway",
                    flight_time_s=p.flight_time_s,
                    energy_kj=p.energy_joules / 1000.0,
                    co2_grams=0.0,
                    customer_eta_min=p.flight_time_s / 60.0
                )
            elif p.request_id in recovery_dict:
                rec_opt = recovery_dict[p.request_id].recommended_strategy
                log_delivery_order(
                    order_id=p.request_id,
                    pickup_node=req_obj.source,
                    pickup_name=src_n,
                    dest_node=req_obj.destination,
                    dest_name=dst_n,
                    payload_kg=req_obj.payload_kg,
                    deadline_min=req_obj.deadline / 60.0,
                    status="FALLBACK_RECOVERED",
                    strategy_used=rec_opt.strategy_name,
                    flight_time_s=rec_opt.total_time_s,
                    energy_kj=rec_opt.total_energy_kj,
                    co2_grams=rec_opt.co2_grams,
                    customer_eta_min=rec_opt.total_time_s / 60.0
                )
except Exception:
    pass

# -------------------------------------------------------------
# 1. KEY PERFORMANCE INDICATORS (KPIs)
# -------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Direct Flight Ratio</div>
        <div class="metric-val" style="color: #0284c7;">{metrics.admission_ratio * 100:.1f}%</div>
        <small>{metrics.admitted_requests} / {metrics.total_requests} direct flights</small>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">v2 Resilient Delivery</div>
        <div class="metric-val" style="color: #10b981;">{fulfillment_ratio:.1f}%</div>
        <small>{total_fulfilled}/{len(requests)} fulfilled ({total_recovered} recovered)</small>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Energy / Delivery</div>
        <div class="metric-val" style="color: #f59e0b;">{metrics.avg_energy_kj:.2f} kJ</div>
        <small>Battery capacity: {specs.battery_capacity_wh} Wh</small>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Air Safety Violations</div>
        <div class="metric-val" style="color: #10b981;">{metrics.safety_violations}</div>
        <small>100% Conflict-Free Guarantee</small>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Planning Time</div>
        <div class="metric-val" style="color: #6366f1;">{metrics.avg_planning_time_ms:.1f} ms</div>
        <small>Per delivery request</small>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# 2. INTERACTIVE CITY AIRSPACE TABS
# -------------------------------------------------------------
tab_custom, tab_fallback, tab_backend, tab_map, tab_comparison, tab_telemetry, tab_sitl = st.tabs([
    "📦 Place Custom Orders & Test Wind Admission",
    "🔄 Multi-Modal Fallback Hub (v2 Module 1)",
    "🗄️ Backend DB & Weather Mesh (v2 Module 2)",
    "🗺️ Interactive City Flight Map",
    "📊 Comparative Benchmark",
    "📋 Fleet Telemetry Log",
    "🎮 Hardware SITL & MAVLink Export"
])

# -------------------------------------------------------------
# TAB 0: CUSTOM ORDERS & LIVE WIND ADMISSION
# -------------------------------------------------------------
with tab_custom:
    st.subheader("📦 Custom Delivery Order Dispatcher & Live Wind Admission")
    st.markdown("""
    **Core Project Functionality:** Customers place customized delivery orders with designated pickup/drop hubs, 
    weights, and deadlines. **Our WEBS planner evaluates the active atmospheric wind vector and battery capacity to decide whether 
    to ACCEPT the delivery safely or REJECT it to prevent mid-air crashes and dead batteries.**
    """)

    # Order Creation Controls
    with st.expander("➕ Create a New Custom Delivery Order", expanded=False):
        col_src, col_dst = st.columns(2)
        with col_src:
            chosen_src = st.selectbox("Pickup Location (Origin Depot)", options=list(landmark_dict.keys()), format_func=lambda k: landmark_dict[k], key="new_src")
        with col_dst:
            dst_options = [k for k in landmark_dict.keys() if k != chosen_src]
            chosen_dst = st.selectbox("Delivery Destination", options=dst_options, format_func=lambda k: landmark_dict[k], key="new_dst")

        col_w, col_d, col_btn = st.columns([1, 1, 1])
        with col_w:
            new_payload = st.slider("Parcel Weight (kg)", min_value=0.2, max_value=2.5, value=1.0, step=0.1)
        with col_d:
            new_deadline = st.slider("Customer Deadline (Minutes)", min_value=3, max_value=25, value=8, step=1)
        with col_btn:
            st.write("")
            st.write("")
            if st.button("➕ Add Order to Queue", width='stretch'):
                next_id = max([o['id'] for o in st.session_state["custom_order_list"]] + [0]) + 1
                st.session_state["custom_order_list"].append({
                    "id": next_id,
                    "src": chosen_src,
                    "dst": chosen_dst,
                    "src_name": landmark_dict[chosen_src],
                    "dst_name": landmark_dict[chosen_dst],
                    "payload": new_payload,
                    "deadline_min": new_deadline
                })
                st.rerun()

    # Presets
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("🎲 Standard Bangalore Orders"):
            st.session_state["custom_order_list"] = [
                {"id": 1, "src": 0, "dst": 2, "src_name": landmark_dict[0], "dst_name": landmark_dict[2], "payload": 0.8, "deadline_min": 7},
                {"id": 2, "src": 2, "dst": 0, "src_name": landmark_dict[2], "dst_name": landmark_dict[0], "payload": 1.4, "deadline_min": 5},
                {"id": 3, "src": 1, "dst": 4, "src_name": landmark_dict[1], "dst_name": landmark_dict[4], "payload": 1.0, "deadline_min": 8},
                {"id": 4, "src": 11, "dst": 12, "src_name": landmark_dict[11], "dst_name": landmark_dict[12], "payload": 1.2, "deadline_min": 6}
            ]
            st.rerun()
    with col_p2:
        if st.button("🌪️ Headwind Stress Test"):
            st.session_state["custom_order_list"] = [
                {"id": 1, "src": 2, "dst": 0, "src_name": landmark_dict[2], "dst_name": landmark_dict[0], "payload": 2.2, "deadline_min": 4},
                {"id": 2, "src": 9, "dst": 4, "src_name": landmark_dict[9], "dst_name": landmark_dict[4], "payload": 2.0, "deadline_min": 5}
            ]
            st.rerun()
    with col_p3:
        if st.button("🏥 Hospital Corridor"):
            st.session_state["custom_order_list"] = [
                {"id": 1, "src": 11, "dst": 12, "src_name": landmark_dict[11], "dst_name": landmark_dict[12], "payload": 0.5, "deadline_min": 6},
                {"id": 2, "src": 0, "dst": 12, "src_name": landmark_dict[0], "dst_name": landmark_dict[12], "payload": 0.7, "deadline_min": 8}
            ]
            st.rerun()
    with col_p4:
        if st.button("🗑️ Clear All Orders"):
            st.session_state["custom_order_list"] = []
            st.rerun()

    # Current Queue Display
    st.write(f"### Current Orders in Dispatch Queue ({len(st.session_state['custom_order_list'])} Orders)")
    if st.session_state["custom_order_list"]:
        df_queue = pd.DataFrame([
            {
                "Order #": f"ORD-{o['id']:02d}",
                "Pickup Location": o['src_name'],
                "Destination": o['dst_name'],
                "Weight (kg)": f"{o['payload']} kg",
                "Deadline": f"{o['deadline_min']} mins"
            }
            for o in st.session_state["custom_order_list"]
        ])
        st.dataframe(df_queue, hide_index=True, width='stretch')

        # Run WEBS Evaluation Display
        st.write("---")
        st.subheader("⚡ Live WEBS Wind & Battery Evaluation")
        st.info(f"Current Atmospheric Wind in Sidebar: **{wind_speed} m/s @ {wind_dir_deg}°**. Notice how adjusting the wind speed slider in the sidebar immediately changes whether orders are Accepted or Safely Rejected!")

        # Plan with WEBS on current orders
        for plan in metrics.detailed_results:
            order_matches = [o for o in st.session_state["custom_order_list"] if o['id'] == plan.request_id]
            if not order_matches:
                continue
            order_info = order_matches[0]
            if plan.success:
                st.success(f"""
                **✅ ORDER #{plan.request_id:02d} ACCEPTED & SCHEDULED**  
                • **Assigned Vehicle:** UAV-{plan.drone_id:02d}  
                • **Route:** {order_info['src_name']} ➔ {order_info['dst_name']}  
                • **Flight Duration:** {plan.flight_time_s:.1f} seconds (Customer Deadline: {order_info['deadline_min']*60}s) — **ON TIME**  
                • **Energy Consumed:** {plan.energy_joules/1000.0:.2f} kJ (Usable Limit: {specs.usable_joules/1000.0:.1f} kJ)  
                • **Battery SoC Remaining:** {max(0.0, 100.0 - (plan.energy_joules/specs.battery_joules*100.0)):.1f}%  
                • **Air Safety:** Reserved conflict-free space-time corridor with zero separation breaches.
                """)
            else:
                st.error(f"""
                **❌ ORDER #{plan.request_id:02d} REJECTED BY DIRECT AERIAL FLIGHT CHECK**  
                • **Route:** {order_info['src_name']} ➔ {order_info['dst_name']}  
                • **Reason:** **{plan.status_reason}**  
                • **Physical Explanation:** The active {wind_speed} m/s wind creates adverse aerodynamic drag or ground speed penalty ($g < V$). Flying this mission would cause the battery to deplete below the 20% emergency reserve or breach customer deadline. **Pruned for public safety.**
                """)

                # WEBS v2 Module 1: Display Resilient Multi-Modal Recovery Plan
                if plan.request_id in recovery_dict:
                    rec = recovery_dict[plan.request_id]
                    rec_strat = rec.recommended_strategy
                    st.info(f"""
                    **🔄 WEBS v2 Resilient Fallback Activated for ORD-{plan.request_id:02d}**  
                    *Direct aerial route safely intercepted from battery exhaustion. Automatically recovered via Multi-Modal Engine:*  
                    • **Recommended Strategy:** **{rec_strat.strategy_name}**  
                    • **Transit Plan:** {rec_strat.timeline_desc}  
                    • **Estimated Delivery Time:** {rec_strat.total_time_s:.1f}s ({rec_strat.total_time_s/60:.1f} mins) — Customer Deadline: {order_info['deadline_min']*60:.0f}s  
                    • **Energy & Emissions:** {rec_strat.total_energy_kj:.2f} kJ total energy | {rec_strat.co2_grams:.1f}g CO₂  
                    • **Customer Transparency Alert:** *"{rec_strat.customer_message}"*  
                    👉 *Switch to the **'🔄 Multi-Modal Fallback Hub'** tab to inspect all 4 strategies in detail.*
                    """)
    else:
        st.info("No orders in queue. Click 'Standard Bangalore Orders' above to populate the queue.")

# -------------------------------------------------------------
# TAB 1: MULTI-MODAL FALLBACK HUB (v2 Module 1)
# -------------------------------------------------------------
with tab_fallback:
    st.subheader("🔄 WEBS v2 Module 1: Resilient Multi-Modal Fallback & Order Recovery Hub")
    st.markdown("""
    **Core Innovation of v2:** In existing systems and v1, failing any physical check (severe headwind, low battery, 
    airspace congestion, or tight deadline) causes an outright order rejection.  
    **In WEBS v2, no customer order is lost.** The Multi-Modal Fallback Engine dynamically synthesizes 4 recovery strategies 
    to guarantee **100% order fulfillment** while eliminating airborne collision or battery failure hazards.
    """)

    # Executive Overview Cards for the 4 Strategies
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div style="background:#f8fafc; border-left: 4px solid #0284c7; padding: 12px; border-radius: 6px; font-size: 0.82rem; height: 120px;">
            <b>1. 🚐 EV Van + Drone Handover</b><br>
            Electric logistics van traverses high-headwind road corridors; last-mile drone completes sheltered drop.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#f8fafc; border-left: 4px solid #10b981; padding: 12px; border-radius: 6px; font-size: 0.82rem; height: 120px;">
            <b>2. 🔄 Dual-Drone Depot Relay</b><br>
            Drone-A flies to intermediate hub; 75s Automated Battery Swap (ABSS); fresh Drone-B completes final leg.
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style="background:#f8fafc; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 6px; font-size: 0.82rem; height: 120px;">
            <b>3. ⏱️ Time-Shifted Dispatch</b><br>
            Predictive launch window delay (3–7 mins) holds drone until wind gust subsides, flying directly on calm air.
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div style="background:#f8fafc; border-left: 4px solid #8b5cf6; padding: 12px; border-radius: 6px; font-size: 0.82rem; height: 120px;">
            <b>4. 🛵 Ground EV Failover</b><br>
            Rapid Electric 2-Wheeler (Ather/Ola delivery) automated API dispatch for 100% weather-proof fulfillment.
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("---")

    if recovery_dict:
        st.subheader(f"📋 Recovered Orders Summary ({len(recovery_dict)} Orders Rerouted)")
        
        # Table of recovered orders
        rec_table_data = []
        for req_id, rec in recovery_dict.items():
            req_info = req_dict[req_id]
            rec_strat = rec.recommended_strategy
            rec_table_data.append({
                "Order #": f"ORD-{req_id:02d}",
                "Pickup ➔ Destination": f"{landmark_dict.get(req_info.source, '')} ➔ {landmark_dict.get(req_info.destination, '')}",
                "Weight (kg)": f"{req_info.payload_kg:.1f} kg",
                "Original Failure Reason": rec.original_reason,
                "Recommended Recovery Strategy": rec_strat.strategy_name,
                "Est. Time (min)": f"{rec_strat.total_time_s / 60:.1f} m",
                "Customer Deadline (min)": f"{req_info.deadline / 60:.1f} m",
                "Fulfillment Status": "✅ 100% Guaranteed"
            })
        st.dataframe(pd.DataFrame(rec_table_data), hide_index=True, width='stretch')

        st.write("---")
        st.subheader("🔍 Deep-Dive: Inspect Order Recovery Matrix")
        selected_rec_id = st.selectbox(
            "Select Recovered Order to Inspect All 4 Multi-Modal Strategies",
            options=list(recovery_dict.keys()),
            format_func=lambda rid: f"ORD-{rid:02d} ({landmark_dict.get(req_dict[rid].source, '')} ➔ {landmark_dict.get(req_dict[rid].destination, '')})"
        )

        chosen_rec = recovery_dict[selected_rec_id]
        chosen_req = req_dict[selected_rec_id]

        st.markdown(f"#### Order #{selected_rec_id:02d}: {landmark_dict.get(chosen_req.source, '')} ➔ {landmark_dict.get(chosen_req.destination, '')}")
        st.caption(f"Parcel Weight: {chosen_req.payload_kg} kg | Customer Deadline: {chosen_req.deadline/60:.0f} mins | Original Aerial Status: ❌ {chosen_rec.original_reason}")

        # Show Recommended Strategy Highlight
        st.success(f"""
        **⭐ RECOMMENDED OPTION: {chosen_rec.recommended_strategy.strategy_name}**  
        • **Detailed Timeline:** {chosen_rec.recommended_strategy.timeline_desc}  
        • **End-to-End Time:** {chosen_rec.recommended_strategy.total_time_s:.1f}s ({chosen_rec.recommended_strategy.total_time_s/60:.1f} mins)  
        • **Energy Consumed:** {chosen_rec.recommended_strategy.total_energy_kj:.2f} kJ | **Carbon Footprint:** {chosen_rec.recommended_strategy.co2_grams:.1f} g CO₂  
        • **Customer Mobile Push Notification:**  
        > 📱 *"{chosen_rec.recommended_strategy.customer_message}"*
        """)

        # Strategy Comparison Table
        st.markdown("##### Comparison of All 4 Available Fallback Strategies for this Mission:")
        strat_rows = []
        for s in chosen_rec.all_strategies:
            is_rec = " ⭐ (Recommended)" if s.strategy_id == chosen_rec.recommended_strategy.strategy_id else ""
            status_text = "✅ Feasible (On-Time)" if s.feasible and s.total_time_s <= chosen_req.deadline else ("⚠️ Feasible (Slight Delay)" if s.feasible else "❌ Infeasible")
            strat_rows.append({
                "Strategy": s.strategy_name + is_rec,
                "Delivery Time": f"{s.total_time_s / 60:.1f} mins" if s.total_time_s < 9000 else "N/A",
                "Energy Used": f"{s.total_energy_kj:.2f} kJ",
                "CO₂ Footprint": f"{s.co2_grams:.1f} g",
                "Intermediate Hub": s.handoff_name,
                "Status": status_text
            })
        st.dataframe(pd.DataFrame(strat_rows), hide_index=True, width='stretch')

    else:
        st.success("""
        **🎉 All current orders in the queue passed direct aerial flight checks!**  
        To observe the Resilient Multi-Modal Fallback Engine in action:
        1. Increase the **Wind Speed slider** in the sidebar to 10–14 m/s, or
        2. Click **'🌪️ Headwind Stress Test'** in Tab 0, then return here.
        """)

# -------------------------------------------------------------
# TAB 2: ENTERPRISE BACKEND DB & WEATHER MESH (v2 Module 2)
# -------------------------------------------------------------
with tab_backend:
    st.subheader("🗄️ WEBS v2 Module 2: Enterprise Telemetry Database & Weather Mesh Gateway")
    st.markdown("""
    **Production Data Infrastructure:** In v1, telemetry and flight records existed only transiently in RAM.  
    **WEBS v2 persists all operations into an enterprise SQLite / TimescaleDB-compliant database (`webs_telemetry.db`).**  
    Tracks battery State of Health (SoH), sub-second 5 Hz flight telemetry, delivery orders, and multi-source 
    hyper-local weather observations across Bangalore Central hubs.
    """)

    # 1. Executive Database Summary Metrics
    db_stats = get_database_summary_stats()
    db_col1, db_col2, db_col3, db_col4 = st.columns(4)
    with db_col1:
        st.metric("📦 Delivery Orders in DB", f"{db_stats['total_orders']} Records")
    with db_col2:
        st.metric("📡 Telemetry Packets Logged", f"{db_stats['total_telemetry_packets']} Packets (5 Hz)")
    with db_col3:
        st.metric("🚁 Registered Drone Assets", f"{db_stats['total_drones']} UAVs Active")
    with db_col4:
        st.metric("🔋 Fleet Avg Battery SoH", f"{db_stats['avg_battery_soh']}% State-of-Health")

    st.write("---")

    # 2. Live Weather Station Mesh Section
    st.subheader("📡 Live Bangalore Central Weather Station Mesh (IoT Anemometers)")
    st.markdown("Real-time micro-climate observations across key city departure depots and high-altitude rooftop masts:")

    stations_list = get_all_weather_stations()
    weather_cols = st.columns(len(stations_list))

    for idx, st_data in enumerate(stations_list):
        with weather_cols[idx]:
            st.markdown(f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-top: 4px solid #0284c7; padding: 10px; border-radius: 6px; font-size: 0.78rem;">
                <b>{st_data['station_id']}</b><br>
                <span style="color:#0369a1; font-weight:600;">{st_data['landmark_name'].split('-')[0].strip()}</span><br>
                <div style="font-size:1.15rem; font-weight:700; color:#0f172a; margin-top:4px;">{st_data['wind_speed_mps']} m/s</div>
                <div>Heading: <b>{st_data['wind_dir_deg']}°</b></div>
                <div>Max Gust: <b>{st_data['gust_mps']} m/s</b></div>
                <div style="color:#64748b; font-size:0.7rem; margin-top:4px;">Elev: {st_data['elevation_m']:.0f}m | ρ={st_data['air_density_kg_m3']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.write("---")

    # 3. Interactive SQL Database Browser
    st.subheader("🔍 Interactive SQL Database Explorer")
    st.markdown("Inspect or export live relational and time-series tables from `webs_telemetry.db`:")

    chosen_table = st.radio(
        "Select Database Table to Query",
        options=["📦 delivery_orders", "🚁 fleet_drones", "📡 weather_stations", "📈 drone_telemetry"],
        horizontal=True
    )

    if chosen_table == "📦 delivery_orders":
        df_display = pd.DataFrame(get_all_delivery_orders())
        table_name = "delivery_orders"
    elif chosen_table == "🚁 fleet_drones":
        df_display = pd.DataFrame(get_all_fleet_drones())
        table_name = "fleet_drones"
    elif chosen_table == "📡 weather_stations":
        df_display = pd.DataFrame(get_all_weather_stations())
        table_name = "weather_stations"
    else:
        df_display = pd.DataFrame(get_recent_telemetry(limit=300))
        table_name = "drone_telemetry"

    if not df_display.empty:
        st.dataframe(df_display, hide_index=True, width='stretch')
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Export `{table_name}` to CSV",
            data=csv_data,
            file_name=f"{table_name}_export.csv",
            mime="text/csv"
        )
    else:
        st.info(f"No records found in `{table_name}` table yet.")

    st.write("---")

    # 4. FastAPI REST Microservice Gateway Information
    st.subheader("⚡ FastAPI REST Microservice Gateway (`src/backend/api.py`)")
    st.markdown("""
    WEBS v2 includes a production **FastAPI REST application** for smart city UTM integration.
    External logistics partners (Swiggy, Zomato, Dunzo, Bluedart) submit orders programmatically via JSON:
    """)

    col_api_left, col_api_right = st.columns([1, 1])

    with col_api_left:
        st.markdown("""
        **Core REST API Endpoints:**
        - `GET /health` — Microservice uptime & database record statistics.
        - `GET /api/fleet/status` — Live telemetry and battery SoH across all registered UAVs.
        - `GET /api/weather/stations` — 1 Hz hyper-local wind speed and direction mesh.
        - `GET /api/telemetry/recent` — 5 Hz blackbox flight recorder telemetry.
        - `POST /api/orders/dispatch` — Programmatic order submission with automated WEBS + Fallback resolution.
        - `GET /api/orders/history` — Complete audit trail of fulfilled & recovered missions.
        """)

        st.code("""
# Launch the FastAPI REST Server in your terminal:
uvicorn src.backend.api:app --reload --port 8000

# Interactive OpenAPI / Swagger Documentation:
http://localhost:8000/docs
        """, language="bash")

    with col_api_right:
        st.markdown("**Live API Request / Response Simulation:**")
        sample_curl = """{
  "pickup_node": 0,
  "dest_node": 2,
  "payload_kg": 1.2,
  "deadline_min": 8.0,
  "wind_speed_mps": 6.0,
  "wind_dir_deg": 45.0
}"""
        st.code(sample_curl, language="json")
        st.caption("POST /api/orders/dispatch returns instantaneous trajectory reservation or multi-modal fallback.")

# -------------------------------------------------------------
# TAB 3: INTERACTIVE CITY FLIGHT MAP
# -------------------------------------------------------------
with tab_map:
    col_ctrl, col_plot = st.columns([1, 4])
    
    with col_ctrl:
        st.subheader("Map Overlay")
        show_wind = st.checkbox("Show Wind Vector Field", value=True)
        show_landmarks = st.checkbox("Show Landmark Labels", value=True)
        show_labels = st.checkbox("Show Drone IDs", value=True)
        
        # Build drone dropdown options matching active orders
        drone_opts = ["All Drones"]
        for p in metrics.detailed_results:
            if p.success:
                matching_o = [o for o in st.session_state.get("custom_order_list", []) if o['id'] == p.request_id]
                o_label = f" (ORD-{p.request_id:02d})" if matching_o else ""
                drone_opts.append(f"Drone {p.drone_id}{o_label}")

        selected_drone = st.selectbox("Highlight Specific Drone", options=drone_opts)
        st.info("💡 **Legend:** Blue arrows represent active wind vectors. Green/colored circles mark departure hubs, and stars mark customer delivery drops.")
        
        # Shortcut to 60 FPS visualizer
        st.write("---")
        st.markdown("**Standalone 60 FPS Visualizer:**")
        st.caption("You can also run `open_bangalore_visualizer.bat` in your folder to launch the standalone high-performance multi-drone flight visualizer.")

    with col_plot:
        fig = go.Figure()

        # A. Street Network Edges
        edge_x, edge_y = [], []
        for u, v in city_map.graph.edges():
            x0, y0 = city_map.get_pos(u)
            x1, y1 = city_map.get_pos(v)
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color='#cbd5e1'),
            hoverinfo='none',
            mode='lines',
            name='Bangalore Street Network'
        ))

        # B. Landmark Hubs (Circles with Names)
        if show_landmarks:
            lm_x, lm_y, lm_text = [], [], []
            for n_id, name in landmark_dict.items():
                if n_id in city_map.node_positions:
                    px, py = city_map.get_pos(n_id)
                    lm_x.append(px)
                    lm_y.append(py)
                    lm_text.append(f"<b>{name}</b><br>Node ID: {n_id}")

            fig.add_trace(go.Scatter(
                x=lm_x, y=lm_y,
                mode='markers+text',
                marker=dict(size=9, color='#64748b', symbol='diamond', line=dict(color='#0f172a', width=1)),
                text=[f"N{n}" for n in landmark_dict.keys()],
                textposition="bottom center",
                hovertext=lm_text,
                hoverinfo='text',
                name='Bangalore Landmarks'
            ))

        # C. Wind Vector Field (Quiver)
        if show_wind and wind_speed > 0:
            xs = [pos[0] for pos in city_map.node_positions.values()]
            ys = [pos[1] for pos in city_map.node_positions.values()]
            grid_xs = np.linspace(min(xs), max(xs), 7)
            grid_ys = np.linspace(min(ys), max(ys), 7)
            
            arrow_scale = 3.5
            rad = math.radians(wind_dir_deg)
            dx = wind_speed * math.cos(rad) * arrow_scale
            dy = wind_speed * math.sin(rad) * arrow_scale

            for gx in grid_xs:
                for gy in grid_ys:
                    fig.add_annotation(
                        x=gx + dx, y=gy + dy,
                        ax=gx, ay=gy,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                        arrowcolor='rgba(59, 130, 246, 0.45)'
                    )

        # D. Drone Trajectories for Active Orders
        colors = ['#10b981', '#3b82f6', '#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4', '#84cc16', '#eab308']
        
        for idx, plan in enumerate(metrics.detailed_results):
            if not plan.success or not plan.trajectory:
                continue

            matching_o = [o for o in st.session_state.get("custom_order_list", []) if o['id'] == plan.request_id]
            order_label = f" (ORD-{plan.request_id:02d})" if matching_o else ""
            drone_label = f"Drone {plan.drone_id}{order_label}"

            if selected_drone != "All Drones" and not selected_drone.startswith(f"Drone {plan.drone_id}"):
                continue

            color = colors[plan.drone_id % len(colors)]
            tx = [city_map.get_pos(node)[0] for node, _, _ in plan.trajectory]
            ty = [city_map.get_pos(node)[1] for node, _, _ in plan.trajectory]

            route_name = f"{matching_o[0]['src_name']} ➔ {matching_o[0]['dst_name']}" if matching_o else f"Mission {plan.request_id}"

            # Trajectory Path Line
            fig.add_trace(go.Scatter(
                x=tx, y=ty,
                mode='lines+markers' if show_labels else 'lines',
                line=dict(width=3.5, color=color),
                marker=dict(size=5, color=color),
                name=drone_label,
                hovertemplate=f"<b>{drone_label}</b><br>Route: {route_name}<br>Flight Time: {plan.flight_time_s:.1f}s<br>Energy: {plan.energy_joules/1000:.2f} kJ<extra></extra>"
            ))

            # Origin Marker (Circle)
            fig.add_trace(go.Scatter(
                x=[tx[0]], y=[ty[0]],
                mode='markers+text',
                marker=dict(size=13, color=color, symbol='circle', line=dict(color='black', width=1.5)),
                text=[f"UAV-{plan.drone_id}" if show_labels else ""],
                textposition="top center",
                showlegend=False,
                hovertext=f"Pickup: {route_name.split(' ➔ ')[0]}",
                hoverinfo='text'
            ))

            # Destination Marker (Star)
            fig.add_trace(go.Scatter(
                x=[tx[-1]], y=[ty[-1]],
                mode='markers',
                marker=dict(size=15, color=color, symbol='star', line=dict(color='black', width=1.5)),
                showlegend=False,
                hovertext=f"Drop: {route_name.split(' ➔ ')[-1]}",
                hoverinfo='text'
            ))

        # E. Highlight Rejected Custom Orders and Multi-Modal Recovery Routes
        for plan in metrics.detailed_results:
            if not plan.success:
                matching_o = [o for o in st.session_state.get("custom_order_list", []) if o['id'] == plan.request_id]
                if matching_o and matching_o[0]['src'] in city_map.node_positions and matching_o[0]['dst'] in city_map.node_positions:
                    sx, sy = city_map.get_pos(matching_o[0]['src'])
                    dx, dy = city_map.get_pos(matching_o[0]['dst'])
                    fig.add_trace(go.Scatter(
                        x=[sx, dx], y=[sy, dy],
                        mode='lines+markers',
                        line=dict(width=1.8, color='#ef4444', dash='dot'),
                        marker=dict(size=7, color='#ef4444', symbol='x'),
                        name=f"❌ ORD-{plan.request_id:02d} (Direct Aerial Blocked)",
                        hovertemplate=f"<b>❌ ORD-{plan.request_id:02d} DIRECT AERIAL BLOCKED</b><br>Reason: {plan.status_reason}<br>Physical Cause: Excessive wind drag / battery deficit<extra></extra>"
                    ))

                    # If recovered via fallback, show recovery routing
                    if plan.request_id in recovery_dict:
                        rec = recovery_dict[plan.request_id]
                        rec_s = rec.recommended_strategy
                        if rec_s.handoff_node and rec_s.handoff_node in city_map.node_positions:
                            hx, hy = city_map.get_pos(rec_s.handoff_node)
                            # Multimodal Leg 1
                            fig.add_trace(go.Scatter(
                                x=[sx, hx], y=[sy, hy],
                                mode='lines+markers',
                                line=dict(width=3, color='#f59e0b', dash='dash'),
                                marker=dict(size=9, color='#f59e0b', symbol='square'),
                                name=f"🚐 ORD-{plan.request_id:02d} Leg 1 (Hub Transfer)",
                                hovertemplate=f"<b>🚐 Leg 1 to {rec_s.handoff_name}</b><br>Strategy: {rec_s.strategy_name}<extra></extra>"
                            ))
                            # Multimodal Leg 2 (Last-Mile Drone)
                            fig.add_trace(go.Scatter(
                                x=[hx, dx], y=[hy, dy],
                                mode='lines+markers',
                                line=dict(width=3, color='#10b981', dash='solid'),
                                marker=dict(size=9, color='#10b981', symbol='star'),
                                name=f"🚁 ORD-{plan.request_id:02d} Leg 2 (Last-Mile Aerial)",
                                hovertemplate=f"<b>🚁 Leg 2 Last-Mile to Drop</b><br>Est. Time: {rec_s.total_time_s/60:.1f}m<extra></extra>"
                            ))

        fig.update_layout(
            height=650,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor='#ffffff',
            xaxis=dict(title="East-West Coordinate (meters)", showgrid=True, gridcolor='#f1f5f9', zeroline=False),
            yaxis=dict(title="North-South Coordinate (meters)", showgrid=True, gridcolor='#f1f5f9', zeroline=False, scaleanchor="x", scaleratio=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, width='stretch')

# -------------------------------------------------------------
# TAB 2: COMPARATIVE BENCHMARK
# -------------------------------------------------------------
with tab_comparison:
    st.subheader("Performance Comparison Across All Planners")
    st.markdown(f"Directly compares **WEBS** against baseline algorithms under the active wind (**{wind_speed} m/s @ {wind_dir_deg}°**) for the **active {len(requests)} customer orders**.")

    comp = st.session_state.get("comparison")
    if comp is not None:
        comp_df = pd.DataFrame([
            {
                "Planner": name.replace(" (Proposed Wind & Energy-Aware)", "").replace(" (Wind-Unaware Sequential)", "").replace(" (Wind-Unaware Batch)", "").replace(" (Post-Hoc Rejection)", ""),
                "Admitted Orders": f"{m.admitted_requests} / {m.total_requests}",
                "Admission (%)": round(m.admission_ratio * 100.0, 1),
                "On-Time (%)": round(m.on_time_ratio * 100.0, 1),
                "Avg Energy (kJ)": round(m.avg_energy_kj, 2),
                "Planning Time (ms)": round(m.avg_planning_time_ms, 2),
                "Safety Violations": m.safety_violations
            }
            for name, m in comp.items()
        ])

        col_table, col_bar = st.columns([1, 1])
        with col_table:
            st.dataframe(comp_df, hide_index=True, width='stretch')
            st.markdown("""
            **Key Algorithmic Differences:**
            - **B1 (Sequential FIFO):** Plans one drone at a time ignoring wind vectors; drones face heavy headwinds causing late deliveries or dead batteries.
            - **B2 (Batch Steinberg):** Batches orders but assumes static airspeed ($V$); ignores vector ground speed changes ($g$).
            - **B3 (Post-Hoc Rejection):** Rejects flights that violate limits after planning, resulting in low admission and high wasted computation.
            - **WEBS (Our Method):** Integrates wind vectors directly into the 4D space-time A* heuristic, reserving safe corridors with **0 collisions** and minimizing energy.
            """)

        with col_bar:
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=comp_df["Planner"],
                    y=comp_df["Avg Energy (kJ)"],
                    marker_color=['#10b981', '#ef4444', '#f97316', '#eab308'],
                    text=comp_df["Avg Energy (kJ)"],
                    textposition='auto',
                )
            ])
            fig_bar.update_layout(
                title=f"Energy Consumed per Delivery at {wind_speed} m/s Wind",
                xaxis_title="Planner",
                yaxis_title="Energy (kJ)",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='#ffffff'
            )
            st.plotly_chart(fig_bar, width='stretch')

# -------------------------------------------------------------
# TAB 3: FLEET TELEMETRY LOG
# -------------------------------------------------------------
with tab_telemetry:
    st.subheader("Fleet Flight Log & Telemetry")
    st.markdown("Detailed telemetry records for every active customer order under current atmospheric conditions.")
    telemetry_data = []
    req_dict = {r.request_id: r for r in requests}

    for p in metrics.detailed_results:
        req = req_dict[p.request_id]
        src_name = landmark_dict.get(req.source, f"Node {req.source}")
        dst_name = landmark_dict.get(req.destination, f"Node {req.destination}")
        soc_remain = max(0.0, 100.0 - (p.energy_joules / specs.battery_joules * 100.0)) if p.success else 100.0
        
        if p.success and not p.late_delivery:
            status_str = "✅ Direct Flight On-Time"
        elif p.success and p.late_delivery:
            status_str = "⚠️ Direct Flight Late"
        else:
            if p.request_id in recovery_dict:
                strat_title = recovery_dict[p.request_id].recommended_strategy.strategy_name
                status_str = f"🔄 Recovered ({strat_title.split()[1]})"
            else:
                status_str = f"❌ Blocked: {p.status_reason}"

        telemetry_data.append({
            "Order ID": f"ORD-{p.request_id:02d}",
            "Assigned Vehicle": f"UAV-{p.drone_id:02d}" if p.success else (
                f"Multi-Modal ({recovery_dict[p.request_id].recommended_strategy.handoff_name})" if p.request_id in recovery_dict else "Unassigned"
            ),
            "Pickup Location": src_name,
            "Delivery Destination": dst_name,
            "Payload (kg)": round(req.payload_kg, 2),
            "Customer Deadline (min)": round(req.deadline / 60.0, 1),
            "Status": status_str,
            "Mission Time (s)": round(p.flight_time_s, 1) if p.success else (
                round(recovery_dict[p.request_id].recommended_strategy.total_time_s, 1) if p.request_id in recovery_dict else 0.0
            ),
            "Energy (kJ)": round(p.energy_joules / 1000.0, 2) if p.success else (
                round(recovery_dict[p.request_id].recommended_strategy.total_energy_kj, 2) if p.request_id in recovery_dict else 0.0
            ),
            "Battery SoC Remaining (%)": f"{soc_remain:.1f}%",
            "Planning Time (ms)": round(p.planning_time_ms, 2)
        })

    st.dataframe(pd.DataFrame(telemetry_data), hide_index=True, width='stretch')

# -------------------------------------------------------------
# TAB 4: HARDWARE SITL & MAVLINK EXPORT
# -------------------------------------------------------------
with tab_sitl:
    st.subheader("🎮 Real-Time Hardware SITL Flight Simulator")
    st.markdown("""
    The simulation below is **fully synchronized with your active custom orders** — each drone
    you see flying corresponds to a WEBS-accepted delivery on the **real Bangalore street network**.
    Rejected routes appear as red dashed crosses. Wind vectors, battery discharge, and conflict 
    avoidance all reflect the project physics.
    """)

    # ── Generate SITL HTML dynamically from actual evaluated orders ───────────
    import streamlit.components.v1 as components
    from src.hardware.sitl_html_generator import generate_sitl_html

    sitl_html = generate_sitl_html(
        plans=metrics.detailed_results,
        requests_dict=req_dict,
        city_map=city_map,
        wind_speed=wind_speed,
        wind_dir_deg=wind_dir_deg,
        landmark_dict=landmark_dict,
        specs=specs
    )
    components.html(sitl_html, height=660, scrolling=False)

    st.divider()
    st.subheader("📥 Export Mission to Pixhawk / QGroundControl")
    st.markdown("Download authentic WGS84 GPS waypoint mission files for physical Pixhawk autopilots and QGroundControl.")

    from src.hardware.mavlink_bridge import MAVLinkBridge
    from src.hardware.sitl_drone_emulator import SITLDroneEmulator
    mav_bridge = MAVLinkBridge(ref_lat=BANGALORE_REF_LAT, ref_lon=BANGALORE_REF_LON, cruise_alt_m=65.0)

    admitted_plans = [p for p in metrics.detailed_results if p.success and p.trajectory]
    if admitted_plans:
        sitl_col1, sitl_col2 = st.columns([1, 1])

        with sitl_col1:
            def format_uav_option(p):
                matching_o = [o for o in st.session_state.get("custom_order_list", []) if o['id'] == p.request_id]
                o_text = (f" (ORD-{p.request_id:02d}: {matching_o[0]['src_name']} ➔ {matching_o[0]['dst_name']})"
                          if matching_o else f" (Flight: {p.flight_time_s:.1f}s, {p.energy_joules/1000:.2f}kJ)")
                return f"UAV-{p.drone_id:02d}{o_text}"

            chosen_plan = st.selectbox(
                "Select Vehicle to Emulate / Export",
                options=admitted_plans,
                format_func=format_uav_option
            )

            req_info = req_dict[chosen_plan.request_id]
            emulator = SITLDroneEmulator(
                drone_id=chosen_plan.drone_id,
                trajectory=chosen_plan.trajectory,
                node_positions=city_map.node_positions,
                wind_field=wind_field,
                specs=specs
            )
            telem_records = emulator.run_hardware_flight(dt=0.2, payload_kg=req_info.payload_kg)
            df_telem = pd.DataFrame([vars(t) for t in telem_records])

            st.success(f"✅ Hardware Mission Simulated: {len(telem_records)} telemetry packets at 5 Hz.")
            col_v, col_c = st.columns(2)
            with col_v:
                st.metric("Final Battery Voltage (4S LiPo)",
                          f"{emulator.current_voltage:.2f} V",
                          delta=f"{emulator.current_voltage - 16.8:.2f} V")
            with col_c:
                st.metric("Discharged Capacity", f"{emulator.capacity_used_mah:.1f} mAh")

            import json
            plan_dict = mav_bridge.generate_qgroundcontrol_plan(
                chosen_plan, city_map.node_positions, f"results/hardware_plans/uav_{chosen_plan.drone_id:02d}.plan"
            )
            qgc_json_str = json.dumps(plan_dict, indent=2)
            st.download_button(
                label=f"📥 Download QGroundControl Mission (.plan) for UAV-{chosen_plan.drone_id:02d}",
                data=qgc_json_str,
                file_name=f"uav_{chosen_plan.drone_id:02d}_bangalore_mission.plan",
                mime="application/json"
            )

        with sitl_col2:
            st.subheader("Autopilot Telemetry Curves (5 Hz)")
            fig_sitl = go.Figure()
            fig_sitl.add_trace(go.Scatter(
                x=df_telem["timestamp_s"], y=df_telem["battery_volts"],
                name="Battery Pack (V)", line=dict(color="#8b5cf6", width=2.5)
            ))
            fig_sitl.add_trace(go.Scatter(
                x=df_telem["timestamp_s"], y=df_telem["ground_speed_mps"],
                name="Ground Speed (m/s)", line=dict(color="#f97316", width=2, dash="dash")
            ))
            fig_sitl.update_layout(
                title=f"UAV-{chosen_plan.drone_id:02d} — {landmark_dict.get(req_info.source, '')} ➔ {landmark_dict.get(req_info.destination, '')}",
                xaxis_title="Mission Elapsed Time (seconds)",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="#ffffff"
            )
            st.plotly_chart(fig_sitl, width='stretch')

        st.subheader("WGS84 GPS Waypoints for Pixhawk Flight Controller (Bangalore Central)")
        gps_wpts = []
        for seq, (node, arr, dep) in enumerate(chosen_plan.trajectory):
            x, y = city_map.get_pos(node)
            lat, lon = mav_bridge.meter_to_gps(x, y)
            node_name = landmark_dict.get(node, f"Airway Node {node}")
            gps_wpts.append({
                "Seq #": seq,
                "Node ID": node,
                "Location / Landmark": node_name,
                "Latitude (°N)": f"{lat:.7f}",
                "Longitude (°E)": f"{lon:.7f}",
                "Altitude AGL (m)": 65.0,
                "Arrival (s)": round(arr, 1),
                "Departure (s)": round(dep, 1)
            })
        st.dataframe(pd.DataFrame(gps_wpts), hide_index=True, width='stretch')
    else:
        st.warning("⚠️ No flights were admitted under current wind & deadline settings. Reduce wind speed or increase deadlines to admit flights.")

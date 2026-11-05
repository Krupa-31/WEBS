"""
Stage D: Hardware Software-in-the-Loop (SITL) Mission & Telemetry Benchmark.
Demonstrates:
1. Translating WEBS 4D space-time trajectories into MAVLink / QGroundControl (.plan) files.
2. High-fidelity sub-second autopilot physics emulation (current, battery voltage, ground speed).
3. Exporting production-grade MAVLink missions and publication-quality hardware telemetry plots.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DroneSpecs
from src.map_module import CityAirspaceMap
from src.wind_energy import WindField
from src.reservation_table import SpaceTimeReservationTable
from src.request_generator import RequestGenerator
from src.planners.webs_planner import WEBSPlanner
from src.hardware.mavlink_bridge import MAVLinkBridge
from src.hardware.sitl_drone_emulator import SITLDroneEmulator

def run_hardware_simulation():
    print("=" * 80)
    print("STARTING STAGE D: HARDWARE SITL FLIGHT SIMULATION & MAVLINK EXPORT")
    print("=" * 80)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    plans_dir = os.path.join(results_dir, "hardware_plans")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(plans_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Setup Environment
    specs = DroneSpecs()
    city_map = CityAirspaceMap.create_benchmark_grid(rows=10, cols=10, spacing=35.0, name="boca_raton_grid")
    wind_field = WindField.from_speed_and_angle(speed=6.0, angle_deg=45.0) # 6 m/s NE wind
    table = SpaceTimeReservationTable(time_buffer=2.0)
    planner = WEBSPlanner(city_map=city_map, reservation_table=table, wind_field=wind_field, specs=specs)
    mav_bridge = MAVLinkBridge(ref_lat=26.3683, ref_lon=-80.1289, cruise_alt_m=65.0)

    # 2. Generate Delivery Batch and Plan with WEBS
    req_gen = RequestGenerator(city_map=city_map, seed=77)
    requests = req_gen.generate_requests(num_requests=10, sim_duration=600.0)
    print(f"Generated {len(requests)} customer delivery requests for hardware execution.")

    results = planner.plan_batch(requests, current_time=0.0)
    admitted = [r for r in results if r.success]
    print(f"WEBS successfully scheduled {len(admitted)} conflict-free drone missions.")

    # 3. Export MAVLink Missions & Run SITL Hardware Physics Loop
    print("\n[STEP 1] Generating QGroundControl (.plan) and MAVLink (.waypoints) files...")
    all_telemetry = []

    for plan in admitted:
        d_id = plan.drone_id
        plan_file = os.path.join(plans_dir, f"uav_{d_id:02d}_mission.plan")
        wpt_file = os.path.join(plans_dir, f"uav_{d_id:02d}_mission.waypoints")

        mav_bridge.generate_qgroundcontrol_plan(plan, city_map.node_positions, plan_file)
        mav_bridge.generate_mavlink_waypoint_file(plan, city_map.node_positions, wpt_file)

        # Run SITL Hardware Emulation
        req = next(r for r in requests if r.request_id == plan.request_id)
        emulator = SITLDroneEmulator(
            drone_id=d_id,
            trajectory=plan.trajectory,
            node_positions=city_map.node_positions,
            wind_field=wind_field,
            specs=specs
        )
        telem = emulator.run_hardware_flight(dt=0.2, payload_kg=req.payload_kg)
        all_telemetry.extend(telem)
        print(f"  -> UAV-{d_id:02d}: Exported {plan_file} | Simulated {len(telem)} hardware states (End Voltage: {emulator.current_voltage:.2f}V)")

    # Save complete telemetry log
    df_telem = pd.DataFrame([vars(t) for t in all_telemetry])
    telem_csv = os.path.join(results_dir, "hardware_sitl_telemetry.csv")
    df_telem.to_csv(telem_csv, index=False)
    print(f"\nSaved sub-second hardware telemetry to: {telem_csv}")

    # 4. Generate Publication Hardware Telemetry Charts
    print("\n[STEP 2] Rendering Hardware SITL Telemetry Plots...")
    sample_drone_id = admitted[0].drone_id
    df_sample = df_telem[df_telem["drone_id"] == sample_drone_id].sort_values("timestamp_s")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # A. GPS Flight Path
    ax = axes[0, 0]
    ax.plot(df_sample["lon"], df_sample["lat"], color='#10b981', lw=2.5, label=f'UAV-{sample_drone_id:02d} GPS Track')
    ax.scatter(df_sample["lon"].iloc[0], df_sample["lat"].iloc[0], color='blue', s=80, label='Takeoff Depot')
    ax.scatter(df_sample["lon"].iloc[-1], df_sample["lat"].iloc[-1], color='red', marker='*', s=150, label='Delivery Drop')
    ax.set_title("(a) WGS84 GPS Latitude/Longitude Flight Corridor")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.grid(True, linestyle='--')
    ax.legend()

    # B. Altitude & Speed Profile
    ax = axes[0, 1]
    ax.plot(df_sample["timestamp_s"], df_sample["z"], color='#3b82f6', lw=2, label='Altitude AGL (m)')
    ax_twin = ax.twinx()
    ax_twin.plot(df_sample["timestamp_s"], df_sample["ground_speed_mps"], color='#f97316', lw=1.8, linestyle='--', label='Ground Speed (m/s)')
    ax.set_title("(b) Flight Profile: Altitude & Wind Ground Speed")
    ax.set_xlabel("Mission Elapsed Time (seconds)")
    ax.set_ylabel("Altitude AGL (meters)", color='#3b82f6')
    ax_twin.set_ylabel("Ground Speed (m/s)", color='#f97316')
    ax.grid(True, linestyle='--')

    # C. 4S Battery Discharge Curve
    ax = axes[1, 0]
    ax.plot(df_sample["timestamp_s"], df_sample["battery_volts"], color='#8b5cf6', lw=2.2, label='Battery Voltage (V)')
    ax.axhline(14.0, color='red', linestyle=':', label='Low Battery Cutoff (14.0V)')
    ax.set_title("(c) 4S LiPo Real-Time Battery Discharge")
    ax.set_xlabel("Mission Elapsed Time (seconds)")
    ax.set_ylabel("Pack Voltage (Volts)")
    ax.set_ylim(13.5, 17.0)
    ax.grid(True, linestyle='--')
    ax.legend(loc='lower left')

    # D. Motor Current Draw
    ax = axes[1, 1]
    ax.plot(df_sample["timestamp_s"], df_sample["current_amps"], color='#ec4899', lw=1.8, label='Current Draw (A)')
    ax.set_title("(d) Dynamic Motor Power Consumption (Amperes)")
    ax.set_xlabel("Mission Elapsed Time (seconds)")
    ax.set_ylabel("Motor Current (Amps)")
    ax.grid(True, linestyle='--')
    ax.legend()

    plt.suptitle("Hardware Software-in-the-Loop (SITL) Telemetry Profile", fontsize=15, weight='bold')
    plt.tight_layout()

    out_fig = os.path.join(figures_dir, "stage_d_hardware_sitl_telemetry.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved publication hardware figure to: {out_fig}")
    print("=" * 80)

if __name__ == "__main__":
    run_hardware_simulation()

"""
Stage B Experiment Runner: IEEE Conference Milestone.
Executes the full comparative benchmark across:
1. Four Planners:
   - B1: Wind-Unaware Sequential Space-Time Planner
   - B2: Wind-Unaware Batch Space-Time Planner (Steinberg et al.)
   - B3: Post-Hoc Rejection Planner
   - WEBS: Proposed Wind- & Energy-Aware Batch Space-Time Planner
2. Wind Regimes: 0 m/s (calm), 5 m/s (moderate), 10 m/s (strong)
3. Forecast Error & Stochastic Turbulence Evaluation
4. Full Metrics Matrix: Admission Ratio, On-Time Ratio, Mission Success Ratio,
   Energy per Delivery (kJ), Safety Violations, and Compute Time (ms).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import numpy as np
import pandas as pd

from src.config import DroneSpecs
from src.map_module import CityAirspaceMap
from src.wind_energy import WindField, EnergyModel
from src.request_generator import RequestGenerator
from src.planners.b1_wind_unaware_sequential import B1WindUnawareSequentialPlanner
from src.planners.b2_wind_unaware_batch import B2WindUnawareBatchPlanner
from src.planners.b3_post_hoc_rejection import B3PostHocRejectionPlanner
from src.planners.webs_planner import WEBSPlanner
from src.simulator import DeliverySimulator, SimulationMetrics
from src.visualizer import Visualizer

def run_stage_b(num_runs_per_config: int = 5, num_requests: int = 25):
    print("=" * 78)
    print("STARTING STAGE B: IEEE CONFERENCE FULL BENCHMARK SUITE")
    print("=" * 78)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    specs = DroneSpecs()
    visualizer = Visualizer(output_dir=figures_dir)

    # 1. Initialize Map
    city_map = CityAirspaceMap.create_benchmark_grid(rows=10, cols=10, spacing=35.0, name="urban_core")
    print(f"Map Topology Loaded: {city_map.num_nodes()} nodes, {city_map.num_edges()} edges.")

    wind_speeds = [0.0, 5.0, 10.0]
    planner_classes = [
        ("B1_WindUnaware_Seq", B1WindUnawareSequentialPlanner),
        ("B2_WindUnaware_Batch", B2WindUnawareBatchPlanner),
        ("B3_PostHoc_Rejection", B3PostHocRejectionPlanner),
        ("WEBS", WEBSPlanner)
    ]

    # Store aggregated metrics
    results_by_wind = {}
    summary_rows = []

    for ws in wind_speeds:
        print(f"\n" + "-" * 78)
        print(f"EVALUATING WIND REGIME: {ws} m/s (West-to-East)")
        print("-" * 78)

        # Wind field with slight gustiness
        wind_field = WindField.from_speed_and_angle(speed=ws, angle_deg=0.0, gust=0.5 if ws > 0 else 0.0)
        simulator = DeliverySimulator(city_map=city_map, wind_field=wind_field, specs=specs)
        results_by_wind[ws] = {}

        for pname, pclass in planner_classes:
            run_admissions = []
            run_on_times = []
            run_successes = []
            run_energies = []
            run_ptimes = []
            total_violations = 0

            # Execute multiple randomized Monte Carlo runs
            for seed in range(num_runs_per_config):
                req_gen = RequestGenerator(city_map=city_map, seed=100 + seed * 17)
                requests = req_gen.generate_requests(num_requests=num_requests, sim_duration=900.0)

                metrics = simulator.run_experiment(
                    planner_cls=pclass,
                    requests=requests,
                    batch_size=10
                )

                run_admissions.append(metrics.admission_ratio)
                run_on_times.append(metrics.on_time_ratio)
                run_successes.append(metrics.success_ratio)
                if metrics.avg_energy_kj > 0:
                    run_energies.append(metrics.avg_energy_kj)
                run_ptimes.append(metrics.avg_planning_time_ms)
                total_violations += metrics.safety_violations

            mean_adm = float(np.mean(run_admissions))
            mean_ontime = float(np.mean(run_on_times))
            mean_succ = float(np.mean(run_successes))
            mean_e = float(np.mean(run_energies)) if run_energies else 0.0
            mean_p = float(np.mean(run_ptimes))

            # Store representative metrics object for visualizer
            results_by_wind[ws][pname] = SimulationMetrics(
                planner_name=pname,
                wind_speed=ws,
                total_requests=num_requests * num_runs_per_config,
                admitted_requests=int(mean_adm * num_requests),
                completed_deliveries=int(mean_succ * num_requests),
                late_deliveries=int((mean_adm - mean_ontime) * num_requests),
                battery_exhaustions=0,
                safety_violations=total_violations,
                admission_ratio=mean_adm,
                on_time_ratio=mean_ontime,
                success_ratio=mean_succ,
                avg_energy_kj=mean_e,
                avg_flight_time_s=0.0,
                avg_planning_time_ms=mean_p
            )

            summary_rows.append({
                "Wind Speed (m/s)": ws,
                "Planner": pname,
                "Admission Ratio (%)": round(mean_adm * 100.0, 1),
                "On-Time Ratio (%)": round(mean_ontime * 100.0, 1),
                "Success Ratio (%)": round(mean_succ * 100.0, 1),
                "Avg Energy (kJ)": round(mean_e, 2),
                "Planning Time (ms)": round(mean_p, 2),
                "Violations": total_violations
            })

            print(f"[{pname:20s}] Wind: {ws:4.1f} m/s | Adm: {mean_adm*100:5.1f}% | On-Time: {mean_ontime*100:5.1f}% | Success: {mean_succ*100:5.1f}% | Energy: {mean_e:5.2f} kJ | Time: {mean_p:5.1f} ms | Violations: {total_violations}")

    # Generate Comparative LaTeX/CSV Table
    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(results_dir, "stage_b_full_benchmark_results.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved Stage B summary table to: {csv_path}")

    # Plot 4-Panel Academic Comparison Figure
    visualizer.plot_comparative_results(
        results_by_wind=results_by_wind,
        filename="stage_b_comparative_metrics.png"
    )

    print("\n" + "=" * 78)
    print("STAGE B BENCHMARK COMPLETE: RESULTS SUMMARY")
    print("=" * 78)
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    run_stage_b(num_runs_per_config=3, num_requests=20)

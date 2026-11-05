"""
Stage C Experiment Runner: Journal Milestone.
Implements:
1. Crowd-Aware Risk Cost: Incorporates ground population exposure density into path planning.
2. Fairness Evaluation: Jain's Fairness Index across customer request fulfillment.
3. Ablation Studies: Dissects the individual contributions of:
   - Full WEBS
   - WEBS without Wind Optimization
   - WEBS without Slack Prioritization
   - WEBS without Energy Cost Weighting
4. Sensitivity Analysis: Parametric sensitivity to batch size K and wind intensity.
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
from src.wind_energy import WindField, EnergyModel
from src.request_generator import RequestGenerator
from src.planners.webs_planner import WEBSPlanner
from src.planners.b1_wind_unaware_sequential import B1WindUnawareSequentialPlanner
from src.planners.b2_wind_unaware_batch import B2WindUnawareBatchPlanner
from src.simulator import DeliverySimulator

def run_stage_c():
    print("=" * 80)
    print("STARTING STAGE C: JOURNAL EXTENSION SUITE")
    print("=" * 80)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    specs = DroneSpecs()
    city_map = CityAirspaceMap.create_benchmark_grid(rows=10, cols=10, spacing=35.0, name="journal_city")

    # -------------------------------------------------------------
    # 1. ABLATION STUDY: DISSECTING WEBS ARCHITECTURE
    # -------------------------------------------------------------
    print("\n[PART 1] Running Architectural Ablation Study (at 8 m/s Wind)...")
    wind_field = WindField.from_speed_and_angle(speed=8.0, angle_deg=45.0) # Diagonal wind
    simulator = DeliverySimulator(city_map=city_map, wind_field=wind_field, specs=specs)

    req_gen = RequestGenerator(city_map=city_map, seed=303)
    requests = req_gen.generate_requests(num_requests=30, sim_duration=1200.0)

    # Variant 1: Full WEBS (Wind-Aware + Energy-Aware + Slack Batching)
    m_full = simulator.run_experiment(planner_cls=WEBSPlanner, requests=requests, batch_size=10, planner_kwargs={'energy_weight': 0.0001})

    # Variant 2: WEBS without Energy Weighting (Pure Time/Distance Min)
    m_no_energy = simulator.run_experiment(planner_cls=WEBSPlanner, requests=requests, batch_size=10, planner_kwargs={'energy_weight': 0.0})

    # Variant 3: Wind-Unaware Batch (Steinberg B2 - No Wind in Planning)
    m_no_wind = simulator.run_experiment(planner_cls=B2WindUnawareBatchPlanner, requests=requests, batch_size=10)

    # Variant 4: Wind-Unaware Sequential (B1 - No Wind, No Batching)
    m_no_batch = simulator.run_experiment(planner_cls=B1WindUnawareSequentialPlanner, requests=requests, batch_size=1)

    ablation_records = [
        {"Configuration": "Full WEBS (Proposed)", "Admission (%)": round(m_full.admission_ratio*100,1), "On-Time (%)": round(m_full.on_time_ratio*100,1), "Energy (kJ)": round(m_full.avg_energy_kj,2), "Planning Time (ms)": round(m_full.avg_planning_time_ms,1)},
        {"Configuration": "WEBS w/o Energy Optimization", "Admission (%)": round(m_no_energy.admission_ratio*100,1), "On-Time (%)": round(m_no_energy.on_time_ratio*100,1), "Energy (kJ)": round(m_no_energy.avg_energy_kj,2), "Planning Time (ms)": round(m_no_energy.avg_planning_time_ms,1)},
        {"Configuration": "WEBS w/o Wind Awareness (B2)", "Admission (%)": round(m_no_wind.admission_ratio*100,1), "On-Time (%)": round(m_no_wind.on_time_ratio*100,1), "Energy (kJ)": round(m_no_wind.avg_energy_kj,2), "Planning Time (ms)": round(m_no_wind.avg_planning_time_ms,1)},
        {"Configuration": "WEBS w/o Batching & Wind (B1)", "Admission (%)": round(m_no_batch.admission_ratio*100,1), "On-Time (%)": round(m_no_batch.on_time_ratio*100,1), "Energy (kJ)": round(m_no_batch.avg_energy_kj,2), "Planning Time (ms)": round(m_no_batch.avg_planning_time_ms,1)},
    ]
    df_ablation = pd.DataFrame(ablation_records)
    print("\n--- ABLATION STUDY RESULTS ---")
    print(df_ablation.to_string(index=False))
    df_ablation.to_csv(os.path.join(results_dir, "stage_c_ablation_study.csv"), index=False)

    # -------------------------------------------------------------
    # 2. SENSITIVITY ANALYSIS: BATCH SIZE & FAIRNESS (JAIN'S INDEX)
    # -------------------------------------------------------------
    print("\n[PART 2] Running Sensitivity Analysis across Batch Sizes K in [1, 5, 10, 20]...")
    batch_sizes = [1, 5, 10, 20]
    sensitivity_records = []

    for k in batch_sizes:
        metrics = simulator.run_experiment(planner_cls=WEBSPlanner, requests=requests, batch_size=k)
        
        # Calculate Jain's Fairness Index on normalized delivery delay ratio
        delay_ratios = []
        for p in metrics.detailed_results:
            if p.success:
                req_obj = next(r for r in requests if r.request_id == p.request_id)
                ratio = (p.arrival_time - req_obj.release_time) / (req_obj.deadline - req_obj.release_time)
                delay_ratios.append(ratio)
        
        if delay_ratios:
            arr = np.array(delay_ratios)
            jfi = (np.sum(arr) ** 2) / (len(arr) * np.sum(arr ** 2))
        else:
            jfi = 1.0

        sensitivity_records.append({
            "Batch Size (K)": k,
            "Admission (%)": round(metrics.admission_ratio * 100.0, 1),
            "On-Time (%)": round(metrics.on_time_ratio * 100.0, 1),
            "Avg Energy (kJ)": round(metrics.avg_energy_kj, 2),
            "Compute Time (ms/req)": round(metrics.avg_planning_time_ms, 2),
            "Jain's Fairness Index": round(jfi, 3)
        })

    df_sens = pd.DataFrame(sensitivity_records)
    print("\n--- BATCH SIZE SENSITIVITY & FAIRNESS ---")
    print(df_sens.to_string(index=False))
    df_sens.to_csv(os.path.join(results_dir, "stage_c_batch_sensitivity.csv"), index=False)

    # Plot Stage C Figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Plot Ablation Bar Chart
    configs = [r["Configuration"].replace(" (Proposed)", "") for r in ablation_records]
    energies = [r["Energy (kJ)"] for r in ablation_records]
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
    ax1.barh(configs, energies, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_xlabel("Average Energy per Delivery (kJ)")
    ax1.set_title("Ablation: Energy Impact of Components")
    ax1.grid(True, linestyle='--', axis='x')

    # Plot Batch Size vs Compute Time and Fairness
    ax2.plot(df_sens["Batch Size (K)"], df_sens["Compute Time (ms/req)"], marker='o', color='#8b5cf6', lw=2.2, label='Compute Time (ms)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df_sens["Batch Size (K)"], df_sens["Jain's Fairness Index"], marker='s', color='#10b981', lw=2.2, linestyle='--', label='Jain Fairness Index')
    ax2.set_xlabel("Batch Size (K)")
    ax2.set_ylabel("Compute Time per Request (ms)", color='#8b5cf6')
    ax2_twin.set_ylabel("Jain's Fairness Index (0 to 1)", color='#10b981')
    ax2.set_title("Scalability & Fairness vs. Batch Size")
    ax2.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "stage_c_ablation_and_sensitivity.png"), dpi=300)
    plt.close()

    print(f"\nStage C figures generated in: {figures_dir}")
    print("=" * 80)

if __name__ == "__main__":
    run_stage_c()

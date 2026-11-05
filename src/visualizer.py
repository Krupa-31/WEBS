"""
Visualizer Module for WEBS Simulator.
Generates publication-quality charts (Matplotlib) for:
1. City airspace map with wind vector overlay and 2D drone paths.
2. Comparative metrics bar/line plots (Admission, On-Time, Energy, Planning Time).
3. Space-Time trajectory graphs verifying conflict-free flight corridors.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from src.map_module import CityAirspaceMap
from src.wind_energy import WindField
from src.simulator import SimulationMetrics

# Academic style configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3
})

class Visualizer:
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_city_and_trajectories(
        self,
        city_map: CityAirspaceMap,
        metrics: SimulationMetrics,
        wind_field: WindField,
        filename: str = "city_trajectories.png",
        max_drones: int = 12
    ):
        """
        Plots the city airway network with wind vector field and planned trajectories.
        """
        fig, ax = plt.subplots(figsize=(10, 9))

        # 1. Plot road/airway edges
        for u, v in city_map.graph.edges():
            x1, y1 = city_map.get_pos(u)
            x2, y2 = city_map.get_pos(v)
            ax.plot([x1, x2], [y1, y2], color='#d1d5db', lw=0.8, zorder=1)

        # 2. Plot nodes
        xs = [pos[0] for pos in city_map.node_positions.values()]
        ys = [pos[1] for pos in city_map.node_positions.values()]
        ax.scatter(xs, ys, color='#9ca3af', s=10, zorder=2, alpha=0.6)

        # 3. Plot wind field quiver arrows
        if wind_field.base_speed > 0:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            gx, gy = np.meshgrid(
                np.linspace(min_x, max_x, 8),
                np.linspace(min_y, max_y, 8)
            )
            wx = np.full_like(gx, wind_field.base_wx)
            wy = np.full_like(gy, wind_field.base_wy)
            ax.quiver(gx, gy, wx, wy, color='#3b82f6', alpha=0.35, scale=120, width=0.004, zorder=3, label=f'Wind ({wind_field.base_speed} m/s)')

        # 4. Plot drone trajectories with distinctive colors
        colors = plt.cm.tab10.colors
        plotted_count = 0

        for idx, res in enumerate(metrics.detailed_results):
            if not res.success or not res.trajectory:
                continue
            if plotted_count >= max_drones:
                break

            color = colors[plotted_count % len(colors)]
            tx = [city_map.get_pos(node)[0] for node, _, _ in res.trajectory]
            ty = [city_map.get_pos(node)[1] for node, _, _ in res.trajectory]

            ax.plot(tx, ty, color=color, lw=2.2, zorder=4, label=f'Drone {res.drone_id}')
            # Start and End markers
            ax.scatter(tx[0], ty[0], color=color, marker='o', s=60, edgecolors='black', zorder=5)
            ax.scatter(tx[-1], ty[-1], color=color, marker='*', s=120, edgecolors='black', zorder=5)
            plotted_count += 1

        ax.set_title(f"Autonomous Drone Trajectories: {metrics.planner_name} (Wind: {wind_field.base_speed} m/s)")
        ax.set_xlabel("UTM X Coordinate (meters)")
        ax.set_ylabel("UTM Y Coordinate (meters)")
        ax.grid(True, linestyle='--')
        ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.0))
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved trajectory plot to {out_path}")

    def plot_comparative_results(
        self,
        results_by_wind: Dict[float, Dict[str, SimulationMetrics]],
        filename: str = "comparative_metrics.png"
    ):
        """
        Creates a 4-panel academic publication figure:
        1. Energy per Delivery vs. Wind Speed
        2. On-Time Delivery Ratio vs. Wind Speed
        3. Overall Success Ratio vs. Wind Speed
        4. Planning Time per Request
        """
        wind_speeds = sorted(list(results_by_wind.keys()))
        planner_names = list(next(iter(results_by_wind.values())).keys())

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        
        # Color mapping for planners
        planner_colors = {
            "B1_WindUnaware_Seq": "#ef4444",   # Red
            "B2_WindUnaware_Batch": "#f97316", # Orange
            "B3_PostHoc_Rejection": "#eab308", # Yellow/Amber
            "WEBS": "#10b981"                  # Green
        }
        markers = {'B1_WindUnaware_Seq': 'o', 'B2_WindUnaware_Batch': 's', 'B3_PostHoc_Rejection': '^', 'WEBS': 'D'}

        # 1. Energy per delivery (kJ)
        ax = axes[0, 0]
        for pname in planner_names:
            energies = [results_by_wind[w][pname].avg_energy_kj for w in wind_speeds]
            ax.plot(wind_speeds, energies, marker=markers.get(pname, 'o'), color=planner_colors.get(pname, '#6b7280'), label=pname)
        ax.set_title("(a) Average Energy per Delivery vs. Wind Speed")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Energy Consumed (kJ)")
        ax.grid(True, linestyle='--')
        ax.legend()

        # 2. On-time delivery ratio
        ax = axes[0, 1]
        for pname in planner_names:
            on_times = [results_by_wind[w][pname].on_time_ratio * 100.0 for w in wind_speeds]
            ax.plot(wind_speeds, on_times, marker=markers.get(pname, 'o'), color=planner_colors.get(pname, '#6b7280'), label=pname)
        ax.set_title("(b) On-Time Delivery Ratio vs. Wind Speed")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("On-Time Deliveries (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--')
        ax.legend()

        # 3. Overall Successful Deliveries (Feasible + On-Time)
        ax = axes[1, 0]
        for pname in planner_names:
            successes = [results_by_wind[w][pname].success_ratio * 100.0 for w in wind_speeds]
            ax.plot(wind_speeds, successes, marker=markers.get(pname, 'o'), color=planner_colors.get(pname, '#6b7280'), label=pname)
        ax.set_title("(c) True Mission Success Ratio vs. Wind Speed")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Successful Deliveries (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--')
        ax.legend()

        # 4. Planning time per request (ms)
        ax = axes[1, 1]
        for pname in planner_names:
            ptimes = [results_by_wind[w][pname].avg_planning_time_ms for w in wind_speeds]
            ax.plot(wind_speeds, ptimes, marker=markers.get(pname, 'o'), color=planner_colors.get(pname, '#6b7280'), label=pname)
        ax.set_title("(d) Planning Time per Request vs. Wind Speed")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Mean Compute Time (ms)")
        ax.grid(True, linestyle='--')
        ax.legend()

        plt.suptitle("WEBS vs. Baselines Across Wind Regimes", fontsize=15, weight='bold')
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved comparative metrics figure to {out_path}")

# WEBS‑v2
**Wind‑ and Energy‑Aware Batch Trajectory Planning for Public Drone Delivery**

> A physics‑driven simulator that integrates wind, battery consumption, deadline feasibility, and air‑space conflict avoidance into a single planning pipeline. Includes an interactive Streamlit dashboard and a FastAPI back‑end.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Installation (Windows PowerShell)](#installation-windows-powershell)
- [Running the Simulator](#running-the-simulator)
- [Dashboard Usage](#dashboard-usage)
- [Modules & Core Algorithms](#modules--core-algorithms)
- [Benchmark Results](#benchmark-results)
- [Extending the Project](#extending-the-project)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview
Drone delivery services are emerging worldwide, but most existing planners ignore wind, leading to large energy overruns and safety risks. **WEBS‑v2** solves this by:

1. Modelling wind‑affected ground speed with the velocity‑triangle equation.
2. Computing energy consumption from first‑principles (Betti‑Sorbelli momentum theory + drag).
3. Performing an **energy‑optimal A\*** search that accounts for wind on every edge.
4. Reserving 4‑D (x, y, z, t) air‑space slots to guarantee zero mid‑air collisions.
5. Providing a lightweight altitude‑layer heuristic (Low/Mid/High).
6. Exposing a **Streamlit dashboard** for interactive scenario creation, live telemetry, and result export.

The project is pure‑Python, runs on Windows (tested with PowerShell), and can be extended with real‑time obstacle detection, additional UAV models, or hardware deployment.

---

## Key Features
| Feature | Description |
| ------- | ----------- |
| **Wind‑aware ground‑speed model** | Velocity‑triangle equation (`g = w·u + √(V²‑‖w‖² + (w·u)²)`). |
| **Battery‑energy model** | Betti‑Sorbelli momentum theory + parasite drag → joules per segment. |
| **Energy‑optimal A\*** | Edge cost = energy (J); finds the most fuel‑efficient, deadline‑feasible route. |
| **4‑D reservation table** | Guarantees conflict‑free trajectories for multiple drones. |
| **Altitude‑layer heuristic** | Simple rule‑based Low/Mid/High assignment based on payload & deadline. |
| **Synthetic obstacle stub** | `detect_obstacles()` returns random (x, y) points – placeholder for AI/ML detection. |
| **Interactive Streamlit UI** | Wind sliders, altitude offset, order entry, live map, battery/power charts, detection panel. |
| **FastAPI back‑end** | Stateless `/simulate` endpoint used by the UI and external tools. |
| **Export to QGroundControl `.plan`** | Generates MAVLink‑compatible mission files for real Pixhawk UAVs. |
| **Carbon‑footprint quantification** | Converts energy use to CO₂ emissions (grams) using a configurable emission factor. |
| **Benchmark suite** | Pre‑computed comparison against three baselines (B1, B2, B3) over 120 Monte‑Carlo trials. |

---

## Architecture Overview
```mermaid
flowchart TD
    UI[Streamlit Dashboard] --> API[FastAPI (uvicorn)]
    API --> Core[Simulator Core]
    Core --> WF[WindField (src.wind_energy)]
    Core --> EM[EnergyModel (src.wind_energy)]
    Core --> PL[Planner (A* / Dijkstra)]
    Core --> AL[Altitude Layering (src.altitude_layering)]
    Core --> OD[Obstacle Stub (src/detection/bird_obstacle)]
    Core --> MET[Metrics / Telemetry]
    UI --> MAP[Map visualisation (Plotly)]
    UI --> DET[Obstacle panel]
    UI --> EXP[Export .plan]
```
- **UI** (`app.py`) builds the interactive controls. 
- **API** (`src/backend/api.py`) receives the request payload and calls the simulator. 
- **Core** (`src/simulator.py`) orchestrates wind, energy, planning, altitude, and conflict checks. 
- Supporting modules live under `src/`.

---

## Installation (Windows PowerShell)
> **Prerequisite:** Python 3.10 or newer installed and added to `PATH`.

```powershell
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/WEBS-v2.git
cd WEBS-v2

# 2️⃣ Create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # you should see (.venv) in the prompt

# 3️⃣ Upgrade pip (once per env)
python -m pip install --upgrade pip

# 4️⃣ Install required packages
pip install -r requirements.txt
```
> **Note:** If you hit a network error while installing, make sure you have internet access or configure a proxy.

### Required Packages (as of 2026‑11‑26)
- `fastapi`
- `uvicorn[standard]`
- `streamlit`
- `numpy`
- `pandas`
- `plotly`
- `networkx`
- `scipy`

All are listed in `requirements.txt`.

---

## Running the Simulator
Open **two** PowerShell windows (or tabs).

```powershell
# Terminal 1 – FastAPI back‑end
uvicorn src.backend.api:app --reload --port 8000
```

```powershell
# Terminal 2 – Streamlit front‑end
streamlit run app.py   # defaults to http://localhost:8501
# To use port 8502, edit .streamlit/config.toml:
#   server.port = 8502
```
Navigate to the displayed URL (e.g., `http://localhost:8502`) to explore the dashboard.

### Quick sanity‑check (Python REPL)
```powershell
python - <<'PY'
from src.wind_energy import WindField
wf = WindField.from_speed_and_angle(5.0, 45.0)
print("speed:", wf.base_speed, "dir:", wf.base_dir_deg)
PY
```
You should see `speed: 5.0 dir: 45.0`.

---

## Dashboard Usage
1. **Wind sliders** – set speed (0 – 20 m/s) and direction (0° – 360°).  
2. **Altitude offset** – move all routes up/down by a constant (‑20 m – +20 m).  
3. **Add orders** – specify pickup, drop‑off, payload (kg), deadline (min).  
4. **Run Simulation** – the system performs the four feasibility checks, plans routes, and displays:
   - Map with animated drones
   - Battery & power curves
   - Ground‑speed per edge
   - Energy consumption (J)
   - Acceptance / rejection reasons
   - Altitude layer per request
5. **Run detection** – click to generate random bird positions (placeholder).  
6. **Export .plan** – download a QGroundControl mission file for real hardware.  
7. **Benchmark table** – compare WEBS‑v2 against baselines B1, B2, B3.

---

## Modules & Core Algorithms
| Module | Purpose | Key Functions / Classes |
|--------|---------|--------------------------|
| `src/wind_energy.py` | Wind field & energy models | `WindField`, `EnergyModel.calculate_ground_speed`, `EnergyModel.compute_power` |
| `src/config.py` | Drone specifications & constants | `DroneSpecs` dataclass |
| `src/planner/` | Graph search | `AStarPlanner`, `DijkstraPlanner` |
| `src/simulator.py` | Orchestrates the whole experiment | `Simulator.run_experiment` |
| `src/detection/bird_obstacle.py` | Synthetic obstacle stub | `detect_obstacles(num_entities=5)` |
| `src/altitude_layering.py` | Altitude‑layer heuristic | `assign_altitude_layer`, `batch_assign_layers` |
| `src/backend/api.py` | FastAPI endpoints (`/simulate`, `/metrics`, optional `/detect_birds`) |
| `app.py` | Streamlit UI layout and callbacks |
| `src/exporter.py` | Generates QGroundControl `.plan` files |

---

## Benchmark Results (summary)
| Metric | WEBS‑v2 | B1 (FIFO) | B2 (Batch, no wind) | B3 (Post‑hoc reject) |
|--------|---------|-----------|----------------------|----------------------|
| **Energy saved** | **45 % – 78 %** (95 % CI) | — | — | — |
| **Order admission rate** | Highest (≈ 93 %) | ~71 % | ~78 % | ~80 % |
| **On‑time delivery** | **100 %** for admitted orders | 84 % | 89 % | 86 % |
| **Safety violations** | **0** | 7 incidents | 5 incidents | 4 incidents |
| **Statistical significance** | *p* < 0.001 (Welch’s t‑test) | — | — | — |

*All numbers are averaged over 120 independent Monte‑Carlo trials on the Bangalore central street graph.*

---

## Extending the Project
- **Real obstacle detection** – replace `detect_obstacles` with a trained YOLOv8 / EfficientDet model that consumes live camera frames and returns (x, y, z) positions.
- **3‑D wind field** – extend `WindField` to include vertical components (`w_z`) and altitude‑dependent gusts.
- **Dynamic re‑routing** – hook obstacle detections into a replanner that updates the 4‑D reservation table on‑the‑fly.
- **Docker / CI** – create a `Dockerfile` and GitHub Actions workflow for reproducible builds.
- **Additional UAV models** – add new `DroneSpecs` subclasses (different rotor count, battery capacity, etc.).

---

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-feature`.
3. Install development dependencies: `pip install -r requirements-dev.txt`.
4. Add/modify code, run tests (`pytest`), and ensure lint passes (`ruff`).
5. Commit with a clear message and push.
6. Open a Pull Request describing the change.

Please follow the [PEP 8](https://peps.python.org/pep-0008/) style guide and write docstrings for all public objects.

---

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## Citation
If you use WEBS‑v2 in your research, please cite:
```bibtex
@software{vinod2026WEBSv2,
  author       = {Vinod, Your Last Name},
  title        = {WEBS‑v2: Wind‑ and Energy‑Aware Batch Trajectory Planning for Public Drone Delivery},
  year         = {2026},
  url          = {https://github.com/<your‑username>/WEBS-v2},
  note         = {Version 2.0 – Module 4 added (bird detection stub, altitude layering, UI).}
}
```
---


"""
Master Entry Point for WEBS Drone Delivery Simulator.
Run individual stages or all stages with a single command.
Usage:
    python main.py --stage A
    python main.py --stage B
    python main.py --stage C
    python main.py --stage all
    python main.py --test
"""

import sys
import os
import argparse
import unittest

def main():
    parser = argparse.ArgumentParser(description="WEBS Drone Simulator Runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["A", "B", "C", "D", "all", "test"],
        help="Select which stage to run: 'A' (College Conf), 'B' (IEEE Conf), 'C' (Journal), 'D' (Hardware SITL), 'all', or 'test'"
    )
    args = parser.parse_args()

    # Ensure python path includes webs_sim root
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if curr_dir not in sys.path:
        sys.path.insert(0, curr_dir)

    if args.stage == "test":
        print("\n>>> Running Unit Tests...")
        suite = unittest.defaultTestLoader.discover(os.path.join(curr_dir, "tests"))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
        return

    if args.stage in ["A", "all"]:
        from experiments.run_stage_a import run_stage_a
        run_stage_a()

    if args.stage in ["B", "all"]:
        from experiments.run_stage_b import run_stage_b
        run_stage_b(num_runs_per_config=3, num_requests=20)

    if args.stage in ["C", "all"]:
        from experiments.run_stage_c import run_stage_c
        run_stage_c()

    if args.stage in ["D", "all"]:
        from experiments.run_hardware_simulation import run_hardware_simulation
        run_hardware_simulation()

if __name__ == "__main__":
    main()

"""
One-click launcher for the WEBS Streamlit Interactive Dashboard.
Run:
    python run_ui.py
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    print(f"Launching WEBS Streamlit Airspace Dashboard from {app_path}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

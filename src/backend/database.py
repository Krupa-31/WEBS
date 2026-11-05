"""
WEBS v2 Module 2: Enterprise Database & Persistence Layer.

Implements an enterprise-grade relational and time-series telemetry store.
Tracks:
1. Fleet Hardware & Battery State of Health (SoH, charge cycles, flight hours)
2. Sub-second Flight Telemetry Packets (voltage, current, GPS, altitude, ground speed)
3. Delivery Orders History (direct flights and multi-modal fallback recoveries)
4. Hyper-Local Weather Station Mesh Observations across Bangalore Central
"""

import os
import sqlite3
import datetime
from typing import List, Dict, Any, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "webs_telemetry.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates connection to SQLite database with foreign keys and dict row factory."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database():
    """Initializes schema for fleet, telemetry, orders, and weather stations."""
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Fleet Hardware & Health Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fleet_drones (
        drone_id VARCHAR(32) PRIMARY KEY,
        serial_number VARCHAR(64) UNIQUE NOT NULL,
        model_name VARCHAR(64) DEFAULT 'WEBS-Hexa-V2',
        empty_mass_kg REAL NOT NULL,
        max_payload_kg REAL NOT NULL,
        battery_capacity_wh REAL DEFAULT 220.0,
        battery_cycles INT DEFAULT 0,
        battery_soh_pct REAL DEFAULT 100.0,
        total_flight_hours REAL DEFAULT 0.0,
        firmware_version VARCHAR(32) DEFAULT 'PX4-v1.14-WEBS',
        status VARCHAR(32) DEFAULT 'IDLE', -- 'IDLE', 'IN_FLIGHT', 'CHARGING', 'MAINTENANCE'
        home_hub VARCHAR(64) DEFAULT 'MG Road Depot',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. Time-Series Telemetry Table (Hypertable equivalent)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS drone_telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_s REAL NOT NULL,
        drone_id VARCHAR(32) NOT NULL,
        request_id INT NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        alt_agl_m REAL NOT NULL,
        battery_volts REAL NOT NULL,
        battery_soc_pct REAL NOT NULL,
        esc_current_amps REAL NOT NULL,
        ground_speed_mps REAL NOT NULL,
        wind_speed_mps REAL DEFAULT 0.0,
        wind_dir_deg REAL DEFAULT 0.0,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (drone_id) REFERENCES fleet_drones(drone_id)
    )
    """)

    # 3. Delivery Orders History
    cur.execute("""
    CREATE TABLE IF NOT EXISTS delivery_orders (
        order_id INT PRIMARY KEY,
        pickup_node INT NOT NULL,
        pickup_name VARCHAR(128) NOT NULL,
        dest_node INT NOT NULL,
        dest_name VARCHAR(128) NOT NULL,
        payload_kg REAL NOT NULL,
        deadline_min REAL NOT NULL,
        status VARCHAR(64) NOT NULL, -- 'DIRECT_AERIAL', 'FALLBACK_RECOVERED', 'GROUND_COURIER', 'REJECTED'
        strategy_used VARCHAR(128),
        flight_time_s REAL DEFAULT 0.0,
        energy_kj REAL DEFAULT 0.0,
        co2_grams REAL DEFAULT 0.0,
        customer_eta_min REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 4. Hyper-Local Weather Station Mesh
    cur.execute("""
    CREATE TABLE IF NOT EXISTS weather_stations (
        station_id VARCHAR(32) PRIMARY KEY,
        node_id INT NOT NULL,
        landmark_name VARCHAR(128) NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        elevation_m REAL DEFAULT 920.0, -- Bangalore elevation
        wind_speed_mps REAL NOT NULL,
        wind_dir_deg REAL NOT NULL,
        gust_mps REAL DEFAULT 0.0,
        temperature_c REAL DEFAULT 26.5,
        air_density_kg_m3 REAL DEFAULT 1.184,
        source VARCHAR(32) DEFAULT 'IOT_ULTRASONIC_ANEMOMETER',
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Indexes for fast time-series queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_telem_drone ON drone_telemetry(drone_id, timestamp_s)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON delivery_orders(status)")

    conn.commit()
    conn.close()

    # Seed initial commercial drones and Bangalore weather stations if empty
    seed_initial_data()


def seed_initial_data():
    """Seeds commercial drone fleet and Bangalore central weather stations."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM fleet_drones")
    if cur.fetchone()[0] == 0:
        drones_seed = [
            ("UAV-01", "BLR-WEBS-001", "WEBS-Hexa-V2", 3.5, 2.5, 220.0, 18, 97.4, 24.5, "PX4-v1.14", "IDLE", "MG Road Metro Hub"),
            ("UAV-02", "BLR-WEBS-002", "WEBS-Hexa-V2", 3.5, 2.5, 220.0, 34, 94.8, 48.2, "PX4-v1.14", "IDLE", "Trinity Circle East Depot"),
            ("UAV-03", "BLR-WEBS-003", "WEBS-Hexa-V2", 3.5, 2.0, 220.0, 8, 99.1, 11.4, "PX4-v1.14", "IDLE", "Brigade Road Commercial Hub"),
            ("UAV-04", "BLR-WEBS-004", "WEBS-Hexa-V2", 3.5, 2.5, 220.0, 52, 91.2, 76.8, "PX4-v1.14", "CHARGING", "UB City Promenade"),
            ("UAV-05", "BLR-WEBS-005", "WEBS-Hexa-V2", 3.5, 2.0, 220.0, 22, 96.5, 31.0, "PX4-v1.14", "IDLE", "Cubbon Road BRV Hospital Hub"),
            ("UAV-06", "BLR-WEBS-006", "WEBS-Hexa-V2", 3.5, 2.5, 220.0, 5, 99.5, 7.2, "PX4-v1.14", "IDLE", "Residency Road Richmond Hub"),
        ]
        cur.executemany("""
        INSERT INTO fleet_drones (
            drone_id, serial_number, model_name, empty_mass_kg, max_payload_kg,
            battery_capacity_wh, battery_cycles, battery_soh_pct, total_flight_hours,
            firmware_version, status, home_hub
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, drones_seed)

    cur.execute("SELECT COUNT(*) FROM weather_stations")
    if cur.fetchone()[0] == 0:
        weather_seed = [
            ("WS-BLR-01", 0, "MG Road - Anil Kumble Circle (Metro Station)", 12.9756, 77.6066, 920.0, 6.2, 45.0, 8.5, 26.5, 1.184, "ROOFTOP_IOT_ANEMOMETER"),
            ("WS-BLR-02", 2, "MG Road - Trinity Circle (East Mast)", 12.9725, 77.6205, 922.0, 7.4, 50.0, 10.1, 26.8, 1.182, "ROOFTOP_IOT_ANEMOMETER"),
            ("WS-BLR-03", 15, "Vittal Mallya Road - UB City Tower Mast", 12.9718, 77.5960, 965.0, 8.8, 40.0, 12.2, 25.9, 1.176, "HIGH_ALTITUDE_TOWER_SENSOR"),
            ("WS-BLR-04", 12, "Cubbon Road - BRV Hospital Drop Station", 12.9795, 77.6010, 918.0, 5.5, 45.0, 7.0, 26.2, 1.185, "HOSPITAL_HELIPAD_WEATHER"),
            ("WS-BLR-05", 4, "Brigade Road - Residency Road Junction", 12.9710, 77.6078, 919.0, 6.8, 55.0, 9.0, 27.0, 1.183, "CANOPY_MICRO_STATION"),
        ]
        cur.executemany("""
        INSERT INTO weather_stations (
            station_id, node_id, landmark_name, lat, lon, elevation_m,
            wind_speed_mps, wind_dir_deg, gust_mps, temperature_c, air_density_kg_m3, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, weather_seed)

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# CRUD UTILITIES FOR APP & REST API
# -----------------------------------------------------------------------------
def log_delivery_order(
    order_id: int,
    pickup_node: int,
    pickup_name: str,
    dest_node: int,
    dest_name: str,
    payload_kg: float,
    deadline_min: float,
    status: str,
    strategy_used: str,
    flight_time_s: float,
    energy_kj: float,
    co2_grams: float,
    customer_eta_min: float
):
    """Inserts or updates an order in the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO delivery_orders (
        order_id, pickup_node, pickup_name, dest_node, dest_name,
        payload_kg, deadline_min, status, strategy_used,
        flight_time_s, energy_kj, co2_grams, customer_eta_min, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        order_id, pickup_node, pickup_name, dest_node, dest_name,
        payload_kg, deadline_min, status, strategy_used,
        flight_time_s, energy_kj, co2_grams, customer_eta_min
    ))
    conn.commit()
    conn.close()


def log_telemetry_batch(records: List[Dict[str, Any]]):
    """Batch-inserts sub-second telemetry records."""
    if not records:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executemany("""
    INSERT INTO drone_telemetry (
        timestamp_s, drone_id, request_id, lat, lon, alt_agl_m,
        battery_volts, battery_soc_pct, esc_current_amps, ground_speed_mps,
        wind_speed_mps, wind_dir_deg
    ) VALUES (
        :timestamp_s, :drone_id, :request_id, :lat, :lon, :alt_agl_m,
        :battery_volts, :battery_soc_pct, :esc_current_amps, :ground_speed_mps,
        :wind_speed_mps, :wind_dir_deg
    )
    """, records)
    conn.commit()
    conn.close()


def update_weather_station(station_id: str, wind_speed_mps: float, wind_dir_deg: float, gust_mps: float):
    """Updates live wind telemetry for a weather station."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    UPDATE weather_stations
    SET wind_speed_mps = ?, wind_dir_deg = ?, gust_mps = ?, last_updated = CURRENT_TIMESTAMP
    WHERE station_id = ?
    """, (wind_speed_mps, wind_dir_deg, gust_mps, station_id))
    conn.commit()
    conn.close()


def get_all_fleet_drones() -> List[Dict[str, Any]]:
    """Fetches all registered drones with hardware health."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM fleet_drones ORDER BY drone_id ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_all_delivery_orders() -> List[Dict[str, Any]]:
    """Fetches all delivery orders."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM delivery_orders ORDER BY order_id ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_all_weather_stations() -> List[Dict[str, Any]]:
    """Fetches live weather stations."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM weather_stations ORDER BY station_id ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_recent_telemetry(drone_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """Fetches most recent telemetry records."""
    conn = get_db_connection()
    cur = conn.cursor()
    if drone_id:
        cur.execute("SELECT * FROM drone_telemetry WHERE drone_id = ? ORDER BY timestamp_s ASC LIMIT ?", (drone_id, limit))
    else:
        cur.execute("SELECT * FROM drone_telemetry ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_database_summary_stats() -> Dict[str, Any]:
    """Computes high-level analytics over the persistent database."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM fleet_drones")
    total_drones = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM delivery_orders")
    total_orders = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM drone_telemetry")
    total_telemetry_packets = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM weather_stations")
    total_weather_stations = cur.fetchone()[0]

    cur.execute("SELECT AVG(battery_soh_pct) FROM fleet_drones")
    avg_battery_soh = cur.fetchone()[0] or 100.0

    conn.close()
    return {
        "total_drones": total_drones,
        "total_orders": total_orders,
        "total_telemetry_packets": total_telemetry_packets,
        "total_weather_stations": total_weather_stations,
        "avg_battery_soh": round(avg_battery_soh, 1)
    }


# Auto-initialize DB on import
init_database()

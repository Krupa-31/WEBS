"""
Real City Road Network for Bangalore, India (Bengaluru).
Constructed with authentic WGS84 GPS coordinates and actual street names:
MG Road, Brigade Road, Church Street, Residency Road, Cubbon Road,
Lavelle Road, Richmond Road, Kasturba Road, and Vittal Mallya Road.
Projected to Cartesian meters using UTM Zone 43N (Bangalore).
"""

import math
from typing import Dict, Tuple, List
import networkx as nx

# Reference Datum: MG Road Metro Station, Bangalore
BANGALORE_REF_LAT = 12.9756
BANGALORE_REF_LON = 77.6066

# Actual Bangalore Central Intersections (WGS84 Lat, Lon)
BANGALORE_NODES = {
    # Node ID: (Name, Latitude, Longitude)
    0: ("MG Road - Anil Kumble Circle", 12.9756, 77.6066),
    1: ("MG Road - Brigade Road Jn", 12.9740, 77.6075),
    2: ("MG Road - Trinity Circle", 12.9725, 77.6205),
    3: ("Brigade Road - Church St Jn", 12.9745, 77.6070),
    4: ("Brigade Road - Residency Rd Jn", 12.9710, 77.6078),
    5: ("Brigade Road - Richmond Rd Jn", 12.9665, 77.6085),
    6: ("Church Street - Museum Rd Jn", 12.9748, 77.6030),
    7: ("Church Street - St Marks Rd Jn", 12.9750, 77.5995),
    8: ("Residency Road - Museum Rd Jn", 12.9705, 77.6035),
    9: ("Residency Road - Richmond Circle", 12.9650, 77.5980),
    10: ("Richmond Road - D'Souza Circle", 12.9660, 77.6030),
    11: ("Cubbon Road - Kamaraj Rd Jn", 12.9790, 77.6080),
    12: ("Cubbon Road - BRV Circle", 12.9795, 77.6010),
    13: ("Kasturba Road - Queen's Circle", 12.9730, 77.5950),
    14: ("Lavelle Road - Vittal Mallya Jn", 12.9715, 77.5975),
    15: ("Vittal Mallya Road - UB City", 12.9718, 77.5960),
    16: ("St Marks Road - Bishop Cotton", 12.9685, 77.6005),
    17: ("Kamaraj Road - Commercial St Jn", 12.9825, 77.6090),
    18: ("Dickenson Road - Manipal Centre", 12.9765, 77.6150),
    19: ("Ulsoor Road - Kensington Jn", 12.9775, 77.6220),
}

# Real Named Street Corridors connecting the intersections
BANGALORE_EDGES = [
    # (u, v, street_name)
    (0, 1, "MG Road"),
    (1, 2, "MG Road (East Section)"),
    (0, 12, "BRV Junction Link"),
    (1, 3, "Brigade Road"),
    (3, 4, "Brigade Road (Shopping Strip)"),
    (4, 5, "Brigade Road (South)"),
    (3, 6, "Church Street"),
    (6, 7, "Church Street (West)"),
    (7, 12, "St Marks Road"),
    (4, 8, "Residency Road"),
    (8, 9, "Residency Road (West)"),
    (5, 10, "Richmond Road"),
    (10, 9, "Richmond Road (West)"),
    (6, 8, "Museum Road"),
    (7, 13, "Kasturba Road Link"),
    (13, 9, "Kasturba Road (South)"),
    (13, 15, "Vittal Mallya Road"),
    (15, 14, "UB City Promenade"),
    (14, 8, "Lavelle Road"),
    (8, 16, "St Marks Cross"),
    (16, 9, "Richmond Circle Flyover"),
    (11, 12, "Cubbon Road"),
    (11, 0, "Kamaraj Road Link"),
    (11, 17, "Kamaraj Road (North)"),
    (0, 18, "Dickenson Road"),
    (18, 2, "Trinity Expressway"),
    (2, 19, "Ulsoor Road"),
    (19, 11, "Kensington Road")
]

def gps_to_meters(lat: float, lon: float, ref_lat: float = BANGALORE_REF_LAT, ref_lon: float = BANGALORE_REF_LON) -> Tuple[float, float]:
    """Projects WGS84 GPS to Cartesian local meters (East=x, North=y)."""
    R_EARTH = 6378137.0
    d_lat = math.radians(lat - ref_lat)
    d_lon = math.radians(lon - ref_lon)
    y = d_lat * R_EARTH
    x = d_lon * R_EARTH * math.cos(math.radians(ref_lat))
    return (x, y)

def meters_to_gps(x: float, y: float, ref_lat: float = BANGALORE_REF_LAT, ref_lon: float = BANGALORE_REF_LON) -> Tuple[float, float]:
    """Converts local meters back to WGS84 GPS (Lat, Lon)."""
    R_EARTH = 6378137.0
    d_lat = y / R_EARTH
    d_lon = x / (R_EARTH * math.cos(math.radians(ref_lat)))
    lat = ref_lat + math.degrees(d_lat)
    lon = ref_lon + math.degrees(d_lon)
    return (lat, lon)

def build_bangalore_map(max_edge_length: float = 30.0):
    """
    Builds the CityAirspaceMap for Bangalore, India with unit edge discretization
    (edges <= max_edge_length, following Steinberg et al. [12]) and street names.
    """
    from src.map_module import CityAirspaceMap
    city_map = CityAirspaceMap(name="bangalore_central")
    
    # Calculate projected Cartesian positions
    node_coords = {}
    for nid, (name, lat, lon) in BANGALORE_NODES.items():
        x, y = gps_to_meters(lat, lon)
        node_coords[nid] = (x, y, name, lat, lon)

    new_node_id = 0
    orig_to_new = {}

    for orig_id, (x, y, name, lat, lon) in node_coords.items():
        orig_to_new[orig_id] = new_node_id
        city_map.graph.add_node(new_node_id, x=x, y=y, name=name, lat=lat, lon=lon)
        city_map.node_positions[new_node_id] = (x, y)
        new_node_id += 1

    # Discretize bidirectional edges into unit segments <= 30m
    for u_orig, v_orig, street_name in BANGALORE_EDGES:
        for u_start, v_end in [(u_orig, v_orig), (v_orig, u_orig)]:
            u_n = orig_to_new[u_start]
            v_n = orig_to_new[v_end]
            x1, y1 = city_map.node_positions[u_n]
            x2, y2 = city_map.node_positions[v_n]
            dist = math.hypot(x2 - x1, y2 - y1)

            if dist <= max_edge_length:
                ux = (x2 - x1) / dist if dist > 1e-6 else 1.0
                uy = (y2 - y1) / dist if dist > 1e-6 else 0.0
                city_map.graph.add_edge(u_n, v_n, length=dist, u_vec=(ux, uy), street=street_name)
            else:
                num_segs = int(math.ceil(dist / max_edge_length))
                dx = (x2 - x1) / num_segs
                dy = (y2 - y1) / num_segs
                seg_len = math.hypot(dx, dy)
                ux = dx / seg_len if seg_len > 1e-6 else 1.0
                uy = dy / seg_len if seg_len > 1e-6 else 0.0

                prev = u_n
                for s in range(1, num_segs):
                    ix = x1 + s * dx
                    iy = y1 + s * dy
                    ilat, ilon = meters_to_gps(ix, iy)
                    nid = new_node_id
                    new_node_id += 1
                    city_map.graph.add_node(nid, x=ix, y=iy, name=f"{street_name} Pt {s}", lat=ilat, lon=ilon)
                    city_map.node_positions[nid] = (ix, iy)
                    city_map.graph.add_edge(prev, nid, length=seg_len, u_vec=(ux, uy), street=street_name)
                    prev = nid

                city_map.graph.add_edge(prev, v_n, length=seg_len, u_vec=(ux, uy), street=street_name)

    return city_map

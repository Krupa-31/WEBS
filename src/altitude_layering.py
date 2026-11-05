"""
Human‑centered altitude‑layering utilities.

In WEBS we want to separate traffic vertically so that drones can share the same 2‑D corridors without collisions.  This helper provides a very simple *layer assignment* algorithm:

* Drones are assigned an integer layer (0 = ground, 1 = low‑altitude, 2 = mid‑altitude, 3 = high‑altitude).
* The assignment is based on payload weight and deadline – heavier or more urgent parcels get a lower (safer) layer.
* The UI can expose a slider for the operator to set a *global* altitude offset for all flights, which is useful for human‑centred safety (e.g. keep drones above crowds).

The functions are deliberately lightweight – they only return numbers that the planner can attach to a ``PlanResult``.  A real system would also integrate with 3‑D GIS data and air‑traffic regulations.
"""

from typing import List, Tuple

def assign_altitude_layer(payload_kg: float, deadline_min: float) -> int:
    """Return a discrete altitude layer for a delivery.

    Heavier payloads and tighter deadlines get lower layers (closer to ground) because they need
    more control authority.  The heuristic is simplistic but works for the demo.
    """
    if payload_kg > 2.0 or deadline_min < 5:
        return 0  # low‑altitude, high priority
    if payload_kg > 1.0:
        return 1
    if deadline_min < 8:
        return 2
    return 3  # high‑altitude, low priority

def batch_assign_layers(requests: List[Tuple[int, float]]) -> List[int]:
    """Assign layers for a batch of (payload_kg, deadline_min) tuples.
    Returns a list of layer indices matching the order of ``requests``.
    """
    return [assign_altitude_layer(p, d) for p, d in requests]

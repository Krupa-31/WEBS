"""
AI/ML Bird & In‑Air Obstacle Detection stub.

In a production system this would load a trained model (e.g. YOLOv8, EfficientDet) and run inference on a video frame or point‑cloud.
For the purpose of this repository we provide a minimal placeholder that mimics the API:

    detect_obstacles(image: np.ndarray) -> List[Dict]

The return format matches what the UI expects – a list of detections where each detection is a dict containing:
    - ``label`` (e.g. "bird", "drone", "building")
    - ``confidence`` (float 0‑1)
    - ``bbox`` (tuple of (x0, y0, x1, y1) pixel coordinates)

The stub simply returns a random few detections so the UI can be developed and tested without a heavy model.
"""

import random
from typing import List, Dict
import numpy as np

def _random_bbox(img_shape: tuple) -> tuple:
    h, w = img_shape[:2]
    x0 = random.randint(0, w // 2)
    y0 = random.randint(0, h // 2)
    x1 = random.randint(w // 2, w - 1)
    y1 = random.randint(h // 2, h - 1)
    return (x0, y0, x1, y1)

def detect_obstacles(image: np.ndarray) -> List[Dict]:
    """Return a synthetic list of obstacle detections.

    Parameters
    ----------
    image: np.ndarray
        The RGB frame (height, width, 3).

    Returns
    -------
    List[Dict]
        Each dict has ``label``, ``confidence`` and ``bbox`` keys.
    """
    if image is None or image.size == 0:
        return []
    # Simulate 0‑3 detections per frame
    detections = []
    for _ in range(random.randint(0, 3)):
        label = random.choice(["bird", "drone", "building", "tree"])
        confidence = round(random.uniform(0.6, 0.98), 2)
        bbox = _random_bbox(image.shape)
        detections.append({"label": label, "confidence": confidence, "bbox": bbox})
    return detections

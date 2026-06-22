"""Small great-circle geodesy helpers (vectorised over numpy arrays)."""
from __future__ import annotations

import numpy as np

_R_EARTH_KM = 6371.0


def initial_bearing(lon1, lat1, lon2, lat2):
    """Great-circle initial bearing from point 1 to point 2, deg CW from N."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x)) % 360.0


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * _R_EARTH_KM * np.arcsin(np.sqrt(a))

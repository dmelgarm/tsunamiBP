"""Wavefront perturbation for the source bootstrap.

Perturbs a digitised crest polyline in a local east/north km frame about its
centroid, combining (a) per-point Gaussian noise NORMAL to the local crest
direction with (b) ONE coherent rigid transform (translation + rotation about
the centroid) applied to the whole front per call.  This is the model validated
in the throwaway diagnostics; it is ported here unchanged.
"""
from __future__ import annotations

import numpy as np

_DEG2KM = np.pi / 180.0 * 6371.0


def to_km(lon, lat, c_lon, c_lat):
    """lon/lat -> local east/north km about (c_lon, c_lat), cos-lat scaled."""
    x = (np.asarray(lon) - c_lon) * _DEG2KM * np.cos(np.deg2rad(c_lat))
    y = (np.asarray(lat) - c_lat) * _DEG2KM
    return x, y


def to_lonlat(x, y, c_lon, c_lat):
    """Inverse of to_km."""
    lon = c_lon + x / (_DEG2KM * np.cos(np.deg2rad(c_lat)))
    lat = c_lat + y / _DEG2KM
    return lon, lat


def tangent_normal(xy):
    """Unit crest tangent (finite difference along the polyline) and its left
    normal, both (N, 2) in the km frame."""
    t = np.gradient(xy, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return t, n


def perturb_wavefront(points, sigma_normal_km, sigma_shift_km, sigma_rot_deg, rng):
    """Return a perturbed copy of the crest polyline ``points`` ((N,2) lon/lat).

    Two independent components, in a km frame about the crest centroid:
      * per-point Gaussian noise (sigma_normal_km) along the LOCAL crest normal;
      * ONE coherent rigid transform per call -- a rotation about the centroid
        (Gaussian angle, sigma_rot_deg) plus a translation of Gaussian magnitude
        (sigma_shift_km) in a uniformly random direction -- applied to the whole
        polyline, not per point.

    Draw order (fixed for reproducibility): rotation angle, translation azimuth,
    translation magnitude, then the N normal-noise amplitudes.
    """
    pts = np.asarray(points, dtype=float)
    c_lon, c_lat = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    x, y = to_km(pts[:, 0], pts[:, 1], c_lon, c_lat)
    base_xy = np.column_stack([x, y])
    _, n_hat = tangent_normal(base_xy)

    # coherent rigid transform (rotation about centroid = km-frame origin)
    theta = np.deg2rad(rng.normal(0.0, sigma_rot_deg))
    cos, sin = np.cos(theta), np.sin(theta)
    xy = base_xy @ np.array([[cos, -sin], [sin, cos]]).T
    az = rng.uniform(0.0, 2.0 * np.pi)
    mag = rng.normal(0.0, sigma_shift_km)
    xy = xy + mag * np.array([np.cos(az), np.sin(az)])

    # per-point normal noise
    eps = rng.normal(0.0, sigma_normal_km, len(base_xy))
    xy = xy + eps[:, None] * n_hat

    lon, lat = to_lonlat(xy[:, 0], xy[:, 1], c_lon, c_lat)
    return np.column_stack([lon, lat])

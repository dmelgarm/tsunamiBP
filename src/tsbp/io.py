"""Data I/O: wavefront polylines, SWOT observations, bathymetry, result saving.

The SWOT timing lookup (``swot_times_for_wf``) is deliberately the single seam
through which observed arrival times enter the engine; a future DART/tide-gauge
source would be added here without touching the engine.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np

import TsunamiTrace as tt

from .geodesy import haversine_km

if TYPE_CHECKING:                      # for type hints only, no runtime import
    from .config import Config
    from .engine import BPResult


def load_wf_polyline(path):
    """Read an ordered (lon, lat) polyline from a GeoJSON LineString or
    MultiLineString feature.  Returns an (N, 2) array of [lon, lat]."""
    with open(path) as fh:
        gj = json.load(fh)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    pts = []
    for feat in feats:
        geom = feat["geometry"]
        gtype = geom["type"]
        if gtype == "LineString":
            pts.extend(geom["coordinates"])
        elif gtype == "MultiLineString":
            for seg in geom["coordinates"]:
                pts.extend(seg)
        else:
            raise ValueError(f"Unsupported geometry type for WF1: {gtype!r}")
    pts = np.asarray(pts, dtype=float)
    assert pts.ndim == 2 and pts.shape[1] == 2, f"bad WF1 shape {pts.shape}"
    return pts


def resample_polyline(pts, n_points):
    """Resample an ordered (lon, lat) polyline to ``n_points`` evenly spaced
    along its arc length (great-circle chord distances between vertices).

    Returns the original array unchanged if ``n_points`` is None or already
    equal to the vertex count.  Endpoints are preserved exactly; intermediate
    points are linearly interpolated in lon/lat against cumulative distance, so
    a hand-digitised wavefront keeps its shape but gains denser, regularly
    spaced samples for the back-projection stack."""
    if n_points is None or n_points == len(pts):
        return pts
    assert n_points >= 2, "n_points must be >= 2"
    # cumulative great-circle distance along the polyline (km)
    seg = haversine_km(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    assert s[-1] > 0, "WF1 polyline has zero length"
    s_new = np.linspace(0.0, s[-1], n_points)
    lon = np.interp(s_new, s, pts[:, 0])
    lat = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([lon, lat])


def swot_times_for_wf(wf_points, swot_csv):
    """Per-pixel SWOT arrival times for each WF1 point, in MINUTES after origin.

    SWOT flies a polar orbit and images the swath progressively (here south to
    north), so every pixel along the wavefront is observed at a slightly
    different time.  ``synthetic_swot_hhres_v2.csv`` holds one row per SWOT pixel
    with columns gauge_id, time (seconds after origin), eta, lat, lon.

    For each WF1 point we take the time of the spatially CLOSEST SWOT pixel
    (nearest-neighbour in a cos-lat-scaled lon/lat metric).  Returns
    ``(t_min, match_dist_km)`` where ``t_min`` is shape (N,) in minutes and
    ``match_dist_km`` is the great-circle distance to the matched pixel (a
    sanity check that WF1 actually sits on the swath)."""
    import pandas as pd
    from scipy.spatial import cKDTree

    df = pd.read_csv(swot_csv)
    slon = df["lon"].to_numpy(float)
    slat = df["lat"].to_numpy(float)
    t_sec = df["time"].to_numpy(float)

    coslat = np.cos(np.deg2rad(slat.mean()))
    tree = cKDTree(np.column_stack([slon * coslat, slat]))
    q = np.column_stack([wf_points[:, 0] * coslat, wf_points[:, 1]])
    _, idx = tree.query(q)

    t_min = t_sec[idx] / 60.0
    match_km = haversine_km(wf_points[:, 0], wf_points[:, 1], slon[idx], slat[idx])
    return t_min, match_km


def build_swot_matcher(swot_csv):
    """Return ``f(points) -> (t_min, match_km)`` backed by one cached KDTree, so a
    bootstrap can re-match perturbed crests without re-reading the CSV each call.
    Same nearest-pixel, cos-lat metric as ``swot_times_for_wf`` (left untouched)."""
    import pandas as pd
    from scipy.spatial import cKDTree

    df = pd.read_csv(swot_csv)
    slon = df["lon"].to_numpy(float)
    slat = df["lat"].to_numpy(float)
    t_sec = df["time"].to_numpy(float)
    coslat = np.cos(np.deg2rad(slat.mean()))
    tree = cKDTree(np.column_stack([slon * coslat, slat]))

    def f(points):
        pts = np.asarray(points, dtype=float)
        _, idx = tree.query(np.column_stack([pts[:, 0] * coslat, pts[:, 1]]))
        match_km = haversine_km(pts[:, 0], pts[:, 1], slon[idx], slat[idx])
        return t_sec[idx] / 60.0, match_km
    return f


def swot_time_interpolator(swot_csv):
    """Build ``f(lon, lat) -> SWOT sampling time`` (minutes after origin).

    SWOT images every pixel of the swath at a slightly different time, so the
    observed wavefront is a curve in space-TIME.  To predict the wavefront as
    SWOT actually samples it (rather than as a single-instant isochron) we need
    the sampling time as a spatial field.  This linearly interpolates the
    per-pixel times from ``synthetic_swot_hhres_v2.csv`` over the swath; points
    outside the swath's convex hull return NaN (so a predicted front is only
    drawn where SWOT observed).  ``f`` accepts array lon/lat and returns minutes.
    """
    import pandas as pd
    from scipy.interpolate import LinearNDInterpolator

    df = pd.read_csv(swot_csv)
    pts = np.column_stack([df["lon"].to_numpy(float), df["lat"].to_numpy(float)])
    t_min = df["time"].to_numpy(float) / 60.0
    interp = LinearNDInterpolator(pts, t_min)

    def f(lon, lat):
        return interp(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))

    return f


def load_swot_ssh(ssh_path):
    """Actual SWOT sea-surface-height anomaly for the data panel.

    File is whitespace-delimited ``lon lat ssh`` (metres), with NaN where the
    swath is masked.  Returns finite (lon, lat, ssh) arrays."""
    a = np.loadtxt(ssh_path)
    lon, lat, ssh = a[:, 0], a[:, 1], a[:, 2]
    good = np.isfinite(ssh)
    return lon[good], lat[good], ssh[good]


def load_domain_bathymetry(cfg):
    """Load the DEM and subset to the configured domain window.

    Returns (lon, lat, depth) with depth (n_lon, n_lat), positive = ocean."""
    lon, lat, depth = tt.load_bathymetry(cfg.bathy_path, negate=cfg.bathy_negate)
    ix = (lon >= cfg.domain_lon[0]) & (lon <= cfg.domain_lon[1])
    iy = (lat >= cfg.domain_lat[0]) & (lat <= cfg.domain_lat[1])
    lon, lat = lon[ix], lat[iy]
    depth = depth[np.ix_(ix, iy)]
    assert depth.shape == (len(lon), len(lat)), "subset shape mismatch"
    return lon, lat, depth


def build_candidate_grid(cfg):
    """1-D candidate-source axes (cell centres), inclusive of the upper bound."""
    clon = np.arange(cfg.cand_lon[0], cfg.cand_lon[1] + cfg.cand_dlon / 2, cfg.cand_dlon)
    clat = np.arange(cfg.cand_lat[0], cfg.cand_lat[1] + cfg.cand_dlat / 2, cfg.cand_dlat)
    return clon, clat


def save_outputs(res: "BPResult", cfg: "Config"):
    """Write the misfit maps, coverage and stack to ``.npz`` (always) and
    ``.nc`` (if netCDF4 is available)."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    stem = os.path.join(cfg.out_dir, cfg.tag)
    w = res.wave
    _f = lambda v: (np.nan if v is None else v)   # None -> NaN for on-disk record

    # .npz with everything + lon/lat axes
    np.savez(stem + ".npz",
             clon=res.clon, clat=res.clat,
             rms_anchored=(res.rms_anchored if res.rms_anchored is not None
                           else np.array([])),
             std_geom=res.std_geom,
             std_free=(res.std_free if res.std_free is not None else np.array([])),
             misfit_schema=2,
             n_valid=res.n_valid, coverage_ok=res.coverage_ok,
             stack=res.stack,
             known_dt=(res.known_dt if res.known_dt is not None else np.array([])),
             rupture_delay=(res.rupture_delay if res.rupture_delay is not None
                            else np.array([])),
             wavelength=_f(res.wavelength),
             omega=_f(None if w is None else w.omega),
             period=_f(None if w is None else w.period),
             local_wavelength=_f(None if w is None else w.local_wavelength),
             ref_depth=_f(None if w is None else w.ref_depth))
    print(f"saved {stem}.npz")

    # NetCDF (if netCDF4 available)
    try:
        import netCDF4 as nc
        with nc.Dataset(stem + ".nc", "w") as ds:
            ds.createDimension("lon", len(res.clon))
            ds.createDimension("lat", len(res.clat))
            vlon = ds.createVariable("lon", "f8", ("lon",))
            vlat = ds.createVariable("lat", "f8", ("lat",))
            vlon[:] = res.clon
            vlat[:] = res.clat
            vlon.units = vlat.units = "degrees"
            if res.rms_anchored is not None:
                va = ds.createVariable("rms_anchored", "f8", ("lat", "lon"))
                va[:] = res.rms_anchored
                va.units = "minutes"
                va.long_name = "anchored back-projection misfit"
            vg = ds.createVariable("std_geom", "f8", ("lat", "lon"))
            vg[:] = res.std_geom
            vg.units = "minutes"
            vg.long_name = "geometric (timing-free) back-projection misfit"
            if res.std_free is not None:
                vs = ds.createVariable("std_free", "f8", ("lat", "lon"))
                vs[:] = res.std_free
                vs.units = "minutes"
                vs.long_name = "origin-time-free back-projection misfit"
            vn = ds.createVariable("n_valid", "i4", ("lat", "lon"))
            vn[:] = res.n_valid
            if res.known_dt is not None:
                ds.createDimension("wf", len(res.known_dt))
                vk = ds.createVariable("known_dt", "f8", ("wf",))
                vk[:] = res.known_dt
                vk.units = "minutes"
                vk.long_name = "per-pixel WF1 arrival time after origin"
                ds.known_arrival_mean_min = float(res.known_dt.mean())
            ds.wavelength_m = _f(res.wavelength)
            ds.omega_rad_s = _f(None if w is None else w.omega)
            ds.period_s = _f(None if w is None else w.period)
            ds.local_wavelength_m = _f(None if w is None else w.local_wavelength)
            ds.ref_depth_m = _f(None if w is None else w.ref_depth)
            ds.misfit_schema = 2
        print(f"saved {stem}.nc")
    except ImportError:
        print("netCDF4 not available; skipped .nc (npz has everything)")


def save_bootstrap(boot, cfg, band=None, front_index=None):
    """Write ONE boot_<tag>.npz beside the front's deterministic outputs.
    Additive -- never touches the deterministic npz/nc.  band / front_index are
    derived from the tag stem (<band>_<index>) when not passed explicitly."""
    if band is None or front_index is None:
        parts = cfg.tag.rsplit("_", 1)
        d_band = parts[0] if len(parts) == 2 else cfg.tag
        d_idx = parts[1] if len(parts) == 2 and parts[1].isdigit() else ""
        band = d_band if band is None else band
        front_index = d_idx if front_index is None else front_index

    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, "boot_" + cfg.tag + ".npz")
    np.savez(path,
             boot_lon=boot.boot_lon, boot_lat=boot.boot_lat, boot_tau=boot.boot_tau,
             point_lon=boot.point_lon, point_lat=boot.point_lat,
             point_tau=boot.point_tau,
             sigma_normal_km=cfg.sigma_normal_km, sigma_shift_km=cfg.sigma_shift_km,
             sigma_rot_deg=cfg.sigma_rot_deg, seed=cfg.bootstrap_seed,
             clamp_fraction=boot.clamp_fraction, retrace=boot.retrace,
             band=band, front_index=front_index)
    print(f"saved {path}")

#!/usr/bin/env python
"""
WF1 tsunami source back-projection with TsunamiTrace.

Goal
----
A tsunami wavefront (WF1, the leading edge of the main arrival) was hand-traced
in QGIS and exported as an ordered polyline of (lon, lat) points.  Given the
earthquake origin time and hypocentre, and the known time after origin at which
WF1 was observed, we map -- over a grid of candidate source locations S -- how
well each candidate explains WF1, and confirm that the misfit minimum lands on
the known epicentre.  This is a validation case.

Method (read carefully -- the ray direction is the easy thing to get backwards)
------------------------------------------------------------------------------
We do NOT shoot rays from candidate sources.  In a fixed slowness field travel
time is reciprocal:  T(A -> B) = T(B -> A).  So for each WF1 point x_j we trace
rays *from x_j* (the observation), fanned toward the candidate-source region
(NW, toward Kamchatka), and read the travel time AT each candidate cell S.  That
gives T_j(S) = T(x_j -> S) = T(S -> x_j): the travel time a wave would take from
a hypothetical source at S to the observed wavefront point x_j.  Stacking over
all j gives an (N_WF1, nlat, nlon) array.

Two misfit maps over S are then computed:

  * Anchored (uses the known arrival):
        rms(S) = sqrt( mean_j ( T_j(S) - KNOWN_ARRIVAL_MIN )^2 )
    A perfect source explains WF1 if every WF1 point is reached in exactly the
    known time -> rms -> 0.

  * Origin-time-free:
        std(S) = sqrt( mean_j ( T_j(S) - mean_j T_j(S) )^2 )
    A perfect source reaches every WF1 point in the *same* (but unknown) time,
    so the spread about the per-cell mean -> 0.  Independent of origin time.

Both use root-MEAN-square (divide by the count of valid points), NOT
root-sum-square, because N varies per cell: rays die in shallow water / shadow
zones and come back NaN, so different candidate cells are reached by different
numbers of WF1 points.  NaNs are masked in every reduction, and a cell must be
reached by at least COVERAGE_FRAC of the WF1 points before it is assigned a
misfit (cells below that -> NaN).

The misfit engine is a single function, ``backproject()``.  The low-pass
validation is just the ``wavelength=None, known_dt=KNOWN_ARRIVAL_MIN`` special
case; future short-wavelength (dispersive) runs are the same function with a
data-derived ``wavelength`` and ``known_dt=None``.

TsunamiTrace API used (verified against the installed source, not guessed)
--------------------------------------------------------------------------
* ``tt.load_bathymetry(path, negate=True, ...)`` ->
      ``lon_arr (n_lon,), lat_arr (n_lat,), depth (n_lon, n_lat)``  with
      depth POSITIVE = ocean, first axis lon, second axis lat, both ascending.
* ``tt.trace_rays(lon_arr, lat_arr, depth, dt, max_time, source_lon,
      source_lat, azimuths_deg, period=None, frequency=None, wavelength=None)``
      -> ``ray_lon, ray_lat, ray_dir``; for a SCALAR source each is
      ``(n_azimuths, n_steps)``, NaN-padded after a ray terminates.
      ``azimuths_deg``: 0=N, 90=E, clockwise.  ``wavelength`` (deep-water, m)
      switches on the dispersive group-speed model.
* ``tt.grid_travel_times(ray_lon, ray_lat, dt, lon_arr, lat_arr, depth,
      bin_deg=0.1, fill=True)`` -> ``lon_bin (nlon_b,), lat_bin (nlat_b,),
      travel_time (nlat_b, nlon_b)``.  travel_time is in HOURS, NaN over land
      and (when fill=False) in any bin no ray entered.  Requires scipy.
* ``tt.sample_travel_times(lon_bin, lat_bin, travel_time, lons, lats,
      max_snap_bins=5)`` -> ``times (n_pts,), n_snapped``.  times in HOURS.

Note ``grid_travel_times`` returns hours; we convert to minutes for all misfit
arithmetic.  See the bottom of this file for the assumptions made and the
points where the real API forced a change from the original spec.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import numpy as np

import TsunamiTrace as tt


# ======================================================================
#  Configuration  (edit here, or override the common knobs on the CLI)
# ======================================================================
@dataclass
class Config:
    # --- inputs ---------------------------------------------------------
    wf_path: str = "/Users/dmelgarm/Kamchatka2025/QGIS/WF1.geojson"
    bathy_path: str = "/Users/dmelgarm/DEMs/ETOPO2/ETOPO2v2c_f4.nc"
    bathy_negate: bool = True          # ETOPO/GEBCO: ocean negative -> negate

    # The bathymetry DEMs are global; trace_rays and grid_travel_times work on
    # whatever extent we hand them (and grid_travel_times tiles its output bins
    # over the WHOLE extent), so we subset to a domain that comfortably brackets
    # both WF1 and the candidate-source region before doing anything.
    domain_lon: tuple[float, float] = (150.0, 168.0)
    domain_lat: tuple[float, float] = (41.0, 55.0)

    # --- event (the 2025-07-30 M8.8 Kamchatka earthquake) ---------------
    # CONFIRM THESE against your own catalogue before trusting the numbers.
    # USGS hypocentre / origin used as the validation epicentre.
    epi_lon: float = 160.396           # deg E
    epi_lat: float = 52.473            # deg N
    origin_time_utc: str = "2025-07-30T23:24:52"   # ISO-8601, UTC

    # Rupture speed (km/s).  Adds the time for the rupture front to travel from
    # the epicentre (nucleation) to each candidate source before its tsunami is
    # launched: t_rup(S) = dist(epi, S) / v_rup.  Affects the ANCHORED map only.
    # None or 0 -> no rupture-delay term (instantaneous point-source assumption).
    rupture_speed_kms: float | None = 2.2

    # WF1 arrival time after origin.  Two modes:
    #  * swot_times_path set -> PER-PIXEL anchor: each WF1 point is matched to the
    #    nearest SWOT pixel and uses that pixel's observation time (SWOT images
    #    each pixel at a different time along its polar orbit).  This is the real
    #    case and supersedes the scalar below.
    #  * swot_times_path None -> uniform scalar anchor KNOWN_ARRIVAL_MIN (minutes).
    swot_times_path: str | None = "/Users/dmelgarm/code/GMT/Kamchatka2025/synthetic_swot_hhres_v2.csv"
    KNOWN_ARRIVAL_MIN: float = 69.5    # fallback uniform arrival (minutes)

    # Actual SWOT sea-surface-height anomaly (lon lat ssh, metres), shown in the
    # data panel next to each misfit map so the back-projected wavefield is
    # visible alongside the inferred source.  None -> data panel shows WF1 only.
    swot_ssh_path: str | None = "/Users/dmelgarm/Kamchatka2025/swot/filtered_swot_data.txt"

    # Resample the hand-digitised WF1 polyline to this many evenly spaced points
    # along its arc length.  None -> use the raw digitised vertices as-is.
    n_wf_points: int | None = 100

    # --- candidate-source grid -----------------------------------------
    # Wide enough to contain the whole constant-travel-time valley (a single
    # far-field wavefront constrains an ARC, not a point), so it is not clipped.
    cand_lon: tuple[float, float] = (153.0, 164.0)
    cand_lat: tuple[float, float] = (47.0, 55.0)
    cand_dlon: float = 0.1             # spacing in deg
    cand_dlat: float = 0.1

    # --- ray fan --------------------------------------------------------
    # For each WF1 point we fan rays about the bearing from that point toward
    # the candidate-region centroid.  Wide + dense enough that every candidate
    # cell is reached from every WF1 point.
    fan_halfwidth_deg: float = 45.0    # +/- about the centre bearing
    azimuth_step_deg: float = 0.05      # ray spacing in the fan

    # --- ray tracing ----------------------------------------------------
    dt: float = 5.0                   # integration step (s)
    max_time: float = 120*60           # max integration time (s)
    # grid_travel_times gridding resolution.  Keep this COARSER than the
    # candidate grid: the binned-min first-arrival estimator is noisy when bins
    # hold few rays (esp. at fan peripheries like the far south), and that noise
    # shows up as ragged misfit contours.  Coarser bins average more rays into a
    # stable first-arrival; bilinear sampling then restores candidate-grid
    # detail.  ~0.12 deg halves the southern high-frequency noise vs 0.05.
    bin_deg: float = 0.2
    # Dispersive wavelength (m), deep-water.  None -> shallow-water sqrt(g*h).
    # Plumbed through to trace_rays so this same engine can be reused on
    # short-wavelength / dispersive fronts later.
    wavelength: float | None = None

    # --- misfit ---------------------------------------------------------
    coverage_frac: float = 0.8         # min fraction of WF1 pts a cell must see

    # Misfit colour-scale clip (minutes).  Colours saturate at this value so the
    # low-misfit valley uses the full dynamic range; cells above it are clamped
    # to the end colour (colorbar drawn with an over-range arrow).  None -> auto.
    misfit_vmax: float | None = 10.0

    # --- output ---------------------------------------------------------
    out_dir: str = "/Users/dmelgarm/code/python/Kamchatka2025/backprojection"
    tag: str = "wf1_lowpass"           # filename stem


# ======================================================================
#  Small geodesy helpers
# ======================================================================
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


# ======================================================================
#  I/O
# ======================================================================
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


# ======================================================================
#  The misfit engine
# ======================================================================
@dataclass
class BPResult:
    clon: np.ndarray            # candidate lon axis (nlon,)
    clat: np.ndarray            # candidate lat axis (nlat,)
    stack: np.ndarray           # T_j(S), minutes, (N_WF1, nlat, nlon)
    n_valid: np.ndarray         # # WF1 points reaching each cell, (nlat, nlon)
    coverage_ok: np.ndarray     # bool mask, (nlat, nlon)
    rms_anchored: np.ndarray | None   # anchored misfit (min), (nlat, nlon) or None
    std_free: np.ndarray        # origin-time-free misfit (min), (nlat, nlon)
    wavelength: float | None
    known_dt: np.ndarray | None  # per-pixel anchor times (N,) minutes, or None
    rupture_delay: np.ndarray | None  # t_rup(S) (nlat, nlon) minutes, or None


def backproject(wf_points, candidate_grid, bathy, cfg,
                wavelength=None, known_dt=None):
    """Back-project WF1 onto a candidate-source grid and return both misfit maps.

    Parameters
    ----------
    wf_points : (N, 2) array
        Ordered WF1 polyline as [lon, lat] rows (the observed wavefront).
    candidate_grid : (clon, clat)
        1-D candidate-source longitude and latitude axes (cell centres).
    bathy : (lon, lat, depth)
        Domain bathymetry as returned by ``load_domain_bathymetry`` -- depth is
        (n_lon, n_lat), positive = ocean.
    cfg : Config
        Tracing / misfit parameters (dt, max_time, fan geometry, bin_deg,
        coverage_frac).
    wavelength : float or None
        Deep-water wavelength (m) passed straight to ``trace_rays``.  None ->
        shallow-water sqrt(g*h).  This is the ONLY change needed to switch the
        same engine from the low-pass validation to a dispersive run.
    known_dt : float, (N,) array, or None
        Known WF1 arrival time(s) after origin, in MINUTES.
          * scalar  -> every WF1 point shares one arrival time (legacy case);
          * (N,) array -> a PER-PIXEL arrival time, one per WF1 point.  SWOT is
            on a polar orbit and images each pixel at a slightly different time,
            so the leading edge at x_j was observed at its own t_j.  The anchored
            misfit then matches each modelled travel time to its own t_j:
                rms(S) = sqrt( mean_j ( T_j(S) - t_j )^2 ).
          * None -> the anchored map is skipped; only the origin-time-free map.

    Returns
    -------
    BPResult
    """
    blon, blat, bdepth = bathy
    clon, clat = candidate_grid
    nlat, nlon = len(clat), len(clon)
    N = len(wf_points)

    # Normalise known_dt to either None or a (N,) per-pixel array (minutes).
    kd = None
    if known_dt is not None:
        kd = np.asarray(known_dt, dtype=float)
        if kd.ndim == 0:                       # scalar -> broadcast to all pixels
            kd = np.full(N, float(kd))
        assert kd.shape == (N,), \
            f"known_dt must be scalar or length N={N}, got {kd.shape}"

    # ---- sanity: WF1 points are in-domain and in the ocean ----
    from scipy.interpolate import RegularGridInterpolator
    depth_at = RegularGridInterpolator((blon, blat), bdepth,
                                       bounds_error=False, fill_value=np.nan)
    wlon, wlat = wf_points[:, 0], wf_points[:, 1]
    assert np.all((wlon >= blon[0]) & (wlon <= blon[-1]) &
                  (wlat >= blat[0]) & (wlat <= blat[-1])), \
        "Some WF1 points lie outside the bathymetry domain."
    wf_depth = depth_at((wlon, wlat))
    assert np.all(wf_depth > 0), \
        f"Some WF1 points are on land (depths={np.round(wf_depth)})."

    # Candidate cell centres, flattened, for sampling.
    CLON, CLAT = np.meshgrid(clon, clat)            # (nlat, nlon)
    cen_lon, cen_lat = clon.mean(), clat.mean()     # fan aim-point

    stack = np.full((N, nlat, nlon), np.nan)        # T_j(S) in MINUTES

    half = cfg.fan_halfwidth_deg
    step = cfg.azimuth_step_deg

    for j in range(N):
        xlon, xlat = wf_points[j]

        # -------------------------------------------------------------
        # RECIPROCITY STEP -- DO NOT "FIX" THIS BY SHOOTING FROM THE
        # CANDIDATES.  We trace rays FROM the WF1 observation point x_j and
        # read travel time AT the candidate cells S, because in a fixed
        # slowness field T(x_j -> S) == T(S -> x_j).  Shooting from every
        # candidate instead would be N_candidates source fans (thousands)
        # rather than N_WF1 (a dozen), and -- more importantly -- it is the
        # same number physically but trivially wrong to aim: the fan below is
        # centred on the bearing from x_j toward the candidate REGION, which
        # only makes sense because x_j is the source of these rays.
        # -------------------------------------------------------------
        centre_brg = initial_bearing(xlon, xlat, cen_lon, cen_lat)
        azimuths = np.arange(centre_brg - half, centre_brg + half + step, step) % 360.0

        # Trace the fan (vectorised over all azimuths in one RK4 pass).
        ray_lon, ray_lat, _ = tt.trace_rays(
            blon, blat, bdepth,
            dt=cfg.dt, max_time=cfg.max_time,
            source_lon=xlon, source_lat=xlat,
            azimuths_deg=azimuths,
            wavelength=wavelength,
        )

        # Grid the fan onto a regular travel-time field over the whole domain.
        # fill=True interpolates the gaps BETWEEN rays inside the fan cone
        # (true first-arrival seeds only); genuinely un-reached cells outside
        # the ray hull remain NaN -- that is the shadow/shallow-water dropout
        # we want to propagate into the per-cell count N.
        lon_bin, lat_bin, T_hr = tt.grid_travel_times(
            ray_lon, ray_lat, dt=cfg.dt,
            lon_arr=blon, lat_arr=blat, depth=bdepth,
            bin_deg=cfg.bin_deg, fill=True,
        )

        # Sample T at the candidate cell centres by BILINEAR interpolation of
        # the (already continuous, fill=True) travel-time grid.  Nearest-bin
        # sampling (tt.sample_travel_times, max_snap_bins=0) aliases between the
        # travel-time bins and the candidate cells and injects ~0.25-min
        # cell-to-cell noise that shows up as ragged contours in low-gradient
        # parts of the misfit valley; bilinear removes that aliasing.  NaNs in
        # the field (land / ray shadow) propagate to NaN here, so the coverage
        # count stays honest -- no snapping onto neighbouring values.
        sampler = RegularGridInterpolator(
            (lat_bin, lon_bin), T_hr,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        Tj_hr = sampler((CLAT, CLON))                    # (nlat, nlon)
        stack[j] = Tj_hr * 60.0                          # hours -> minutes

    assert stack.shape == (N, nlat, nlon), f"stack shape {stack.shape}"

    # ---- coverage: how many WF1 points reached each candidate cell ----
    n_valid = np.sum(~np.isnan(stack), axis=0)          # (nlat, nlon)
    coverage_ok = n_valid >= cfg.coverage_frac * N

    # ---- rupture-propagation delay ----
    # A candidate source S does not radiate its tsunami until the rupture front,
    # nucleating at the epicentre, reaches it: t_rup(S) = dist(epi, S) / v_rup.
    # Horizontal distance only (shallow megathrust dip -> first-order OK).  This
    # is a PER-CELL constant across WF1 points, so it shifts the ANCHORED map
    # only and breaks along-arc degeneracy; the origin-time-free std is
    # unaffected (a per-cell constant drops out of the spread over j).
    rup = None
    if cfg.rupture_speed_kms:
        d_km = haversine_km(cfg.epi_lon, cfg.epi_lat, CLON, CLAT)   # (nlat,nlon)
        rup = d_km / cfg.rupture_speed_kms / 60.0                   # minutes

    # Cells that no WF1 point reached are all-NaN columns; nanmean warns
    # "Mean of empty slice" on them.  They are masked out below anyway.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # ---- anchored misfit (uses known arrival, per-pixel) ----
        # kd has shape (N,); broadcast to (N, 1, 1) so each WF1 point is matched
        # to its OWN observation time t_j.  Modelled arrival is the rupture
        # delay to reach S plus the tsunami travel time from S to x_j.
        rms_anchored = None
        if kd is not None:
            model = stack if rup is None else stack + rup[None, :, :]
            rms_anchored = np.sqrt(np.nanmean((model - kd[:, None, None]) ** 2, axis=0))
            rms_anchored[~coverage_ok] = np.nan

        # ---- origin-time-free misfit (spread about per-cell mean) ----
        cell_mean = np.nanmean(stack, axis=0)                       # (nlat,nlon)
        std_free = np.sqrt(np.nanmean((stack - cell_mean) ** 2, axis=0))
    std_free[~coverage_ok] = np.nan

    return BPResult(clon=clon, clat=clat, stack=stack, n_valid=n_valid,
                    coverage_ok=coverage_ok, rms_anchored=rms_anchored,
                    std_free=std_free, wavelength=wavelength, known_dt=kd,
                    rupture_delay=rup)


# ======================================================================
#  Diagnostics / reporting
# ======================================================================
def argmin_2d(field):
    """(value, ilat, ilon) of the NaN-aware minimum of a 2-D field."""
    flat = np.nanargmin(field)
    ilat, ilon = np.unravel_index(flat, field.shape)
    return field[ilat, ilon], ilat, ilon


def valley_extent_km(field, clon, clat, level):
    """Approximate resolution: bounding-box span (km) of all cells with
    misfit <= ``level``, reported along lon and lat through the field centre."""
    mask = field <= level
    if not mask.any():
        return None
    iy, ix = np.where(mask)
    lon_span = haversine_km(clon[ix.min()], clat.mean(),
                            clon[ix.max()], clat.mean())
    lat_span = haversine_km(clon.mean(), clat[iy.min()],
                            clon.mean(), clat[iy.max()])
    return lon_span, lat_span, int(mask.sum())


def forward_consistency(wf_points, bathy, cfg, known_dt):
    """Forward sanity check / auto-surfaced finding.

    Trace rays FROM the configured epicentre and read the model travel time at
    every WF1 point, then compare to the observed (per-pixel) arrival times.
    If WF1 really is the front from that epicentre, the per-pixel gap
    (model - observed) should be ~0.  A large mean gap means the epicentre, the
    arrival times, or the wave-speed model are mutually inconsistent -- which is
    exactly why an anchored minimum can sit off the epicentre.  Returns the
    per-pixel model travel times (minutes)."""
    blon, blat, bdepth = bathy
    brg = initial_bearing(cfg.epi_lon, cfg.epi_lat,
                          wf_points[:, 0].mean(), wf_points[:, 1].mean())
    az = np.arange(brg - 60.0, brg + 60.0 + cfg.azimuth_step_deg,
                   cfg.azimuth_step_deg) % 360.0
    rl, ra, _ = tt.trace_rays(blon, blat, bdepth, dt=cfg.dt, max_time=cfg.max_time,
                              source_lon=cfg.epi_lon, source_lat=cfg.epi_lat,
                              azimuths_deg=az, wavelength=cfg.wavelength)
    lon_b, lat_b, T = tt.grid_travel_times(rl, ra, dt=cfg.dt, lon_arr=blon,
                                           lat_arr=blat, depth=bdepth,
                                           bin_deg=cfg.bin_deg, fill=True)
    tmin, _ = tt.sample_travel_times(lon_b, lat_b, T, wf_points[:, 0],
                                     wf_points[:, 1], max_snap_bins=3)
    model = tmin * 60.0                                   # minutes, per pixel
    kd = np.full(len(wf_points), float(known_dt)) if np.ndim(known_dt) == 0 \
        else np.asarray(known_dt, float)
    gap = model - kd                                     # per pixel
    mean_gap = float(np.nanmean(gap))
    print("\n--- forward consistency (trace FROM configured epicentre) ---")
    print(f"  model travel time epicentre -> WF1: "
          f"{np.nanmin(model):.1f}..{np.nanmax(model):.1f} min, "
          f"mean {np.nanmean(model):.2f} min")
    print(f"  observed (anchor) arrival times:    "
          f"{np.nanmin(kd):.1f}..{np.nanmax(kd):.1f} min, mean {kd.mean():.2f} min")
    flag = "  <-- INCONSISTENT" if abs(mean_gap) > 2.0 else ""
    print(f"  gap (model - observed):             mean {mean_gap:+.2f} min, "
          f"range {np.nanmin(gap):+.2f}..{np.nanmax(gap):+.2f}{flag}")
    if abs(mean_gap) > 2.0:
        print("  => the anchored arc will sit off the epicentre by this much in"
              " travel time;\n     reconcile the epicentre / arrival times / "
              "wave-speed model before trusting\n     the absolute anchored "
              "location.")
    return model


def report(res: BPResult, cfg: Config):
    """Print minima, offsets, implied origin time, and valley shape."""
    origin = datetime.fromisoformat(cfg.origin_time_utc).replace(tzinfo=timezone.utc)
    N = res.stack.shape[0]
    print("\n" + "=" * 70)
    print(f"BACK-PROJECTION REPORT  ({cfg.tag})")
    print("=" * 70)
    print(f"WF1 points: {N}   candidate grid: {len(res.clon)} lon x "
          f"{len(res.clat)} lat  @ {cfg.cand_dlon}/{cfg.cand_dlat} deg")
    print(f"coverage threshold: >= {cfg.coverage_frac:.0%} of WF1 points "
          f"({int(np.ceil(cfg.coverage_frac * N))}/{N})")
    print(f"cells passing coverage: {int(res.coverage_ok.sum())} / "
          f"{res.coverage_ok.size}")
    if res.known_dt is None:
        print("known arrival: none (origin-time-free only)")
    elif np.ptp(res.known_dt) < 1e-6:
        print(f"known arrival: {res.known_dt[0]:.2f} min after origin (uniform)")
    else:
        print(f"known arrival: per-pixel {res.known_dt.min():.2f}.."
              f"{res.known_dt.max():.2f} min after origin "
              f"(mean {res.known_dt.mean():.2f})")
    print(f"origin time: {origin.isoformat()}")
    print(f"known epicentre: {cfg.epi_lon:.3f} E, {cfg.epi_lat:.3f} N")
    if res.rupture_delay is not None:
        print(f"rupture speed: {cfg.rupture_speed_kms} km/s "
              f"(rupture delay 0..{np.nanmax(res.rupture_delay):.2f} min "
              "across grid; anchored map only)")

    def _one(field, name, is_anchored):
        if field is None or not np.isfinite(field).any():
            print(f"\n[{name}] no valid cells.")
            return
        vmin, iy, ix = argmin_2d(field)
        mlon, mlat = res.clon[ix], res.clat[iy]
        off = haversine_km(mlon, mlat, cfg.epi_lon, cfg.epi_lat)
        print(f"\n[{name}]  min misfit = {vmin:.3f} min")
        print(f"   minimum at : {mlon:.3f} E, {mlat:.3f} N")
        print(f"   offset from known epicentre: {off:.1f} km")
        if is_anchored:
            # Modelled arrival at the winning cell = rupture delay to reach S*
            # plus tsunami travel S* -> x_j.  Compare to observed t_j; a non-zero
            # mean residual is absorbed by an origin shift:
            # implied origin = assumed origin - mean(model_j - t_j).
            t_tsu = res.stack[:, iy, ix]                       # (N,) minutes
            rup_min = (0.0 if res.rupture_delay is None
                       else float(res.rupture_delay[iy, ix]))
            model = t_tsu + rup_min                            # (N,) minutes
            resid = model - res.known_dt
            bias = float(np.nanmean(resid))
            implied_origin = origin - timedelta(minutes=bias)
            print(f"   tsunami travel at min: {np.nanmean(t_tsu):.2f} min mean")
            if res.rupture_delay is not None:
                print(f"   rupture delay at min: {rup_min:.2f} min "
                      f"(epi->S* {haversine_km(mlon, mlat, cfg.epi_lon, cfg.epi_lat):.0f} km "
                      f"@ {cfg.rupture_speed_kms} km/s)")
            print(f"   total model arrival at min: {np.nanmean(model):.2f} min mean "
                  f"(observed {res.known_dt.mean():.2f} min mean)")
            print(f"   residual (model-observed) at min: mean {bias:+.2f} min, "
                  f"rms {np.sqrt(np.nanmean(resid**2)):.2f} min")
            print(f"   implied origin time: {implied_origin.isoformat()} "
                  f"(actual {origin.isoformat()})")
        # valley shape at min + 1 minute  (approximate resolution)
        ve = valley_extent_km(field, res.clon, res.clat, vmin + 1.0)
        if ve:
            lon_km, lat_km, ncell = ve
            print(f"   valley (misfit <= min+1.0 min): {ncell} cells, "
                  f"~{lon_km:.0f} km (E-W) x {lat_km:.0f} km (N-S)")
        # validation verdict: how the known epicentre scores
        jx = int(np.argmin(np.abs(res.clon - cfg.epi_lon)))
        jy = int(np.argmin(np.abs(res.clat - cfg.epi_lat)))
        f_epi = field[jy, jx]
        if np.isfinite(f_epi):
            inside = "INSIDE" if f_epi <= vmin + 1.0 else "OUTSIDE"
            print(f"   misfit AT known epicentre: {f_epi:.3f} min "
                  f"(= min + {f_epi - vmin:.3f}); epicentre is {inside} the "
                  f"min+1 valley")
        else:
            print("   known epicentre cell failed the coverage threshold (NaN)")

    _one(res.rms_anchored, "ANCHORED rms", True)
    _one(res.std_free, "ORIGIN-TIME-FREE std", False)


# ======================================================================
#  Raw coverage diagnostic (fill=False)
# ======================================================================
def raw_coverage(wf_points, bathy, cfg):
    """Trace every WF1 fan once, combine, and grid with fill=False to show
    raw ray coverage so we can confirm the candidate grid sits in a
    well-covered region.  Returns (lon_bin, lat_bin, travel_time_hours)."""
    blon, blat, bdepth = bathy
    cen_lon = np.mean(cfg.cand_lon)
    cen_lat = np.mean(cfg.cand_lat)
    half, step = cfg.fan_halfwidth_deg, cfg.azimuth_step_deg

    all_lon, all_lat = [], []
    for xlon, xlat in wf_points:
        brg = initial_bearing(xlon, xlat, cen_lon, cen_lat)
        az = np.arange(brg - half, brg + half + step, step) % 360.0
        rl, ra, _ = tt.trace_rays(blon, blat, bdepth, dt=cfg.dt,
                                  max_time=cfg.max_time, source_lon=xlon,
                                  source_lat=xlat, azimuths_deg=az,
                                  wavelength=cfg.wavelength)
        all_lon.append(rl)
        all_lat.append(ra)
    ray_lon = np.vstack(all_lon)        # (sum n_az, n_steps)
    ray_lat = np.vstack(all_lat)
    return tt.grid_travel_times(ray_lon, ray_lat, dt=cfg.dt,
                                lon_arr=blon, lat_arr=blat, depth=bdepth,
                                bin_deg=cfg.bin_deg, fill=False)


# ======================================================================
#  Saving + plotting
# ======================================================================
def save_outputs(res: BPResult, cfg: Config):
    import os
    os.makedirs(cfg.out_dir, exist_ok=True)
    stem = os.path.join(cfg.out_dir, cfg.tag)

    # .npz with everything + lon/lat axes
    np.savez(stem + ".npz",
             clon=res.clon, clat=res.clat,
             rms_anchored=(res.rms_anchored if res.rms_anchored is not None
                           else np.array([])),
             std_free=res.std_free,
             n_valid=res.n_valid, coverage_ok=res.coverage_ok,
             stack=res.stack,
             known_dt=(res.known_dt if res.known_dt is not None else np.array([])),
             rupture_delay=(res.rupture_delay if res.rupture_delay is not None
                            else np.array([])))
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
            ds.wavelength_m = (np.nan if res.wavelength is None else res.wavelength)
        print(f"saved {stem}.nc")
    except ImportError:
        print("netCDF4 not available; skipped .nc (npz has everything)")


def _scene_extent(cfg, res, wf, margin=0.5):
    """Lon/lat limits framing the whole back-projection scene: candidate grid,
    WF1, epicentre and the located minimum, with a margin."""
    lons = [cfg.cand_lon[0], cfg.cand_lon[1], cfg.epi_lon,
            float(wf[:, 0].min()), float(wf[:, 0].max())]
    lats = [cfg.cand_lat[0], cfg.cand_lat[1], cfg.epi_lat,
            float(wf[:, 1].min()), float(wf[:, 1].max())]
    return (min(lons) - margin, max(lons) + margin), \
           (min(lats) - margin, max(lats) + margin)


def _draw_misfit(ax, fig, plt, res, cfg, field, title):
    """Left panel: the candidate-grid misfit map with epicentre and minimum."""
    cl, ca = res.clon, res.clat
    vmax = cfg.misfit_vmax
    pcm = ax.pcolormesh(cl, ca, field, shading="nearest", cmap="gist_heat_r",
                        vmin=0.0, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="misfit (min)", shrink=0.85,
                 extend="max" if vmax is not None else "neither")
    # contour the valley shape: levels stepped above the minimum so the
    # narrowing toward the best-fit source is easy to read.
    if np.isfinite(field).any():
        vmin, iy, ix = argmin_2d(field)
        # start at +0.5 min (above the ~0.25-min ray-discretization floor) and
        # widen geometrically so the contours show valley shape, not numerical
        # stair-steps in the low-gradient floor.
        levels = vmin + np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
        levels = levels[levels < np.nanmax(field)]
        if levels.size:
            cs = ax.contour(cl, ca, field, levels=levels, colors="k",
                            linewidths=0.7, alpha=0.85, zorder=3)
            ax.clabel(cs, fmt=lambda v: f"+{v - vmin:.2g}", fontsize=6,
                      inline=True, colors="k")
        ax.plot(cl[ix], ca[iy], marker="o", ms=11, mfc="none", mec="magenta",
                mew=2.0, label="misfit min", zorder=6)
    ax.plot(cfg.epi_lon, cfg.epi_lat, marker="*", ms=18, mfc="red", mec="k",
            mew=0.8, label="epicentre", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("lon (E)")
    ax.set_ylabel("lat (N)")
    ax.set_xlim(cl[0], cl[-1])
    ax.set_ylim(ca[0], ca[-1])
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)


def _draw_data_panel(ax, fig, plt, res, cfg, field, wf, swot):
    """Right panel: the DATA being back-projected.  Actual SWOT sea-surface
    height anomaly, the hand-traced WF1 leading edge highlighted on top, and the
    located minimum + epicentre so the input wavefield and the inferred source
    appear in one frame."""
    cl, ca = res.clon, res.clat
    (x0, x1), (y0, y1) = _scene_extent(cfg, res, wf)

    if swot is not None:
        slon, slat, ssh = swot
        inwin = (slon >= x0) & (slon <= x1) & (slat >= y0) & (slat <= y1)
        # symmetric, zero-centred scale from a robust in-window percentile
        if inwin.any():
            vlim = np.percentile(np.abs(ssh[inwin]), 98)
        else:
            vlim = np.percentile(np.abs(ssh), 98)
        sc = ax.scatter(slon[inwin], slat[inwin], c=ssh[inwin], s=3,
                        cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                        alpha=0.8, linewidths=0, zorder=1)
        fig.colorbar(sc, ax=ax, label="SWOT SSH anomaly (m)", shrink=0.85)

    # WF1 leading edge (the actual data back-projected)
    ax.plot(wf[:, 0], wf[:, 1], "-", color="lime", lw=2.0, zorder=4)
    ax.plot(wf[:, 0], wf[:, 1], "o", mfc="white", mec="none", ms=4, lw=0,
            zorder=5, label="WF1 (traced front)")

    # connector WF1 centroid -> located minimum
    if np.isfinite(field).any():
        _, iy, ix = argmin_2d(field)
        mlon, mlat = cl[ix], ca[iy]
        ax.plot([wf[:, 0].mean(), mlon], [wf[:, 1].mean(), mlat], "--",
                color="magenta", lw=1.0, alpha=0.7, zorder=3)
        ax.plot(mlon, mlat, "o", mfc="none", mec="magenta", mew=2.0, ms=12,
                zorder=6, label="misfit min")

    ax.plot(cfg.epi_lon, cfg.epi_lat, marker="*", ms=16, mfc="red", mec="k",
            zorder=6, label="epicentre")
    ax.add_patch(plt.Rectangle((cl[0], ca[0]), cl[-1] - cl[0], ca[-1] - ca[0],
                               fill=False, ec="0.25", lw=0.8, ls=":",
                               label="candidate grid", zorder=2))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title("Data: SWOT swath + WF1  ->  inferred source")
    ax.set_xlabel("lon (E)")
    ax.set_ylabel("lat (N)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def plot_misfit_figure(res, cfg, field, title, suffix, wf, swot):
    """One figure: misfit map (left) beside the SWOT+WF1 data panel (right)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True)
    _draw_misfit(axes[0], fig, plt, res, cfg, field, title)
    _draw_data_panel(axes[1], fig, plt, res, cfg, field, wf, swot)
    out = os.path.join(cfg.out_dir, cfg.tag + suffix + ".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def plot_coverage_figure(res, cfg, coverage, wf, swot):
    """Standalone raw-ray-coverage diagnostic (fill=False) with WF1 overlaid."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon_bin, lat_bin, T_raw = coverage
    cl, ca = res.clon, res.clat
    fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)
    cov = np.isfinite(T_raw).astype(float)
    pcm = ax.pcolormesh(lon_bin, lat_bin, np.where(cov > 0, T_raw * 60, np.nan),
                        shading="nearest", cmap="plasma_r")
    fig.colorbar(pcm, ax=ax, label="first-arrival time (min)", shrink=0.85)
    ax.add_patch(plt.Rectangle((cl[0], ca[0]), cl[-1] - cl[0], ca[-1] - ca[0],
                               fill=False, ec="cyan", lw=1.5,
                               label="candidate grid"))
    ax.plot(wf[:, 0], wf[:, 1], "-o", color="lime", mfc="white", mec="k",
            ms=4, lw=1.5, label="WF1", zorder=5)
    ax.plot(cfg.epi_lon, cfg.epi_lat, marker="*", ms=16, mfc="red", mec="k",
            zorder=5, label="epicentre")
    ax.set_title("Raw ray coverage (fill=False)")
    ax.set_xlabel("lon (E)")
    ax.set_ylabel("lat (N)")
    ax.set_xlim(cfg.domain_lon)
    ax.set_ylim(cfg.domain_lat)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    out = os.path.join(cfg.out_dir, cfg.tag + "_coverage.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


# ======================================================================
#  Driver
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="WF1 tsunami source back-projection")
    p.add_argument("--wf", dest="wf_path")
    p.add_argument("--bathy", dest="bathy_path")
    p.add_argument("--known-dt", type=float, dest="KNOWN_ARRIVAL_MIN",
                   help="known WF1 arrival, minutes after origin")
    p.add_argument("--wavelength", type=float, default=None,
                   help="deep-water wavelength (m); omit for shallow-water")
    p.add_argument("--coverage-frac", type=float, dest="coverage_frac")
    p.add_argument("--fan-halfwidth", type=float, dest="fan_halfwidth_deg")
    p.add_argument("--az-step", type=float, dest="azimuth_step_deg")
    p.add_argument("--dt", type=float)
    p.add_argument("--max-time", type=float)
    p.add_argument("--bin-deg", type=float)
    p.add_argument("--n-wf-points", type=int, dest="n_wf_points",
                   help="resample WF1 polyline to this many points "
                        "(default: use raw digitised vertices)")
    p.add_argument("--tag")
    p.add_argument("--out-dir", dest="out_dir")
    p.add_argument("--free-only", action="store_true",
                   help="origin-time-free map only (known_dt=None)")
    p.add_argument("--uniform-dt", action="store_true",
                   help="ignore per-pixel SWOT times; use scalar KNOWN_ARRIVAL_MIN")
    p.add_argument("--swot-times", dest="swot_times_path",
                   help="CSV of per-pixel SWOT times (cols: time[s],lat,lon)")
    p.add_argument("--swot-ssh", dest="swot_ssh_path",
                   help="SWOT SSH file for the data panel (cols: lon lat ssh)")
    p.add_argument("--rupture-speed", type=float, dest="rupture_speed_kms",
                   help="rupture speed km/s for the epicentre->source delay")
    p.add_argument("--misfit-vmax", type=float, dest="misfit_vmax",
                   help="saturate misfit colour scale at this value (minutes)")
    p.add_argument("--no-rupture", action="store_true",
                   help="disable the rupture-propagation delay term")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    # apply CLI overrides
    for k, v in vars(args).items():
        if k in ("free_only", "uniform_dt", "no_rupture"):
            continue
        if v is not None and hasattr(cfg, k):
            cfg = replace(cfg, **{k: v})
    if args.wavelength is not None:
        cfg = replace(cfg, wavelength=args.wavelength)
    if args.no_rupture:
        cfg = replace(cfg, rupture_speed_kms=None)

    print("Loading WF1 polyline ...")
    wf = load_wf_polyline(cfg.wf_path)
    n_raw = len(wf)
    wf = resample_polyline(wf, cfg.n_wf_points)
    resampled = "" if cfg.n_wf_points is None else f" (resampled from {n_raw})"
    print(f"  {len(wf)} WF1 points{resampled}, "
          f"lon {wf[:,0].min():.2f}..{wf[:,0].max():.2f}, "
          f"lat {wf[:,1].min():.2f}..{wf[:,1].max():.2f}")

    print("Loading + subsetting bathymetry ...")
    bathy = load_domain_bathymetry(cfg)
    blon, blat, bdepth = bathy
    print(f"  domain {blon[0]:.1f}..{blon[-1]:.1f} E, {blat[0]:.1f}..{blat[-1]:.1f} N "
          f"-> depth {bdepth.shape}")

    cand = build_candidate_grid(cfg)
    print(f"  candidate grid {len(cand[0])} x {len(cand[1])} cells")

    # ---- build the arrival-time anchor ----
    #   free-only        -> None (origin-time-free map only)
    #   SWOT csv set     -> per-pixel times matched to each WF1 point
    #   otherwise        -> uniform scalar KNOWN_ARRIVAL_MIN
    if args.free_only:
        known_dt = None
        anchor_desc = "none (free-only)"
    elif cfg.swot_times_path and not args.uniform_dt:
        print(f"Matching SWOT per-pixel times: {cfg.swot_times_path}")
        known_dt, match_km = swot_times_for_wf(wf, cfg.swot_times_path)
        print(f"  per-pixel arrival {known_dt.min():.2f}..{known_dt.max():.2f} min "
              f"(mean {known_dt.mean():.2f}); nearest-pixel match "
              f"{match_km.min():.1f}..{match_km.max():.1f} km")
        if match_km.max() > 25.0:
            print("  WARNING: some WF1 points are >25 km from any SWOT pixel; "
                  "check WF1 lies on the swath.")
        anchor_desc = "per-pixel SWOT"
    else:
        known_dt = cfg.KNOWN_ARRIVAL_MIN
        anchor_desc = f"uniform {known_dt:.2f} min"

    print("Raw coverage diagnostic (fill=False) ...")
    cov = raw_coverage(wf, bathy, cfg)

    if known_dt is not None:
        forward_consistency(wf, bathy, cfg, known_dt)

    print(f"Back-projecting (wavelength={cfg.wavelength}, anchor={anchor_desc}) ...")
    res = backproject(wf, cand, bathy, cfg,
                      wavelength=cfg.wavelength, known_dt=known_dt)

    report(res, cfg)
    save_outputs(res, cfg)

    # SWOT sea-surface-height field for the data panels (None if unavailable)
    swot_pts = None
    if cfg.swot_ssh_path and os.path.exists(cfg.swot_ssh_path):
        swot_pts = load_swot_ssh(cfg.swot_ssh_path)

    # Separate figure per misfit map, each beside its SWOT+WF1 data panel.
    if res.rms_anchored is not None:
        plot_misfit_figure(res, cfg, res.rms_anchored,
                           "Anchored rms (per-pixel arrival)", "_anchored",
                           wf, swot_pts)
    plot_misfit_figure(res, cfg, res.std_free,
                       "Origin-time-free std", "_free", wf, swot_pts)
    plot_coverage_figure(res, cfg, cov, wf, swot_pts)


if __name__ == "__main__":
    main()

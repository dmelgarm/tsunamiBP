"""The back-projection misfit engine.

Method (read carefully -- the ray direction is the easy thing to get backwards)
------------------------------------------------------------------------------
We do NOT shoot rays from candidate sources.  In a fixed slowness field travel
time is reciprocal:  T(A -> B) = T(B -> A).  So for each wavefront point x_j we
trace rays *from x_j* (the observation), fanned toward the candidate-source
region, and read the travel time AT each candidate cell S.  That gives
T_j(S) = T(x_j -> S) = T(S -> x_j): the travel time a wave would take from a
hypothetical source at S to the observed wavefront point x_j.  Stacking over all
j gives an (N_WF, nlat, nlon) array.

Two misfit maps over S are then computed:

  * Anchored (uses the known per-pixel arrival):
        rms(S) = sqrt( mean_j ( t_rup(S) + T_j(S) - t_j )^2 )
  * Origin-time-free:
        std(S) = sqrt( mean_j ( T_j(S) - mean_j T_j(S) )^2 )

Both use root-MEAN-square with NaN masking (N varies per cell because rays die
in shallow water / shadow zones), and a cell must be reached by at least
``coverage_frac`` of the wavefront points before it is assigned a misfit.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

import TsunamiTrace as tt

from .geodesy import haversine_km, initial_bearing
from .progress import maybe_track


@dataclass
class BPResult:
    clon: np.ndarray            # candidate lon axis (nlon,)
    clat: np.ndarray            # candidate lat axis (nlat,)
    stack: np.ndarray           # T_j(S), minutes, (N_WF, nlat, nlon)
    n_valid: np.ndarray         # # WF points reaching each cell, (nlat, nlon)
    coverage_ok: np.ndarray     # bool mask, (nlat, nlon)
    rms_anchored: np.ndarray | None   # anchored misfit (min), (nlat, nlon) or None
    std_free: np.ndarray        # origin-time-free misfit (min), (nlat, nlon)
    wavelength: float | None
    known_dt: np.ndarray | None  # per-pixel anchor times (N,) minutes, or None
    rupture_delay: np.ndarray | None  # t_rup(S) (nlat, nlon) minutes, or None


def backproject(wf_points, candidate_grid, bathy, cfg,
                wavelength=None, known_dt=None, progress_label=None):
    """Back-project one wavefront onto a candidate-source grid; return both maps.

    Parameters
    ----------
    wf_points : (N, 2) array
        Ordered wavefront polyline as [lon, lat] rows (the observed front).
    candidate_grid : (clon, clat)
        1-D candidate-source longitude and latitude axes (cell centres).
    bathy : (lon, lat, depth)
        Domain bathymetry -- depth (n_lon, n_lat), positive = ocean.
    cfg : Config
        Tracing / misfit parameters (dt, max_time, fan geometry, bin_deg,
        coverage_frac, epicentre, rupture_speed_kms).
    wavelength : float or None
        Deep-water wavelength (m) passed straight to ``trace_rays``.  None ->
        shallow-water sqrt(g*h).  This is the ONLY change needed to switch the
        same engine from a long-wave front to a dispersive (short-wavelength)
        front.
    known_dt : float, (N,) array, or None
        Known arrival time(s) after origin, in MINUTES.
          * scalar  -> every wavefront point shares one arrival time;
          * (N,) array -> a PER-PIXEL arrival time, one per wavefront point;
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

    # ---- sanity: wavefront points are in-domain and in the ocean ----
    from scipy.interpolate import RegularGridInterpolator
    depth_at = RegularGridInterpolator((blon, blat), bdepth,
                                       bounds_error=False, fill_value=np.nan)
    wlon, wlat = wf_points[:, 0], wf_points[:, 1]
    assert np.all((wlon >= blon[0]) & (wlon <= blon[-1]) &
                  (wlat >= blat[0]) & (wlat <= blat[-1])), \
        "Some wavefront points lie outside the bathymetry domain."
    wf_depth = depth_at((wlon, wlat))
    assert np.all(wf_depth > 0), \
        f"Some wavefront points are on land (depths={np.round(wf_depth)})."

    # Candidate cell centres, flattened, for sampling.
    CLON, CLAT = np.meshgrid(clon, clat)            # (nlat, nlon)
    cen_lon, cen_lat = clon.mean(), clat.mean()     # fan aim-point

    stack = np.full((N, nlat, nlon), np.nan)        # T_j(S) in MINUTES

    half = cfg.fan_halfwidth_deg
    step = cfg.azimuth_step_deg

    # progress is over the wavefront points -- the actual (vectorised-per-point)
    # work loop; all candidate cells are computed at once inside each iteration.
    for j in maybe_track(range(N), N, progress_label):
        xlon, xlat = wf_points[j]

        # -------------------------------------------------------------
        # RECIPROCITY STEP -- DO NOT "FIX" THIS BY SHOOTING FROM THE
        # CANDIDATES.  We trace rays FROM the wavefront observation point x_j
        # and read travel time AT the candidate cells S, because in a fixed
        # slowness field T(x_j -> S) == T(S -> x_j).  Shooting from every
        # candidate instead would be N_candidates source fans (thousands)
        # rather than N_WF (a dozen), and -- more importantly -- it is the
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

    # ---- coverage: how many wavefront points reached each candidate cell ----
    n_valid = np.sum(~np.isnan(stack), axis=0)          # (nlat, nlon)
    coverage_ok = n_valid >= cfg.coverage_frac * N

    # ---- rupture-propagation delay ----
    # A candidate source S does not radiate its tsunami until the rupture front,
    # nucleating at the epicentre, reaches it: t_rup(S) = dist(epi, S) / v_rup.
    # Horizontal distance only (shallow megathrust dip -> first-order OK).  This
    # is a PER-CELL constant across wavefront points, so it shifts the ANCHORED
    # map only and breaks along-arc degeneracy; the origin-time-free std is
    # unaffected (a per-cell constant drops out of the spread over j).
    rup = None
    if cfg.rupture_speed_kms:
        d_km = haversine_km(cfg.epi_lon, cfg.epi_lat, CLON, CLAT)   # (nlat,nlon)
        rup = d_km / cfg.rupture_speed_kms / 60.0                   # minutes

    # Cells that no wavefront point reached are all-NaN columns; nanmean warns
    # "Mean of empty slice" on them.  They are masked out below anyway.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # ---- anchored misfit (uses known arrival, per-pixel) ----
        # kd has shape (N,); broadcast to (N, 1, 1) so each wavefront point is
        # matched to its OWN observation time t_j.  Modelled arrival is the
        # rupture delay to reach S plus the tsunami travel time from S to x_j.
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

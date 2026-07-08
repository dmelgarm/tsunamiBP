"""Diagnostics and text reporting: minima, offsets, resolution, consistency."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np

import TsunamiTrace as tt

from .engine import resolve_wave
from .geodesy import haversine_km, initial_bearing
from .progress import maybe_track

if TYPE_CHECKING:
    from .config import Config
    from .engine import BPResult


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


def forward_consistency(wf_points, bathy, cfg, known_dt, wave=None):
    """Forward sanity check / auto-surfaced finding.

    Trace rays FROM the configured epicentre and read the model travel time at
    every wavefront point, then compare to the observed (per-pixel) arrival
    times.  A large mean gap (model - observed) means the epicentre, the arrival
    times, or the wave-speed model are mutually inconsistent -- which is exactly
    why an anchored minimum can sit off the epicentre (e.g. finite-source
    directivity).  Returns the per-pixel model travel times (minutes)."""
    blon, blat, bdepth = bathy
    if wave is None:
        wave = resolve_wave(wf_points, bathy, wavelength=cfg.wavelength)
    brg = initial_bearing(cfg.epi_lon, cfg.epi_lat,
                          wf_points[:, 0].mean(), wf_points[:, 1].mean())
    az = np.arange(brg - 60.0, brg + 60.0 + cfg.azimuth_step_deg,
                   cfg.azimuth_step_deg) % 360.0
    rl, ra, _ = tt.trace_rays(blon, blat, bdepth, dt=cfg.dt, max_time=cfg.max_time,
                              source_lon=cfg.epi_lon, source_lat=cfg.epi_lat,
                              azimuths_deg=az, **wave.trace_kwargs)
    lon_b, lat_b, T = tt.grid_travel_times(rl, ra, dt=cfg.dt, lon_arr=blon,
                                           lat_arr=blat, depth=bdepth,
                                           bin_deg=cfg.bin_deg, fill=True)
    tmin, _ = tt.sample_travel_times(lon_b, lat_b, T, wf_points[:, 0],
                                     wf_points[:, 1], max_snap_bins=3)
    model = tmin * 60.0                                   # minutes, per pixel
    kd = np.full(len(wf_points), float(known_dt)) if np.ndim(known_dt) == 0 \
        else np.asarray(known_dt, float)
    print("\n--- forward consistency (trace FROM configured epicentre) ---")
    if not np.isfinite(model).any():
        # No ray from the epicentre reached the wavefront within max_time (the
        # epicentre is farther than max_time allows, common for slow dispersive
        # fronts).  This is only a diagnostic and does not affect the
        # back-projection minima, which trace FROM the wavefront.
        print(f"  no epicentre ray reached the wavefront within max_time="
              f"{cfg.max_time:.0f}s (~{cfg.max_time/60:.0f} min); "
              "skipping the gap check.")
        print("  => raise max_time above the epicentre->WF travel time to run "
              "this diagnostic.")
        return model
    gap = model - kd                                     # per pixel
    mean_gap = float(np.nanmean(gap))
    print(f"  model travel time epicentre -> WF: "
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


def report(res: "BPResult", cfg: "Config"):
    """Print minima, offsets, implied origin time, and valley shape."""
    origin = datetime.fromisoformat(cfg.origin_time_utc).replace(tzinfo=timezone.utc)
    N = res.stack.shape[0]
    print("\n" + "=" * 70)
    print(f"BACK-PROJECTION REPORT  ({cfg.tag})")
    print("=" * 70)
    print(f"WF points: {N}   candidate grid: {len(res.clon)} lon x "
          f"{len(res.clat)} lat  @ {cfg.cand_dlon}/{cfg.cand_dlat} deg")
    print(f"coverage threshold: >= {cfg.coverage_frac:.0%} of WF points "
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


def raw_coverage(wf_points, bathy, cfg, wave=None, progress_label=None):
    """Trace every wavefront fan once, combine, and grid with fill=False to show
    raw ray coverage so we can confirm the candidate grid sits in a
    well-covered region.  Returns (lon_bin, lat_bin, travel_time_hours)."""
    blon, blat, bdepth = bathy
    if wave is None:
        wave = resolve_wave(wf_points, bathy, wavelength=cfg.wavelength)
    cen_lon = np.mean(cfg.cand_lon)
    cen_lat = np.mean(cfg.cand_lat)
    half, step = cfg.fan_halfwidth_deg, cfg.azimuth_step_deg

    all_lon, all_lat = [], []
    for xlon, xlat in maybe_track(wf_points, len(wf_points), progress_label):
        brg = initial_bearing(xlon, xlat, cen_lon, cen_lat)
        az = np.arange(brg - half, brg + half + step, step) % 360.0
        rl, ra, _ = tt.trace_rays(blon, blat, bdepth, dt=cfg.dt,
                                  max_time=cfg.max_time, source_lon=xlon,
                                  source_lat=xlat, azimuths_deg=az,
                                  **wave.trace_kwargs)
        all_lon.append(rl)
        all_lat.append(ra)
    ray_lon = np.vstack(all_lon)        # (sum n_az, n_steps)
    ray_lat = np.vstack(all_lat)
    return tt.grid_travel_times(ray_lon, ray_lat, dt=cfg.dt,
                                lon_arr=blon, lat_arr=blat, depth=bdepth,
                                bin_deg=cfg.bin_deg, fill=False)

"""Emission-time search -- a SEPARATE hypothesis from rupture-anchoring.

Treat the front as radiating from a source S at a single emission time
tau = origin + Delta (tau >= 0), IGNORING the rupture delay, and ask which
(location, emission-time) best explains the observed per-pixel arrival times:

    misfit(S, tau) = sqrt( mean_j ( tau + T_j(S) - t_j )^2 )

where T_j(S) = res.stack (already ray-traced) and t_j = res.known_dt.  It is a
cheap post-process over the existing stack -- no re-tracing, no engine changes.

Relationship to the origin-time-free map (an EXACT identity): minimising this
misfit over an unconstrained, continuous tau gives

    tau*(S)                = mean_j( t_j - T_j(S) )
    min_tau misfit(S, tau) = std_j( T_j(S) - t_j ) = std_free(S)   (exactly),

so the origin-time-free map IS the tau-marginalised envelope of this family
(std_free under its corrected definition; note this is NOT std_j(T_j), the
timing-free std_geom).  The search still earns its keep through the tau >= 0
constraint -- tau* can come out negative, which is unphysical -- and by
reporting tau as a checkable emission time.
"""
from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .diagnostics import argmin_2d
from .geodesy import haversine_km
from .engine import backproject
from .perturb import perturb_wavefront, to_km, tangent_normal
# Tracer's own dispersive group-speed model: travel time accrues at the GROUP
# speed (slowness = 1/c_group), the celerity that produced the stack.  Imported
# so the cheap-path reconstruction stays IDENTICAL to the validated throwaway.
from TsunamiTrace.raytracing import _dispersive_group_speed

_G = 9.8


@dataclass
class TimeSearchResult:
    taus: np.ndarray            # (M,) emission times swept (minutes)
    misfit_min: np.ndarray      # (M,) minimum misfit over S at each tau
    best_lon: np.ndarray        # (M,) best-fit lon at each tau
    best_lat: np.ndarray        # (M,) best-fit lat at each tau
    best_tau: float             # tau of the global minimum
    best_lon0: float            # location of the global minimum
    best_lat0: float
    best_misfit: float
    best_map: np.ndarray        # (nlat, nlon) misfit map at best_tau
    clon: np.ndarray
    clat: np.ndarray


def time_search(res, cfg):
    """Sweep emission time tau and return the best (location, tau) for one
    wavefront.  ``res`` is a BPResult (needs ``stack``, ``known_dt``,
    ``coverage_ok``); ``cfg`` supplies the tau grid and epicentre."""
    assert res.known_dt is not None, \
        "time search needs observed arrival times (not a free-only run)"
    kd = res.known_dt                                  # (N,)
    stack = res.stack                                  # (N, nlat, nlon)
    clon, clat = res.clon, res.clat
    taus = np.arange(0.0, cfg.time_max_min + cfg.time_step_min / 2.0,
                     cfg.time_step_min)
    M = len(taus)
    misfit_min = np.full(M, np.nan)
    best_lon = np.full(M, np.nan)
    best_lat = np.full(M, np.nan)
    best_tau = None
    best_val = np.inf
    best_map = None
    best_ll = (np.nan, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for k, tau in enumerate(taus):
            # modelled arrival = tau + T_j(S); residual vs observed t_j
            m = np.sqrt(np.nanmean((stack + tau - kd[:, None, None]) ** 2, axis=0))
            m[~res.coverage_ok] = np.nan
            if np.isfinite(m).any():
                v, iy, ix = argmin_2d(m)
                misfit_min[k] = v
                best_lon[k] = clon[ix]
                best_lat[k] = clat[iy]
                if v < best_val:
                    best_val = v
                    best_tau = float(tau)
                    best_map = m
                    best_ll = (float(clon[ix]), float(clat[iy]))

    if best_map is None:
        raise ValueError("time search: no candidate cells passed coverage")

    return TimeSearchResult(
        taus=taus, misfit_min=misfit_min, best_lon=best_lon, best_lat=best_lat,
        best_tau=best_tau, best_lon0=best_ll[0], best_lat0=best_ll[1],
        best_misfit=best_val, best_map=best_map, clon=clon, clat=clat)


def report_time_search(ts, cfg):
    off = haversine_km(ts.best_lon0, ts.best_lat0, cfg.epi_lon, cfg.epi_lat)
    print("\n--- emission-time search (tau = origin + Delta; rupture delay OFF) ---")
    print(f"  tau grid: 0..{ts.taus[-1]:.0f} min, step {cfg.time_step_min:.0f} min")
    print(f"  best emission time: tau = {ts.best_tau:.1f} min after origin")
    print(f"  best location: {ts.best_lon0:.3f} E, {ts.best_lat0:.3f} N "
          f"(offset {off:.0f} km from epicentre)")
    print(f"  min misfit = {ts.best_misfit:.3f} min")
    fin = np.isfinite(ts.misfit_min)
    if fin.sum() > 1:
        span = float(np.nanmax(ts.misfit_min) - ts.best_misfit)
        print(f"  misfit-vs-tau span across the grid: {span:.2f} min "
              f"({'well-constrained' if span > 0.5 else 'flat / weak'} in tau)")


def write_time_search_csv(ts, cfg):
    path = os.path.join(cfg.out_dir, cfg.tag + "_timesearch.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau_min", "min_misfit_min", "best_lon", "best_lat", "offset_km"])
        for k in range(len(ts.taus)):
            if np.isfinite(ts.misfit_min[k]):
                off = haversine_km(ts.best_lon[k], ts.best_lat[k],
                                   cfg.epi_lon, cfg.epi_lat)
                w.writerow([f"{ts.taus[k]:.1f}", f"{ts.misfit_min[k]:.3f}",
                            f"{ts.best_lon[k]:.3f}", f"{ts.best_lat[k]:.3f}",
                            f"{off:.1f}"])
    print(f"saved {path}")


def plot_time_search(ts, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axc, axm) = plt.subplots(1, 2, figsize=(13.0, 5.6),
                                   constrained_layout=True)

    # ---- left: misfit vs tau ----
    fin = np.isfinite(ts.misfit_min)
    axc.plot(ts.taus[fin], ts.misfit_min[fin], "-o", ms=4, color="C0")
    axc.axvline(ts.best_tau, color="magenta", lw=1.0, ls="--")
    axc.plot(ts.best_tau, ts.best_misfit, "o", mfc="magenta", mec="k", ms=9,
             zorder=5, label=f"best tau = {ts.best_tau:.0f} min")
    axc.set_xlabel("emission time tau (min after origin)")
    axc.set_ylabel("min misfit over S (min)")
    axc.set_title("Misfit vs emission time")
    axc.legend(fontsize=8)

    # ---- right: misfit map at the best tau, with per-tau location track ----
    vmax = cfg.misfit_vmax
    pcm = axm.pcolormesh(ts.clon, ts.clat, ts.best_map, shading="nearest",
                         cmap="gist_heat_r", vmin=0.0, vmax=vmax)
    fig.colorbar(pcm, ax=axm, label="misfit (min)", shrink=0.85,
                 extend="max" if vmax is not None else "neither")
    if np.isfinite(ts.best_map).any():
        vmin_, _, _ = argmin_2d(ts.best_map)
        levels = vmin_ + np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
        levels = levels[levels < np.nanmax(ts.best_map)]
        if levels.size:
            axm.contour(ts.clon, ts.clat, ts.best_map, levels=levels, colors="k",
                        linewidths=0.7, alpha=0.85)
    tfin = np.isfinite(ts.best_lon)
    sc = axm.scatter(ts.best_lon[tfin], ts.best_lat[tfin], c=ts.taus[tfin],
                     cmap="viridis", s=18, zorder=4, edgecolors="k",
                     linewidths=0.3)
    fig.colorbar(sc, ax=axm, label="tau (min)", shrink=0.7, location="bottom")
    axm.plot(ts.best_lon0, ts.best_lat0, "o", mfc="none", mec="magenta", mew=2.0,
             ms=12, zorder=6, label=f"best (tau={ts.best_tau:.0f} min)")
    axm.plot(cfg.epi_lon, cfg.epi_lat, marker="*", ms=16, mfc="yellow", mec="k",
             mew=0.8, zorder=6, label="epicentre")
    axm.set_xlim(ts.clon[0], ts.clon[-1])
    axm.set_ylim(ts.clat[0], ts.clat[-1])
    axm.set_xlabel("lon (E)")
    axm.set_ylabel("lat (N)")
    axm.set_title(f"Misfit map at best tau ({ts.best_tau:.0f} min)")
    axm.legend(loc="lower left", fontsize=8, framealpha=0.85)

    out = os.path.join(cfg.out_dir, cfg.tag + "_timesearch.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def run_time_search(res, cfg):
    """time_search + report + CSV + figure."""
    ts = time_search(res, cfg)
    report_time_search(ts, cfg)
    write_time_search_csv(ts, cfg)
    plot_time_search(ts, cfg)
    return ts


# ======================================================================
#  source bootstrap (a wrapper AROUND time_search; no misfit/tracer changes)
# ======================================================================
def _eikonal_gradients(res, wf, bathy, wave, cand):
    """Precompute the per-(point, candidate) travel-time gradient for the cheap
    path: a first-order eikonal expansion of T about each base crest point.
    The tangential gradient dT/ds is measured from the stack; the normal-gradient
    magnitude is sqrt(slowness^2 - (dT/ds)^2) with slowness = 1/c_group from the
    tracer's own group-speed model (the celerity that produced the stack), signed
    by which side of the front the source sits on.  IDENTICAL to the validated
    throwaway -- do not alter or the gamma_9 validation no longer applies.
    Returns (g_t, sn, t_hat, n_hat, base_xy, c_lon, c_lat, clamp_fraction)."""
    blon, blat, bdepth = bathy
    clon, clat = cand
    stack = res.stack
    c_lon, c_lat = float(wf[:, 0].mean()), float(wf[:, 1].mean())
    bx, by = to_km(wf[:, 0], wf[:, 1], c_lon, c_lat)
    base_xy = np.column_stack([bx, by])
    t_hat, n_hat = tangent_normal(base_xy)

    seg = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    g_t = np.gradient(stack, s, axis=0)                  # dT/ds, min/km, per cell

    depth_at = RegularGridInterpolator((blon, blat), bdepth,
                                       bounds_error=False, fill_value=np.nan)
    depth_j = depth_at((wf[:, 0], wf[:, 1]))
    if wave.omega is None:
        cg = np.sqrt(_G * depth_j)
    else:
        cg, _ = _dispersive_group_speed(depth_j, wave.omega)
    sl = (1000.0 / 60.0) / cg                            # slowness, min/km

    sn_mag = np.sqrt(np.maximum(sl[:, None, None] ** 2 - g_t ** 2, 0.0))
    CLON, CLAT = np.meshgrid(clon, clat)
    cx, cy = to_km(CLON, CLAT, c_lon, c_lat)
    sign = np.empty_like(sn_mag)
    for j in range(len(base_xy)):
        sign[j] = np.sign(n_hat[j, 0] * (base_xy[j, 0] - cx) +
                          n_hat[j, 1] * (base_xy[j, 1] - cy))
    sn = sn_mag * sign
    clamp = float(np.mean(sl[:, None, None] ** 2 < g_t ** 2))
    return g_t, sn, t_hat, n_hat, base_xy, c_lon, c_lat, clamp


def _eikonal_apply(stack, g_t, sn, t_hat, n_hat, disp):
    """Reconstruct the stack at perturbed points; disp is (N,2) km displacement."""
    a = np.einsum("ij,ij->i", disp, t_hat)
    b = np.einsum("ij,ij->i", disp, n_hat)
    return stack + a[:, None, None] * g_t + b[:, None, None] * sn


@dataclass
class BootstrapResult:
    boot_lon: np.ndarray        # (n_boot,) recovered source lon per replicate
    boot_lat: np.ndarray        # (n_boot,)
    boot_tau: np.ndarray        # (n_boot,) emission time per replicate
    point_lon: float            # unperturbed point estimate
    point_lat: float
    point_tau: float
    clamp_fraction: float       # cheap path only; NaN on the re-trace path
    retrace: bool               # which path produced this cloud


def bootstrap_source(res, wf, bathy, cand, cfg, wave, known_dt_fn, *,
                     n_boot, sigma_normal_km, sigma_shift_km, sigma_rot_deg,
                     retrace=False, rng=None):
    """Bootstrap the emission-time source: perturb the crest ``n_boot`` times and
    re-run the emission-time search per replicate, returning the raw (lon,lat,tau)
    cloud plus the unperturbed point estimate.

    ``retrace=False`` (default) is the cheap path: the stack is traced ONCE (it is
    ``res.stack``, already computed) and reconstructed at the perturbed points by
    the first-order eikonal expansion above.  ``retrace=True`` re-traces via
    ``backproject`` every replicate (slow, gold standard).  ``known_dt_fn(points)
    -> (N,) minutes`` supplies the observation anchor for each perturbed crest,
    mirroring how the deterministic run derived known_dt.  Enforces tau>=0 via
    ``time_search`` (unchanged).  Does not save anything."""
    if rng is None:
        rng = np.random.default_rng()

    ts0 = time_search(res, cfg)                          # unperturbed estimate
    point = (ts0.best_lon0, ts0.best_lat0, ts0.best_tau)

    clamp = float("nan")
    if not retrace:
        g_t, sn, t_hat, n_hat, base_xy, c_lon, c_lat, clamp = \
            _eikonal_gradients(res, wf, bathy, wave, cand)

    lon = np.full(n_boot, np.nan)
    lat = np.full(n_boot, np.nan)
    tau = np.full(n_boot, np.nan)
    for r in range(n_boot):
        pert = perturb_wavefront(wf, sigma_normal_km, sigma_shift_km,
                                 sigma_rot_deg, rng)
        kd = known_dt_fn(pert)
        if retrace:
            try:
                res_r = backproject(pert, cand, bathy, cfg, wave=wave,
                                    known_dt=kd, progress_label=None)
            except AssertionError as e:
                # perturbed crest left the ocean/domain: invalid replicate; NaN
                # it loudly and keep the arrays aligned.
                print(f"  bootstrap replicate {r}: rejected ({e}); NaN.")
                continue
            ts = time_search(res_r, cfg)
        else:
            dx, dy = to_km(pert[:, 0], pert[:, 1], c_lon, c_lat)
            disp = np.column_stack([dx, dy]) - base_xy
            Tp = _eikonal_apply(res.stack, g_t, sn, t_hat, n_hat, disp)
            ts = time_search(replace(res, stack=Tp, known_dt=kd), cfg)
        lon[r], lat[r], tau[r] = ts.best_lon0, ts.best_lat0, ts.best_tau

    return BootstrapResult(boot_lon=lon, boot_lat=lat, boot_tau=tau,
                           point_lon=point[0], point_lat=point[1],
                           point_tau=point[2], clamp_fraction=clamp,
                           retrace=retrace)

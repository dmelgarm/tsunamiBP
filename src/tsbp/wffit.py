"""Predicted-vs-digitised wavefront-fit figure.

For a back-projection's best-fit source(s), trace rays FORWARD from the source
and draw the **predicted wavefront as SWOT samples it**: SWOT images each pixel
of the swath at a different time, so the observed front is a curve in space-TIME,
not a snapshot.  The predicted front is therefore the locus where the modelled
crest arrives exactly when SWOT samples that location,

    modelled_arrival(x) = t_swot(x)   <=>   offset + T(x) - t_swot(x) = 0,

with T(x) the forward travel time from the source and ``offset`` the source's
emission convention (rupture delay for anchored, best tau for emission-time,
tau* = mean_j(t_j - T_j) for free).  We overlay this zero-locus on the digitised
polyline, colour the points by their timing residual t_j - modelled_arrival, and
export the predicted front as GMT multi-segment xy (``<tag>_wffit_<name>.xy``).

If there are no observed times (geometry-only run) we fall back to the previous
single-instant isochron at the mean modelled travel time; with a uniform scalar
SWOT anchor the SWOT-time field is constant and the construction reduces to that
same isochron automatically.

Panels: the anchored-minimum source, the free-minimum source, and (when a
time-search result is supplied) the emission-time best source.
"""
from __future__ import annotations

import os

import numpy as np

import TsunamiTrace as tt

from .diagnostics import argmin_2d
from .geodesy import haversine_km, initial_bearing


def _forward_field(source_lon, source_lat, wf_points, bathy, cfg, wave):
    """Trace forward from a source toward the wavefront and return the gridded
    travel-time field in MINUTES (for contouring an isochron)."""
    blon, blat, bdepth = bathy
    brg = initial_bearing(source_lon, source_lat,
                          wf_points[:, 0].mean(), wf_points[:, 1].mean())
    half = cfg.fan_halfwidth_deg
    az = np.arange(brg - half, brg + half + cfg.azimuth_step_deg,
                   cfg.azimuth_step_deg) % 360.0
    rl, ra, _ = tt.trace_rays(blon, blat, bdepth, dt=cfg.dt, max_time=cfg.max_time,
                              source_lon=source_lon, source_lat=source_lat,
                              azimuths_deg=az, **wave.trace_kwargs)
    lon_b, lat_b, T = tt.grid_travel_times(rl, ra, dt=cfg.dt, lon_arr=blon,
                                           lat_arr=blat, depth=bdepth,
                                           bin_deg=cfg.bin_deg, fill=True)
    return lon_b, lat_b, T * 60.0                    # minutes


def _minidx(field):
    """(iy, ix) of the NaN-aware minimum, or None if the field has no valid cell."""
    if field is None or not np.isfinite(field).any():
        return None
    _, iy, ix = argmin_2d(field)
    return iy, ix


def _write_front_xy(cfg, name, segs):
    """Write a predicted-front polyline as GMT multi-segment xy (lon lat)."""
    path = os.path.join(cfg.out_dir, f"{cfg.tag}_wffit_{name}.xy")
    with open(path, "w") as fh:
        for seg in segs:
            if len(seg) == 0:
                continue
            fh.write(">\n")
            for lon, lat in seg:
                fh.write(f"{lon:.6f} {lat:.6f}\n")
    print(f"  saved {path}")


def _panel(ax, fig, plt, res, cfg, wf, bathy, iy, ix, name, color,
           offset, t_swot, swot, extra=None):
    slon, slat = res.clon[ix], res.clat[iy]
    Tj = res.stack[:, iy, ix]                         # modelled travel time -> WF pts
    kd = res.known_dt                                 # observed per-pixel times t_j
    off = haversine_km(slon, slat, cfg.epi_lon, cfg.epi_lat)

    m = 0.3
    x0, x1 = wf[:, 0].min() - m, wf[:, 0].max() + m
    y0, y1 = wf[:, 1].min() - m, wf[:, 1].max() + m

    if swot is not None:
        s_lon, s_lat, ssh = swot
        inw = (s_lon >= x0) & (s_lon <= x1) & (s_lat >= y0) & (s_lat <= y1)
        if inw.any():
            vlim = np.percentile(np.abs(ssh[inw]), 98)
            ax.scatter(s_lon[inw], s_lat[inw], c=ssh[inw], cmap="RdBu_r",
                       vmin=-vlim, vmax=vlim, s=6, alpha=0.30, linewidths=0,
                       zorder=1)

    lon_b, lat_b, T = _forward_field(slon, slat, wf, bathy, cfg, res.wave)

    if offset is not None and t_swot is not None and kd is not None:
        # SWOT-sampled predicted front: zero-locus of offset + T(x) - t_swot(x).
        LON, LAT = np.meshgrid(lon_b, lat_b)
        R = offset + T - t_swot(LON, LAT)
        cs = ax.contour(lon_b, lat_b, R, levels=[0.0], colors=[color],
                        linewidths=2.0, zorder=4)
        _write_front_xy(cfg, name, cs.allsegs[0] if cs.allsegs else [])
        resid = kd - (offset + Tj)                    # observed - modelled arrival
        cbar_label = "SWOT − modelled time (min)"
        fit_txt = f"time RMS {float(np.sqrt(np.nanmean(resid ** 2))):.3f} min"
    else:
        # geometry-only fallback: single-instant isochron at the mean travel time
        Tbar = float(np.nanmean(Tj))
        ax.contour(lon_b, lat_b, T, levels=[Tbar], colors=[color],
                   linewidths=2.0, zorder=4)
        resid = Tj - Tbar
        cbar_label = "digitised − predicted (min)"
        fit_txt = (f"shape RMS {float(np.sqrt(np.nanmean(resid ** 2))):.3f} min "
                   f"· isochron @ {Tbar:.1f} min")

    ax.plot(wf[:, 0], wf[:, 1], "-", color="0.4", lw=1.0, zorder=3)
    rlim = max(0.05, float(np.nanpercentile(np.abs(resid), 98)))
    sc = ax.scatter(wf[:, 0], wf[:, 1], c=resid, cmap="coolwarm",
                    vmin=-rlim, vmax=rlim, s=22, edgecolors="k", linewidths=0.3,
                    zorder=5)
    fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.85)

    ax.plot([], [], color=color, lw=2.0, label="predicted wavefront")
    ax.plot([], [], "-", color="0.4", lw=1.0, label="digitised")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("lon (E)")
    ax.set_ylabel("lat (N)")
    title = f"{name} source {slon:.2f} E, {slat:.2f} N ({off:.0f} km)\n"
    if extra:
        title += extra + " · "
    title += fit_txt
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)


def wffit_figure(res, cfg, wf, bathy, swot=None, ts=None):
    """Write ``<tag>_wffit.png``: predicted vs digitised wavefront for the
    anchored and free best-fit sources (and the emission-time best source when
    ``ts`` is given), zoomed on the wavefront."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kd = res.known_dt

    # SWOT sampling-time field (minutes) for the predicted front.  Per-pixel
    # times -> a spatial field; a uniform scalar anchor -> a constant field
    # (isochron); no observed times -> None, so panels fall back to the isochron.
    t_swot = None
    if cfg.swot_times_path is not None:
        from .io import swot_time_interpolator
        t_swot = swot_time_interpolator(cfg.swot_times_path)
    elif kd is not None:
        c = float(cfg.KNOWN_ARRIVAL_MIN)
        t_swot = lambda lon, lat, c=c: np.full(np.shape(lon), c)

    panels = []                            # (iy, ix, name, color, offset, extra)
    a = _minidx(res.rms_anchored)
    if a is not None:
        rup = (0.0 if res.rupture_delay is None
               else float(res.rupture_delay[a[0], a[1]]))
        panels.append((a[0], a[1], "anchored", "magenta", rup, None))
    free_field = res.std_free if res.std_free is not None else res.std_geom
    free_name = "free" if res.std_free is not None else "geometric"
    f = _minidx(free_field)
    if f is not None:
        # free source's optimal emission time tau* = mean_j(t_j - T_j); None for a
        # geometric (timing-free) run so that panel keeps the isochron.
        off_free = (float(np.nanmean(kd - res.stack[:, f[0], f[1]]))
                    if (res.std_free is not None and kd is not None) else None)
        panels.append((f[0], f[1], free_name, "green", off_free, None))
    if ts is not None:
        jx = int(np.argmin(np.abs(res.clon - ts.best_lon0)))
        jy = int(np.argmin(np.abs(res.clat - ts.best_lat0)))
        panels.append((jy, jx, "emission-time", "darkorange", float(ts.best_tau),
                       f"tau = {ts.best_tau:.0f} min"))
    if not panels:
        print("  wffit skipped: no valid minimum")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels), 5.6),
                             constrained_layout=True, squeeze=False)
    for ax, (iy, ix, name, color, offset, extra) in zip(axes[0], panels):
        _panel(ax, fig, plt, res, cfg, wf, bathy, iy, ix, name, color,
               offset, t_swot, swot, extra)

    fig.suptitle("Predicted vs digitised wavefront (SWOT-sampled; best-fit source)")
    out = os.path.join(cfg.out_dir, cfg.tag + "_wffit.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")

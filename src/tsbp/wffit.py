"""Predicted-vs-digitised wavefront-fit figure.

For a back-projection's best-fit source(s), trace rays FORWARD from the source,
draw the **predicted wavefront** -- the isochron at the mean modelled travel time
to the digitised points -- and overlay it on the digitised polyline, with the
points coloured by their travel-time residual about that isochron.  A spatial
goodness-of-fit for the wavefront shape/location, complementing the scalar
misfit.

Panels: the anchored-minimum source, the free-minimum source, and (when a
time-search result is supplied) the emission-time best source.  The residual is a
*shape* residual (spread about the mean isochron), which is independent of the
emission time tau -- so the emission-time panel differs from the free panel only
through its (tau-grid-constrained) location.
"""
from __future__ import annotations

import os

import numpy as np

import TsunamiTrace as tt

from .diagnostics import argmin_2d
from .geodesy import haversine_km, initial_bearing


def _forward_field(source_lon, source_lat, wf_points, bathy, cfg, wavelength):
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
                              azimuths_deg=az, wavelength=wavelength)
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


def _panel(ax, fig, plt, res, cfg, wf, bathy, iy, ix, name, color, swot, extra=None):
    slon, slat = res.clon[ix], res.clat[iy]
    Tj = res.stack[:, iy, ix]                         # modelled travel time -> WF pts
    Tbar = float(np.nanmean(Tj))                      # isochron level
    resid = Tj - Tbar
    rms = float(np.sqrt(np.nanmean(resid ** 2)))
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

    lon_b, lat_b, T = _forward_field(slon, slat, wf, bathy, cfg, res.wavelength)
    ax.contour(lon_b, lat_b, T, levels=[Tbar], colors=[color], linewidths=2.0,
               zorder=4)

    ax.plot(wf[:, 0], wf[:, 1], "-", color="0.4", lw=1.0, zorder=3)
    rlim = max(0.05, float(np.nanpercentile(np.abs(resid), 98)))
    sc = ax.scatter(wf[:, 0], wf[:, 1], c=resid, cmap="coolwarm",
                    vmin=-rlim, vmax=rlim, s=22, edgecolors="k", linewidths=0.3,
                    zorder=5)
    fig.colorbar(sc, ax=ax, label="digitised − predicted (min)", shrink=0.85)

    ax.plot([], [], color=color, lw=2.0, label="predicted wavefront")
    ax.plot([], [], "-", color="0.4", lw=1.0, label="digitised")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("lon (E)")
    ax.set_ylabel("lat (N)")
    title = f"{name} source {slon:.2f} E, {slat:.2f} N ({off:.0f} km)\n"
    if extra:
        title += extra + " · "
    title += f"shape RMS {rms:.3f} min · isochron @ {Tbar:.1f} min"
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)


def wffit_figure(res, cfg, wf, bathy, swot=None, ts=None):
    """Write ``<tag>_wffit.png``: predicted vs digitised wavefront for the
    anchored and free best-fit sources (and the emission-time best source when
    ``ts`` is given), zoomed on the wavefront."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []                                       # (iy, ix, name, color, extra)
    a = _minidx(res.rms_anchored)
    if a is not None:
        panels.append((a[0], a[1], "anchored", "magenta", None))
    f = _minidx(res.std_free)
    if f is not None:
        panels.append((f[0], f[1], "free", "green", None))
    if ts is not None:
        jx = int(np.argmin(np.abs(res.clon - ts.best_lon0)))
        jy = int(np.argmin(np.abs(res.clat - ts.best_lat0)))
        panels.append((jy, jx, "emission-time", "darkorange",
                       f"tau = {ts.best_tau:.0f} min"))
    if not panels:
        print("  wffit skipped: no valid minimum")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels), 5.6),
                             constrained_layout=True, squeeze=False)
    for ax, (iy, ix, name, color, extra) in zip(axes[0], panels):
        _panel(ax, fig, plt, res, cfg, wf, bathy, iy, ix, name, color, swot, extra)

    fig.suptitle("Predicted vs digitised wavefront (best-fit source)")
    out = os.path.join(cfg.out_dir, cfg.tag + "_wffit.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")

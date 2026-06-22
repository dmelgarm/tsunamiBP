"""Figure rendering: per-wavefront misfit + data panels, raw coverage."""
from __future__ import annotations

import os

import numpy as np

from .diagnostics import argmin_2d


def _scene_extent(cfg, res, wf, margin=0.5):
    """Lon/lat limits framing the whole back-projection scene: candidate grid,
    wavefront, epicentre and the located minimum, with a margin."""
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
    height anomaly, the hand-traced wavefront highlighted on top, and the
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

    # wavefront leading edge (the actual data back-projected)
    ax.plot(wf[:, 0], wf[:, 1], "-", color="lime", lw=2.0, zorder=4)
    ax.plot(wf[:, 0], wf[:, 1], "o", mfc="white", mec="none", ms=4, lw=0,
            zorder=5, label="WF (traced front)")

    # connector wavefront centroid -> located minimum
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
    ax.set_title("Data: SWOT swath + WF  ->  inferred source")
    ax.set_xlabel("lon (E)")
    ax.set_ylabel("lat (N)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def plot_misfit_figure(res, cfg, field, title, suffix, wf, swot):
    """One figure: misfit map (left) beside the SWOT+WF data panel (right)."""
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
    """Standalone raw-ray-coverage diagnostic (fill=False) with WF overlaid."""
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
            ms=4, lw=1.5, label="WF", zorder=5)
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

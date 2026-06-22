"""Cross-wavefront comparison -- DESCRIPTIVE ONLY, nothing averaged.

Each wavefront is back-projected independently (its own misfit maps and
minimum).  The hypothesis driving this tool is that dispersed later wavefronts
may radiate from *different* places than the leading front, so we deliberately
do NOT combine them into a joint misfit.  Instead we collate the independent
results so the spatial progression is visible:

  * a summary table (printed) + CSV of each wavefront's minimum, offset from the
    epicentre, resolution (min+1 valley extent) and timing residual;
  * an overlay figure putting every wavefront's traced polyline, minimum and
    min+1 valley on one map, coloured by wavefront, for the anchored and the
    origin-time-free misfits side by side.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from .diagnostics import argmin_2d, valley_extent_km
from .geodesy import haversine_km


def _min_stats(field, res, cfg, anchored):
    """Minimum location/value, offset from epicentre, valley extent and (for the
    anchored map) the timing residual, for one misfit field.  None if empty."""
    if field is None or not np.isfinite(field).any():
        return None
    v, iy, ix = argmin_2d(field)
    mlon, mlat = res.clon[ix], res.clat[iy]
    out = {
        "lon": float(mlon), "lat": float(mlat), "misfit": float(v),
        "offset_km": float(haversine_km(mlon, mlat, cfg.epi_lon, cfg.epi_lat)),
    }
    ve = valley_extent_km(field, res.clon, res.clat, v + 1.0)
    if ve:
        out["valley_ew_km"], out["valley_ns_km"], out["valley_cells"] = \
            float(ve[0]), float(ve[1]), int(ve[2])
    if anchored and res.known_dt is not None:
        rup = 0.0 if res.rupture_delay is None else float(res.rupture_delay[iy, ix])
        resid = (res.stack[:, iy, ix] + rup) - res.known_dt
        out["resid_min"] = float(np.nanmean(resid))
    return out


def summarize(results, cfg):
    """Build one summary row per wavefront.

    ``results`` is a list of (name, wavelength, wf_points, BPResult)."""
    rows = []
    for name, wavelength, wf, res in results:
        anc = _min_stats(res.rms_anchored, res, cfg, anchored=True)
        free = _min_stats(res.std_free, res, cfg, anchored=False)
        rows.append({
            "wavefront": name,
            "wavelength_m": ("shallow" if wavelength is None else wavelength),
            "n_points": len(wf),
            "anchored": anc,
            "free": free,
        })
    return rows


def print_summary_table(rows, cfg):
    """Print a compact per-wavefront comparison table."""
    print("\n" + "=" * 78)
    print(f"WAVEFRONT COMPARISON  (epicentre {cfg.epi_lon:.3f} E, {cfg.epi_lat:.3f} N)")
    print("=" * 78)
    hdr = (f"{'wavefront':<12}{'wavelength':>11}{'npts':>5}   "
           f"{'ANCHORED min (lon,lat)':>24}{'off km':>8}{'res EWxNS km':>14}")
    print(hdr)
    print("-" * 78)
    for r in rows:
        a = r["anchored"]
        if a:
            loc = f"{a['lon']:.2f},{a['lat']:.2f}"
            res_s = (f"{a.get('valley_ew_km', float('nan')):.0f}x"
                     f"{a.get('valley_ns_km', float('nan')):.0f}")
            line = (f"{r['wavefront']:<12}{str(r['wavelength_m']):>11}"
                    f"{r['n_points']:>5}   {loc:>24}{a['offset_km']:>8.0f}"
                    f"{res_s:>14}")
        else:
            line = (f"{r['wavefront']:<12}{str(r['wavelength_m']):>11}"
                    f"{r['n_points']:>5}   {'(no anchored min)':>24}")
        print(line)
    print("-" * 78)
    print("Each wavefront is independent; locations are NOT combined.")


def _rnd(v, n):
    """Round for CSV readability; pass through None."""
    return None if v is None else round(v, n)


def write_summary_csv(rows, path):
    """Flatten the summary rows to a CSV (one row per wavefront)."""
    cols = ["wavefront", "wavelength_m", "n_points",
            "anc_lon", "anc_lat", "anc_misfit_min", "anc_offset_km",
            "anc_valley_ew_km", "anc_valley_ns_km", "anc_resid_min",
            "free_lon", "free_lat", "free_misfit_min", "free_offset_km"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            a, f = r["anchored"] or {}, r["free"] or {}
            w.writerow({
                "wavefront": r["wavefront"],
                "wavelength_m": r["wavelength_m"],
                "n_points": r["n_points"],
                "anc_lon": _rnd(a.get("lon"), 3), "anc_lat": _rnd(a.get("lat"), 3),
                "anc_misfit_min": _rnd(a.get("misfit"), 3),
                "anc_offset_km": _rnd(a.get("offset_km"), 1),
                "anc_valley_ew_km": _rnd(a.get("valley_ew_km"), 0),
                "anc_valley_ns_km": _rnd(a.get("valley_ns_km"), 0),
                "anc_resid_min": _rnd(a.get("resid_min"), 3),
                "free_lon": _rnd(f.get("lon"), 3), "free_lat": _rnd(f.get("lat"), 3),
                "free_misfit_min": _rnd(f.get("misfit"), 3),
                "free_offset_km": _rnd(f.get("offset_km"), 1),
            })
    print(f"saved {path}")


def _overlay(ax, plt, results, cfg, field_attr, title, colors):
    """One overlay panel: every wavefront's polyline, minimum and min+1 valley
    contour drawn in its own colour, for the given misfit field."""
    lons, lats = [cfg.epi_lon], [cfg.epi_lat]
    for (name, _wl, wf, res), c in zip(results, colors):
        field = getattr(res, field_attr)
        lons += [wf[:, 0].min(), wf[:, 0].max(), cfg.cand_lon[0], cfg.cand_lon[1]]
        lats += [wf[:, 1].min(), wf[:, 1].max(), cfg.cand_lat[0], cfg.cand_lat[1]]
        # traced polyline (the input being back-projected)
        ax.plot(wf[:, 0], wf[:, 1], "-", color=c, lw=1.4, alpha=0.7, zorder=3)
        if field is None or not np.isfinite(field).any():
            continue
        v, iy, ix = argmin_2d(field)
        mlon, mlat = res.clon[ix], res.clat[iy]
        # min+1 valley boundary
        ax.contour(res.clon, res.clat, field, levels=[v + 1.0],
                   colors=[c], linewidths=1.3, zorder=4)
        # connector polyline-centroid -> minimum, and the minimum marker
        ax.plot([wf[:, 0].mean(), mlon], [wf[:, 1].mean(), mlat], "--",
                color=c, lw=0.8, alpha=0.5, zorder=3)
        ax.plot(mlon, mlat, "o", mfc=c, mec="k", mew=0.6, ms=9, zorder=6,
                label=name)
    ax.plot(cfg.epi_lon, cfg.epi_lat, marker="*", ms=16, mfc="yellow", mec="k",
            mew=0.8, zorder=7, label="epicentre")
    ax.add_patch(plt.Rectangle((cfg.cand_lon[0], cfg.cand_lat[0]),
                               cfg.cand_lon[1] - cfg.cand_lon[0],
                               cfg.cand_lat[1] - cfg.cand_lat[0],
                               fill=False, ec="0.4", lw=0.8, ls=":", zorder=2))
    m = 0.5
    ax.set_xlim(min(lons) - m, max(lons) + m)
    ax.set_ylim(min(lats) - m, max(lats) + m)
    ax.set_title(title)
    ax.set_xlabel("lon (E)")
    ax.set_ylabel("lat (N)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def plot_comparison(results, cfg):
    """Two-panel overlay (anchored | origin-time-free) of all wavefronts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(results))]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    _overlay(axes[0], plt, results, cfg, "rms_anchored",
             "Anchored minima + min+1 valleys", colors)
    _overlay(axes[1], plt, results, cfg, "std_free",
             "Origin-time-free minima + min+1 valleys", colors)
    fig.suptitle("Per-wavefront source localisation (independent; not combined)")
    out = os.path.join(cfg.out_dir, cfg.tag + "_compare.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def compare_wavefronts(results, cfg):
    """Run the full comparison: table, CSV, overlay figure."""
    rows = summarize(results, cfg)
    print_summary_table(rows, cfg)
    write_summary_csv(rows, os.path.join(cfg.out_dir, cfg.tag + "_summary.csv"))
    plot_comparison(results, cfg)

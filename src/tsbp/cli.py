"""Command-line driver.

This still drives a SINGLE wavefront, identical in behaviour to the original
``backproject_wf1.py``.  The multi-wavefront loop and the YAML config arrive in
the next phase; the engine and helpers it calls are already wavefront-agnostic.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

from .config import Config
from .io import (build_candidate_grid, load_domain_bathymetry, load_swot_ssh,
                 load_wf_polyline, resample_polyline, save_outputs,
                 swot_times_for_wf)
from .engine import backproject
from .diagnostics import forward_consistency, raw_coverage, report
from .plotting import plot_coverage_figure, plot_misfit_figure


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="WF tsunami source back-projection")
    p.add_argument("--wf", dest="wf_path")
    p.add_argument("--bathy", dest="bathy_path")
    p.add_argument("--known-dt", type=float, dest="KNOWN_ARRIVAL_MIN",
                   help="known WF arrival, minutes after origin")
    p.add_argument("--wavelength", type=float, default=None,
                   help="deep-water wavelength (m); omit for shallow-water")
    p.add_argument("--coverage-frac", type=float, dest="coverage_frac")
    p.add_argument("--fan-halfwidth", type=float, dest="fan_halfwidth_deg")
    p.add_argument("--az-step", type=float, dest="azimuth_step_deg")
    p.add_argument("--dt", type=float)
    p.add_argument("--max-time", type=float)
    p.add_argument("--bin-deg", type=float)
    p.add_argument("--n-wf-points", type=int, dest="n_wf_points",
                   help="resample WF polyline to this many points "
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
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
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

    print("Loading WF polyline ...")
    wf = load_wf_polyline(cfg.wf_path)
    n_raw = len(wf)
    wf = resample_polyline(wf, cfg.n_wf_points)
    resampled = "" if cfg.n_wf_points is None else f" (resampled from {n_raw})"
    print(f"  {len(wf)} WF points{resampled}, "
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
    #   SWOT csv set     -> per-pixel times matched to each WF point
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
            print("  WARNING: some WF points are >25 km from any SWOT pixel; "
                  "check WF lies on the swath.")
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

    # Separate figure per misfit map, each beside its SWOT+WF data panel.
    if res.rms_anchored is not None:
        plot_misfit_figure(res, cfg, res.rms_anchored,
                           "Anchored rms (per-pixel arrival)", "_anchored",
                           wf, swot_pts)
    plot_misfit_figure(res, cfg, res.std_free,
                       "Origin-time-free std", "_free", wf, swot_pts)
    plot_coverage_figure(res, cfg, cov, wf, swot_pts)


if __name__ == "__main__":
    main()

"""Command-line driver.

Runs each configured wavefront **independently** (own misfit maps, own minimum)
-- there is deliberately no joint misfit, because dispersed later wavefronts may
radiate from different places than the leading front.  Wavefronts and shared
settings come from a YAML config (``--config``) or, with no config, from the
``Config`` defaults plus the single-wavefront CLI flags.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

from .config import Config, WavefrontSpec, load_config
from .io import (build_candidate_grid, load_domain_bathymetry, load_swot_ssh,
                 load_wf_polyline, resample_polyline, save_outputs,
                 swot_times_for_wf)
from .engine import backproject
from .diagnostics import forward_consistency, raw_coverage, report
from .plotting import plot_coverage_figure, plot_misfit_figure
from .compare import compare_wavefronts
from .timesearch import run_time_search
from .wffit import wffit_figure


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="WF tsunami source back-projection")
    p.add_argument("--config", help="YAML run config (shared settings + wavefronts)")
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
    p.add_argument("--geometry-only", action="store_true", dest="geometry_only",
                   help="geometric misfit only; ignore observed arrival times")
    p.add_argument("--free-only", action="store_true", dest="free_only",
                   help="deprecated alias for --geometry-only")
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
    p.add_argument("--time-search", action="store_true",
                   help="also run the emission-time search (separate hypothesis)")
    p.add_argument("--time-step", type=float, dest="time_step_min",
                   help="emission-time grid step (minutes)")
    p.add_argument("--time-max", type=float, dest="time_max_min",
                   help="emission-time grid maximum (minutes after origin)")
    p.add_argument("--no-wffit", action="store_true",
                   help="skip the predicted-vs-digitised wavefront-fit figure")
    return p.parse_args(argv)


def run_wavefront(cfg, bathy, cand, swot_ssh, geometry_only=False, uniform_dt=False,
                  local_wavelength=None, local_depth=None):
    """Back-project ONE wavefront and write its outputs.

    ``cfg`` is the per-wavefront config (wf_path / wavelength / n_wf_points /
    tag already set for this wavefront); ``bathy`` and ``cand`` are shared and
    loaded once by the caller.  Returns the BPResult."""
    print(f"\n========== wavefront: {cfg.tag} ==========")
    print("Loading WF polyline ...")
    wf = load_wf_polyline(cfg.wf_path)
    n_raw = len(wf)
    wf = resample_polyline(wf, cfg.n_wf_points)
    resampled = "" if cfg.n_wf_points is None else f" (resampled from {n_raw})"
    print(f"  {len(wf)} WF points{resampled}, "
          f"lon {wf[:,0].min():.2f}..{wf[:,0].max():.2f}, "
          f"lat {wf[:,1].min():.2f}..{wf[:,1].max():.2f}")

    # Resolve the wave ONCE; every trace for this wavefront shares it.
    from .engine import resolve_wave
    wave = resolve_wave(wf, bathy, wavelength=cfg.wavelength,
                        local_wavelength=local_wavelength, local_depth=local_depth)

    # ---- build the arrival-time anchor ----
    if geometry_only:
        known_dt = None
        anchor_desc = "none (geometry-only)"
    elif cfg.swot_times_path and not uniform_dt:
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
    cov = raw_coverage(wf, bathy, cfg, wave=wave, progress_label="coverage")

    if known_dt is not None:
        forward_consistency(wf, bathy, cfg, known_dt, wave=wave)

    wave_desc = (f"local_wavelength={wave.local_wavelength:.0f} m @ "
                 f"{wave.ref_depth:.0f} m (period {wave.period:.1f} s)"
                 if wave.local_wavelength is not None else
                 f"wavelength={wave.wavelength}" if wave.wavelength is not None
                 else "shallow-water")
    print(f"Back-projecting (wave={wave_desc}, anchor={anchor_desc}) ...")
    res = backproject(wf, cand, bathy, cfg, wave=wave, known_dt=known_dt,
                      progress_label="back-projecting")

    report(res, cfg)
    save_outputs(res, cfg)

    if res.rms_anchored is not None:
        plot_misfit_figure(res, cfg, res.rms_anchored,
                           "Anchored rms (per-pixel arrival)", "_anchored",
                           wf, swot_ssh)
    if res.std_free is not None:
        plot_misfit_figure(res, cfg, res.std_free,
                           "Origin-time-free std", "_free", wf, swot_ssh)
    plot_misfit_figure(res, cfg, res.std_geom,
                       "Geometric std (timing-free)", "_geom", wf, swot_ssh)
    plot_coverage_figure(res, cfg, cov, wf, swot_ssh)

    # emission-time search (separate hypothesis; opt-in, additive) -- run first
    # so its best source can be shown on the wavefront-fit figure.
    ts = None
    if cfg.time_search:
        if res.known_dt is None:
            print("  time search skipped: needs observed arrival times "
                  "(geometry-only run)")
        else:
            ts = run_time_search(res, cfg)

    # predicted-vs-digitised wavefront-fit figure (spatial goodness-of-fit);
    # adds the emission-time best source as a third panel when available.
    if cfg.wavefront_fit:
        wffit_figure(res, cfg, wf, bathy, swot=swot_ssh, ts=ts)

    return res, wf


def main(argv=None):
    args = parse_args(argv)

    # base config: from YAML if given, else built-in defaults
    cfg = load_config(args.config) if args.config else Config()

    # apply CLI overrides on top (flags win over the config file)
    skip = {"free_only", "uniform_dt", "no_rupture", "config", "time_search",
            "no_wffit", "geometry_only"}
    for k, v in vars(args).items():
        if k in skip:
            continue
        if v is not None and hasattr(cfg, k):
            cfg = replace(cfg, **{k: v})
    if args.wavelength is not None:
        cfg = replace(cfg, wavelength=args.wavelength)
    if args.no_rupture:
        cfg = replace(cfg, rupture_speed_kms=None)
    if args.time_search:
        cfg = replace(cfg, time_search=True)
    if args.no_wffit:
        cfg = replace(cfg, wavefront_fit=False)

    # --free-only is the deprecated spelling of --geometry-only.
    geometry_only = args.geometry_only or args.free_only
    if args.free_only:
        print("WARNING: --free-only is deprecated; use --geometry-only "
              "(geometric misfit only; ignores observed arrival times).")

    # the wavefronts to run: from the config, else a single synthesised one
    wavefronts = cfg.wavefronts or [
        WavefrontSpec(name="WF", path=cfg.wf_path,
                      wavelength=cfg.wavelength, n_points=cfg.n_wf_points,
                      local_wavelength=None, local_depth=cfg.local_depth)
    ]
    print(f"Wavefronts to back-project: {[w.name for w in wavefronts]}")

    # shared inputs, loaded once
    print("Loading + subsetting bathymetry ...")
    bathy = load_domain_bathymetry(cfg)
    blon, blat, bdepth = bathy
    print(f"  domain {blon[0]:.1f}..{blon[-1]:.1f} E, {blat[0]:.1f}..{blat[-1]:.1f} N "
          f"-> depth {bdepth.shape}")
    cand = build_candidate_grid(cfg)
    print(f"  candidate grid {len(cand[0])} x {len(cand[1])} cells")
    swot_ssh = None
    if cfg.swot_ssh_path and os.path.exists(cfg.swot_ssh_path):
        swot_ssh = load_swot_ssh(cfg.swot_ssh_path)

    # run each wavefront independently.  With >1 wavefront the output stem is
    # suffixed by the wavefront name; a single wavefront keeps the base tag.
    multi = len(wavefronts) > 1
    results = []
    for spec in wavefronts:
        tag = f"{cfg.tag}_{spec.name}" if multi else cfg.tag
        wf_cfg = replace(cfg, wf_path=spec.path, wavelength=spec.wavelength,
                         n_wf_points=spec.n_points, tag=tag)
        eff_local_depth = (spec.local_depth if spec.local_depth is not None
                           else cfg.local_depth)
        res, wf = run_wavefront(wf_cfg, bathy, cand, swot_ssh,
                                geometry_only=geometry_only, uniform_dt=args.uniform_dt,
                                local_wavelength=spec.local_wavelength,
                                local_depth=eff_local_depth)
        results.append((spec.name, spec.wavelength, wf, res))

    # cross-wavefront comparison (descriptive overlay + table; never combines).
    if multi:
        compare_wavefronts(results, cfg)


if __name__ == "__main__":
    main()

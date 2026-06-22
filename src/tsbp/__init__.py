"""tsbp -- tsunami source back-projection from observed wavefronts.

Given one or more hand-traced tsunami wavefronts (ordered lon/lat polylines)
and per-pixel observed arrival times, map -- over a grid of candidate source
locations S -- how well each candidate explains each wavefront, using the
TsunamiTrace ray tracer and travel-time reciprocity (rays are traced FROM the
observed wavefront points, never from the candidates).

Public API
----------
- Config                : run configuration (dataclass; YAML loader to come)
- backproject, BPResult : the per-wavefront misfit engine
- load_wf_polyline, resample_polyline, load_domain_bathymetry,
  build_candidate_grid, swot_times_for_wf, load_swot_ssh, save_outputs : I/O
- forward_consistency, report, raw_coverage, argmin_2d, valley_extent_km
- plot_misfit_figure, plot_coverage_figure
- cli.main              : the single-wavefront command-line driver
"""
from __future__ import annotations

from .config import Config
from .engine import BPResult, backproject
from .io import (build_candidate_grid, load_domain_bathymetry, load_swot_ssh,
                 load_wf_polyline, resample_polyline, save_outputs,
                 swot_times_for_wf)
from .diagnostics import (argmin_2d, forward_consistency, raw_coverage, report,
                          valley_extent_km)
from .plotting import plot_coverage_figure, plot_misfit_figure

__version__ = "0.1.0"

__all__ = [
    "Config",
    "backproject", "BPResult",
    "load_wf_polyline", "resample_polyline", "load_domain_bathymetry",
    "build_candidate_grid", "swot_times_for_wf", "load_swot_ssh", "save_outputs",
    "forward_consistency", "report", "raw_coverage", "argmin_2d",
    "valley_extent_km",
    "plot_misfit_figure", "plot_coverage_figure",
]

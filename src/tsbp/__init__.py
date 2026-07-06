"""tsbp -- tsunami source back-projection from observed wavefronts.

Given one or more hand-traced tsunami wavefronts (ordered lon/lat polylines)
and per-pixel observed arrival times, map -- over a grid of candidate source
locations S -- how well each candidate explains each wavefront, using the
TsunamiTrace ray tracer and travel-time reciprocity (rays are traced FROM the
observed wavefront points, never from the candidates).

Public API
----------
- Config, WavefrontSpec : run configuration; load_config reads a YAML file
- backproject, BPResult : the per-wavefront misfit engine
- load_wf_polyline, resample_polyline, load_domain_bathymetry,
  build_candidate_grid, swot_times_for_wf, load_swot_ssh, save_outputs : I/O
- forward_consistency, report, raw_coverage, argmin_2d, valley_extent_km
- plot_misfit_figure, plot_coverage_figure
- cli.main              : CLI driver; back-projects each wavefront independently
"""
from __future__ import annotations

# TsunamiTrace (the ray-tracing core) is a hard requirement but is not on PyPI,
# so fail early with an actionable message rather than a bare ModuleNotFoundError
# deep inside the engine.
try:
    import TsunamiTrace as _tt  # noqa: F401
except ModuleNotFoundError as _e:  # pragma: no cover
    raise ModuleNotFoundError(
        "tsbp requires the TsunamiTrace package (the ray-tracing core), which is "
        "not on PyPI. Install it from source:\n"
        "    git clone https://github.com/dmelgarm/TsunamiTrace.git\n"
        "    pip install -e ./TsunamiTrace\n"
        "or, to fetch it directly:\n"
        '    pip install -e ".[tsunamitrace]"'
    ) from _e

from .config import Config, WavefrontSpec, load_config
from .engine import BPResult, backproject
from .io import (build_candidate_grid, load_domain_bathymetry, load_swot_ssh,
                 load_wf_polyline, resample_polyline, save_outputs,
                 swot_times_for_wf)
from .gpkg import gpkg_to_geojson
from .diagnostics import (argmin_2d, forward_consistency, raw_coverage, report,
                          valley_extent_km)
from .plotting import plot_coverage_figure, plot_misfit_figure
from .compare import compare_wavefronts, summarize
from .timesearch import TimeSearchResult, run_time_search, time_search

__version__ = "0.1.0"

__all__ = [
    "Config", "WavefrontSpec", "load_config",
    "backproject", "BPResult",
    "load_wf_polyline", "resample_polyline", "load_domain_bathymetry",
    "build_candidate_grid", "swot_times_for_wf", "load_swot_ssh", "save_outputs",
    "gpkg_to_geojson",
    "forward_consistency", "report", "raw_coverage", "argmin_2d",
    "valley_extent_km",
    "plot_misfit_figure", "plot_coverage_figure",
    "compare_wavefronts", "summarize",
    "time_search", "run_time_search", "TimeSearchResult",
]

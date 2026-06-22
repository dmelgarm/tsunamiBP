# tsbp — tsunami source back-projection

Locate the source region of a tsunami by **back-projecting observed wavefronts**.
Given one or more hand-traced wavefronts (ordered lon/lat polylines, e.g. digitised
from a SWOT sea-surface-height swath) and the per-pixel times at which they were
observed, `tsbp` maps — over a grid of candidate source locations *S* — how well
each candidate explains each wavefront, using the
[TsunamiTrace](https://github.com/dmelgarm/TsunamiTrace) ray tracer.

Built and validated on the 2025-07-30 M8.8 Kamchatka earthquake, but the code is
event-agnostic — point it at a different earthquake/tsunami via configuration.

## The idea (and the one thing not to get backwards)

Travel time is **reciprocal** in a fixed slowness field: `T(A→B) = T(B→A)`. So we do
**not** shoot rays from candidate sources. For each observed wavefront point `x_j`
we trace rays *from `x_j`* toward the candidate region and read the travel time **at**
each candidate cell `S`. That gives `T_j(S) = T(S→x_j)` — the time a wave from a
hypothetical source at `S` would take to reach the observed point `x_j`. Stacking
over all `j` and comparing to the observed arrival times yields a misfit map over `S`.

Two misfit maps are produced per wavefront:

- **Anchored RMS** — uses the known per-pixel arrival times `t_j`, including a
  rupture-propagation delay `t_rup(S) = dist(epicentre, S) / v_rupture`:
  `rms(S) = sqrt(mean_j (t_rup(S) + T_j(S) − t_j)²)`.
- **Origin-time-free std** — the spread of `T_j(S)` about its own per-cell mean;
  independent of origin time (and of the rupture delay).

Both use root-**mean**-square with NaN masking (ray coverage varies per cell), and a
cell must be reached by at least `coverage_frac` of the wavefront points to be scored.

## Requirements

`tsbp` is a thin layer over **[TsunamiTrace](https://github.com/dmelgarm/TsunamiTrace)**,
which does the actual ray tracing (`import TsunamiTrace`). **TsunamiTrace is required
and is not on PyPI** — install it from source first:

```bash
git clone https://github.com/dmelgarm/TsunamiTrace.git
pip install -e ./TsunamiTrace
```

It is intentionally *not* listed in `tsbp`'s auto-installed dependencies, so that
installing `tsbp` never clobbers an editable development checkout of TsunamiTrace.
If you are not developing TsunamiTrace locally, you can instead let `tsbp` pull it
via the `tsunamitrace` extra (see below). The rest of the dependencies (numpy,
scipy, matplotlib, pandas, pyyaml) install automatically.

## Install

```bash
conda activate tsunamitrace          # an env with TsunamiTrace importable
cd ~/code/python/tsunamiBP
pip install -e .                      # tsbp + numpy/scipy/matplotlib/pandas/pyyaml
```

Optional extras:

```bash
pip install -e ".[tsunamitrace]"      # also fetch TsunamiTrace from its repo
pip install -e ".[netcdf]"            # NetCDF output (netCDF4)
pip install -e ".[dev]"               # pytest
```

## Run

```bash
tsbp                                  # uses the built-in Config defaults
# or, without installing:
python -m tsbp
python backproject_wf1.py             # deprecated shim, same thing
```

Common overrides (see `tsbp --help` for the full list):

```bash
tsbp --tag kamchatka_wf1 --out-dir runs/kamchatka
tsbp --wavelength 25000               # dispersive (short-wavelength) front
tsbp --no-rupture                     # drop the rupture-delay term
tsbp --free-only                      # origin-time-free map only
```

Outputs per run (written to `out_dir`, filename stem = `tag`):

- `<tag>_anchored.png` — anchored misfit map beside a SWOT-SSH + wavefront data panel
- `<tag>_free.png` — origin-time-free misfit map + data panel
- `<tag>_coverage.png` — raw ray-coverage diagnostic
- `<tag>.npz` / `<tag>.nc` — misfit maps, coverage, stack, per-pixel times

## Package layout

```
tsunamiBP/
├── pyproject.toml          # packaging + the `tsbp` entry point
├── backproject_wf1.py      # deprecated shim → tsbp.cli.main
├── configs/                # (next phase) per-run YAML configs
├── src/tsbp/
│   ├── __init__.py         # public API
│   ├── __main__.py         # `python -m tsbp`
│   ├── config.py           # Config (run settings; YAML loader to come)
│   ├── geodesy.py          # great-circle bearing / distance
│   ├── io.py               # wavefront, SWOT, bathymetry loaders; result saving
│   ├── engine.py           # backproject() + BPResult — the misfit engine
│   ├── diagnostics.py      # forward-consistency check, text report, coverage
│   ├── plotting.py         # misfit + data panels, coverage figure
│   └── cli.py              # single-wavefront command-line driver
├── tests/                  # (next phase)
└── runs/                   # outputs (gitignored)
```

## Inputs

- **Wavefront**: GeoJSON `LineString`/`MultiLineString` of ordered `(lon, lat)` points.
- **SWOT per-pixel times**: CSV with columns `time` (s after origin), `lat`, `lon`
  (the polar-orbit pass images each pixel at a slightly different time → each
  wavefront point gets its own observed arrival time).
- **SWOT SSH** (optional, for the data panel): whitespace `lon lat ssh` (m).
- **Bathymetry**: any DEM `tt.load_bathymetry` reads (ETOPO/GEBCO/SRTM `.nc`/`.xyz`);
  subset to a domain bracketing the wavefront and the candidate region.

## Status & roadmap

Current: a single wavefront, configured via the `Config` dataclass / CLI flags.

Next phase (planned): **multiple wavefronts of different wavelengths**, each
processed **independently** (the hypothesis is that dispersed later fronts radiate
from *different* places than the leading front, so they are deliberately **not**
combined into a joint misfit). A YAML config will list the wavefronts; a `compare`
step will overlay each wavefront's polyline, minimum and resolution on one map plus
a summary table — purely descriptive, nothing averaged across wavefronts.

## Validation note (Kamchatka)

The anchored minimum lands ~360 km **SW of the USGS epicentre**, not on it, and the
forward check flags a ~9–10 min model-vs-observed gap. This is **expected physics**,
not an error: the 2025 Kamchatka rupture propagated unilaterally SW, so the tsunami's
effective source is SW of the nucleation point. The tool surfaces this automatically
via the forward-consistency diagnostic.

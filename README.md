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
python -m tsbp                        # equivalent, without installing the command
```

### Example configs (Kamchatka)

Three ready-to-run configs ship in `configs/` for the 2025 Kamchatka event. They
differ only in resolution / cost — start with the low-res one to preview, then run
the hi-res for the final maps:

| config | candidate grid | dt | az step | bin | speed | output |
|--------|:--:|:--:|:--:|:--:|:--|:--|
| `kamchatka_lowres.yaml` | 0.2° | 10 s | 0.10° | 0.30° | fast (~seconds–minute) | `…/back_projection/test_run/` |
| `kamchatka_hires.yaml`  | 0.1° |  5 s | 0.05° | 0.20° | slow (dense ray fan) | `runs/kamchatka/` |

```bash
tsbp --config configs/kamchatka_lowres.yaml      # quick preview
tsbp --config configs/kamchatka_hires.yaml       # full-resolution maps
```

Edit the `bathymetry.path`, `swot.*` and `wavefronts[].path` entries to point at
your own data (the bundled paths are the author's). Any setting can still be
overridden on the command line, e.g. a fast pass of the hi-res config:

```bash
tsbp --config configs/kamchatka_hires.yaml --dt 20 --az-step 0.5 --bin-deg 0.3
```

Common overrides (see `tsbp --help` for the full list):

```bash
tsbp --tag kamchatka_wf1 --out-dir runs/kamchatka
tsbp --wavelength 25000               # dispersive (short-wavelength) front
tsbp --no-rupture                     # drop the rupture-delay term
tsbp --free-only                      # origin-time-free map only
```

### Multiple wavefronts

List several wavefronts in the config and each is back-projected **independently**
— its own misfit maps, minimum and resolution. They are deliberately **not**
combined into a joint misfit (the hypothesis is that dispersed later fronts radiate
from different places than the leading one). Give each its own `wavelength`
(`null` = shallow-water long wave):

```yaml
wavefronts:
  - {name: WF1, path: .../WF1.geojson, wavelength: null,  n_points: 100}
  - {name: WF2, path: .../WF2.geojson, wavelength: 25000, n_points: 100}
```

### Emission-time search (a separate hypothesis)

The two standard maps ask *where*; this asks *where **and** when*. It treats a
front as radiating from a source `S` at a single emission time `τ = origin + Δ`
(τ ≥ 0), **ignoring** the rupture delay, and sweeps τ to score
`rms_j(τ + T_j(S) − t_j)`. Useful for later/dispersed fronts that may have been
emitted after origin. Enable per run:

```yaml
time_search: {enabled: true, step_min: 5, max_min: 60}
```
or on the command line: `tsbp --config … --time-search --time-step 5 --time-max 60`.

It's a cheap post-process over the already-traced `stack` (no re-tracing) and runs
**in addition** to the anchored/free maps, adding `<tag>_timesearch.png` (misfit-vs-τ
curve + the map at the best τ, with the best-location track colored by τ) and
`<tag>_timesearch.csv`. Note the free map is already the τ-optimized envelope of
this family, so the search earns its keep via the τ ≥ 0 constraint and by reporting
τ as a physical, checkable emission time.

### Outputs

Per wavefront (stem = `<tag>` for a single wavefront, `<tag>_<wfname>` for several):

- `..._anchored.png` — anchored misfit map beside a SWOT-SSH + wavefront data panel
- `..._free.png` — origin-time-free misfit map + data panel
- `..._coverage.png` — raw ray-coverage diagnostic
- `....npz` / `....nc` — misfit maps, coverage, stack, per-pixel times

With more than one wavefront, two extra **comparison** outputs (descriptive only):

- `<tag>_compare.png` — every wavefront's polyline, minimum and min+1 valley on one
  map (anchored | origin-time-free), coloured by wavefront
- `<tag>_summary.csv` — per-wavefront minimum, offset from epicentre, resolution,
  timing residual

## Package layout

```
tsunamiBP/
├── pyproject.toml          # packaging + the `tsbp` entry point
├── configs/                # per-run YAML configs (kamchatka_hires / _lowres)
├── src/tsbp/
│   ├── __init__.py         # public API
│   ├── __main__.py         # `python -m tsbp`
│   ├── config.py           # Config + WavefrontSpec + load_config (YAML)
│   ├── geodesy.py          # great-circle bearing / distance
│   ├── io.py               # wavefront, SWOT, bathymetry loaders; result saving
│   ├── engine.py           # backproject() + BPResult — the misfit engine
│   ├── diagnostics.py      # forward-consistency check, text report, coverage
│   ├── plotting.py         # misfit + data panels, coverage figure
│   ├── compare.py          # cross-wavefront overlay + summary CSV (no joint fit)
│   ├── progress.py         # dependency-free progress bar
│   ├── gpkg.py             # split a QGIS GeoPackage layer into per-WF GeoJSON
│   └── cli.py              # command-line driver (each wavefront independent)
├── tests/                  # unit + integration suite
└── runs/                   # outputs (gitignored)
```

## Inputs

- **Wavefront**: GeoJSON `LineString`/`MultiLineString` of ordered `(lon, lat)` points,
  one file per wavefront (a config `wavefronts[].path`).
- **SWOT per-pixel times**: CSV with columns `time` (s after origin), `lat`, `lon`
  (the polar-orbit pass images each pixel at a slightly different time → each
  wavefront point gets its own observed arrival time).
- **SWOT SSH** (optional, for the data panel): whitespace `lon lat ssh` (m).
- **Bathymetry**: any DEM `tt.load_bathymetry` reads (ETOPO/GEBCO/SRTM `.nc`/`.xyz`);
  subset to a domain bracketing the wavefront and the candidate region.

### Wavefronts from a QGIS GeoPackage

Digitising in QGIS naturally gives **one GeoPackage layer with several line
features** (one per wavefront, distinguished by an id attribute such as `wf_id`),
but the tool wants one GeoJSON per wavefront. `tsbp-gpkg` splits them — with no
GDAL/geopandas dependency (a `.gpkg` is a SQLite DB; geometries are parsed with
the standard library):

```bash
tsbp-gpkg path/to/bandpass_40_50.gpkg  out_dir/  --id-field wf_id
# or:  python -m tsbp.gpkg <gpkg> <out_dir> [--layer NAME] [--id-field wf_id] [--prefix STEM]
```

This writes `<layer>_<id>.geojson` per feature (e.g. `bandpass_40_50_1.geojson …_5`).
The layer must be geographic lon/lat (EPSG:4326); reproject in QGIS first if not.
Then list the outputs in a config's `wavefronts:` block, each with the wavelength
from your spectral analysis. Programmatic equivalent: `tsbp.gpkg_to_geojson(...)`.

## Tests

```bash
pytest                 # from the repo root; conftest.py puts src/ on the path
```

- `tests/test_units.py` — fast pure-function tests (geodesy, polyline resampling,
  GeoJSON / SWOT / YAML loaders, diagnostics helpers, compare summary/CSV); no ray
  tracer or data files needed.
- `tests/test_integration.py` — runs the real engine and full CLI on tiny synthetic
  constant-depth bathymetry: a point-source localization check (anchored minimum
  within 40 km of truth) and a two-wavefront end-to-end run that asserts every
  per-wavefront output plus the comparison figure and CSV. Auto-skips if
  TsunamiTrace is not importable. The whole suite runs in a few seconds.

## Validation note (Kamchatka)

The anchored minimum lands ~360 km **SW of the USGS epicentre**, not on it, and the
forward check flags a ~9–10 min model-vs-observed gap. This is **expected physics**,
not an error: the 2025 Kamchatka rupture propagated unilaterally SW, so the tsunami's
effective source is SW of the nucleation point. The tool surfaces this automatically
via the forward-consistency diagnostic.

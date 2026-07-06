"""Integration tests: the real engine / CLI on tiny synthetic constant-depth
bathymetry.  Self-contained (no project data files) and fast (small grid, coarse
tracing).  Skipped automatically if TsunamiTrace is not importable."""
import json

import numpy as np
import pytest

pytest.importorskip("TsunamiTrace")

from tsbp.config import Config
from tsbp.io import build_candidate_grid
from tsbp.engine import backproject
from tsbp.diagnostics import argmin_2d
from tsbp.geodesy import haversine_km
from tsbp.cli import main

_G = 9.8
_DEPTH = 4000.0
_C_KMS = np.sqrt(_G * _DEPTH) / 1000.0      # shallow-water speed, km/s


def _flat_bathy(lo0=0.0, lo1=6.0, la0=0.0, la1=6.0, d=0.1, depth=_DEPTH):
    lon = np.arange(lo0, lo1 + d / 2, d)
    lat = np.arange(la0, la1 + d / 2, d)
    dep = np.full((len(lon), len(lat)), float(depth))
    return lon, lat, dep


def _arc_around(source, radius_deg=2.0, n=15, az0=-80, az1=80):
    """A broad wavefront arc (n points) ~radius north of the source, giving good
    angular aperture so the back-projection is well localised."""
    az = np.deg2rad(np.linspace(az0, az1, n))
    lon = source[0] + radius_deg * np.sin(az)
    lat = source[1] + radius_deg * np.cos(az)
    return np.column_stack([lon, lat])


def test_engine_localizes_point_source():
    bathy = _flat_bathy()
    source = (3.0, 2.0)
    wf = _arc_around(source)
    # true (great-circle) travel time from the source to each wavefront point
    known_dt = haversine_km(source[0], source[1], wf[:, 0], wf[:, 1]) / _C_KMS / 60.0

    cfg = Config(epi_lon=source[0], epi_lat=source[1], rupture_speed_kms=None,
                 cand_lon=(2.0, 4.0), cand_lat=(1.0, 3.0),
                 cand_dlon=0.1, cand_dlat=0.1,
                 fan_halfwidth_deg=70.0, azimuth_step_deg=0.5,
                 dt=10.0, max_time=2500.0, bin_deg=0.1,
                 coverage_frac=0.6, misfit_vmax=None)
    cand = build_candidate_grid(cfg)
    res = backproject(wf, cand, bathy, cfg, wavelength=None, known_dt=known_dt)

    # shapes / basic invariants
    assert res.rms_anchored.shape == (len(cand[1]), len(cand[0]))
    assert res.std_free.shape == res.rms_anchored.shape
    assert np.isfinite(res.rms_anchored).any()

    # the anchored minimum should land near the true source
    _, iy, ix = argmin_2d(res.rms_anchored)
    off = haversine_km(res.clon[ix], res.clat[iy], source[0], source[1])
    assert off < 40.0, f"anchored min {off:.1f} km from the true source"


def test_engine_known_dt_none_skips_anchored():
    bathy = _flat_bathy()
    source = (3.0, 2.0)
    wf = _arc_around(source, n=9)
    cfg = Config(epi_lon=source[0], epi_lat=source[1], rupture_speed_kms=None,
                 cand_lon=(2.0, 4.0), cand_lat=(1.0, 3.0),
                 cand_dlon=0.2, cand_dlat=0.2,
                 fan_halfwidth_deg=70.0, azimuth_step_deg=1.0,
                 dt=20.0, max_time=2500.0, bin_deg=0.2, coverage_frac=0.6)
    cand = build_candidate_grid(cfg)
    res = backproject(wf, cand, bathy, cfg, wavelength=None, known_dt=None)
    assert res.rms_anchored is None
    assert np.isfinite(res.std_free).any()


# ── full CLI run, incl. the multi-wavefront comparison ──────────────────────
def _write_xyz(path, bathy):
    lon, lat, dep = bathy
    LON, LAT = np.meshgrid(lon, lat)            # (nlat, nlon)
    rows = np.column_stack([LON.ravel(), LAT.ravel(),
                            -dep.T.ravel()])     # negative = ocean (negate=true)
    np.savetxt(path, rows, fmt="%.4f")


def _write_geojson(path, coords):
    doc = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "LineString", "coordinates": coords}}]}
    path.write_text(json.dumps(doc))


def _write_swot_csv(path, wf_points, source):
    import pandas as pd
    t_sec = haversine_km(source[0], source[1],
                         wf_points[:, 0], wf_points[:, 1]) / _C_KMS
    pd.DataFrame({
        "gauge_id": np.arange(len(wf_points)),
        "time": t_sec,
        "eta": np.full(len(wf_points), 0.1),
        "lat": wf_points[:, 1],
        "lon": wf_points[:, 0],
    }).to_csv(path, index=False)


def test_cli_end_to_end_two_wavefronts(tmp_path):
    bathy = _flat_bathy()
    source = (3.0, 2.0)
    wf1 = _arc_around(source, radius_deg=2.0, n=11)
    wf2 = _arc_around(source, radius_deg=1.6, n=11)   # a second, closer front

    xyz = tmp_path / "bathy.xyz"
    _write_xyz(xyz, bathy)
    p1 = tmp_path / "wf1.geojson"
    p2 = tmp_path / "wf2.geojson"
    _write_geojson(p1, wf1.tolist())
    _write_geojson(p2, wf2.tolist())
    swot = tmp_path / "swot.csv"
    _write_swot_csv(swot, np.vstack([wf1, wf2]), source)
    out = tmp_path / "out"

    cfg_yaml = tmp_path / "run.yaml"
    cfg_yaml.write_text(f"""
event: {{epi_lon: 3.0, epi_lat: 2.0, origin_time_utc: "2025-01-01T00:00:00", rupture_speed_kms: null}}
bathymetry: {{path: {xyz}, negate: true, domain_lon: [0, 6], domain_lat: [0, 6]}}
swot: {{times: {swot}, ssh: null}}
candidate: {{lon: [2.0, 4.0], lat: [1.0, 3.0], dlon: 0.2, dlat: 0.2}}
tracing: {{dt: 20.0, max_time: 2500, bin_deg: 0.2, fan_halfwidth_deg: 70.0, azimuth_step_deg: 1.0}}
misfit: {{coverage_frac: 0.6}}
plot: {{misfit_vmax: 10.0}}
time_search: {{enabled: true, step_min: 5, max_min: 15}}
output: {{out_dir: {out}, tag: itest}}
wavefronts:
  - {{name: WF1, path: {p1}, wavelength: null, n_points: 11}}
  - {{name: WF2, path: {p2}, wavelength: null, n_points: 11}}
""")

    main(["--config", str(cfg_yaml)])

    # per-wavefront outputs (name-suffixed) + comparison outputs
    for stem in ["itest_WF1", "itest_WF2"]:
        for suf in ["_anchored.png", "_free.png", "_coverage.png",
                    "_wffit.png", "_timesearch.png", "_timesearch.csv", ".npz"]:
            assert (out / (stem + suf)).exists(), f"missing {stem}{suf}"
    assert (out / "itest_compare.png").exists()
    assert (out / "itest_summary.csv").exists()

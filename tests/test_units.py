"""Fast unit tests for the pure-Python helpers (no ray tracer, no data files)."""
import json

import numpy as np
import pytest

from tsbp.geodesy import haversine_km, initial_bearing
from tsbp.io import (build_candidate_grid, load_wf_polyline, resample_polyline,
                     swot_times_for_wf)
from tsbp.config import Config, WavefrontSpec, load_config
from tsbp.diagnostics import argmin_2d, valley_extent_km
from tsbp.compare import summarize, write_summary_csv
from tsbp.engine import BPResult
from tsbp.gpkg import _parse_gpkg_geometry, gpkg_to_geojson


# ── geodesy ───────────────────────────────────────────────────────────────
def test_haversine_known_distance():
    # 1 degree of latitude is ~111.2 km
    assert haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.19, abs=0.5)
    assert haversine_km(0.0, 0.0, 0.0, 0.0) == 0.0


def test_initial_bearing_cardinals():
    assert initial_bearing(0, 0, 0, 1) == pytest.approx(0.0, abs=1e-6)    # north
    assert initial_bearing(0, 0, 1, 0) == pytest.approx(90.0, abs=1e-6)   # east
    assert initial_bearing(0, 0, 0, -1) == pytest.approx(180.0, abs=1e-6)  # south


# ── polyline resampling ─────────────────────────────────────────────────────
def test_resample_polyline_count_and_endpoints():
    pts = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    out = resample_polyline(pts, 21)
    assert out.shape == (21, 2)
    np.testing.assert_allclose(out[0], pts[0])
    np.testing.assert_allclose(out[-1], pts[-1])
    # evenly spaced along a straight meridian -> lat increments are equal
    dlat = np.diff(out[:, 1])
    np.testing.assert_allclose(dlat, dlat[0], rtol=1e-6)


def test_resample_polyline_passthrough():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert resample_polyline(pts, None) is pts


# ── GeoJSON loading ─────────────────────────────────────────────────────────
def _write_geojson(path, coords, multi=False):
    geom = ({"type": "MultiLineString", "coordinates": [coords]} if multi
            else {"type": "LineString", "coordinates": coords})
    doc = {"type": "FeatureCollection",
           "features": [{"type": "Feature", "properties": {}, "geometry": geom}]}
    path.write_text(json.dumps(doc))


def test_load_wf_polyline_linestring_and_multi(tmp_path):
    coords = [[160.0, 50.0], [160.5, 50.5], [161.0, 51.0]]
    p1 = tmp_path / "ls.geojson"
    _write_geojson(p1, coords, multi=False)
    a = load_wf_polyline(str(p1))
    assert a.shape == (3, 2)
    np.testing.assert_allclose(a[0], [160.0, 50.0])

    p2 = tmp_path / "mls.geojson"
    _write_geojson(p2, coords, multi=True)
    b = load_wf_polyline(str(p2))
    np.testing.assert_allclose(a, b)


# ── candidate grid ──────────────────────────────────────────────────────────
def test_build_candidate_grid_bounds_and_spacing():
    cfg = Config(cand_lon=(150.0, 151.0), cand_lat=(40.0, 41.0),
                 cand_dlon=0.25, cand_dlat=0.5)
    clon, clat = build_candidate_grid(cfg)
    assert clon[0] == pytest.approx(150.0)
    assert clon[-1] == pytest.approx(151.0)
    assert np.allclose(np.diff(clon), 0.25)
    assert np.allclose(np.diff(clat), 0.5)


# ── SWOT nearest-pixel time matching ────────────────────────────────────────
def test_swot_times_for_wf(tmp_path):
    import pandas as pd
    # three pixels at distinct locations/times
    df = pd.DataFrame({
        "gauge_id": [1, 2, 3],
        "time": [600.0, 1200.0, 1800.0],      # seconds -> 10, 20, 30 min
        "eta": [0.1, 0.2, 0.3],
        "lat": [50.0, 51.0, 52.0],
        "lon": [160.0, 160.0, 160.0],
    })
    csv = tmp_path / "swot.csv"
    df.to_csv(csv, index=False)
    # query points nearest to pixels 1 and 3
    wf = np.array([[160.01, 50.0], [160.0, 51.98]])
    t_min, match_km = swot_times_for_wf(wf, str(csv))
    np.testing.assert_allclose(t_min, [10.0, 30.0])
    assert np.all(match_km < 5.0)


# ── YAML config loader ──────────────────────────────────────────────────────
def test_load_config_mapping(tmp_path):
    yaml_text = """
event: {epi_lon: 160.4, epi_lat: 52.5, origin_time_utc: "2025-07-30T23:24:52", rupture_speed_kms: 2.2}
bathymetry: {path: /tmp/b.nc, negate: true, domain_lon: [150, 168], domain_lat: [41, 55]}
candidate: {lon: [153, 164], lat: [47, 55], dlon: 0.2, dlat: 0.2}
tracing: {dt: 10, bin_deg: 0.15}
output: {out_dir: /tmp/out, tag: run1}
wavefronts:
  - {name: WF1, path: /tmp/wf1.geojson, wavelength: null, n_points: 50}
  - {name: WF2, path: /tmp/wf2.geojson, wavelength: 25000, n_points: 80}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = load_config(str(p))
    assert cfg.epi_lon == 160.4 and cfg.epi_lat == 52.5
    assert isinstance(cfg.origin_time_utc, str)            # not a datetime
    assert cfg.domain_lon == (150, 168)                    # list -> tuple
    assert isinstance(cfg.cand_lon, tuple)
    assert cfg.dt == 10 and cfg.bin_deg == 0.15
    assert cfg.tag == "run1"
    assert [w.name for w in cfg.wavefronts] == ["WF1", "WF2"]
    assert cfg.wavefronts[0].wavelength is None
    assert cfg.wavefronts[1].wavelength == 25000
    # unspecified settings keep their defaults
    assert cfg.coverage_frac == Config().coverage_frac


# ── diagnostics helpers ─────────────────────────────────────────────────────
def test_argmin_2d_and_valley_extent():
    clon = np.linspace(0, 1, 11)
    clat = np.linspace(0, 1, 11)
    field = np.full((11, 11), 10.0)
    field[3, 7] = 1.0                       # min at (iy=3, ix=7)
    field[3, 6] = 1.4
    val, iy, ix = argmin_2d(field)
    assert (val, iy, ix) == (1.0, 3, 7)
    ve = valley_extent_km(field, clon, clat, level=1.5)
    assert ve is not None
    _, _, ncells = ve
    assert ncells == 2                       # the 1.0 and 1.4 cells


def test_argmin_2d_handles_nans():
    field = np.array([[np.nan, 5.0], [2.0, np.nan]])
    val, iy, ix = argmin_2d(field)
    assert val == 2.0 and (iy, ix) == (1, 0)


# ── compare summary + CSV ───────────────────────────────────────────────────
def _toy_result():
    clon = np.linspace(155.0, 160.0, 11)
    clat = np.linspace(48.0, 53.0, 11)
    rms = np.full((11, 11), 9.0)
    rms[4, 5] = 0.2
    std = np.full((11, 11), 4.0)
    std[6, 3] = 0.1
    stack = np.full((5, 11, 11), 30.0)
    return BPResult(clon=clon, clat=clat, stack=stack,
                    n_valid=np.full((11, 11), 5), coverage_ok=np.ones((11, 11), bool),
                    rms_anchored=rms, std_geom=std, std_free=std, wavelength=None,
                    known_dt=np.full(5, 30.0), rupture_delay=None)


def test_summarize_and_csv(tmp_path):
    res = _toy_result()
    wf = np.array([[160.0, 50.0], [160.5, 50.5]])
    cfg = Config(epi_lon=160.0, epi_lat=52.0)
    rows = summarize([("WF1", None, wf, res)], cfg)
    assert len(rows) == 1
    r = rows[0]
    assert r["wavefront"] == "WF1"
    assert r["anchored"]["misfit"] == pytest.approx(0.2)
    assert r["anchored"]["offset_km"] > 0

    out = tmp_path / "summary.csv"
    write_summary_csv(rows, str(out))
    text = out.read_text()
    assert "wavefront" in text and "WF1" in text
    assert len(text.strip().splitlines()) == 2     # header + one row


# ── GeoPackage export ───────────────────────────────────────────────────────
import sqlite3
import struct


def _gpkg_blob(points, srs_id=4326):
    """Build a GeoPackage geometry blob (no envelope, little-endian) wrapping a
    2-D WKB LineString of ``points``."""
    header = b"GP" + bytes([0, 0x01]) + struct.pack("<i", srs_id)   # env_code 0
    wkb = struct.pack("<BI", 1, 2) + struct.pack("<I", len(points))  # LE LineString
    for x, y in points:
        wkb += struct.pack("<dd", x, y)
    return header + wkb


def test_parse_gpkg_geometry_roundtrip():
    pts = [(160.0, 50.0), (160.5, 50.5), (161.0, 51.0)]
    out, srs = _parse_gpkg_geometry(_gpkg_blob(pts))
    assert srs == 4326
    np.testing.assert_allclose(out, pts)


def _make_minimal_gpkg(path, layer, features):
    """Write a minimal GeoPackage with one line layer and (wf_id, points) rows."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE gpkg_geometry_columns (table_name TEXT, "
                "column_name TEXT, geometry_type_name TEXT, srs_id INTEGER, "
                "z TINYINT, m TINYINT)")
    con.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,?,?)",
                (layer, "geom", "LINESTRING", 4326, 0, 0))
    con.execute(f'CREATE TABLE "{layer}" (fid INTEGER PRIMARY KEY, '
                'wf_id INTEGER, geom BLOB)')
    for i, (wf_id, pts) in enumerate(features, start=1):
        con.execute(f'INSERT INTO "{layer}" VALUES (?,?,?)',
                    (i, wf_id, _gpkg_blob(pts)))
    con.commit()
    con.close()


def test_gpkg_to_geojson_splits_features(tmp_path):
    gpkg = tmp_path / "fronts.gpkg"
    feats = [(1, [(160.0, 50.0), (160.5, 50.5)]),
             (2, [(161.0, 51.0), (161.5, 51.5), (162.0, 52.0)])]
    _make_minimal_gpkg(str(gpkg), "fronts", feats)

    out = tmp_path / "geojson"
    paths = gpkg_to_geojson(str(gpkg), str(out))     # sole layer, wf_id
    assert len(paths) == 2
    a1 = load_wf_polyline(paths[0])
    a2 = load_wf_polyline(paths[1])
    assert a1.shape == (2, 2) and a2.shape == (3, 2)
    np.testing.assert_allclose(a2[-1], [162.0, 52.0])
    # filenames carry the wf_id
    assert any(p.endswith("fronts_1.geojson") for p in paths)


# ── emission-time search ────────────────────────────────────────────────────
def test_time_search_recovers_source_and_time():
    from tsbp.timesearch import time_search
    clon = np.linspace(155.0, 161.0, 13)
    clat = np.linspace(47.0, 53.0, 13)
    LON, LAT = np.meshgrid(clon, clat)          # (13, 13)
    iy0, ix0 = 6, 6                             # true source = grid centre
    Clon, Clat = clon[ix0], clat[iy0]
    # wavefront points on a circle of radius R about the source -> the source is
    # the unique point equidistant (in "travel time") from all of them
    R, N = 1.5, 8
    ang = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    plon, plat = Clon + R * np.cos(ang), Clat + R * np.sin(ang)
    stack = np.stack([np.sqrt((LON - plon[j]) ** 2 + (LAT - plat[j]) ** 2)
                      for j in range(N)])        # T_j(S) = distance
    tau_true = 10.0
    kd = tau_true + np.full(N, R)                # observed = tau_true + R at C

    res = BPResult(clon=clon, clat=clat, stack=stack,
                   n_valid=np.full((13, 13), N),
                   coverage_ok=np.ones((13, 13), bool),
                   rms_anchored=None, std_geom=np.zeros((13, 13)),
                   std_free=np.zeros((13, 13)),
                   wavelength=None, known_dt=kd, rupture_delay=None)
    cfg = Config(epi_lon=Clon, epi_lat=Clat, time_step_min=5.0, time_max_min=30.0)

    ts = time_search(res, cfg)
    assert ts.best_tau == pytest.approx(tau_true)
    assert ts.best_lon0 == pytest.approx(Clon)
    assert ts.best_lat0 == pytest.approx(Clat)
    assert ts.best_misfit < 1e-6
    # the misfit-vs-tau curve bottoms out at tau_true
    assert ts.taus[int(np.nanargmin(ts.misfit_min))] == pytest.approx(tau_true)


# ── wave resolution (in-situ wavelength) ─────────────────────────────────────
def test_resolve_wave_local_and_passthrough():
    from tsbp.engine import resolve_wave
    lon = np.linspace(0.0, 6.0, 61)
    lat = np.linspace(0.0, 6.0, 61)
    depth = np.full((61, 61), 5500.0)
    wf = np.array([[3.0, 3.0], [3.5, 3.2]])
    # local_depth omitted -> derived as the median WF bathymetric depth
    w = resolve_wave(wf, (lon, lat, depth), local_wavelength=30e3)
    assert w.ref_depth == pytest.approx(5500.0)
    assert w.omega == pytest.approx(0.040985, rel=1e-4)
    assert w.trace_kwargs == {"local_wavelength": 30000.0, "local_depth": 5500.0}
    # deep-water wavelength passes straight through unchanged
    w2 = resolve_wave(wf, (lon, lat, depth), wavelength=40e3)
    assert w2.trace_kwargs == {"wavelength": 40000.0} and w2.ref_depth is None
    # both given is an error
    with pytest.raises(ValueError):
        resolve_wave(wf, (lon, lat, depth), wavelength=40e3, local_wavelength=30e3)

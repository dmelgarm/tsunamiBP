"""Bootstrap feature tests: off-by-default additivity, the perturbation model,
re-trace vs cheap ensemble agreement, and reproducibility.  Tiny constant-depth
synthetic (reuses the integration-test helpers)."""
import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("TsunamiTrace")

from tsbp.config import Config
from tsbp.io import build_candidate_grid
from tsbp.engine import backproject, resolve_wave
from tsbp.geodesy import haversine_km
from tsbp.perturb import perturb_wavefront, to_km, tangent_normal
from tsbp.timesearch import bootstrap_source
from tsbp.cli import main

_G = 9.8
_DEPTH = 4000.0
_C_KMS = np.sqrt(_G * _DEPTH) / 1000.0


def _flat_bathy(d=0.1):
    lon = np.arange(0.0, 6.0 + d / 2, d)
    lat = np.arange(0.0, 6.0 + d / 2, d)
    return lon, lat, np.full((len(lon), len(lat)), _DEPTH)


def _arc(source, radius_deg=2.0, n=13):
    az = np.deg2rad(np.linspace(-80, 80, n))
    return np.column_stack([source[0] + radius_deg * np.sin(az),
                            source[1] + radius_deg * np.cos(az)])


def _grad_bathy(gx=0.2, gy=0.1, base=4000.0, d=0.1):
    """A SMOOTH but non-flat bed (gentle depth gradient).  A perfectly flat,
    symmetric bed is pathological for the cheap vs re-trace comparison -- a
    coherent front translation maps almost exactly onto a tau shift there, so the
    cheap cloud collapses.  A gradient breaks that degeneracy the way real
    bathymetry does (cf. the gamma_9 validation)."""
    lon = np.arange(0.0, 6.0 + d / 2, d)
    lat = np.arange(0.0, 6.0 + d / 2, d)
    LON, LAT = np.meshgrid(lon, lat, indexing="ij")
    return lon, lat, base * (1.0 + gx * (LON - 3.0) / 3.0 + gy * (LAT - 3.0) / 3.0)


def _tiny_cfg(source):
    return Config(epi_lon=source[0], epi_lat=source[1], rupture_speed_kms=None,
                  cand_lon=(2.0, 4.0), cand_lat=(1.0, 3.0),
                  cand_dlon=0.1, cand_dlat=0.1,
                  fan_halfwidth_deg=80.0, azimuth_step_deg=1.0,
                  dt=20.0, max_time=4000.0, bin_deg=0.2, coverage_frac=0.5,
                  time_step_min=1.0, time_max_min=25.0)


# ---- 2. perturbation model -------------------------------------------------
def test_perturb_pure_normal_noise():
    rng = np.random.default_rng(0)
    pts = _arc((3.0, 2.0))
    pert = perturb_wavefront(pts, sigma_normal_km=2.0, sigma_shift_km=0.0,
                             sigma_rot_deg=0.0, rng=rng)
    c_lon, c_lat = pts[:, 0].mean(), pts[:, 1].mean()
    bx, by = to_km(pts[:, 0], pts[:, 1], c_lon, c_lat)
    px, py = to_km(pert[:, 0], pert[:, 1], c_lon, c_lat)
    base = np.column_stack([bx, by])
    disp = np.column_stack([px, py]) - base
    t_hat, _ = tangent_normal(base)
    along = np.abs(np.einsum("ij,ij->i", disp, t_hat))
    assert np.max(along) < 1e-6            # displacement is purely along the normal
    assert np.std(disp) > 0                # ... and non-trivial


def test_perturb_pure_rigid_transform():
    rng = np.random.default_rng(0)
    pts = _arc((3.0, 2.0))
    pert = perturb_wavefront(pts, sigma_normal_km=0.0, sigma_shift_km=3.0,
                             sigma_rot_deg=0.5, rng=rng)
    c_lon, c_lat = pts[:, 0].mean(), pts[:, 1].mean()
    bx, by = to_km(pts[:, 0], pts[:, 1], c_lon, c_lat)
    px, py = to_km(pert[:, 0], pert[:, 1], c_lon, c_lat)
    base = np.column_stack([bx, by])
    moved = np.column_stack([px, py])
    # a rigid transform preserves all inter-point distances
    d0 = np.linalg.norm(base[:, None] - base[None], axis=-1)
    d1 = np.linalg.norm(moved[:, None] - moved[None], axis=-1)
    assert np.allclose(d0, d1, atol=1e-6)


# ---- shared synthetic for the driver tests ---------------------------------
def _setup(source):
    """Self-consistent synthetic: the observed arrival times are the modelled
    travel times FROM the true source (so the true source is the exact minimum),
    held fixed while the crest is perturbed.  Isolates the model-T reconstruction
    (cheap vs re-trace) -- the thing under test."""
    bathy = _grad_bathy()
    wf = _arc(source)
    cfg = _tiny_cfg(source)
    cand = build_candidate_grid(cfg)
    wave = resolve_wave(wf, bathy)
    res0 = backproject(wf, cand, bathy, cfg, wave=wave, known_dt=None)
    ix = int(np.argmin(np.abs(cand[0] - source[0])))
    iy = int(np.argmin(np.abs(cand[1] - source[1])))
    tj = res0.stack[:, iy, ix]                       # true source-to-crest times
    res = replace(res0, known_dt=tj)
    kdfn = lambda pts, tj=tj: tj                      # observations fixed
    return res, wf, bathy, cand, cfg, wave, kdfn


# ---- 4. reproducibility ----------------------------------------------------
def test_bootstrap_reproducible():
    res, wf, bathy, cand, cfg, wave, kdfn = _setup((3.0, 2.0))
    kw = dict(n_boot=8, sigma_normal_km=2.0, sigma_shift_km=3.0,
              sigma_rot_deg=0.5, retrace=False)
    a = bootstrap_source(res, wf, bathy, cand, cfg, wave, kdfn,
                         rng=np.random.default_rng(1), **kw)
    b = bootstrap_source(res, wf, bathy, cand, cfg, wave, kdfn,
                         rng=np.random.default_rng(1), **kw)
    assert np.array_equal(a.boot_lon, b.boot_lon, equal_nan=True)
    assert np.array_equal(a.boot_lat, b.boot_lat, equal_nan=True)
    assert np.array_equal(a.boot_tau, b.boot_tau, equal_nan=True)


# ---- 3. re-trace vs cheap agree in the ENSEMBLE (not per-replicate) ---------
def test_bootstrap_retrace_vs_cheap_ensemble():
    res, wf, bathy, cand, cfg, wave, kdfn = _setup((3.0, 2.0))
    kw = dict(n_boot=50, sigma_normal_km=3.0, sigma_shift_km=5.0,
              sigma_rot_deg=0.8)
    cheap = bootstrap_source(res, wf, bathy, cand, cfg, wave, kdfn,
                             rng=np.random.default_rng(2), retrace=False, **kw)
    exact = bootstrap_source(res, wf, bathy, cand, cfg, wave, kdfn,
                             rng=np.random.default_rng(2), retrace=True, **kw)

    def mean_long(b):
        lon, lat = b.boot_lon, b.boot_lat
        m = np.isfinite(lon) & np.isfinite(lat)
        x, y = to_km(lon[m], lat[m], np.mean(lon[m]), np.mean(lat[m]))
        ev = np.linalg.eigvalsh(np.cov(np.vstack([x, y])))   # ascending
        return np.mean(lon[m]), np.mean(lat[m]), float(np.sqrt(ev[-1]))  # long axis km

    ch, ex = mean_long(cheap), mean_long(exact)
    assert abs(ch[0] - ex[0]) < 0.15 and abs(ch[1] - ex[1]) < 0.15   # cloud mean
    # long-axis length agrees to a loose factor on smooth bathymetry
    assert 0.5 < (ch[2] + 1e-6) / (ex[2] + 1e-6) < 2.0


# ---- 1. off by default: no boot file, deterministic outputs intact ----------
def _write_case(tmp_path):
    bathy = _flat_bathy()
    src = (3.0, 2.0)
    wf = _arc(src, radius_deg=2.0, n=11)
    lon, lat, dep = bathy
    LON, LAT = np.meshgrid(lon, lat)
    np.savetxt(tmp_path / "b.xyz",
               np.column_stack([LON.ravel(), LAT.ravel(), -dep.T.ravel()]),
               fmt="%.4f")
    (tmp_path / "wf.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature",
         "properties": {}, "geometry": {"type": "LineString",
         "coordinates": wf.tolist()}}]}))
    import pandas as pd
    t = haversine_km(src[0], src[1], wf[:, 0], wf[:, 1]) / _C_KMS
    pd.DataFrame({"gauge_id": np.arange(len(wf)), "time": t,
                  "eta": 0.1, "lat": wf[:, 1], "lon": wf[:, 0]}
                 ).to_csv(tmp_path / "swot.csv", index=False)
    return tmp_path


def _yaml(tmp_path, out, extra=""):
    return f"""
event: {{epi_lon: 3.0, epi_lat: 2.0, origin_time_utc: "2025-01-01T00:00:00", rupture_speed_kms: null}}
bathymetry: {{path: {tmp_path/'b.xyz'}, negate: true, domain_lon: [0, 6], domain_lat: [0, 6]}}
swot: {{times: {tmp_path/'swot.csv'}, ssh: null}}
candidate: {{lon: [2.0, 4.0], lat: [1.0, 3.0], dlon: 0.2, dlat: 0.2}}
tracing: {{dt: 20.0, max_time: 2500, bin_deg: 0.2, fan_halfwidth_deg: 70.0, azimuth_step_deg: 1.0}}
misfit: {{coverage_frac: 0.6}}
plot: {{misfit_vmax: 10.0}}
time_search: {{enabled: true, step_min: 5, max_min: 15}}
{extra}output: {{out_dir: {out}, tag: btest}}
wavefronts:
  - {{name: WF1, path: {tmp_path/'wf.geojson'}, wavelength: null, n_points: 11}}
"""


def test_cli_bootstrap_off_by_default(tmp_path):
    _write_case(tmp_path)
    out = tmp_path / "out"
    (tmp_path / "run.yaml").write_text(_yaml(tmp_path, out))
    main(["--config", str(tmp_path / "run.yaml")])
    assert (out / "btest.npz").exists()                     # deterministic intact
    assert not list(out.glob("boot_*.npz"))                 # additive: nothing written


def test_cli_bootstrap_writes_sibling(tmp_path):
    _write_case(tmp_path)
    out = tmp_path / "out"
    (tmp_path / "run.yaml").write_text(_yaml(tmp_path, out))
    main(["--config", str(tmp_path / "run.yaml"), "--bootstrap", "12"])
    assert (out / "btest.npz").exists()                     # deterministic untouched
    boot = out / "boot_btest.npz"
    assert boot.exists()
    z = np.load(boot)
    assert len(z["boot_lon"]) == 12 and len(z["boot_tau"]) == 12
    assert z["retrace"] == False
    assert np.isfinite(z["clamp_fraction"])

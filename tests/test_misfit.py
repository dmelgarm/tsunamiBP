"""Misfit-map semantics: std_geom (timing-free) vs std_free (uses t_j).

Synthetic stacks only -- no ray tracing.  The two candidate cells are the whole
point of the corrected definition: an isochron source (A) and a source whose
along-front travel-time tilt matches SWOT's acquisition tilt (B) are DIFFERENT
cells, and the two maps pick opposite ones."""
import numpy as np
import pytest

from tsbp.engine import BPResult, _std_maps


def _two_cells():
    N = 60
    s = np.linspace(0.0, 1.0, N)
    t_j = 70.0 + 0.4 * s
    stack = np.empty((N, 1, 2))
    stack[:, 0, 0] = 70.0 + 0.0 * s     # cell A: front is a perfect isochron
    stack[:, 0, 1] = 70.0 + 0.4 * s     # cell B: tilt matches SWOT acquisition
    cov = np.ones((1, 2), bool)
    return stack, t_j, cov, s


def test_std_geom_values():
    stack, t_j, cov, s = _two_cells()
    std_geom, _ = _std_maps(stack, t_j, cov)
    tilt = np.std(0.4 * s)                        # population std of the tilt
    assert std_geom[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert std_geom[0, 1] == pytest.approx(tilt, rel=1e-6)
    assert tilt == pytest.approx(0.11742, abs=2e-4)


def test_std_free_values():
    stack, t_j, cov, s = _two_cells()
    _, std_free = _std_maps(stack, t_j, cov)
    tilt = np.std(0.4 * s)
    assert std_free[0, 0] == pytest.approx(tilt, rel=1e-6)   # A: ~0.11742
    assert std_free[0, 1] == pytest.approx(0.0, abs=1e-12)   # B: 0


def test_argmins_pick_opposite_cells():
    stack, t_j, cov, _ = _two_cells()
    std_geom, std_free = _std_maps(stack, t_j, cov)
    assert int(np.argmin(std_geom.ravel())) == 0            # geometric -> A
    assert int(np.argmin(std_free.ravel())) == 1            # timing    -> B


def test_std_free_is_tau_optimised_rms():
    stack, t_j, cov, _ = _two_cells()
    _, std_free = _std_maps(stack, t_j, cov)
    for cell in (0, 1):
        T = stack[:, 0, cell]
        tau_star = np.mean(t_j - T)
        rms = np.sqrt(np.mean((tau_star + T - t_j) ** 2))
        assert std_free[0, cell] == pytest.approx(rms, rel=1e-12)


def test_constant_t_makes_free_equal_geom():
    stack, _, cov, _ = _two_cells()
    t_const = np.full(stack.shape[0], 70.0)
    std_geom, std_free = _std_maps(stack, t_const, cov)
    np.testing.assert_allclose(std_free, std_geom, rtol=0.0, atol=1e-12)


def test_per_cell_constant_invariance():
    stack, t_j, cov, _ = _two_cells()
    g0, f0 = _std_maps(stack, t_j, cov)
    bump = np.array([[3.0, -1.5]])[None, :, :]              # (1,1,2), const over j
    g1, f1 = _std_maps(stack + bump, t_j, cov)
    np.testing.assert_allclose(g1, g0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(f1, f0, rtol=0.0, atol=1e-12)


def test_known_dt_none_gives_no_std_free():
    stack, _, cov, _ = _two_cells()
    std_geom, std_free = _std_maps(stack, None, cov)
    assert std_free is None
    assert np.isfinite(std_geom).all()


def test_timesearch_min_ge_std_free_at_cell():
    """The grid + tau>=0 search can only do worse than the unconstrained
    tau-optimum, which is std_free.  Here tau* is off the 1-min grid."""
    from tsbp.config import Config
    from tsbp.timesearch import time_search

    N = 60
    s = np.linspace(0.0, 1.0, N)
    stack = np.empty((N, 1, 2))
    stack[:, 0, 0] = 70.0 + 0.0 * s
    stack[:, 0, 1] = 70.0 + 0.4 * s
    kd = 71.3 + 0.4 * s                                     # tau* off the grid
    cov = np.ones((1, 2), bool)
    std_geom, std_free = _std_maps(stack, kd, cov)

    clon, clat = np.array([0.0, 1.0]), np.array([0.0])
    res = BPResult(clon=clon, clat=clat, stack=stack, n_valid=np.full((1, 2), N),
                   coverage_ok=cov, rms_anchored=None, std_geom=std_geom,
                   std_free=std_free, wavelength=None, known_dt=kd,
                   rupture_delay=None)
    cfg = Config(epi_lon=0.0, epi_lat=0.0, time_step_min=1.0, time_max_min=5.0)
    ts = time_search(res, cfg)
    ix = int(np.argmin(np.abs(clon - ts.best_lon0)))
    iy = int(np.argmin(np.abs(clat - ts.best_lat0)))
    assert ts.best_misfit >= std_free[iy, ix] - 1e-9

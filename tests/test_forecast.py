import numpy as np

from aftershock_pulse.evaluate import b_value, completeness_maxc, match_catalog
from aftershock_pulse.forecast import (
    RJModel,
    completeness_after_mainshock,
    fit_rj,
    omori_integral,
)


def _simulate(model: RJModel, m_min: float, t_end: float, seed=0):
    """Thinning simulation of a Reasenberg and Jones sequence above m_min."""
    rng = np.random.default_rng(seed)
    k = 10 ** (model.a + model.b * (model.mainshock_mag - m_min))
    lam_max = k * model.c ** -model.p
    t, times = 0.0, []
    while True:
        t += rng.exponential(1 / lam_max)
        if t > t_end:
            break
        if rng.uniform() < (t + model.c) ** -model.p / model.c ** -model.p:
            times.append(t)
    times = np.array(times)
    mags = m_min + rng.exponential(1 / (model.b * np.log(10)), size=times.size)
    return times, mags


def test_omori_integral_matches_numeric():
    t = np.linspace(0.1, 5, 200001)
    for p in (0.8, 1.0, 1.2):
        numeric = np.trapezoid((t + 0.05) ** -p, t)
        assert abs(omori_integral(0.1, 5, p, 0.05) - numeric) < 1e-4


def test_fit_recovers_parameters():
    true = RJModel(a=-1.8, b=1.0, p=1.1, c=0.03, mainshock_mag=7.0)
    t, m = _simulate(true, 3.0, 10.0, seed=4)
    fit = fit_rj(t, m, 3.0, 1.0, 7.0, 0.05, 10.0, time_dependent_mc=False)
    assert abs(fit.p - 1.1) < 0.15  # p and c trade off, the forecast is what matters
    expected = true.expected(4.0, 3.0, 10.0)
    assert abs(fit.expected(4.0, 3.0, 10.0) - expected) / expected < 0.25


def test_time_dependent_completeness():
    mc = completeness_after_mainshock(np.array([0.01, 0.1, 1.0, 100.0]), 7.7, 3.0)
    np.testing.assert_allclose(mc, [4.7, 3.95, 3.2, 3.0])


def test_time_dependent_fit_is_robust_to_early_gaps():
    true = RJModel(a=-1.8, b=1.0, p=1.1, c=0.03, mainshock_mag=7.0)
    t, m = _simulate(true, 2.5, 10.0, seed=5)
    # remove what a real network would miss early on
    seen = m >= completeness_after_mainshock(t, 7.0, 2.5)
    naive = fit_rj(t[seen], m[seen], 2.5, 1.0, 7.0, 0.05, 3.0, time_dependent_mc=False)
    aware = fit_rj(t[seen], m[seen], 2.5, 1.0, 7.0, 0.05, 3.0, time_dependent_mc=True)
    assert abs(aware.p - 1.1) < abs(naive.p - 1.1)


def test_b_value_and_mc():
    rng = np.random.default_rng(2)
    m = np.round(1.95 + rng.exponential(1 / np.log(10), size=5000), 1)
    b, se, n = b_value(m, 2.0)
    assert abs(b - 1.0) < 0.06 and n == 5000
    assert 1.9 <= completeness_maxc(m) <= 2.3


def test_match_is_one_to_one():
    det = np.array([10.0, 11.0, 50.0])
    cat = np.array([10.5, 10.6, 80.0])
    out = match_catalog(det, cat, tol_s=2.0)
    assert sorted(out[:2].tolist()) == [0, 1] and out[2] == -1

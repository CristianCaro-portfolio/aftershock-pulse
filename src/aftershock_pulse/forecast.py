"""Reasenberg and Jones aftershock forecast, fitted by maximum likelihood.

Rate of aftershocks with magnitude >= M at time t days after a mainshock of magnitude Mm:
    lambda(t, M) = 10 ** (a + b * (Mm - M)) * (t + c) ** -p
The fit uses events above a completeness magnitude between t_start and T, with b fixed from the
same events. The forecast is the expected number of M >= Mf events in a later window, with a
Poisson interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

GENERIC_CALIFORNIA = {"a": -1.67, "b": 0.91, "p": 1.08, "c": 0.05}  # Reasenberg and Jones 1989


def omori_integral(t1: float, t2: float, p: float, c: float) -> float:
    if abs(p - 1.0) < 1e-6:
        return float(np.log((t2 + c) / (t1 + c)))
    return float(((t2 + c) ** (1 - p) - (t1 + c) ** (1 - p)) / (1 - p))


@dataclass
class RJModel:
    a: float
    b: float
    p: float
    c: float
    mainshock_mag: float
    n_fit: int = 0
    m_min: float = float("nan")

    def expected(self, m: float, t1: float, t2: float) -> float:
        k = 10 ** (self.a + self.b * (self.mainshock_mag - m))
        return k * omori_integral(t1, t2, self.p, self.c)

    def interval(self, m: float, t1: float, t2: float, level: float = 0.95):
        mu = self.expected(m, t1, t2)
        lo, hi = poisson.ppf([(1 - level) / 2, 1 - (1 - level) / 2], mu)
        return mu, int(lo), int(hi)

    def prob_at_least_one(self, m: float, t1: float, t2: float) -> float:
        return float(1 - np.exp(-self.expected(m, t1, t2)))


def completeness_after_mainshock(t_days, mainshock_mag: float, floor: float):
    """Short term incompleteness after a large mainshock: coda and overlapping waveforms hide
    small events. Helmstetter, Kagan and Jackson (2006): Mc(t) = Mm - 4.5 - 0.75 log10(t)."""
    t = np.maximum(np.asarray(t_days, dtype=float), 1e-6)
    return np.maximum(floor, mainshock_mag - 4.5 - 0.75 * np.log10(t))


def fit_rj(t_days: np.ndarray, mags: np.ndarray, m_min: float, b: float, mainshock_mag: float,
           t_start: float, t_end: float, time_dependent_mc: bool = True) -> RJModel:
    """Maximum likelihood a, p and c with b fixed.

    With time_dependent_mc the threshold at time t is max(m_min, Mc(t)), so the hours right
    after the mainshock count only the events the network could actually see. Without it, the
    missing small events early on flatten the decay and the fit underestimates p.
    """
    t_days, mags = np.asarray(t_days, dtype=float), np.asarray(mags, dtype=float)
    win = (t_days >= t_start) & (t_days < t_end)
    if time_dependent_mc:
        thr = completeness_after_mainshock(t_days, mainshock_mag, m_min)
    else:
        thr = np.full(t_days.shape, m_min)
    sel = win & (mags >= thr)
    t = t_days[sel]
    n = t.size
    if n < 3:
        raise ValueError(f"only {n} complete events before day {t_end}")
    shift = b * (mainshock_mag - m_min)
    # integration grid, log spaced because the rate changes fastest at the start
    tg = np.geomspace(t_start, t_end, 600)
    if time_dependent_mc:
        drop = completeness_after_mainshock(tg, mainshock_mag, m_min) - m_min
    else:
        drop = np.zeros_like(tg)
    drop_events = (thr[sel] - m_min)

    def nll(x):
        a, p, log_c = x
        c = 10**log_c
        log_k = (a + shift) * np.log(10)
        log_rate = log_k - b * drop_events * np.log(10) - p * np.log(t + c)
        rate_grid = np.exp(log_k - b * drop * np.log(10) - p * np.log(tg + c))
        return -(np.sum(log_rate) - np.trapezoid(rate_grid, tg))

    best = None
    for p0 in (0.9, 1.1, 1.3):
        for lc0 in (-2.0, -1.0):
            res = minimize(nll, x0=[-1.5, p0, lc0], method="L-BFGS-B",
                           bounds=[(-6, 3), (0.5, 2.0), (-3.5, 0.0)])
            if best is None or res.fun < best.fun:
                best = res
    a, p, log_c = best.x
    return RJModel(a=float(a), b=b, p=float(p), c=float(10**log_c),
                   mainshock_mag=mainshock_mag, n_fit=int(n), m_min=m_min)


def fit_productivity(t_days: np.ndarray, mags: np.ndarray, m_min: float, b: float,
                     mainshock_mag: float, t_start: float, t_end: float, p: float,
                     c: float) -> RJModel:
    """Maximum likelihood a with the decay (p, c) fixed, closed form: K = N / integral.

    Used for the hybrid forecast: the dense sensor catalog says how fast the sequence decays,
    the authoritative catalog says how productive it is in its own magnitude scale.
    """
    t_days, mags = np.asarray(t_days, dtype=float), np.asarray(mags, dtype=float)
    thr = completeness_after_mainshock(t_days, mainshock_mag, m_min)
    n = int(((t_days >= t_start) & (t_days < t_end) & (mags >= thr)).sum())
    if n < 1:
        raise ValueError(f"no complete events before day {t_end}")
    tg = np.geomspace(t_start, t_end, 600)
    drop = completeness_after_mainshock(tg, mainshock_mag, m_min) - m_min
    integral = np.trapezoid(10 ** (-b * drop) * (tg + c) ** -p, tg)
    a = np.log10(n / integral) - b * (mainshock_mag - m_min)
    return RJModel(a=float(a), b=b, p=p, c=c, mainshock_mag=mainshock_mag, n_fit=n, m_min=m_min)


def poisson_log_score(mu: float, observed: int) -> float:
    return float(poisson.logpmf(observed, mu))

"""Amplitudes per station and a magnitude scale calibrated against the reference catalog.

The scale is M = beta * log10(A) + alpha * log10(R) + c_station, where A is the peak band passed
ground velocity after the onset and R the hypocentral distance. alpha is shared, every station
gets its own constant, which absorbs site effects and any gain the metadata gets wrong.

beta is fixed at 1 by default, as in any magnitude scale. Fitting it by regressing catalog
magnitude on a noisy log amplitude shrinks it toward zero (regression dilution): on this data a
free beta came out near 0.3, every event looked like M4.5 and the b value doubled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .associate import Grid


def event_amplitudes(events: pd.DataFrame, picks: pd.DataFrame, triggers: pd.DataFrame,
                     grid: Grid, depth_km: float) -> pd.DataFrame:
    """One row per (event, station) pick with the trigger amplitude and hypocentral distance."""
    if picks.empty:
        return pd.DataFrame(columns=["origin", "station", "peak_ms", "snr", "hypo_km"])
    node = {
        (round(la, 4), round(lo, 4)): i
        for i, (la, lo) in enumerate(zip(grid.lats, grid.lons, strict=True))
    }
    col = {code: j for j, code in enumerate(grid.stations)}
    ev = events.reset_index(drop=True)
    rows = []
    for p in picks.itertuples(index=False):
        e = ev.iloc[p.event]
        n = node[(round(e.latitude, 4), round(e.longitude, 4))]
        trig = triggers.loc[p.trigger]
        rows.append(
            {
                "origin": e.origin,
                "station": p.station,
                "peak_ms": trig.peak_ms,
                "snr": trig.snr,
                "hypo_km": float(np.hypot(grid.dist_km[n, col[p.station]], depth_km)),
            }
        )
    return pd.DataFrame(rows)


class MagnitudeScale:
    def __init__(self, beta: float = 1.0, alpha: float = 1.5,
                 constants: dict[str, float] | None = None, fit_beta: bool = False):
        self.beta = beta
        self.fit_beta = fit_beta
        self.alpha = alpha
        self.constants = constants or {}

    def fit(self, amps: pd.DataFrame, ref_mag: pd.Series) -> MagnitudeScale:
        """amps rows carry an 'origin' that indexes ref_mag (the catalog magnitude)."""
        df = amps[amps.origin.isin(ref_mag.index)].copy()
        if df.empty:
            raise ValueError("no amplitudes to calibrate on")
        stations = sorted(df.station.unique())
        s_idx = df.station.map({s: i for i, s in enumerate(stations)}).to_numpy()
        logA = np.log10(df.peak_ms.to_numpy())
        logR = np.log10(df.hypo_km.to_numpy())
        target = ref_mag.loc[df.origin].to_numpy()

        fixed_beta = self.beta

        def unpack(x):
            if self.fit_beta:
                return x[0], x[1], x[2:]
            return fixed_beta, x[0], x[1:]

        def resid(x):
            beta, alpha, const = unpack(x)
            return beta * logA + alpha * logR + const[s_idx] - target

        start = [1.0, 1.5] if self.fit_beta else [1.5]
        x0 = np.concatenate([start, np.full(len(stations), 6.0)])
        sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.3)
        beta, alpha, const = unpack(sol.x)
        self.beta, self.alpha = float(beta), float(alpha)
        self.constants = {s: float(c) for s, c in zip(stations, const, strict=True)}
        return self

    def station_magnitudes(self, amps: pd.DataFrame) -> pd.Series:
        c = amps.station.map(self.constants)
        return self.beta * np.log10(amps.peak_ms) + self.alpha * np.log10(amps.hypo_km) + c

    def event_magnitudes(self, amps: pd.DataFrame, min_snr: float = 3.0) -> pd.Series:
        ok = amps[(amps.snr >= min_snr) & amps.station.isin(self.constants)].copy()
        ok["m"] = self.station_magnitudes(ok)
        return ok.groupby("origin")["m"].median()

    def to_dict(self) -> dict:
        return {"beta": self.beta, "alpha": self.alpha, "constants": self.constants}


def cross_validated_mae(amps: pd.DataFrame, ref_mag: pd.Series, folds: int = 5,
                        seed: int = 7) -> tuple[float, pd.DataFrame]:
    """K fold by event. Returns the MAE and the out of fold predictions."""
    origins = np.array(sorted(ref_mag.index))
    rng = np.random.default_rng(seed)
    rng.shuffle(origins)
    preds = []
    for k in range(folds):
        test = set(origins[k::folds])
        train_ref = ref_mag[[o for o in ref_mag.index if o not in test]]
        scale = MagnitudeScale().fit(amps, train_ref)
        m = scale.event_magnitudes(amps[amps.origin.isin(test)])
        preds.append(pd.DataFrame({"origin": m.index, "pred": m.values}))
    out = pd.concat(preds, ignore_index=True)
    out["ref"] = ref_mag.loc[out.origin].to_numpy()
    return float(np.mean(np.abs(out.pred - out.ref))), out

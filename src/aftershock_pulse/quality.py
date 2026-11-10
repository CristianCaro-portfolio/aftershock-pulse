"""Hourly data quality per station: coverage, noise level and dead channels."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .store import CHUNK_SECONDS, WaveStore


def hourly_quality(store: WaveStore, code: str) -> pd.DataFrame:
    data = store.read(code, store.start, store.end)
    per = int(CHUNK_SECONDS * store.rate)
    n_hours = len(data) // per
    blocks = data[: n_hours * per].reshape(n_hours, per)
    finite = np.isfinite(blocks)
    cov = finite.mean(axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rms = np.sqrt(np.nanmean(np.where(finite, blocks, np.nan) ** 2, axis=1))
        p05 = np.nanpercentile(np.abs(blocks), 5, axis=1)
    flat = (cov > 0) & (np.nan_to_num(rms) < 1e-12)
    hours = [store.start + h * CHUNK_SECONDS for h in range(n_hours)]
    return pd.DataFrame(
        {
            "station": code,
            "hour": [str(h)[:13] for h in hours],
            "coverage": cov.round(4),
            "rms_ms": rms,
            "p05_abs_ms": p05,
            "flatline": flat,
        }
    )


def summarize(qc: pd.DataFrame) -> pd.DataFrame:
    g = qc.groupby("station")
    return pd.DataFrame(
        {
            "coverage": g.coverage.mean().round(4),
            "hours_without_data": g.coverage.apply(lambda c: int((c == 0).sum())),
            "flatline_hours": g.flatline.sum().astype(int),
            "median_noise_rms_ms": g.rms_ms.median(),
        }
    )

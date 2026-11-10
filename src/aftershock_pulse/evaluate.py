"""Detection quality against the reference catalog and magnitude statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def match_catalog(det_origins: np.ndarray, cat_times: np.ndarray, tol_s: float) -> np.ndarray:
    """Greedy one to one match by time. Returns, per catalog event, the index of the matched
    detection or -1."""
    det_origins = np.asarray(det_origins, dtype=float)
    order = np.argsort(det_origins)
    sorted_det = det_origins[order]
    used = np.zeros(len(det_origins), dtype=bool)
    out = np.full(len(cat_times), -1, dtype=int)
    for i, t in enumerate(cat_times):
        lo = np.searchsorted(sorted_det, t - tol_s)
        hi = np.searchsorted(sorted_det, t + tol_s, side="right")
        best, best_dt = -1, np.inf
        for k in range(lo, hi):
            j = order[k]
            if not used[j] and abs(sorted_det[k] - t) < best_dt:
                best, best_dt = j, abs(sorted_det[k] - t)
        if best >= 0:
            used[best] = True
            out[i] = best
    return out


def recall_by_magnitude(cat_mag: np.ndarray, matched: np.ndarray,
                        bins=(4.0, 4.5, 5.0, 5.5, 10.0)) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = (cat_mag >= lo) & (cat_mag < hi)
        n = int(sel.sum())
        hit = int((matched[sel] >= 0).sum())
        rows.append({"mag_bin": f"{lo:.1f}-{hi:.1f}" if hi < 10 else f">={lo:.1f}",
                     "n_catalog": n, "n_detected": hit,
                     "recall": hit / n if n else np.nan})
    return pd.DataFrame(rows)


def completeness_maxc(mags: np.ndarray, dm: float = 0.1, correction: float = 0.2) -> float:
    """Magnitude of completeness by maximum curvature (Wiemer and Wyss 2000) plus correction."""
    mags = np.asarray(mags)
    mags = mags[np.isfinite(mags)]
    if mags.size < 10:
        return float("nan")
    bins, counts = np.unique(np.round(mags / dm).astype(int), return_counts=True)
    return float(np.round(bins[np.argmax(counts)] * dm + correction, 1))


def b_value(mags: np.ndarray, mc: float, dm: float = 0.1) -> tuple[float, float, int]:
    """Aki and Utsu maximum likelihood b value with Shi and Bolt standard error."""
    m = np.asarray(mags)
    m = m[np.isfinite(m) & (m >= mc - 1e-9)]
    n = m.size
    if n < 10:
        return float("nan"), float("nan"), n
    b = np.log10(np.e) / (m.mean() - (mc - dm / 2))
    se = 2.3 * b**2 * np.sqrt(np.sum((m - m.mean()) ** 2) / (n * (n - 1)))
    return float(b), float(se), int(n)

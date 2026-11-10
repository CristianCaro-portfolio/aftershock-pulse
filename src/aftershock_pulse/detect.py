"""Single station event triggers with a recursive STA/LTA detector.

The characteristic function is computed once per station over every contiguous stretch of
data. Triggers in the first LTA window of a stretch are discarded, because right after a gap
the long term average has not settled and every bit of noise looks like an event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

from .config import Detection
from .store import WaveStore


@dataclass
class StationCft:
    code: str
    rate: float
    cft: np.ndarray  # NaN where there is no data or the LTA is still warming up
    data: np.ndarray  # the band passed velocity the cft was computed from


def finite_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and end indices of the runs of True values in mask."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2], strict=True))


def characteristic_function(data: np.ndarray, rate: float, det: Detection) -> np.ndarray:
    nsta, nlta = int(det.sta_s * rate), int(det.lta_s * rate)
    cft = np.full(data.shape, np.nan, dtype="float64")
    for lo, hi in finite_segments(np.isfinite(data)):
        if hi - lo < 3 * nlta:
            continue
        seg = recursive_sta_lta(data[lo:hi].astype("float64"), nsta, nlta)
        seg[:nlta] = np.nan
        cft[lo:hi] = seg
    return cft


def station_cft(store: WaveStore, code: str, det: Detection) -> StationCft:
    data = store.read(code, store.start, store.end)
    return StationCft(code, store.rate, characteristic_function(data, store.rate, det), data)


TRIGGER_COLUMNS = ["station", "t_on", "t_off", "cft_peak", "peak_ms", "snr"]
NOISE_S = 20.0


def amplitude(data: np.ndarray, rate: float, on: int, off: int) -> tuple[float, float]:
    """Peak band passed velocity after the onset (long enough to include the S wave) and its
    ratio to the noise RMS just before the onset."""
    win = int(np.clip(off - on + 5 * rate, 10 * rate, 60 * rate))
    sig = data[max(on - int(rate), 0) : on + win]
    noise = data[max(on - int((NOISE_S + 1) * rate), 0) : max(on - int(rate), 0)]
    sig, noise = sig[np.isfinite(sig)], noise[np.isfinite(noise)]
    if sig.size == 0:
        return float("nan"), float("nan")
    peak = float(np.max(np.abs(sig)))
    nrms = float(np.sqrt(np.mean(noise**2))) if noise.size else 0.0
    return peak, (peak / nrms if nrms > 0 else float("nan"))


def triggers_from_cft(sc: StationCft, det: Detection, t0_epoch: float) -> pd.DataFrame:
    cft = np.nan_to_num(sc.cft, nan=0.0)
    on_off = trigger_onset(cft, det.trigger_on, det.trigger_off)
    rows = []
    min_len = det.min_duration_s * sc.rate
    for on, off in on_off:
        if off - on < min_len:
            continue
        peak, snr = amplitude(sc.data, sc.rate, on, off)
        rows.append(
            {
                "station": sc.code,
                "t_on": t0_epoch + on / sc.rate,
                "t_off": t0_epoch + off / sc.rate,
                "cft_peak": float(cft[on : off + 1].max()),
                "peak_ms": peak,
                "snr": snr,
            }
        )
    return pd.DataFrame(rows, columns=TRIGGER_COLUMNS)


def detect_all(store: WaveStore, det: Detection, cfts: dict[str, StationCft] | None = None,
               stations: list[str] | None = None) -> pd.DataFrame:
    cfts = cfts if cfts is not None else {}
    frames = []
    for code in stations or store.stations():
        if code not in cfts:
            cfts[code] = station_cft(store, code, det)
        frames.append(triggers_from_cft(cfts[code], det, store.start.timestamp))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.sort_values("t_on").reset_index(drop=True)

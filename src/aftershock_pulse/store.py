"""Continuous waveform store on Zarr v3.

One float32 array per station covers the whole study window at a fixed sample rate. Chunks are
one hour long, so a write touches only the hours it covers and a reader can pull any slice
without loading a day of data. Samples that were never written, or fell in a gap, read as NaN.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr
from obspy import UTCDateTime

CHUNK_SECONDS = 3600


class WaveStore:
    def __init__(self, path: str | Path, start: UTCDateTime, end: UTCDateTime, rate_hz: float):
        self.path = Path(path)
        self.start = UTCDateTime(start)
        self.end = UTCDateTime(end)
        self.rate = float(rate_hz)
        self.n_samples = int(round((self.end - self.start) * self.rate))
        self.root = zarr.open_group(str(self.path), mode="a")
        meta = dict(self.root.attrs)
        if meta:
            if (meta["start"], meta["rate_hz"]) != (str(self.start), self.rate):
                raise ValueError(f"{self.path} was created with another window or rate")
        else:
            self.root.attrs.update(
                {"start": str(self.start), "end": str(self.end), "rate_hz": self.rate}
            )

    @classmethod
    def open(cls, path: str | Path) -> WaveStore:
        root = zarr.open_group(str(path), mode="r")
        meta = dict(root.attrs)
        return cls(path, UTCDateTime(meta["start"]), UTCDateTime(meta["end"]), meta["rate_hz"])

    # station level -------------------------------------------------------------------------
    def stations(self) -> list[str]:
        return sorted(name for name, _ in self.root.arrays())

    def _array(self, code: str, create: bool = False):
        if code in self.root:
            return self.root[code]
        if not create:
            raise KeyError(code)
        return self.root.create_array(
            code,
            shape=(self.n_samples,),
            chunks=(int(CHUNK_SECONDS * self.rate),),
            dtype="float32",
            fill_value=np.nan,
        )

    def set_station_meta(self, code: str, meta: dict) -> None:
        self._array(code, create=True).attrs.update(meta)

    def station_meta(self, code: str) -> dict:
        return dict(self._array(code).attrs)

    # sample level --------------------------------------------------------------------------
    def index(self, t: UTCDateTime) -> int:
        return int(round((UTCDateTime(t) - self.start) * self.rate))

    def time_of(self, idx: int) -> UTCDateTime:
        return self.start + idx / self.rate

    def write(self, code: str, t0: UTCDateTime, data: np.ndarray) -> None:
        arr = self._array(code, create=True)
        i0 = self.index(t0)
        data = np.asarray(data, dtype="float32")
        lo, hi = max(i0, 0), min(i0 + len(data), self.n_samples)
        if hi <= lo:
            return
        arr[lo:hi] = data[lo - i0 : hi - i0]

    def read(self, code: str, t0: UTCDateTime, t1: UTCDateTime) -> np.ndarray:
        arr = self._array(code)
        lo, hi = max(self.index(t0), 0), min(self.index(t1), self.n_samples)
        if hi <= lo:
            return np.empty(0, dtype="float32")
        return np.asarray(arr[lo:hi])


class Manifest:
    """Records which (station, chunk) pairs were fetched so ingestion can resume."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(code: str, t0: UTCDateTime) -> str:
        return f"{code}|{UTCDateTime(t0).isoformat()}"

    def done(self, code: str, t0: UTCDateTime) -> bool:
        entry = self.entries.get(self.key(code, t0))
        return bool(entry) and entry["status"] in {"ok", "empty"}

    def record(self, code: str, t0: UTCDateTime, status: str, **extra) -> None:
        self.entries[self.key(code, t0)] = {"status": status, **extra}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

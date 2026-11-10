"""Download continuous waveforms from FDSN services into the Zarr store.

Each station is fetched in fixed chunks with padding on both sides, cleaned and filtered, and
written into the store. Requests are retried with backoff because public services (Raspberry
Shake in particular) answer 502 under load. Progress is kept in a manifest so a rerun only asks
for the chunks that are still missing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from obspy import Inventory, Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from .config import Config, Processing, StationSpec
from .store import Manifest, WaveStore

log = logging.getLogger(__name__)

PAD_S = 60.0
MAX_ATTEMPTS = 4


def chunks(start: UTCDateTime, end: UTCDateTime, hours: int) -> Iterator[UTCDateTime]:
    t = UTCDateTime(start)
    while t < end:
        yield t
        t += hours * 3600


def with_retry(fn, *, attempts: int = MAX_ATTEMPTS, base_delay: float = 2.0, sleep=time.sleep):
    """Call fn, retrying transient failures with exponential backoff. No data is not retried."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except FDSNNoDataException:
            raise
        except Exception as exc:  # network errors, 5xx, truncated responses
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("attempt %d failed (%s), retrying in %.0fs", attempt, exc, delay)
            sleep(delay)
    raise RuntimeError("unreachable")


def preprocess(stream: Stream, inventory: Inventory | None, proc: Processing) -> Stream:
    """Gap aware cleaning: every contiguous piece is detrended, filtered and resampled alone."""
    stream = stream.copy()
    stream.merge(method=1, fill_value=None)
    pieces = stream.split()
    out = Stream()
    for tr in pieces:
        if tr.stats.npts < tr.stats.sampling_rate * 10:
            continue  # pieces shorter than 10 s are dropped, the filter would ring over them
        tr.data = tr.data.astype("float64")
        if inventory is not None:
            tr.remove_sensitivity(inventory)
        tr.detrend("demean")
        tr.detrend("linear")
        tr.taper(max_percentage=None, max_length=2.0)
        tr.filter("bandpass", freqmin=proc.freqmin_hz, freqmax=proc.freqmax_hz, corners=4)
        if tr.stats.sampling_rate != proc.target_rate_hz:
            # start on the store's sample grid so every station shares the same time axis
            ts = tr.stats.starttime.timestamp
            aligned = UTCDateTime(np.ceil(ts * proc.target_rate_hz) / proc.target_rate_hz)
            tr.interpolate(sampling_rate=proc.target_rate_hz, method="lanczos", a=20,
                           starttime=aligned)
        out.append(tr)
    return out


def station_metadata(client: Client, spec: StationSpec, t0: UTCDateTime, t1: UTCDateTime):
    inv = with_retry(
        lambda: client.get_stations(
            network=spec.network,
            station=spec.station,
            location=spec.location,
            channel=spec.channel,
            starttime=t0,
            endtime=t1,
            level="response",
        )
    )
    sta = inv[0][0]
    # coordinates are kept to 0.01 degrees: enough for travel times, and citizen stations sit
    # at people's homes, so the exact spot is not repeated anywhere in this project
    meta = {
        "network": spec.network,
        "station": spec.station,
        "channel": spec.channel,
        "provider": spec.provider,
        "latitude": round(float(sta.latitude), 2),
        "longitude": round(float(sta.longitude), 2),
        "elevation_m": round(float(sta.elevation)),
        "redistributable": spec.redistributable,
    }
    return inv, meta


def ingest_station(cfg: Config, spec: StationSpec, store: WaveStore, manifest: Manifest,
                   client: Client | None = None) -> dict:
    # service discovery is skipped: under load it can fail and leave the client without dataselect
    client = client or Client(spec.provider, timeout=120, _discover_services=False)
    inv, meta = station_metadata(client, spec, cfg.window.start, cfg.window.end)
    store.set_station_meta(spec.code, meta)
    stats = {"ok": 0, "empty": 0, "failed": 0, "skipped": 0}
    for t0 in chunks(cfg.window.start, cfg.window.end, cfg.window.chunk_hours):
        if manifest.done(spec.code, t0):
            stats["skipped"] += 1
            continue
        t1 = min(t0 + cfg.window.chunk_hours * 3600, cfg.window.end)
        try:
            raw = with_retry(
                lambda t0=t0, t1=t1: client.get_waveforms(
                    spec.network, spec.station, spec.location, spec.channel, t0 - PAD_S, t1 + PAD_S
                )
            )
        except FDSNNoDataException:
            manifest.record(spec.code, t0, "empty")
            stats["empty"] += 1
            continue
        except Exception as exc:
            log.error("%s %s failed: %s", spec.code, t0, exc)
            manifest.record(spec.code, t0, "failed", error=str(exc)[:200])
            stats["failed"] += 1
            continue
        clean = preprocess(raw, inv, cfg.processing)
        written = 0
        for tr in clean:
            tr.trim(t0, t1, nearest_sample=False)
            if tr.stats.npts:
                store.write(spec.code, tr.stats.starttime, tr.data)
                written += tr.stats.npts
        coverage = written / ((t1 - t0) * cfg.processing.target_rate_hz)
        manifest.record(spec.code, t0, "ok", coverage=round(min(coverage, 1.0), 4))
        stats["ok"] += 1
        manifest.save()
    manifest.save()
    return stats


def ingest(cfg: Config, workdir: Path, stations: list[str] | None = None,
           workers: int = 4) -> dict:
    store = WaveStore(workdir / "waveforms.zarr", cfg.window.start, cfg.window.end,
                      cfg.processing.target_rate_hz)
    specs = [s for s in cfg.stations if stations is None or s.code in stations]
    results: dict[str, dict] = {}

    def run_station(spec: StationSpec) -> None:
        manifest = Manifest(workdir / "manifests" / f"{spec.code}.json")
        results[spec.code] = ingest_station(cfg, spec, store, manifest)
        log.info("%s %s", spec.code, results[spec.code])

    # one worker per station: each writes its own Zarr array and manifest, so nothing is
    # shared, and the retry backoff keeps the load on each data center modest
    with ThreadPoolExecutor(max_workers=min(len(specs), workers) or 1) as pool:
        list(pool.map(run_station, specs))
    return results


def coverage(store: WaveStore, code: str, t0: UTCDateTime, t1: UTCDateTime) -> float:
    data = store.read(code, t0, t1)
    return float(np.isfinite(data).mean()) if data.size else 0.0

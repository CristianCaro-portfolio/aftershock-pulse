# ADR 001: Zarr v3 for continuous waveforms

## Context

The pipeline needs 14 days of four stations on one shared time axis. Detection reads whole
stations, magnitude and QC read slices, and ingestion writes 6 hour chunks in any order, retries
some of them and resumes after a crash. Raw miniSEED keeps each station in its own sample rate
and time stamps, with gaps as separate records.

## Decision

After cleaning, each station is resampled onto a 25 Hz grid that starts at the window start and
written into one float32 Zarr v3 array per station, chunked by hour, with NaN as fill value.

## Why

* A chunk write is idempotent and touches only the hours it covers, so a retried or resumed
  download never rewrites a day.
* Gaps are explicit. Anything never written reads as NaN, and every later stage (STA/LTA restart,
  QC coverage, amplitude windows) handles NaN instead of guessing where a gap was.
* One axis for all stations means association and plotting index by time without per station
  bookkeeping.
* It is a plain directory of chunks, readable from Python, xarray or any Zarr client, and it would
  move to object storage unchanged.

## Alternatives

* **miniSEED files per day.** The native format, but every reader would have to merge, align and
  find gaps again.
* **Parquet with one row per sample.** Columnar but wasteful for dense regular series, about 30x
  the size here.
* **HDF5.** Fine on one machine, but single writer and not chunk addressable on object storage.

## Consequences

About 400 MB for 14 days of four stations at 25 Hz. The original sample rates are lost above
12.5 Hz, which is acceptable for a 2 to 10 Hz detector at regional distances. Raw waveforms stay
local because Raspberry Shake does not allow redistribution; the repo ships derived triggers.

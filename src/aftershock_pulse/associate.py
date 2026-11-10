"""Grid association: turn single station triggers into events from the aftershock zone.

Every node of a grid over the aftershock zone predicts a P travel time to every station. A
trigger at one station, paired with a node, implies an origin time. The node that lines up the
most other stations within the tolerance wins, and the event is kept if enough stations agree.
This is a small back projection: it rejects noise that hits a single sensor and events from
outside the zone whose moveout does not fit any node.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .catalog import haversine_km
from .config import Association, Region

KM_PER_DEG = 111.195


@lru_cache(maxsize=8)
def p_travel_time_table(depth_km: float, max_deg: float = 8.0, step_deg: float = 0.02):
    """First P arrival (s) against epicentral distance (deg) from TauP with iasp91."""
    from obspy.taup import TauPyModel

    model = TauPyModel("iasp91")
    dists = np.arange(0.0, max_deg + step_deg, step_deg)
    times = np.empty_like(dists)
    for i, d in enumerate(dists):
        arrivals = model.get_travel_times(depth_km, d, phase_list=["p", "P", "Pn", "Pg"])
        times[i] = min(a.time for a in arrivals)
    return dists, times


@dataclass
class Grid:
    lats: np.ndarray  # (n_nodes,)
    lons: np.ndarray
    tt: np.ndarray  # (n_nodes, n_stations) P travel time in s
    dist_km: np.ndarray  # (n_nodes, n_stations)
    stations: list[str]


def build_grid(region: Region, stations: dict[str, tuple[float, float]]) -> Grid:
    lat_axis = np.arange(region.lat_min, region.lat_max + 1e-9, region.step_deg)
    lon_axis = np.arange(region.lon_min, region.lon_max + 1e-9, region.step_deg)
    lats, lons = (a.ravel() for a in np.meshgrid(lat_axis, lon_axis, indexing="ij"))
    dists, times = p_travel_time_table(region.depth_km)
    codes = sorted(stations)
    dist_km = np.stack(
        [haversine_km(lats, lons, *stations[c]) for c in codes], axis=1
    )
    tt = np.interp(dist_km / KM_PER_DEG, dists, times)
    return Grid(lats, lons, tt, dist_km, codes)


def associate(triggers: pd.DataFrame, grid: Grid, assoc: Association):
    """Return (events, picks).

    events has one row per event with origin time, best node and fit quality. picks links each
    event to the trigger row (index into triggers) that every agreeing station contributed.
    """
    cols = ["origin", "latitude", "longitude", "n_stations", "rms_s", "stations"]
    empty = (pd.DataFrame(columns=cols), pd.DataFrame(columns=["event", "station", "trigger"]))
    if triggers.empty:
        return empty
    per_sta: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code in grid.stations:
        sub = triggers[triggers.station == code].sort_values("t_on")
        per_sta[code] = (sub.t_on.to_numpy(), sub.index.to_numpy())
    idx = {code: i for i, code in enumerate(grid.stations)}
    tol = assoc.tolerance_s
    candidates = []
    for code, (onsets, rows) in per_sta.items():
        k = idx[code]
        for t, row in zip(onsets, rows, strict=True):
            origins = t - grid.tt[:, k]  # implied origin time at every node
            count = np.ones(len(origins), dtype=int)
            sq = np.zeros(len(origins))
            for other, j in idx.items():
                arr = per_sta[other][0]
                if other == code or arr.size == 0:
                    continue
                resid = _nearest(arr, origins + grid.tt[:, j]) - (origins + grid.tt[:, j])
                hit = np.abs(resid) <= tol
                count += hit
                sq += np.where(hit, resid**2, 0.0)
            best_count = int(count.max())
            if best_count < assoc.min_stations:
                continue
            nodes = np.flatnonzero(count == best_count)
            rms = np.sqrt(sq[nodes] / max(best_count - 1, 1))
            n = int(nodes[np.argmin(rms)])
            picks = {code: int(row)}
            for other, j in idx.items():
                arr, arr_rows = per_sta[other]
                if other == code or arr.size == 0:
                    continue
                pred = origins[n] + grid.tt[n, j]
                pos = int(np.argmin(np.abs(arr - pred)))
                if abs(arr[pos] - pred) <= tol:
                    picks[other] = int(arr_rows[pos])
            candidates.append(
                {
                    "origin": float(origins[n]),
                    "latitude": float(grid.lats[n]),
                    "longitude": float(grid.lons[n]),
                    "n_stations": best_count,
                    "rms_s": float(rms.min()),
                    "stations": ",".join(sorted(picks)),
                    "pick_map": picks,
                }
            )
    if not candidates:
        return empty
    cand = pd.DataFrame(candidates).sort_values(["n_stations", "rms_s"], ascending=[False, True])
    events = _dedupe(cand, assoc.min_separation_s)
    picks = pd.DataFrame(
        [
            {"event": i, "station": sta, "trigger": trig}
            for i, p in enumerate(events["pick_map"])
            for sta, trig in p.items()
        ]
    )
    return events[cols], picks


def _nearest(sorted_arr: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_arr.size == 1:
        return np.full(values.shape, sorted_arr[0])
    pos = np.clip(np.searchsorted(sorted_arr, values), 1, len(sorted_arr) - 1)
    left, right = sorted_arr[pos - 1], sorted_arr[pos]
    return np.where(np.abs(right - values) < np.abs(left - values), right, left)


def _dedupe(cand: pd.DataFrame, min_sep: float) -> pd.DataFrame:
    """Greedy: best candidates first, drop anything within min_sep of an accepted origin."""
    kept: list[float] = []
    rows = []
    for row in cand.itertuples(index=False):
        pos = np.searchsorted(kept, row.origin)
        near = [kept[i] for i in (pos - 1, pos) if 0 <= i < len(kept)]
        if any(abs(row.origin - k) < min_sep for k in near):
            continue
        kept.insert(pos, row.origin)
        rows.append(row._asdict())
    return pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)

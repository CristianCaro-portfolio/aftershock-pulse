import numpy as np
import pandas as pd

from aftershock_pulse.associate import associate, build_grid
from aftershock_pulse.config import Association, Region

REGION = Region(lat_min=7.0, lat_max=8.0, lon_min=-81.5, lon_max=-80.0, step_deg=0.1)
STATIONS = {"AM.A": (7.7, -80.7), "AM.B": (7.95, -80.4), "AM.C": (8.9, -79.8),
            "CU.D": (9.2, -79.8)}


def _triggers(grid, node, origin, stations, jitter=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for code in stations:
        j = grid.stations.index(code)
        t_on = origin + grid.tt[node, j] + rng.uniform(-jitter, jitter)
        rows.append({"station": code, "t_on": t_on, "peak_ms": 1e-5, "snr": 10.0})
    return rows


def test_event_found_at_the_right_node_with_noise_rejected():
    grid = build_grid(REGION, STATIONS)
    node = int(np.argmin(np.hypot(grid.lats - 7.5, grid.lons + 80.9)))
    origin = 1_000_000.0
    rows = _triggers(grid, node, origin, list(STATIONS))
    # lone noise triggers at single stations, far in time from anything else
    rows += [{"station": "AM.A", "t_on": origin + 500, "peak_ms": 1e-6, "snr": 4.0},
             {"station": "CU.D", "t_on": origin + 1500, "peak_ms": 1e-6, "snr": 4.0}]
    trig = pd.DataFrame(rows)
    events, picks = associate(trig, grid, Association(min_stations=3, tolerance_s=1.5))
    assert len(events) == 1
    ev = events.iloc[0]
    assert ev.n_stations == 4
    assert abs(ev.origin - origin) < 1.5
    assert abs(ev.latitude - 7.5) <= 0.3 and abs(ev.longitude + 80.9) <= 0.3
    assert set(picks.station) == set(STATIONS)


def test_min_stations_and_dedupe():
    grid = build_grid(REGION, STATIONS)
    node = 10
    rows = _triggers(grid, node, 2_000_000.0, ["AM.A", "AM.B"])
    trig = pd.DataFrame(rows)
    ev2, _ = associate(trig, grid, Association(min_stations=2))
    ev3, _ = associate(trig, grid, Association(min_stations=3))
    assert len(ev2) == 1  # both anchors build the same event, dedupe keeps one
    assert len(ev3) == 0


def test_empty_input():
    grid = build_grid(REGION, STATIONS)
    events, picks = associate(pd.DataFrame(columns=["station", "t_on"]), grid, Association())
    assert events.empty and picks.empty

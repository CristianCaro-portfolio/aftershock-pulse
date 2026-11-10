"""End to end from the shipped triggers: the README numbers must reproduce exactly."""

import json

import pandas as pd
import pytest

from aftershock_pulse.catalog import load_catalog
from aftershock_pulse.pipeline import analyze


@pytest.fixture(scope="module")
def result(cfg, root):
    triggers = pd.read_csv(root / "data/derived/triggers.csv.gz")
    stations = json.loads((root / "data/derived/stations.json").read_text())
    catalog = load_catalog(root / "data/reference/usgs_comcat.csv")
    return analyze(cfg, triggers, stations, catalog)


def _close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, float | int) and not isinstance(a, bool):
        return a == pytest.approx(b, rel=2e-3, abs=2e-3)
    return a == b


def test_reproduces_committed_metrics(result, root):
    committed = json.loads((root / "reports/metrics.json").read_text())
    m = json.loads(json.dumps(result["metrics"], default=float))
    for key in ("n_events", "catalog_recall_overall", "background_events_per_day",
                "completeness", "magnitude", "forecast_mean_log_score"):
        assert _close(m[key], committed[key]), key


def test_sanity(result):
    m = result["metrics"]
    assert m["n_events"] > 10 * m["catalog_events_after_mainshock"]
    assert m["catalog_recall_overall"] > 0.7
    events = result["events"]
    assert (events.n_stations >= 3).all()
    assert events.origin.is_monotonic_increasing
    bt = result["backtest"]
    assert set(bt.source) == {"generic", "usgs", "hybrid"}
    assert (bt.lo95 <= bt.hi95).all()

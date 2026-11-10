import numpy as np
from obspy import UTCDateTime, read, read_inventory

from aftershock_pulse.catalog import load_catalog
from aftershock_pulse.config import Detection
from aftershock_pulse.detect import characteristic_function, detect_all, finite_segments
from aftershock_pulse.evaluate import match_catalog
from aftershock_pulse.ingest import preprocess
from aftershock_pulse.pipeline import to_epoch
from aftershock_pulse.store import WaveStore


def test_finite_segments():
    mask = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert finite_segments(mask) == [(1, 3), (5, 8)]


def test_bursts_trigger_and_warmup_is_ignored(tmp_path):
    rate = 25.0
    rng = np.random.default_rng(1)
    x = rng.normal(scale=1.0, size=int(1800 * rate)).astype("float32")
    for t in (300, 900, 1500):  # three synthetic events
        i = int(t * rate)
        x[i : i + int(5 * rate)] += rng.normal(scale=25.0, size=int(5 * rate))
    x[int(1000 * rate) : int(1060 * rate)] = np.nan  # a gap
    t0 = UTCDateTime("2026-10-01")
    store = WaveStore(tmp_path / "w.zarr", t0, t0 + 1800, rate)
    store.write("XX.A", t0, x)
    det = Detection(sta_s=1.0, lta_s=20.0, trigger_on=4.0, trigger_off=1.5, min_duration_s=1.0)
    trig = detect_all(store, det)
    onsets = trig.t_on.to_numpy() - t0.timestamp
    for t in (300, 900, 1500):
        assert np.any(np.abs(onsets - t) < 1.0)
    assert len(trig) == 3
    assert (trig.snr > 5).all()


def test_cft_is_nan_during_lta_warmup():
    det = Detection(sta_s=1.0, lta_s=20.0)
    cft = characteristic_function(np.ones(5000, dtype="float32"), 25.0, det)
    assert np.isnan(cft[: int(20 * 25)]).all()


def test_real_bcip_sample_finds_catalog_aftershocks(tmp_path, cfg, root):
    """Two hours of CU.BCIP (public domain) on 10 October 2026, seven USGS aftershocks."""
    st = read(str(root / "data/sample/CU.BCIP.00.BHZ.2026-10-10T00.mseed"))
    inv = read_inventory(str(root / "data/sample/CU.BCIP.xml"))
    t0, t1 = UTCDateTime("2026-10-10T00:00:00"), UTCDateTime("2026-10-10T02:00:00")
    store = WaveStore(tmp_path / "w.zarr", t0, t1, cfg.processing.target_rate_hz)
    for tr in preprocess(st, inv, cfg.processing):
        tr.trim(t0, t1)
        store.write("CU.BCIP", tr.stats.starttime, tr.data)
    trig = detect_all(store, cfg.detection)
    cat = load_catalog(root / "data/reference/usgs_comcat.csv")
    cat = cat[(cat.time >= "2026-10-10T00:01:00Z") & (cat.time < "2026-10-10T01:59:00Z")]
    epoch = to_epoch(cat.time)
    # BCIP is about 200 km away: P arrives roughly 30 s after the origin
    matched = match_catalog(trig.t_on.to_numpy() - 30.0, epoch, tol_s=8.0)
    assert len(cat) >= 5
    assert (matched >= 0).mean() >= 0.8

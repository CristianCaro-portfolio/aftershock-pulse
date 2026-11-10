import numpy as np
import pytest
from obspy import UTCDateTime

from aftershock_pulse.store import Manifest, WaveStore

T0 = UTCDateTime("2026-10-01T00:00:00")


def test_roundtrip_and_gaps_read_as_nan(tmp_path):
    store = WaveStore(tmp_path / "w.zarr", T0, T0 + 3 * 3600, 10.0)
    store.write("XX.A", T0 + 100, np.arange(50, dtype="float32"))
    out = store.read("XX.A", T0 + 95, T0 + 110)
    assert np.isnan(out[:50]).all()
    np.testing.assert_array_equal(out[50:100], np.arange(50))
    assert np.isnan(out[100:]).all()


def test_writes_are_clipped_to_the_window(tmp_path):
    store = WaveStore(tmp_path / "w.zarr", T0, T0 + 3600, 10.0)
    store.write("XX.A", T0 - 1, np.ones(30, dtype="float32"))
    out = store.read("XX.A", T0, T0 + 3)
    assert np.isfinite(out[:20]).all() and np.isnan(out[20:]).all()


def test_reopen_and_mismatch(tmp_path):
    path = tmp_path / "w.zarr"
    WaveStore(path, T0, T0 + 3600, 10.0).set_station_meta("XX.A", {"latitude": 1.0})
    again = WaveStore.open(path)
    assert again.stations() == ["XX.A"]
    assert again.station_meta("XX.A")["latitude"] == 1.0
    with pytest.raises(ValueError):
        WaveStore(path, T0, T0 + 3600, 20.0)


def test_manifest_resume(tmp_path):
    m = Manifest(tmp_path / "m.json")
    m.record("XX.A", T0, "ok", coverage=1.0)
    m.record("XX.A", T0 + 3600, "failed", error="502")
    m.save()
    again = Manifest(tmp_path / "m.json")
    assert again.done("XX.A", T0)
    assert not again.done("XX.A", T0 + 3600)

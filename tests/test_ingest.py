import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException

from aftershock_pulse.config import Processing
from aftershock_pulse.ingest import chunks, preprocess, with_retry


def test_chunks_cover_window():
    t0 = UTCDateTime("2026-10-01")
    out = list(chunks(t0, t0 + 13 * 3600, 6))
    assert out == [t0, t0 + 6 * 3600, t0 + 12 * 3600]


def test_retry_recovers_from_transient_errors():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("502 bad gateway")
        return "ok"

    assert with_retry(flaky, sleep=lambda s: None) == "ok"
    assert len(calls) == 3


def test_retry_gives_up_and_does_not_retry_no_data():
    with pytest.raises(ConnectionError):
        with_retry(lambda: (_ for _ in ()).throw(ConnectionError("x")), attempts=2,
                   sleep=lambda s: None)
    calls = []

    def nodata():
        calls.append(1)
        raise FDSNNoDataException("204")

    with pytest.raises(FDSNNoDataException):
        with_retry(nodata, sleep=lambda s: None)
    assert len(calls) == 1


def test_preprocess_keeps_gaps_and_aligns_to_grid():
    rng = np.random.default_rng(0)
    t0 = UTCDateTime("2026-10-01T00:00:00.013")
    a = Trace(rng.normal(size=100 * 120).astype("int32"), header={"sampling_rate": 100.0,
                                                                    "starttime": t0})
    b = Trace(rng.normal(size=100 * 120).astype("int32"), header={"sampling_rate": 100.0,
                                                                    "starttime": t0 + 200})
    out = preprocess(Stream([a, b]), None, Processing(target_rate_hz=25.0))
    assert len(out) == 2  # the 80 s gap is not interpolated over
    for tr in out:
        assert tr.stats.sampling_rate == 25.0
        frac = (tr.stats.starttime.timestamp * 25.0) % 1
        assert min(frac, 1 - frac) < 1e-6

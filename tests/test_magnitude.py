import numpy as np
import pandas as pd

from aftershock_pulse.magnitude import MagnitudeScale, cross_validated_mae


def _synthetic(n=150, seed=3):
    rng = np.random.default_rng(seed)
    consts = {"AM.A": 5.0, "AM.B": 5.6, "CU.C": 7.1}
    mags = rng.uniform(3.5, 5.8, size=n)
    rows = []
    for i, m in enumerate(mags):
        for code, c in consts.items():
            r = rng.uniform(10, 300)
            log_a = m - 1.3 * np.log10(r) - c + rng.normal(scale=0.1)
            rows.append({"origin": float(i), "station": code, "peak_ms": 10**log_a, "snr": 20.0,
                         "hypo_km": r})
    return pd.DataFrame(rows), pd.Series(mags, index=np.arange(n, dtype=float)), consts


def test_scale_recovers_the_law():
    amps, ref, consts = _synthetic()
    scale = MagnitudeScale().fit(amps, ref)
    assert abs(scale.alpha - 1.3) < 0.1
    assert abs(scale.constants["CU.C"] - scale.constants["AM.A"] - 2.1) < 0.1
    pred = scale.event_magnitudes(amps)
    assert np.mean(np.abs(pred - ref.loc[pred.index])) < 0.1


def test_low_snr_is_ignored_and_cv_is_out_of_fold():
    amps, ref, _ = _synthetic()
    amps.loc[amps.station == "AM.B", "snr"] = 1.0
    scale = MagnitudeScale().fit(amps, ref)
    pred = scale.event_magnitudes(amps, min_snr=3.0)
    assert len(pred) == len(ref)
    mae, oof = cross_validated_mae(amps, ref, folds=5)
    assert len(oof) == len(ref) and mae < 0.15

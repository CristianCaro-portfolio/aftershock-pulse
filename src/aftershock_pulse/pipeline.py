"""End to end stages: triggers from the store, then everything downstream from the triggers."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .associate import associate, build_grid
from .config import Config
from .detect import detect_all
from .evaluate import b_value, completeness_maxc, match_catalog, recall_by_magnitude
from .forecast import (
    GENERIC_CALIFORNIA,
    RJModel,
    completeness_after_mainshock,
    fit_productivity,
    fit_rj,
    poisson_log_score,
)
from .magnitude import MagnitudeScale, cross_validated_mae, event_amplitudes
from .quality import hourly_quality
from .store import WaveStore

log = logging.getLogger(__name__)

MATCH_TOL_S = 10.0
CALIBRATION_MAX_MAG = 6.0  # bigger events clip the citizen sensors and saturate mb
FIT_START_DAYS = 0.05
# In the first hours aftershocks overlap (about one every two minutes) and the single trigger
# per station design saturates, so the sensor catalog is only used for the decay after this.
SENSOR_FIT_START_DAYS = 0.5
SENSOR_MAG_MARGIN = 0.5  # fit sensor decay on events this far above the sensor Mc
FORECAST_HORIZON_DAYS = 7.0
FORECAST_FIT_ENDS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
FORECAST_MAG = 5.0
MIN_EVENTS_FOR_B = 50
GENERIC_B = 1.0


@dataclass
class Timer:
    seconds: dict[str, float] = field(default_factory=dict)

    def stage(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.seconds[name] = round(time.perf_counter() - self.t, 3)

        return _Ctx()


def station_coords(store: WaveStore) -> dict[str, dict]:
    out = {}
    for code in store.stations():
        meta = store.station_meta(code)
        out[code] = {k: meta[k] for k in ("latitude", "longitude", "provider", "channel",
                                           "redistributable") if k in meta}
    return out


def detect_stage(cfg: Config, store: WaveStore, timer: Timer | None = None):
    timer = timer or Timer()
    with timer.stage("quality"):
        qc = pd.concat([hourly_quality(store, c) for c in store.stations()], ignore_index=True)
    with timer.stage("detect"):
        triggers = detect_all(store, cfg.detection)
    return triggers, qc


def to_epoch(times: pd.Series) -> np.ndarray:
    """Seconds since 1970 for a tz aware datetime column, whatever its resolution."""
    return (times - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def analyze(cfg: Config, triggers: pd.DataFrame, stations: dict[str, dict],
            catalog: pd.DataFrame, timer: Timer | None = None) -> dict:
    timer = timer or Timer()
    ms = cfg.mainshock
    t_main = ms.time.timestamp
    cat = catalog.copy()
    cat["epoch"] = to_epoch(cat["time"])
    cat = cat[(cat.epoch >= cfg.window.start.timestamp) & (cat.epoch < cfg.window.end.timestamp)]
    cat = cat[cat.id != ms.event_id].reset_index(drop=True)  # the mainshock is not a target

    with timer.stage("associate"):
        coords = {c: (m["latitude"], m["longitude"]) for c, m in stations.items()}
        grid = build_grid(cfg.region, coords)
        events, picks = associate(triggers, grid, cfg.association)

    with timer.stage("magnitude"):
        amps = event_amplitudes(events, picks, triggers, grid, cfg.region.depth_km)
        matched = match_catalog(events.origin.to_numpy(), cat.epoch.to_numpy(), MATCH_TOL_S)
        cat["det_index"] = matched
        cal = cat[(matched >= 0) & (cat.mag < CALIBRATION_MAX_MAG)]
        ref = pd.Series(cal.mag.to_numpy(), index=events.origin.to_numpy()[cal.det_index])
        cv_mae, cv_pred = cross_validated_mae(amps, ref)
        scale = MagnitudeScale().fit(amps, ref)
        mags = scale.event_magnitudes(amps)
        events["mag"] = events.origin.map(mags)
        events["t_days"] = (events.origin - t_main) / 86400.0
        events["catalog_id"] = None
        events.loc[cat.det_index[cat.det_index >= 0].to_numpy(), "catalog_id"] = (
            cat.id[cat.det_index >= 0].to_numpy()
        )

    with timer.stage("evaluate"):
        after = cat[cat.epoch >= t_main]
        recall = recall_by_magnitude(after.mag.to_numpy(), after.det_index.to_numpy())
        pre = events[(events.origin < t_main - 3600)]
        pre_days = (t_main - 3600 - cfg.window.start.timestamp) / 86400
        post_ev = events[events.t_days >= FIT_START_DAYS]
        post_cat_t = (after.epoch.to_numpy() - t_main) / 86400
        post_cat = after[post_cat_t >= FIT_START_DAYS]
        mc_det = completeness_maxc(post_ev.mag.to_numpy())
        mc_cat = completeness_maxc(post_cat.mag.to_numpy())
        b_det = b_value(post_ev.mag.to_numpy(), mc_det)
        b_cat = b_value(post_cat.mag.to_numpy(), mc_cat)

    with timer.stage("forecast"):
        sensor_ok = post_ev.mag.notna()
        backtest = forecast_backtest(
            ms.magnitude,
            usgs=(post_cat_t[post_cat_t >= FIT_START_DAYS], post_cat.mag.to_numpy(), mc_cat),
            sensors=(post_ev.t_days.to_numpy()[sensor_ok], post_ev.mag.to_numpy()[sensor_ok],
                     mc_det + SENSOR_MAG_MARGIN),
            observed_t=(after.epoch.to_numpy() - t_main) / 86400,
            observed_m=after.mag.to_numpy(),
            window_end_days=(cfg.window.end.timestamp - t_main) / 86400,
        )

    n_recall = recall[recall.n_catalog > 0]
    metrics = {
        "n_triggers": int(len(triggers)),
        "triggers_per_station": triggers.station.value_counts().sort_index().to_dict(),
        "n_events": int(len(events)),
        "n_events_after_mainshock": int((events.t_days >= 0).sum()),
        "n_events_with_magnitude": int(events.mag.notna().sum()),
        "events_by_n_stations": events.n_stations.value_counts().sort_index().to_dict(),
        "catalog_events_after_mainshock": int(len(after)),
        "catalog_recall_overall": float(n_recall.n_detected.sum() / n_recall.n_catalog.sum()),
        "recall_by_magnitude": recall.to_dict(orient="records"),
        "background_events_per_day": round(len(pre) / pre_days, 2),
        "background_days": round(pre_days, 2),
        "magnitude": {
            "calibration_events": int(len(ref)),
            "cv_mae": round(cv_mae, 3),
            "constant_baseline_mae": round(float(np.mean(np.abs(ref - ref.median()))), 3),
            "cv_pearson_r": round(float(np.corrcoef(cv_pred.pred, cv_pred.ref)[0, 1]), 3),
            "cv_mae_m_ge_5": round(float(np.mean(np.abs(
                cv_pred.pred[cv_pred.ref >= 5] - cv_pred.ref[cv_pred.ref >= 5]))), 3),
            "beta": round(scale.beta, 3),
            "alpha": round(scale.alpha, 3),
            "station_constants": {k: round(v, 3) for k, v in scale.constants.items()},
        },
        "completeness": {
            "sensors_mc": mc_det, "sensors_b": round(b_det[0], 3),
            "sensors_b_se": round(b_det[1], 3), "sensors_n_above_mc": b_det[2],
            "usgs_mc": mc_cat, "usgs_b": round(b_cat[0], 3),
            "usgs_b_se": round(b_cat[1], 3), "usgs_n_above_mc": b_cat[2],
        },
        "forecast": backtest.to_dict(orient="records"),
        "forecast_mean_log_score": backtest.groupby("source").log_score.mean().round(3).to_dict(),
        "forecast_inside_95": backtest.groupby("source").inside_95.mean().round(3).to_dict(),
        "stage_seconds": timer.seconds,
    }
    return {"metrics": metrics, "events": events, "picks": picks, "amplitudes": amps,
            "catalog": cat, "cv_predictions": cv_pred, "backtest": backtest,
            "scale": scale, "grid": grid}


def _b_for(t: np.ndarray, m: np.ndarray, mc: float, mainshock_mag: float, t_fit: float):
    """b from the complete events seen so far, the generic value until there are enough."""
    seen = (t < t_fit) & (m >= completeness_after_mainshock(t, mainshock_mag, mc))
    if seen.sum() >= MIN_EVENTS_FOR_B:
        return b_value(m[seen], mc)[0]
    return GENERIC_B


def forecast_backtest(mainshock_mag: float, usgs: tuple, sensors: tuple | None,
                      observed_t: np.ndarray, observed_m: np.ndarray,
                      window_end_days: float) -> pd.DataFrame:
    """Fit with the data available at T, forecast M >= FORECAST_MAG in [T, T + horizon] and
    score against what USGS recorded.

    generic: Reasenberg and Jones California parameters, no data used.
    usgs: a, p and c fitted on ComCat above its Mc, with Mc(t) after the mainshock.
    hybrid: decay (p, c) from the sensor catalog, productivity a from ComCat. The sensor catalog
        is much denser but its magnitude scale is not mb, so it only provides the timing.
    """
    rows = []
    generic = RJModel(mainshock_mag=mainshock_mag, **GENERIC_CALIFORNIA)
    t_u, m_u, mc_u = usgs
    for t_fit in FORECAST_FIT_ENDS:
        t2 = min(t_fit + FORECAST_HORIZON_DAYS, window_end_days)
        if t2 <= t_fit:
            continue
        obs = int(((observed_t >= t_fit) & (observed_t < t2) & (observed_m >= FORECAST_MAG)).sum())
        models = {"generic": generic}
        b_u = _b_for(t_u, m_u, mc_u, mainshock_mag, t_fit)
        try:
            models["usgs"] = fit_rj(t_u, m_u, mc_u, b_u, mainshock_mag, FIT_START_DAYS, t_fit)
        except ValueError as exc:
            log.info("no usgs fit at T=%s: %s", t_fit, exc)
        if sensors is not None and t_fit > SENSOR_FIT_START_DAYS:
            t_s, m_s, mmin_s = sensors
            try:
                decay = fit_rj(t_s, m_s, mmin_s, GENERIC_B, mainshock_mag,
                               SENSOR_FIT_START_DAYS, t_fit, time_dependent_mc=False)
                models["hybrid"] = fit_productivity(
                    t_u, m_u, mc_u, b_u, mainshock_mag, FIT_START_DAYS, t_fit, decay.p, decay.c
                )
                models["hybrid"].n_fit = decay.n_fit
            except ValueError as exc:
                log.info("no hybrid fit at T=%s: %s", t_fit, exc)
        for name, model in models.items():
            mu, lo, hi = model.interval(FORECAST_MAG, t_fit, t2)
            rows.append(
                {
                    "fit_end_days": t_fit, "horizon_end_days": round(t2, 3), "source": name,
                    "n_fit": model.n_fit, "a": round(model.a, 3), "b": round(model.b, 3),
                    "p": round(model.p, 3), "c": round(model.c, 4),
                    "expected": round(mu, 2), "lo95": lo, "hi95": hi, "observed": obs,
                    "inside_95": bool(lo <= obs <= hi),
                    "log_score": round(poisson_log_score(mu, obs), 3),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, default=_json_default), encoding="utf-8"
    )
    ev = result["events"].copy()
    ev["origin_utc"] = pd.to_datetime(ev.origin, unit="s", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ").str[:-4] + "Z"
    ev = ev[["origin_utc", "t_days", "latitude", "longitude", "mag", "n_stations", "rms_s",
             "stations", "catalog_id"]]
    ev.round({"t_days": 5, "mag": 2, "rms_s": 3}).to_csv(out_dir / "events.csv", index=False)
    result["backtest"].to_csv(out_dir / "forecast_backtest.csv", index=False)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(type(obj))

"""Command line entry point: python -m aftershock_pulse <command>."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import tracking
from .catalog import fetch_usgs, load_catalog
from .config import Config, load_config
from .detect import detect_all
from .ingest import ingest
from .pipeline import Timer, analyze, detect_stage, station_coords, write_outputs
from .quality import summarize
from .report import magnitude_figure, results_figure
from .store import WaveStore

log = logging.getLogger("aftershock_pulse")

DEFAULT_CONFIG = "configs/panama_2026.toml"


def cmd_ingest(cfg: Config, args) -> None:
    stats = ingest(cfg, Path(args.work), stations=args.stations)
    print(json.dumps(stats, indent=2))


def cmd_catalog(cfg: Config, args) -> None:
    df = fetch_usgs(cfg)
    Path(args.catalog).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.catalog, index=False)
    print(f"{len(df)} events written to {args.catalog}")


def cmd_detect(cfg: Config, args) -> None:
    store = WaveStore.open(Path(args.work) / "waveforms.zarr")
    triggers, qc = detect_stage(cfg, store)
    derived = Path(args.derived)
    derived.mkdir(parents=True, exist_ok=True)
    out = triggers.round({"t_on": 2, "t_off": 2, "cft_peak": 3, "snr": 2})
    out["peak_ms"] = out.peak_ms.map(lambda v: float(f"{v:.5g}"))
    out.to_csv(derived / "triggers.csv.gz", index=False)
    qc.to_csv(derived / "station_qc.csv", index=False, float_format="%.6g")
    (derived / "stations.json").write_text(json.dumps(station_coords(store), indent=2))
    print(summarize(qc).to_string())
    print(f"{len(triggers)} triggers written to {derived}")


def load_derived(derived: Path):
    triggers = pd.read_csv(derived / "triggers.csv.gz")
    stations = json.loads((derived / "stations.json").read_text())
    return triggers, stations


def cmd_analyze(cfg: Config, args) -> dict:
    triggers, stations = load_derived(Path(args.derived))
    catalog = load_catalog(args.catalog)
    timer = Timer()
    result = analyze(cfg, triggers, stations, catalog, timer)
    reports = Path(args.reports)
    write_outputs(result, reports)
    figures = Path(args.figures)
    qc_path = Path(args.derived) / "station_qc.csv"
    qc = pd.read_csv(qc_path) if qc_path.exists() else None
    results_figure(result, cfg.mainshock.time.timestamp, figures / "results.png", qc)
    magnitude_figure(result["cv_predictions"], figures / "magnitude_calibration.png",
                     result["metrics"]["magnitude"]["cv_mae"])
    m = result["metrics"]
    with tracking.run(f"aftershock-pulse-{cfg.name}", "analyze") as mlf:
        tracking.log_flat(
            mlf,
            params={**_params(cfg)},
            metrics={
                "n_events": m["n_events"],
                "catalog_recall": m["catalog_recall_overall"],
                "background_per_day": m["background_events_per_day"],
                "sensors_mc": m["completeness"]["sensors_mc"],
                "magnitude_cv_mae": m["magnitude"]["cv_mae"],
            },
        )
        if mlf is not None:
            mlf.log_artifact(str(reports / "metrics.json"))
            mlf.log_artifact(str(figures / "results.png"))
    _print_summary(m)
    return result


def cmd_sweep(cfg: Config, args) -> None:
    """Detector threshold against association strictness, each combination an MLflow run."""
    store = WaveStore.open(Path(args.work) / "waveforms.zarr")
    stations = station_coords(store)
    catalog = load_catalog(args.catalog)
    cfts: dict = {}
    rows = []
    for on, min_sta in itertools.product(args.trigger_on, args.min_stations):
        c = replace(
            cfg,
            detection=replace(cfg.detection, trigger_on=on),
            association=replace(cfg.association, min_stations=min_sta),
        )
        triggers = detect_all(store, c.detection, cfts=cfts)
        m = analyze(c, triggers, stations, catalog)["metrics"]
        row = {
            "trigger_on": on, "min_stations": min_sta, "n_events": m["n_events"],
            "catalog_recall": round(m["catalog_recall_overall"], 3),
            "background_per_day": m["background_events_per_day"],
            "sensors_mc": m["completeness"]["sensors_mc"],
            "magnitude_cv_mae": m["magnitude"]["cv_mae"],
        }
        rows.append(row)
        with tracking.run(f"aftershock-pulse-{cfg.name}-sweep", f"on{on}_sta{min_sta}") as mlf:
            tracking.log_flat(mlf, params=_params(c), metrics={
                k: v for k, v in row.items() if k not in ("trigger_on", "min_stations")
            })
        log.info("%s", row)
    table = pd.DataFrame(rows)
    Path(args.reports).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(args.reports) / "sweep.csv", index=False)
    print(table.to_string(index=False))


def _params(cfg: Config) -> dict:
    return {
        "sta_s": cfg.detection.sta_s, "lta_s": cfg.detection.lta_s,
        "trigger_on": cfg.detection.trigger_on, "trigger_off": cfg.detection.trigger_off,
        "min_stations": cfg.association.min_stations,
        "tolerance_s": cfg.association.tolerance_s,
        "stations": ",".join(s.code for s in cfg.stations),
    }


def _print_summary(m: dict) -> None:
    print(f"events: {m['n_events']} ({m['n_events_after_mainshock']} after the mainshock)")
    print(f"USGS events recovered: {m['catalog_recall_overall']:.3f}")
    print(f"background detections per day before the mainshock: "
          f"{m['background_events_per_day']}")
    c = m["completeness"]
    print(f"Mc sensors {c['sensors_mc']} (b {c['sensors_b']}), Mc USGS {c['usgs_mc']} "
          f"(b {c['usgs_b']})")
    print(f"magnitude CV MAE: {m['magnitude']['cv_mae']}")
    fc = pd.DataFrame(m["forecast"])
    print(fc[["fit_end_days", "source", "n_fit", "expected", "lo95", "hi95", "observed",
              "inside_95", "log_score"]].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aftershock-pulse", description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--work", default="data/work", help="raw store, not committed")
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--catalog", default="data/reference/usgs_comcat.csv")
    p.add_argument("--reports", default="reports")
    p.add_argument("--figures", default="docs")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="download waveforms into the Zarr store")
    ing.add_argument("--stations", nargs="*")
    sub.add_parser("catalog", help="download the USGS reference catalog")
    sub.add_parser("detect", help="quality checks and STA/LTA triggers from the store")
    sub.add_parser("analyze", help="associate, magnitudes, evaluation and forecast")
    sub.add_parser("run", help="detect then analyze")
    sw = sub.add_parser("sweep", help="detector threshold sweep tracked in MLflow")
    sw.add_argument("--trigger-on", type=float, nargs="+", default=[3.0, 3.5, 4.0, 5.0])
    sw.add_argument("--min-stations", type=int, nargs="+", default=[2, 3])
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = load_config(args.config)
    handlers = {
        "ingest": cmd_ingest, "catalog": cmd_catalog, "detect": cmd_detect,
        "analyze": cmd_analyze, "sweep": cmd_sweep,
    }
    if args.command == "run":
        cmd_detect(cfg, args)
        cmd_analyze(cfg, args)
    else:
        handlers[args.command](cfg, args)


if __name__ == "__main__":
    main()

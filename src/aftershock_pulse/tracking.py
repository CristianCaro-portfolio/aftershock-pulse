"""Optional MLflow tracking. Runs log to a local SQLite file, and nothing breaks without MLflow."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def _mlflow():
    if os.environ.get("AFTERSHOCK_PULSE_TRACKING", "1") == "0":
        return None
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow
    except ImportError:
        return None
    return mlflow


@contextlib.contextmanager
def run(experiment: str, run_name: str, db_path: str | Path = "mlflow.db"):
    mlflow = _mlflow()
    if mlflow is None:
        yield None
        return
    # the file store is in maintenance mode since MLflow 3, SQLite is the local default now
    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{Path(db_path).resolve()}"
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        active = mlflow.start_run(run_name=run_name)
    except Exception as exc:  # tracking is a side channel, never a reason to fail a run
        log.warning("MLflow tracking disabled for this run: %s", exc)
        yield None
        return
    with active:
        yield mlflow
        log.info("mlflow run %s logged to %s", active.info.run_id, uri)


def log_flat(mlflow, params: dict | None = None, metrics: dict | None = None) -> None:
    if mlflow is None:
        return
    if params:
        mlflow.log_params({k: str(v) for k, v in params.items()})
    if metrics:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
        mlflow.log_metrics(clean)

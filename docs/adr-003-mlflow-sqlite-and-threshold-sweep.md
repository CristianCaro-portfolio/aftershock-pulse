# ADR 003: MLflow on SQLite for the threshold sweep

## Context

The two settings that decide the catalog are the STA/LTA trigger level and how many stations must
agree. They trade recall against false events, and the right choice has to be shown with numbers,
not picked by eye.

## Decision

`python -m aftershock_pulse sweep` runs every combination, each as an MLflow run with its
parameters and metrics (events, USGS recall, detections per day before the mainshock, Mc,
magnitude error), and writes the same table to reports/sweep.csv. The STA/LTA characteristic
function is computed once per station and reused across thresholds, so nine runs take about 90 s.

Tracking goes to a local SQLite file (`sqlite:///mlflow.db`). MLflow 3 put the file store
(`./mlruns`) in maintenance mode and refuses to start on it by default, so the database backend is
the default from the start. `MLFLOW_TRACKING_URI` points it elsewhere, and
`AFTERSHOCK_PULSE_TRACKING=0` turns it off. A tracking failure logs a warning and never stops a
run.

## Outcome

| on | stations | events | recall | per day before mainshock |
|---|---|---|---|---|
| 4.0 | 3 | 3,788 | 0.823 | 17.5 |
| 5.0 | 3 | 2,641 | 0.810 | 3.8 |
| 5.0 | 2 | 8,927 | 0.912 | 157.1 |

5.0 with three stations was chosen: 1.3 points of recall for a 4.6x cleaner background. Two
stations is never enough with four noisy sensors.

## Alternatives

Weights and Biases or a hosted MLflow server need accounts or a running service. A CSV alone loses
the run history and comparison UI. MLflow on SQLite is local, free and one command to browse.

"""Figures for the README: sequence timeline, magnitude-frequency and the forecast backtest."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SENSORS = "#1f6f8b"
USGS = "#d1495b"
GENERIC = "#8d8d8d"


def results_figure(result: dict, mainshock_epoch: float, path: Path,
                   qc: pd.DataFrame | None = None) -> None:
    ev = result["events"]
    cat = result["catalog"]
    bt = result["backtest"]
    m = result["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=130)

    # 1. hourly counts
    ax = axes[0]
    t_ev = (ev.origin.to_numpy() - mainshock_epoch) / 86400
    t_cat = (cat.epoch.to_numpy() - mainshock_epoch) / 86400
    bins = np.arange(np.floor(min(t_ev.min(), t_cat.min())), np.ceil(t_ev.max()) + 1 / 24, 1 / 24)
    ax.hist(t_ev, bins=bins, color=SENSORS, alpha=0.85, label=f"sensor events ({len(ev)})")
    ax.hist(t_cat, bins=bins, color=USGS, alpha=0.85, label=f"USGS ComCat ({len(cat)})")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    if qc is not None and not qc.empty:
        hours = pd.to_datetime(qc.hour, format="%Y-%m-%dT%H", utc=True)
        qc = qc.assign(t=(hours - pd.Timestamp(mainshock_epoch, unit="s", tz="UTC"))
                       .dt.total_seconds() / 86400, up=qc.coverage >= 0.5)
        up = qc.groupby("t").up.sum()
        down = up[up < qc.station.nunique()].index
        for i, t in enumerate(down):
            ax.axvspan(t, t + 1 / 24, color="#f0c987", alpha=0.35, lw=0,
                       label="a station offline" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("days since the M7.7 mainshock")
    ax.set_ylabel("events per hour")
    ax.set_title("What the sensors see, hour by hour")
    ax.legend(frameon=False, fontsize=9)

    # 2. magnitude frequency
    ax = axes[1]
    grid = np.arange(1.0, 7.0, 0.1)
    post_ev = ev[ev.t_days >= 0.05].mag.dropna().to_numpy()
    post_cat = cat[t_cat >= 0.05].mag.to_numpy()
    for mags, color, label, mc in (
        (post_ev, SENSORS, "sensor catalog", m["completeness"]["sensors_mc"]),
        (post_cat, USGS, "USGS ComCat", m["completeness"]["usgs_mc"]),
    ):
        n = [(mags >= g).sum() for g in grid]
        ax.semilogy(grid, np.maximum(n, 0.8), color=color, lw=2, label=f"{label}, Mc {mc:.1f}")
        ax.axvline(mc, color=color, lw=0.8, ls=":")
    ax.set_ylim(0.8, None)
    ax.set_xlabel("magnitude (calibrated to USGS mb)")
    ax.set_ylabel("events with magnitude >= M")
    ax.set_title("Magnitude of completeness")
    ax.legend(frameon=False, fontsize=9)

    # 3. forecast backtest
    ax = axes[2]
    offsets = {"generic": -0.18, "usgs": 0.0, "hybrid": 0.18}
    colors = {"generic": GENERIC, "usgs": USGS, "hybrid": SENSORS}
    labels = {"generic": "generic parameters", "usgs": "fit on USGS",
              "hybrid": "hybrid: sensor decay, USGS scale"}
    fit_ends = sorted(bt.fit_end_days.unique())
    xpos = {t: i for i, t in enumerate(fit_ends)}
    for source, sub in bt.groupby("source"):
        x = np.array([xpos[t] for t in sub.fit_end_days]) + offsets[source]
        mu = sub.expected.to_numpy()
        ax.errorbar(x, mu, yerr=[mu - sub.lo95, sub.hi95 - mu], fmt="o", color=colors[source],
                    capsize=3, label=labels[source])
    obs = bt.drop_duplicates("fit_end_days").sort_values("fit_end_days")
    ax.scatter([xpos[t] for t in obs.fit_end_days], obs.observed, marker="_", s=500,
               color="black", label="observed (USGS)", zorder=5)
    ax.set_xticks(range(len(fit_ends)), [f"{t:g} d" for t in fit_ends])
    ax.set_xlabel("data available when the forecast is made")
    ax.set_ylabel("M>=5 aftershocks in the next 7 days")
    ax.set_yscale("log")
    ax.set_title("Forecast vs what happened (95% Poisson)")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def magnitude_figure(cv_pred: pd.DataFrame, path: Path, mae: float) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=130)
    ax.scatter(cv_pred.ref, cv_pred.pred, s=14, color=SENSORS, alpha=0.8)
    lims = [min(cv_pred.ref.min(), cv_pred.pred.min()) - 0.2,
            max(cv_pred.ref.max(), cv_pred.pred.max()) + 0.2]
    ax.plot(lims, lims, color="black", lw=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("USGS magnitude")
    ax.set_ylabel("sensor magnitude (out of fold)")
    ax.set_title(f"Magnitude calibration, CV MAE {mae:.2f}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

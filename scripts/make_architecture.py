"""Draw docs/architecture.png. Run: python scripts/make_architecture.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

STAGES = [
    ("Sources", "Raspberry Shake FDSN\n3 citizen sensors\nEarthScope FDSN\nCU.BCIP (USGS)\nUSGS ComCat events", "#e8eef3"),
    ("Ingest", "ObsPy clients\n6 h chunks, retries\nresponse removed\n2-10 Hz, 25 Hz grid\nresumable manifest", "#d9e8f0"),
    ("Store", "Zarr v3\none array per station\n1 h chunks\ngaps read as NaN\nhourly QC", "#cfe3ec"),
    ("Detect", "recursive STA/LTA\nper station\npeak velocity, SNR\nwarm up after gaps\nis discarded", "#c4dde8"),
    ("Associate", "TauP iasp91 grid\n24 x 15 nodes\nmoveout must fit\n>= N stations\nrough location", "#f6e3c6"),
    ("Magnitude", "M = b log A + a log R\n+ station term\ncalibrated on USGS\nout of fold check", "#f3d9b1"),
    ("Evaluate", "recall vs USGS\nbackground rate\nMc and b value", "#efc9a0"),
    ("Forecast", "Reasenberg-Jones\nMc(t) after mainshock\nbacktest M>=5\nnext 7 days", "#e9b8a0"),
]


def main(path: Path = Path("docs/architecture.png")) -> None:
    fig, ax = plt.subplots(figsize=(17, 4.4), dpi=130)
    ax.set_xlim(0, len(STAGES) * 2.1)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    for i, (title, body, color) in enumerate(STAGES):
        x = 0.15 + i * 2.1
        ax.add_patch(FancyBboxPatch((x, 0.5), 1.8, 3.0, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=color, ec="#3d5a6c", lw=1.2))
        ax.text(x + 0.9, 3.2, title, ha="center", va="center", fontsize=12, weight="bold",
                color="#1d3340")
        ax.text(x + 0.9, 1.85, body, ha="center", va="center", fontsize=8.6, color="#1d3340",
                linespacing=1.35)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.86, 2.0), (x + 2.19, 2.0), arrowstyle="-|>",
                                         mutation_scale=14, color="#3d5a6c", lw=1.4))
    ax.text(0.15, 4.05, "aftershock-pulse: citizen seismometers and open networks to aftershock "
            "catalog and forecast", fontsize=13, weight="bold", color="#1d3340")
    ax.text(0.15, 0.15, "MLflow tracks the detector threshold sweep. Raw waveforms stay local; "
            "the repo ships triggers and results so every number reproduces without downloads.",
            fontsize=9, color="#3d5a6c")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

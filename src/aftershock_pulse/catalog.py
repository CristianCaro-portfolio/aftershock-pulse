"""Reference catalog from the USGS ComCat FDSN event service, plus geometry helpers."""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

USGS_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def fetch_usgs(cfg: Config, timeout: float = 60) -> pd.DataFrame:
    params = {
        "format": "csv",
        "starttime": cfg.window.start.isoformat(),
        "endtime": cfg.window.end.isoformat(),
        "latitude": cfg.mainshock.latitude,
        "longitude": cfg.mainshock.longitude,
        "maxradiuskm": cfg.region.catalog_radius_km,
        "orderby": "time-asc",
    }
    url = f"{USGS_EVENT_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
        body = resp.read().decode("utf-8")
    return normalize_usgs(pd.read_csv(io.StringIO(body)))


def normalize_usgs(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["time", "latitude", "longitude", "depth", "mag", "magType", "id"]
    df = df[keep].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df.sort_values("time").reset_index(drop=True)


def load_catalog(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df

"""Typed project configuration loaded from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from obspy import UTCDateTime


@dataclass(frozen=True)
class Mainshock:
    event_id: str
    time: UTCDateTime
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float


@dataclass(frozen=True)
class Window:
    start: UTCDateTime
    end: UTCDateTime
    chunk_hours: int = 6


@dataclass(frozen=True)
class Region:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    step_deg: float = 0.1
    depth_km: float = 10.0
    catalog_radius_km: float = 150.0


@dataclass(frozen=True)
class Processing:
    target_rate_hz: float = 25.0
    freqmin_hz: float = 2.0
    freqmax_hz: float = 10.0


@dataclass(frozen=True)
class Detection:
    sta_s: float = 1.0
    lta_s: float = 20.0
    trigger_on: float = 4.0
    trigger_off: float = 1.5
    min_duration_s: float = 1.0


@dataclass(frozen=True)
class Association:
    min_stations: int = 2
    tolerance_s: float = 2.0
    min_separation_s: float = 6.0


@dataclass(frozen=True)
class StationSpec:
    provider: str
    network: str
    station: str
    channel: str
    location: str = "*"
    redistributable: bool = False

    @property
    def code(self) -> str:
        return f"{self.network}.{self.station}"


@dataclass(frozen=True)
class Config:
    name: str
    mainshock: Mainshock
    window: Window
    region: Region
    processing: Processing
    detection: Detection
    association: Association
    stations: tuple[StationSpec, ...] = field(default_factory=tuple)

    def station(self, code: str) -> StationSpec:
        for spec in self.stations:
            if spec.code == code:
                return spec
        raise KeyError(code)


def load_config(path: str | Path) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    ms = raw["mainshock"]
    win = raw["window"]
    return Config(
        name=raw["name"],
        mainshock=Mainshock(
            event_id=ms["event_id"],
            time=UTCDateTime(ms["time"]),
            latitude=ms["latitude"],
            longitude=ms["longitude"],
            depth_km=ms["depth_km"],
            magnitude=ms["magnitude"],
        ),
        window=Window(
            start=UTCDateTime(win["start"]),
            end=UTCDateTime(win["end"]),
            chunk_hours=win.get("chunk_hours", 6),
        ),
        region=Region(**raw["region"]),
        processing=Processing(**raw.get("processing", {})),
        detection=Detection(**raw.get("detection", {})),
        association=Association(**raw.get("association", {})),
        stations=tuple(StationSpec(**s) for s in raw["stations"]),
    )

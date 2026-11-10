from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg(root):
    from aftershock_pulse.config import load_config

    return load_config(root / "configs" / "panama_2026.toml")

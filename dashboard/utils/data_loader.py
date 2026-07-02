"""Cached loaders for the processed CSVs used by pages 5-7.

Paths resolve relative to the repo root (this file's grandparent's parent),
so loaders work regardless of the current working directory.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _ROOT / "data" / "processed"


def _read(name, **kw):
    path = _PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Place the 13 processed CSVs in data/processed/ "
            "(they are the data team's artifact)."
        )
    return pd.read_csv(path, **kw)


@lru_cache(maxsize=None)
def country_summary():
    return _read("country_summary.csv")


@lru_cache(maxsize=None)
def offices():
    return _read("offices.csv")


@lru_cache(maxsize=None)
def investor_summary():
    return _read("investor_summary.csv")


@lru_cache(maxsize=None)
def investor_sector_matrix():
    return _read("investor_sector_matrix.csv")


@lru_cache(maxsize=None)
def sector_summary():
    return _read("sector_summary.csv")


@lru_cache(maxsize=None)
def companies():
    # 462k rows, mixed-type date columns -> silence the dtype warning.
    return _read("companies.csv", low_memory=False)


@lru_cache(maxsize=None)
def funding_timeline():
    return _read("funding_timeline.csv")

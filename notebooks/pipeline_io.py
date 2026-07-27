"""
CalRetail — build-time storage for the data pipeline.

The three pipeline stages used to hand work to each other as CSV files on
disk. They now hand it over as SQLite tables:

    generate_data.py  --write-->  raw.db        (throwaway, gitignored)
    clean_data.py     --read--->  raw.db
                      --write-->  calretail.db  (the shipped database)
    feature_engineering.py  --read/write-->  calretail.db

Only ``calretail.db`` is committed. ``raw.db`` is an intermediate and is
deleted once a build finishes, so a checkout never carries two copies of the
same data.

Writes go through :func:`save_raw` / :func:`save_processed` rather than
``to_sql`` directly so that column dtypes are normalised in one place —
otherwise a column that happens to be all-NULL in one build lands as TEXT and
compares badly against a REAL in the next.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BUILD_DIR = DATA_DIR / "build"

RAW_DB = BUILD_DIR / "raw.db"

# CALRETAIL_DB redirects the build, which is how you produce a full-scale
# database without overwriting the committed demo one. The runtime honours the
# same variable via settings.DATABASE_PATH, so the app can then be pointed at
# whichever build you want.
MAIN_DB = Path(os.environ.get("CALRETAIL_DB") or (DATA_DIR / "calretail.db"))


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    # These are build-time only. Durability does not matter for a file we can
    # regenerate from a seed, and turning it off makes bulk loading far faster.
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = -128000")
    return conn


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dtypes so SQLite stores compact, correctly-typed values.

    Two things matter here: datetimes must become ISO-8601 TEXT (SQLite has no
    date type, and pandas would otherwise write them as nanosecond ints that
    read back as garbage), and float columns holding only whole numbers become
    nullable Int64 so they store as INTEGER instead of 8-byte REAL.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            df[col] = s.dt.strftime("%Y-%m-%d %H:%M:%S").where(s.notna(), None)
        elif pd.api.types.is_bool_dtype(s):
            df[col] = s.astype("Int64")
        elif pd.api.types.is_float_dtype(s):
            non_null = s.dropna()
            if len(non_null) and (non_null % 1 == 0).all():
                df[col] = s.astype("Int64")
    return df


def _save(df: pd.DataFrame, name: str, path: Path) -> None:
    conn = _connect(path)
    try:
        # executemany rather than method="multi": a multi-row INSERT binds one
        # variable per cell and SQLite caps a statement at 32766 of them, which
        # a wide table blows through in well under one chunk.
        _normalise(df).to_sql(name, conn, index=False, if_exists="replace",
                              chunksize=20_000)
        conn.commit()
    finally:
        conn.close()


def _load(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found — run the pipeline in order "
            f"(python -m notebooks.build_db)"
        )
    conn = _connect(path)
    try:
        return pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
    finally:
        conn.close()


# ── stage 1: generate → raw ──────────────────────────────────────────────────

def save_raw(df: pd.DataFrame, name: str) -> None:
    _save(df, name, RAW_DB)


def load_raw(name: str) -> pd.DataFrame:
    return _load(name, RAW_DB)


# ── stage 2 & 3: clean / features → main ─────────────────────────────────────

def save_processed(df: pd.DataFrame, name: str) -> None:
    _save(df, name, MAIN_DB)


def load_processed(name: str) -> pd.DataFrame:
    return _load(name, MAIN_DB)


def reset_raw() -> None:
    """Start a build from a clean slate."""
    if RAW_DB.exists():
        RAW_DB.unlink()


def reset_main() -> None:
    if MAIN_DB.exists():
        MAIN_DB.unlink()


def drop_raw() -> None:
    """Remove the intermediate once the shipped database is built."""
    if RAW_DB.exists():
        RAW_DB.unlink()
    if BUILD_DIR.exists() and not any(BUILD_DIR.iterdir()):
        BUILD_DIR.rmdir()

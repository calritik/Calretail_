"""
CalRetail — SQLite data access layer.

The database at ``settings.DATABASE_PATH`` is the single source of truth for
every dataset. It is produced by ``python -m notebooks.build_db``.

Two access styles are offered, and the distinction matters for memory:

* :func:`load_df` returns a whole table as a DataFrame and memoises it, which
  is what the capability modules want — they do wide pandas work over the full
  frame and build their state once per process.
* :func:`query` and :func:`read_table` push filtering down into SQLite so a
  route that needs one customer's rows does not materialise a whole table.
  The indexes created by the build script exist to serve exactly these.

Connections are per-thread (uvicorn and Dash both hit this from worker
threads) and opened read-only, so a request can never mutate the demo data.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backend.config.settings import settings

DB_PATH = Path(settings.DATABASE_PATH)

# Columns stored as ISO-8601 TEXT that callers expect back as datetimes.
# SQLite has no native date type, so this is applied on read.
DATE_COLS = {
    "transaction_date", "timestamp", "shipped_date", "delivered_date",
    "start_date", "end_date", "review_date", "created_at", "updated_at",
    "order_date", "delivery_date", "estimated_delivery", "effective_date",
    "date", "open_date", "last_purchase_date", "signup_date", "resolved_at",
    "return_date", "movement_date", "session_start", "session_end",
    "last_restock_date", "expected_delivery",
}

_MISSING_DB = (
    "CalRetail database not found at {path}.\n"
    "Build it first:\n"
    "    python -m notebooks.build_db            # demo scale (~35 MB)\n"
    "    python -m notebooks.build_db --scale full   # full fidelity"
)


class _Connections(threading.local):
    """One SQLite connection per thread; sqlite3 objects are not shareable."""

    conn: sqlite3.Connection | None = None


_local = _Connections()


def database_exists() -> bool:
    return DB_PATH.exists() and DB_PATH.stat().st_size > 0


def connect() -> sqlite3.Connection:
    """Return this thread's read-only connection, opening it if needed."""
    if _local.conn is not None:
        return _local.conn

    if not database_exists():
        raise FileNotFoundError(_MISSING_DB.format(path=DB_PATH))

    # mode=ro keeps request handlers from writing; immutable would be faster
    # still but would break the build script re-pointing at the same file.
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA cache_size = -64000")   # 64 MB page cache
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 268435456")  # 256 MB
    _local.conn = conn
    return conn


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in DATE_COLS:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# Object-dtype strings are the single largest cost in this process. Every
# "C00001", every product name and city is an individual Python str with ~49
# bytes of header, so a table costs 3-5x more in pandas than the bytes SQLite
# stores. Measured across the seven largest tables: 119.7 MB as objects,
# 47.3 MB with Arrow-backed strings.
#
# Arrow rather than categorical, which saves marginally more (68% against 60%)
# but is not transparent: groupby on a categorical column keeps unused
# categories by default, so filtering then grouping silently gains empty rows.
# Arrow strings behave exactly like object strings — same comparisons, same
# .str accessor, same groupby — which is what makes this safe to apply to all
# sixteen capabilities without touching any of them.
#
# Integer downcasting rides along: most of these columns are small counts
# stored as int64.
#
# Floats are deliberately left at float64. Narrowing them changed the route
# optimiser's answer — latitude and longitude lose significant digits at
# float32, the haversine distances shift, and the 2-opt heuristic follows a
# different path to a different total (3,361 km became 3,637 km). Floats were
# only a small part of the saving; correctness is not worth trading for it.
_COMPACT = os.environ.get("CALRETAIL_COMPACT_DTYPES", "1") == "1"

try:
    import pyarrow  # noqa: F401
    _HAVE_ARROW = True
except ImportError:                       # pragma: no cover - depends on install
    _HAVE_ARROW = False


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink a freshly-read frame in place. Values and semantics unchanged."""
    if not _COMPACT:
        return df

    for col in df.columns:
        kind = df[col].dtype.kind
        if kind == "O" and _HAVE_ARROW:
            try:
                df[col] = df[col].astype("string[pyarrow]")
            except Exception:
                pass          # mixed-type column; leave it as objects
        elif kind == "i":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def query(sql: str, params: Sequence[Any] | dict[str, Any] = (),
          parse_dates: bool = True) -> pd.DataFrame:
    """Run arbitrary SQL and return a DataFrame, parsing date columns by default."""
    df = pd.read_sql_query(sql, connect(), params=params)
    df = _parse_dates(df) if parse_dates else df
    return _compact(df)


def read_table(
    name: str,
    columns: Sequence[str] | None = None,
    where: str | None = None,
    params: Sequence[Any] = (),
    order_by: str | None = None,
    limit: int | None = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Read a table with optional pushdown.

    Prefer this over :func:`load_df` whenever the caller only needs a slice —
    it lets SQLite use its indexes instead of scanning the table into pandas.
    """
    if not table_exists(name):
        raise KeyError(f"Table {name!r} not found in {DB_PATH.name}")

    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    sql = f'SELECT {cols} FROM "{name}"'
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return query(sql, params, parse_dates=parse_dates)


# How many whole tables stay memoised. Unbounded, this is a slow leak: each
# capability pulls a different set, nothing is ever released, and the process
# grows by roughly 200 MB as a visitor works through the console — enough to
# get OOM-killed on a 512 MiB host.
#
# Bounding it only releases tables no warm notebook still references, which is
# exactly the intent: a table is re-read from SQLite in milliseconds, so the
# cache is a latency optimisation, not a correctness one.
_TABLE_CACHE = max(4, int(os.environ.get("CALRETAIL_TABLE_CACHE", "8")))


@lru_cache(maxsize=_TABLE_CACHE)
def load_df(name: str) -> pd.DataFrame:
    """
    Load a full table and memoise it.

    Mirrors the old CSV loader's contract, including returning the *same*
    object to every caller while it stays cached — services were written
    against that and some of them assign derived columns onto it.
    """
    return read_table(name)


@lru_cache(maxsize=_TABLE_CACHE)
def load_table(name: str) -> pd.DataFrame:
    """
    Full table with date columns left as ISO-8601 strings.

    This is what the capability notebooks use. They were written against
    ``pd.read_csv``, which handed them date columns as text and left parsing to
    the notebook — several then feed those values straight into ``json.dumps``.
    Returning parsed Timestamps here instead would break them in ways the
    notebook loader swallows silently, so the original contract is kept.
    """
    return read_table(name, parse_dates=False)


@lru_cache(maxsize=1)
def table_names() -> tuple[str, ...]:
    rows = connect().execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return tuple(r[0] for r in rows)


def table_exists(name: str) -> bool:
    return name in table_names()


def row_count(name: str) -> int:
    return connect().execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]


def summary() -> dict[str, int]:
    """Table -> row count, for startup logging and the /health payload."""
    return {t: row_count(t) for t in table_names()}


def clear_cache() -> None:
    """Drop memoised frames — used by tests and after a rebuild."""
    load_df.cache_clear()
    table_names.cache_clear()

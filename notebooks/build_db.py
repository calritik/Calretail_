"""
CalRetail — database builder.

Runs the whole pipeline and leaves a single indexed SQLite file at
``data/calretail.db``:

    generate_data  →  raw.db  →  clean_data  →  calretail.db  →  feature_engineering

Usage
-----
    python -m notebooks.build_db                  # demo scale, the committed default
    python -m notebooks.build_db --scale full     # full fidelity (~300 MB)
    python -m notebooks.build_db --scale 0.5      # anything in between
    python -m notebooks.build_db --keep-raw       # keep the intermediate for debugging

Scale applies to event-log tables only. Customers, products, stores and
inventory are always generated at full size, so a demo build still presents a
complete catalogue and customer base — only the behavioural history behind
them is thinner.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# Scale presets. "demo" is tuned to land the finished database comfortably
# under GitHub's 100 MB per-file limit so it can be committed and deployed to a
# Hugging Face Space with no LFS and no build step.
SCALES = {"demo": 0.12, "full": 1.0}
DEFAULT_SCALE = "demo"


def _human(n_bytes: int) -> str:
    return f"{n_bytes / 1_048_576:.1f} MB"


def _rule(title: str = "") -> None:
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


# Columns worth an index: the join keys the application actually filters on
# (see the pushdown helpers in backend/utils/data_loader.py). Indexing every
# *_id column instead costs ~30 MB on a demo build and buys nothing — most
# tables are read whole into pandas by the capability notebooks, and a scan of
# a few thousand rows is free.
INDEXED_KEYS = {
    "customer_id", "product_id", "store_id", "warehouse_id",
    "order_id", "supplier_id", "promo_id", "ticket_id",
}

# Below this a sequential scan beats maintaining an index, and SQLite's planner
# will usually choose the scan anyway.
MIN_ROWS_FOR_INDEX = 10_000

# Date indexes earn their space only on the genuinely large event logs.
MIN_ROWS_FOR_DATE_INDEX = 100_000


def build_indexes(db_path: Path) -> int:
    """
    Index the finished database.

    An index is created where the column is a real lookup key on a table big
    enough for it to matter. Where the key is unique it is created UNIQUE,
    which gives SQLite's planner the same information a declared primary key
    would without rebuilding all 31 tables' DDL.
    """
    conn = sqlite3.connect(db_path)
    made = 0
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]

        for table in tables:
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if total < MIN_ROWS_FOR_INDEX:
                continue

            targets = [c for c in cols if c in INDEXED_KEYS]
            # The table's own identifier (transactions.transaction_id and the
            # like) is a lookup key even though it is not in the shared set.
            singular = table.rstrip("s").replace("feature_", "")
            targets += [c for c in cols
                        if c not in targets and c.endswith("_id")
                        and c.startswith(singular[:6])]
            if total >= MIN_ROWS_FOR_DATE_INDEX:
                targets += [c for c in cols if c not in targets
                            and ("date" in c.lower() or c.lower() == "timestamp")]

            for col in targets:
                distinct = conn.execute(
                    f'SELECT COUNT(DISTINCT "{col}") FROM "{table}"').fetchone()[0]
                nulls = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IS NULL').fetchone()[0]
                unique = distinct == total and nulls == 0
                kind = "UNIQUE INDEX" if unique else "INDEX"
                conn.execute(f'CREATE {kind} IF NOT EXISTS "ix_{table}_{col}" '
                             f'ON "{table}" ("{col}")')
                made += 1

        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()
    return made


def compact(db_path: Path) -> None:
    """VACUUM reclaims the space left by the pipeline's repeated table replaces."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()


def report(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        n_idx = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
            "AND name NOT LIKE 'sqlite_%'").fetchone()[0]
        print(f"{'table':<34}{'rows':>12}")
        print("-" * 46)
        grand = 0
        for t in tables:
            n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            grand += n
            print(f"{t:<34}{n:>12,}")
        print("-" * 46)
        print(f"{'TOTAL':<34}{grand:>12,}")
        print(f"\n  {len(tables)} tables · {n_idx} indexes · {_human(db_path.stat().st_size)}")
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the CalRetail SQLite database.")
    ap.add_argument("--scale", default=DEFAULT_SCALE,
                    help="demo | full | a float such as 0.5 (default: demo)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep data/build/raw.db after the build")
    args = ap.parse_args()

    try:
        scale = SCALES.get(args.scale, None) or float(args.scale)
    except ValueError:
        ap.error(f"--scale must be one of {sorted(SCALES)} or a number, got {args.scale!r}")

    # generate_data reads this at import time, so it must be set before the
    # pipeline modules are imported below.
    os.environ["CALRETAIL_SCALE"] = str(scale)

    from notebooks import pipeline_io

    _rule(f"CalRetail — building database (scale: {args.scale} = {scale:g})")
    started = time.time()

    try:
        pipeline_io.reset_raw()
        pipeline_io.reset_main()
    except PermissionError:
        # Windows will not unlink a file another process has open, and the
        # usual cause is the app still running against the database.
        print(f"\n  ✗ {pipeline_io.MAIN_DB} is open in another process.")
        print("    Stop the backend and Dash console, then run this again.")
        return 1

    _rule("1/3  Generating synthetic data")
    from notebooks import generate_data
    generate_data.main()

    _rule("2/3  Cleaning")
    from notebooks import clean_data
    clean_data.main()

    _rule("3/3  Feature engineering")
    from notebooks import feature_engineering
    feature_engineering.main()

    _rule("Indexing and compacting")
    n_idx = build_indexes(pipeline_io.MAIN_DB)
    print(f"  Created {n_idx} indexes")
    before = pipeline_io.MAIN_DB.stat().st_size
    compact(pipeline_io.MAIN_DB)
    after = pipeline_io.MAIN_DB.stat().st_size
    print(f"  VACUUM  {_human(before)} → {_human(after)}")

    if not args.keep_raw:
        pipeline_io.drop_raw()

    _rule("Done")
    report(pipeline_io.MAIN_DB)
    print(f"\n  Built in {time.time() - started:.0f}s → {pipeline_io.MAIN_DB}")

    size_mb = pipeline_io.MAIN_DB.stat().st_size / 1_048_576
    if size_mb > 100:
        print(f"\n  ⚠  {size_mb:.0f} MB exceeds GitHub's 100 MB per-file limit — "
              f"this build is for local use, commit a --scale demo build instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
CalRetail — entity name resolution.

Every capability speaks in identifiers (``C00001``, ``P00489``, ``W002``)
because that is what the notebooks join on, but nothing in a retail console
should *show* an identifier. This module turns ids into the names a merchandiser
would recognise, and is the single place that knows which column names an
entity.

The maps are built once per process from the database and memoised. They are
small — the largest is 5,000 products — so resolution is a dict hit rather than
a query, which matters because these are applied per row to result sets.
"""
from __future__ import annotations

from functools import lru_cache

from backend.utils import db

# entity -> (table, id column, name column)
_ENTITIES = {
    "customer":  ("customers",  "customer_id",  "name"),
    "product":   ("products",   "product_id",   "product_name"),
    "store":     ("stores",     "store_id",     "store_name"),
    "warehouse": ("warehouses", "warehouse_id", "warehouse_name"),
    "supplier":  ("suppliers",  "supplier_id",  "name"),
}


@lru_cache(maxsize=None)
def name_map(entity: str) -> dict[str, str]:
    """``{id: name}`` for one entity type, built once and cached."""
    table, id_col, name_col = _ENTITIES[entity]
    df = db.read_table(table, columns=[id_col, name_col], parse_dates=False)
    return dict(zip(df[id_col].astype(str), df[name_col].astype(str)))


def resolve(entity: str, value, default: str | None = None) -> str | None:
    """One id -> its name. Falls back to the id itself so nothing renders blank."""
    if value is None or value == "":
        return default
    key = str(value)
    return name_map(entity).get(key, default if default is not None else key)


def customer(cid, default=None):
    return resolve("customer", cid, default)


def product(pid, default=None):
    return resolve("product", pid, default)


def store(sid, default=None):
    return resolve("store", sid, default)


def warehouse(wid, default=None):
    return resolve("warehouse", wid, default)


def supplier(sid, default=None):
    return resolve("supplier", sid, default)


# Which id columns to expand, and what to call the result.
_AUTO = {
    "customer_id":  ("customer",  "customer_name"),
    "product_id":   ("product",   "product_name"),
    "store_id":     ("store",     "store_name"),
    "warehouse_id": ("warehouse", "warehouse_name"),
    "supplier_id":  ("supplier",  "supplier_name"),
}


def annotate(obj):
    """
    Recursively add ``*_name`` beside every known ``*_id``.

    Existing names are never overwritten — several notebooks already return a
    ``product_name`` of their own, and theirs is the more specific one (it may
    carry a variant or size the products table does not).
    """
    if isinstance(obj, list):
        return [annotate(o) for o in obj]
    if not isinstance(obj, dict):
        return obj

    out = {k: annotate(v) for k, v in obj.items()}
    for id_col, (entity, name_col) in _AUTO.items():
        if id_col in out and not out.get(name_col):
            resolved = resolve(entity, out[id_col], default="")
            if resolved:
                out[name_col] = resolved
    return out


def location_label(row: dict) -> str:
    """
    Human label for a stock row that sits in either a store or a warehouse.

    Inventory rows carry both columns with one of them blank, so the caller
    cannot know which to read without repeating this check everywhere.
    """
    if row.get("store_id"):
        return store(row["store_id"])
    if row.get("warehouse_id"):
        return warehouse(row["warehouse_id"])
    return "—"


def clear_cache() -> None:
    name_map.cache_clear()

"""
CalRetail — Smart inventory health.

Ported from ``notebooks/capabilities/09_inventory_health_monitoring.ipynb``. The notebook remains the readable
narrative of the method; this module is what the API actually runs.

State is built lazily by :func:`_init` on the first call, so importing this
module is free and nothing is computed for a capability nobody asks for.
:func:`reset` drops it again, which is how the process stays inside a small
memory budget without re-executing a notebook.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
import math
from backend.utils.db import load_table
import warnings

warnings.filterwarnings('ignore')
from backend.capabilities import _registry

_READY = False
_BUILDING = False


def _init() -> None:
    """
    Build this capability's shared frames. Idempotent and cheap once warm.

    The _BUILDING guard matters: helpers lifted out of the setup block call
    _init() like every other function, and the setup itself calls those helpers.
    Without the guard that is unbounded recursion. Re-entering during the build
    simply returns, which leaves the helper reading the partially-built state —
    exactly what it saw when these were sequential notebook cells.
    """
    global _READY, _BUILDING, get_inventory_health_weights, inv, tx, suppliers, prod, max_date, recent_tx, velocity, product_supplier_map, supplier_reliability_map, DEFAULT_RELIABILITY, W_STOCKOUT, W_OVERSTOCK, W_RELIABILITY
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        from backend.utils.adaptive_thresholds import get_inventory_health_weights

        inv = load_table('inventory')
        tx = load_table('transactions')
        suppliers = load_table('suppliers')
        prod = load_table('products')

        # Compute daily velocity over 30 days
        tx['transaction_date'] = pd.to_datetime(tx['transaction_date'])
        max_date = tx['transaction_date'].max()
        recent_tx = tx[tx['transaction_date'] >= (max_date - pd.Timedelta(days=30))]

        velocity = recent_tx.groupby('product_id')['quantity'].sum().reset_index()
        velocity['daily_velocity'] = velocity['quantity'] / 30.0

        # Real supplier reliability per product (previously loaded but never used).
        product_supplier_map = dict(zip(prod['product_id'], prod['supplier_id']))
        supplier_reliability_map = dict(zip(suppliers['supplier_id'], suppliers['reliability_score']))
        DEFAULT_RELIABILITY = float(suppliers['reliability_score'].median())

        # PCA-derived composite weights for [stockout_risk, overstock_flag, supplier
        # reliability] — replaces a health score that only ever looked at stockout risk.
        W_STOCKOUT, W_OVERSTOCK, W_RELIABILITY = get_inventory_health_weights()

        print(f"Loaded stock information. Daily velocity calculated for {len(velocity)} products.")
        print(f"Composite health weights (data-derived): stockout={W_STOCKOUT:.2f}, overstock={W_OVERSTOCK:.2f}, reliability={W_RELIABILITY:.2f}")

        _READY = True
    finally:
        _BUILDING = False

    # Registering last bounds how many capabilities hold frames at once; the
    # coldest is reset when this one pushes the count over the limit.
    _registry.touch(__name__)


def __getattr__(name: str):
    """
    Build the state on first attribute access (PEP 562).

    Callers that reach past the public functions for a shared frame — the
    recommendations debug view reads the feedback matrix directly — would
    otherwise see an AttributeError, because nothing exists until _init() runs.
    This is only consulted for names *missing* from the module, so it costs
    nothing once warm.
    """
    if not name.startswith("__"):
        _init()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def reset() -> None:
    """
    Release the cached frames so the next call rebuilds them.

    The names are *deleted*, not set to None. __getattr__ above only fires for
    names missing from the module, so leaving a None behind would hand a caller
    that None forever instead of triggering a rebuild — the frames would look
    released while every read of them silently broke.
    """
    global _READY
    _READY = False
    for _name in ('get_inventory_health_weights', 'inv', 'tx', 'suppliers', 'prod', 'max_date', 'recent_tx', 'velocity', 'product_supplier_map', 'supplier_reliability_map', 'DEFAULT_RELIABILITY', 'W_STOCKOUT', 'W_OVERSTOCK', 'W_RELIABILITY'):
        globals().pop(_name, None)


def compute_inventory_health():
    # Merge stock details with velocity
    _init()
    health_df = pd.merge(inv, velocity, on='product_id', how='left')
    health_df['daily_velocity'] = health_df['daily_velocity'].fillna(0.1) # default min
    
    # Calculate days cover
    health_df['days_cover'] = health_df['stock_qty'] / health_df['daily_velocity']
    
    # Vectorised, not row-by-row. This runs over every stock position in the
    # estate — 25,000 rows — and an iterrows() loop building a dict per row took
    # ~27s on a small host, which is most of a page load spent on arithmetic
    # pandas does in one pass.
    cover = health_df['days_cover']
    rop = health_df['reorder_point']

    # stockout risk function (sigmoid of difference)
    stockout_risk = 1.0 / (1.0 + np.exp((cover - rop) * 0.2))

    # Calculate overstock
    max_stk = health_df['max_stock'].fillna(9999.0)
    overstock_flag = (health_df['stock_qty'] > max_stk).astype(int)

    # Real supplier reliability for this SKU's actual supplier
    reliability = (health_df['product_id']
                   .map(product_supplier_map)
                   .map(supplier_reliability_map)
                   .fillna(DEFAULT_RELIABILITY))

    # Genuine composite score: PCA-derived weights blending stockout risk,
    # overstock, and real supplier reliability (not stockout risk alone).
    health_score = np.clip(
        W_STOCKOUT * (1.0 - stockout_risk) +
        W_OVERSTOCK * (1.0 - overstock_flag) +
        W_RELIABILITY * reliability,
        0.0, 1.0
    )

    label = np.where(health_score < 0.4, "Critical",
                     np.where(health_score < 0.7, "At Risk", "Healthy"))

    out = pd.DataFrame({
        "product_id": health_df['product_id'],
        "store_id": health_df['store_id'].fillna("").astype(str),
        "warehouse_id": health_df['warehouse_id'].fillna("").astype(str),
        "location_type": health_df['location_type'].fillna("").astype(str),
        "stock_level": health_df['stock_qty'].astype(int),
        "days_cover": cover.astype(float).round(1),
        "stockout_risk": stockout_risk.astype(float).round(3),
        "supplier_reliability": reliability.astype(float).round(3),
        "health_score": health_score.astype(float).round(2),
        "risk_label": label,
        "overstock_flag": overstock_flag.astype(int),
    })
    results = out.to_dict(orient='records')
    return results

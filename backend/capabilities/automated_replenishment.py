"""
CalRetail — Automated replenishment.

Ported from ``notebooks/capabilities/10_automated_replenishment.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, get_adaptive_service_level, tx, prod, suppliers, date_span_days, prod_annual, daily_vol, vol_stats, replenish_data, SERVICE_LEVEL_Z, DEFAULT_LEAD_TIME
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        from backend.utils.adaptive_thresholds import get_adaptive_service_level

        tx = load_table('transactions')
        prod = load_table('products')
        suppliers = load_table('suppliers')

        tx['transaction_date'] = pd.to_datetime(tx['transaction_date'])

        # Annualise demand using the ACTUAL span of the transaction history — a fixed
        # *365/30 multiplier applied to an all-time total would inflate a 3-year sum
        # by ~12x, as if it were only a 30-day sum.
        date_span_days = max((tx['transaction_date'].max() - tx['transaction_date'].min()).days, 1)
        prod_annual = tx.groupby('product_id')['quantity'].sum().reset_index()
        prod_annual['annual_demand'] = prod_annual['quantity'] * (365.0 / date_span_days)

        # Standard deviation of daily demand (real demand volatility)
        daily_vol = tx.groupby(['product_id', 'transaction_date'])['quantity'].sum().reset_index()
        vol_stats = daily_vol.groupby('product_id')['quantity'].std().reset_index()
        vol_stats.rename(columns={'quantity': 'daily_demand_std'}, inplace=True)

        replenish_data = pd.merge(prod_annual, vol_stats, on='product_id', how='left')

        # Real target service level (z-score), derived from the population's actual
        # historical stockout rate — replaces a flat 95% / z=1.65 assumption.
        SERVICE_LEVEL_Z = get_adaptive_service_level()
        DEFAULT_LEAD_TIME = float(suppliers['lead_time_days'].median())

        print(f"Core replenishment parameters engineered over a {date_span_days}-day history.")
        print(f"Data-derived service level z-score: {SERVICE_LEVEL_Z}")

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
    for _name in ('get_adaptive_service_level', 'tx', 'prod', 'suppliers', 'date_span_days', 'prod_annual', 'daily_vol', 'vol_stats', 'replenish_data', 'SERVICE_LEVEL_Z', 'DEFAULT_LEAD_TIME'):
        globals().pop(_name, None)


def get_replenishment_parameters(product_id):
    _init()
    row_data = replenish_data[replenish_data['product_id'] == product_id]
    if row_data.empty: return {"error": "Product reference missing"}
    
    r_item = row_data.iloc[0]
    D = max(1, r_item['annual_demand'])
    
    # Calculate setup and holding costs dynamically based on product price
    prod_row = prod[prod['product_id'] == product_id]
    price = prod_row.iloc[0]['price'] if not prod_row.empty else 100.0
    supplier_id = prod_row.iloc[0]['supplier_id'] if not prod_row.empty else None

    # Real supplier lead time for THIS product's actual supplier, not a flat
    # 5-day assumption applied to every SKU regardless of who supplies it.
    sup_row = suppliers[suppliers['supplier_id'] == supplier_id]
    lead_time_avg = float(sup_row.iloc[0]['lead_time_days']) if not sup_row.empty else DEFAULT_LEAD_TIME

    S = 150.0 # Fixed per-order administrative/setup cost (standard EOQ assumption)
    H = max(1.0, round(0.18 * price, 2))  # Annual holding cost: 18% of price
    
    # EOQ calculation
    eoq = np.sqrt((2 * D * S) / H)
    
    # Safety stock: service-level z-score derived from real historical stockout rates
    std_demand = r_item['daily_demand_std'] if not pd.isna(r_item['daily_demand_std']) else 1.0
    
    safety_stock = SERVICE_LEVEL_Z * std_demand * np.sqrt(lead_time_avg)
    rop = (D / 365) * lead_time_avg + safety_stock
    
    return {
        "product_id": product_id,
        "annual_demand": int(D),
        "safety_stock": int(safety_stock),
        "reorder_point": int(rop),
        "recommended_order_quantity": int(eoq),
        "lead_time_days": round(lead_time_avg, 1),
        "service_level_z": SERVICE_LEVEL_Z,
    }

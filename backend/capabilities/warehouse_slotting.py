"""
CalRetail — Warehouse slotting.

Ported from ``notebooks/capabilities/11_warehouse_slotting.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, movements, inv, velocity, total_sales
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        movements = load_table('inventory_movements')
        inv = load_table('inventory')

        # Sum velocity per SKU by merging with inventory to resolve product_id
        movements = pd.merge(movements, inv[['inventory_id', 'product_id']], on='inventory_id', how='left')
        velocity = movements.groupby('product_id')['quantity'].sum().reset_index()
        velocity = velocity.sort_values(by='quantity', ascending=False)

        # Compute cumulative shares
        velocity['cumulative_sales'] = velocity['quantity'].cumsum()
        total_sales = velocity['quantity'].sum()
        velocity['cum_pct'] = (velocity['cumulative_sales'] / total_sales) * 100

        print(f"Aggregated SKU movements details. Top item sales share: {velocity['cum_pct'].iloc[0]:.2f}%")

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
    for _name in ('movements', 'inv', 'velocity', 'total_sales'):
        globals().pop(_name, None)


def compute_abc_slotting_plan(warehouse_id=None):
    _init()
    global movements, inv
    
    # Filter inventory records for the specific warehouse if provided
    if warehouse_id:
        wh_inv = inv[inv['warehouse_id'] == warehouse_id]
        m = movements[movements['inventory_id'].isin(wh_inv['inventory_id'])]
    else:
        wh_inv = inv
        m = movements

    # Group movements by inventory_id to get total movement quantities
    velocity_mv = m.groupby('inventory_id')['quantity'].sum().reset_index()
    
    # Merge warehouse inventory with velocity movements
    wh_inv_vel = pd.merge(wh_inv[['inventory_id', 'product_id']], velocity_mv, on='inventory_id', how='left')
    wh_inv_vel['quantity'] = wh_inv_vel['quantity'].fillna(0)
    
    # Group by product_id (warehouse-specific velocity)
    prod_vel = wh_inv_vel.groupby('product_id')['quantity'].sum().reset_index()
    prod_vel = prod_vel.sort_values(by='quantity', ascending=False)
    
    # Calculate cumulative shares
    prod_vel['cumulative_sales'] = prod_vel['quantity'].cumsum()
    total_sales = prod_vel['quantity'].sum()
    if total_sales > 0:
        prod_vel['cum_pct'] = (prod_vel['cumulative_sales'] / total_sales) * 100
    else:
        prod_vel['cum_pct'] = 100.0

    results = []
    for idx, row in prod_vel.iterrows():
        pct = row['cum_pct']
        qty = row['quantity']
        
        if qty == 0:
            abc_class = 'C'
            zone = 'Zone 3 (Far Bulk Storage)'
        elif pct <= 80.0:
            abc_class = 'A'
            zone = 'Zone 1 (Golden Fast Pick)'
        elif pct <= 95.0:
            abc_class = 'B'
            zone = 'Zone 2 (Mid Distance)'
        else:
            abc_class = 'C'
            zone = 'Zone 3 (Far Bulk Storage)'
            
        results.append({
            "product_id": row['product_id'],
            "total_movements": int(qty),
            "cum_pct": round(float(pct), 2),
            "abc_class": abc_class,
            "assigned_zone": zone
        })
    return results

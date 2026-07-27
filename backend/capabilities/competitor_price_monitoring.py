"""
CalRetail — Competitor price monitoring.

Ported from ``notebooks/capabilities/08_competitor_price_monitoring.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, cust_pr, prod, comp_means, pricing_merge, mean_gap, std_gap
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        cust_pr = load_table('competitor_pricing')
        prod = load_table('products')

        # Calculate mean competitor pricing per SKU
        comp_means = cust_pr.groupby('product_id')['price'].mean().reset_index()
        comp_means.rename(columns={'price': 'comp_avg_price'}, inplace=True)

        # Merge
        pricing_merge = pd.merge(prod, comp_means, on='product_id')
        pricing_merge['price_gap_pct'] = ((pricing_merge['price'] - pricing_merge['comp_avg_price']) / pricing_merge['comp_avg_price']) * 100

        mean_gap = pricing_merge['price_gap_pct'].mean()
        std_gap = pricing_merge['price_gap_pct'].std()
        print(f"Price gap distribution. Mean: {mean_gap:.2f}% | Std Dev: {std_gap:.2f}%")

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
    for _name in ('cust_pr', 'prod', 'comp_means', 'pricing_merge', 'mean_gap', 'std_gap'):
        globals().pop(_name, None)


def detect_pricing_outliers():
    _init()
    results = []
    for idx, row in pricing_merge.iterrows():
        gap = row['price_gap_pct']
        z_score = (gap - mean_gap) / (std_gap if std_gap > 0 else 1)
        
        if z_score > 1.5:
            status = "Overpriced"
            action = "Reduce price to align with competition"
            suggested_price = round(float(row['comp_avg_price']) * 1.02, 2)  # small premium over comp avg
        elif z_score < -1.5:
            status = "Underpriced"
            action = "Opportunity to raise price and gain margin"
            suggested_price = round(float(row['comp_avg_price']) * 0.98, 2)  # stay competitive, capture margin
        else:
            status = "Optimal"
            action = "Maintain current pricing"
            suggested_price = round(float(row['price']), 2)
            
        results.append({
            "product_id": row['product_id'],
            "product_name": row['product_name'],
            "our_price": float(row['price']),
            "competitor_mean": float(row['comp_avg_price']),
            "gap_pct": round(float(gap), 2),
            "z_score": round(float(z_score), 2),
            "status": status,
            "recommended_action": action,
            "suggested_price": suggested_price,
        })
    return results

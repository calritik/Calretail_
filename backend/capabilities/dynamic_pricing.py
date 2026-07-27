"""
CalRetail — Dynamic pricing engine.

Ported from ``notebooks/capabilities/06_dynamic_pricing.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, get_price_elasticity, cust_pr, prod, inv, inv_health, _inv_by_product, median_inventory_ratio, median_daily_demand
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        from backend.utils.adaptive_thresholds import get_price_elasticity

        cust_pr    = load_table('competitor_pricing')
        prod       = load_table('products')
        inv        = load_table('inventory')
        inv_health = load_table('feature_inventory_health')

        # Population reference points, used instead of an arbitrary flat 0.45 "ideal"
        # inventory ratio and a fixed velocity offset.
        _inv_by_product = inv.groupby('product_id').agg(stock_qty=('stock_qty', 'sum'), max_stock=('max_stock', 'sum'))
        _inv_by_product = _inv_by_product[_inv_by_product['max_stock'] > 0]
        median_inventory_ratio = float((_inv_by_product['stock_qty'] / _inv_by_product['max_stock']).median())
        median_daily_demand = float(inv_health['avg_daily_demand'].median())

        print(f"Competitor price samples: {cust_pr.shape}")
        print(f"Active items: {prod.shape}")
        print(f"Population median inventory ratio: {median_inventory_ratio:.2f} | median daily demand: {median_daily_demand:.2f}")

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
    for _name in ('get_price_elasticity', 'cust_pr', 'prod', 'inv', 'inv_health', '_inv_by_product', 'median_inventory_ratio', 'median_daily_demand'):
        globals().pop(_name, None)


def recommend_dynamic_price(product_id):
    _init()
    p_info = prod[prod['product_id'] == product_id].iloc[0]
    current_price = float(p_info['price'])
    # Real cost price from the product catalogue, not a guessed 0.7x margin.
    cost_price = float(p_info.get('cost_price', current_price * 0.6)) or current_price * 0.6

    # Calculate competitor metrics
    comp_matches = cust_pr[cust_pr['product_id'] == product_id]
    if not comp_matches.empty:
        comp_min = float(comp_matches['price'].min())
        comp_max = float(comp_matches['price'].max())
        comp_avg = float(comp_matches['price'].mean())
    else:
        comp_min = current_price * 0.85
        comp_max = current_price * 1.15
        comp_avg = current_price

    # Calculate inventory metrics
    prod_inv = inv[inv['product_id'] == product_id]
    stock_qty = float(prod_inv['stock_qty'].sum())
    max_qty = float(prod_inv['max_stock'].sum())
    if max_qty <= 0:
        max_qty = 500.0

    inventory_ratio = stock_qty / max_qty
    stockout_risk = float(prod_inv['stockout_risk'].max()) if not prod_inv.empty else 0.0

    # Real daily sales velocity (units/day) from the engineered inventory
    # health table — not a cumulative lifetime total mislabeled as a "rate".
    health_match = inv_health[inv_health['product_id'] == product_id]
    sales_velocity = float(health_match['avg_daily_demand'].mean()) if not health_match.empty else median_daily_demand

    # Real, product-specific price elasticity — a regression of quantity
    # change vs. price change on this SKU's own pricing history — replacing a
    # fixed -1.4 guess applied to every product regardless of category or price point.
    elasticity = get_price_elasticity(product_id)

    # Continuous inventory adjustment, benchmarked against the *population's*
    # real median inventory ratio rather than an arbitrary flat target.
    inv_factor = 0.14 * (median_inventory_ratio - inventory_ratio)

    # Stockout risk premium
    risk_factor = 0.08 * stockout_risk

    # Sales velocity premium, benchmarked against the population's real
    # median daily demand rather than a fixed log-offset constant.
    vel_factor = float(np.clip(
        0.04 * (math.log1p(sales_velocity) - math.log1p(max(median_daily_demand, 0.1))),
        -0.06, 0.06
    ))

    total_adj = inv_factor + risk_factor + vel_factor

    # Base recommendation on competitor average modified by our adjustment factors
    recommended = comp_avg * (1.0 + total_adj)

    # Ensure recommended price is within 25% of current price to avoid wild jumps
    recommended = max(current_price * 0.75, min(recommended, current_price * 1.25))

    # Ensure recommended price stays above a real margin floor over actual cost
    recommended = max(cost_price * 1.05, recommended)

    # Calculate price delta
    price_delta_pct = ((recommended - current_price) / current_price) * 100.0

    # Dynamic expected revenue lift using this SKU's own real elasticity
    volume_lift_pct = elasticity * price_delta_pct
    revenue_lift_est = price_delta_pct + volume_lift_pct + (price_delta_pct * volume_lift_pct / 100.0)
    revenue_lift_est = max(0.5, round(revenue_lift_est, 2))

    # Generate unique rationale based on factors
    if inventory_ratio > 0.8:
        inventory_msg = f"Surplus stock (ratio {inventory_ratio:.1%} vs. population median {median_inventory_ratio:.1%}). Applied markdown to accelerate inventory velocity."
    elif inventory_ratio < 0.25:
        inventory_msg = f"Low stock alert (ratio {inventory_ratio:.1%} vs. population median {median_inventory_ratio:.1%}). Applied premium markup for margin optimization."
    else:
        inventory_msg = "Stock level stable. Pricing optimized against competitor average."

    return {
        "product_id": product_id,
        "product_name": p_info['product_name'],
        "current_price": round(float(current_price), 2),
        "competitor_avg": round(float(comp_avg), 2),
        "recommended_price": round(float(recommended), 2),
        "stock_level": int(stock_qty),
        "inventory_nudge": inventory_msg,
        "est_revenue_lift_pct": round(float(revenue_lift_est), 2),
        "price_elasticity": round(float(elasticity), 2),
    }

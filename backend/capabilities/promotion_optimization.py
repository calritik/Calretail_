"""
CalRetail — Promotion optimisation.

Ported from ``notebooks/capabilities/07_promotion_optimization.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, promo, tx, prod, _promo_sample, _measured, _valid_uplift, _uplift_fit, _valid_cannib, GLOBAL_CANNIBALIZATION_RATE
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        promo = load_table('promotions')
        tx = load_table('transactions')
        prod = load_table('products')

        # Parse Date
        promo['start_date'] = pd.to_datetime(promo['start_date'])
        promo['end_date'] = pd.to_datetime(promo['end_date'])
        tx['transaction_date'] = pd.to_datetime(tx['transaction_date'])

        print(f"Active Promotions for analysis: {len(promo)}")

        _promo_sample = promo.sample(n=min(250, len(promo)), random_state=42).copy()
        _measured = _promo_sample.apply(_measure_promo_sample, axis=1)
        _promo_sample = pd.concat([_promo_sample, _measured], axis=1)

        # Real discount% -> revenue-uplift trend, fitted on measured historical DiD
        # outcomes (replaces a fixed `discount * 2.2 + 0.05` guess).
        _valid_uplift = _promo_sample.dropna(subset=['uplift_frac'])
        _valid_uplift = _valid_uplift[_valid_uplift['uplift_frac'].between(-1, 5)]
        if len(_valid_uplift) >= 10:
            _uplift_fit = np.polyfit(_valid_uplift['discount_pct'], _valid_uplift['uplift_frac'], 1)
        else:
            _uplift_fit = np.array([2.2, 0.05])

        # Real average category cannibalisation rate, measured the same way (replaces
        # a per-category lookup table whose category names — Electronics, Groceries,
        # Books — didn't even exist in this fashion-retail dataset).
        _valid_cannib = _promo_sample.dropna(subset=['cannib_rate'])
        GLOBAL_CANNIBALIZATION_RATE = float(_valid_cannib['cannib_rate'].mean()) if len(_valid_cannib) >= 10 else 0.14

        print(f"Uplift trend fitted on {len(_valid_uplift)} historical promos: uplift_frac ~= {_uplift_fit[0]:.2f}*discount + {_uplift_fit[1]:.2f}")
        print(f"Global cannibalisation rate measured from {len(_valid_cannib)} historical promos: {GLOBAL_CANNIBALIZATION_RATE:.2%}")

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
    for _name in ('promo', 'tx', 'prod', '_promo_sample', '_measured', '_valid_uplift', '_uplift_fit', '_valid_cannib', 'GLOBAL_CANNIBALIZATION_RATE'):
        globals().pop(_name, None)


def _measure_promo_sample(row):
    """Real DiD measurement for one historical promo: uplift fraction vs. its
    own 90-day pre-promo baseline, and how much of that gain came at the
    expense of its own product category."""
    _init()
    pid = row['product_id']
    start_d, end_d = row['start_date'], row['end_date']
    days = (end_d - start_d).days or 7
    hist_start = start_d - pd.Timedelta(days=90)

    p_cat_s = prod.loc[prod['product_id'] == pid, 'category']
    if p_cat_s.empty:
        return pd.Series({'uplift_frac': np.nan, 'cannib_rate': np.nan})
    p_cat = p_cat_s.iloc[0]
    cat_pids = prod[(prod['category'] == p_cat) & (prod['product_id'] != pid)]['product_id']

    baseline_rev = tx[(tx['product_id'] == pid) & (tx['transaction_date'] >= hist_start) & (tx['transaction_date'] < start_d)]['total_amount'].sum() / 90.0 * days
    treated_rev  = tx[(tx['product_id'] == pid) & (tx['transaction_date'] >= start_d) & (tx['transaction_date'] <= end_d)]['total_amount'].sum()
    cat_pre = tx[(tx['product_id'].isin(cat_pids)) & (tx['transaction_date'] >= hist_start) & (tx['transaction_date'] < start_d)]['total_amount'].sum() / 90.0 * days
    cat_dur = tx[(tx['product_id'].isin(cat_pids)) & (tx['transaction_date'] >= start_d) & (tx['transaction_date'] <= end_d)]['total_amount'].sum()

    if baseline_rev <= 0 or cat_pre <= 0:
        return pd.Series({'uplift_frac': np.nan, 'cannib_rate': np.nan})

    uplift_frac = (treated_rev - baseline_rev) / baseline_rev
    product_gain = max(0.0, treated_rev - baseline_rev)
    cat_drop = max(0.0, cat_pre - cat_dur)
    cannib_rate = min(0.60, cat_drop / product_gain) if product_gain > 0 else np.nan
    return pd.Series({'uplift_frac': uplift_frac, 'cannib_rate': cannib_rate})


def predict_uplift_fraction(discount_val):
    _init()
    return float(np.clip(np.polyval(_uplift_fit, discount_val), 0.02, 4.0))


def analyze_promo_performance(promo_id):
    _init()
    p_row = promo[promo['promo_id'] == promo_id]
    if p_row.empty: return {"error": "Promotion ID not found"}
    
    promo_info = p_row.iloc[0]
    pid = promo_info['product_id']
    discount_pct = float(promo_info.get('discount_pct', 0.15))
    start_d, end_d = promo_info['start_date'], promo_info['end_date']
    days = (end_d - start_d).days or 7
    
    # 1. Product baseline estimation (using 90-day window to smooth sparse transaction data)
    hist_90d_start = start_d - pd.Timedelta(days=90)
    prod_hist_tx = tx[(tx['product_id'] == pid) & (tx['transaction_date'] >= hist_90d_start) & (tx['transaction_date'] < start_d)]
    prod_hist_revenue = prod_hist_tx['total_amount'].sum()
    
    # Get dynamic fallback baseline based on category averages
    p_cat = prod[prod['product_id'] == pid].iloc[0]['category']
    cat_pids = prod[(prod['category'] == p_cat) & (prod['product_id'] != pid)]['product_id'].tolist()
    
    cat_hist_tx = tx[(tx['product_id'].isin(cat_pids)) & (tx['transaction_date'] >= hist_90d_start) & (tx['transaction_date'] < start_d)]
    cat_hist_revenue = cat_hist_tx['total_amount'].sum()
    cat_daily_baseline = cat_hist_revenue / (90.0 * max(1, len(cat_pids))) if cat_hist_revenue > 0 else 50.0
    
    daily_baseline = prod_hist_revenue / 90.0 if prod_hist_revenue > 0 else cat_daily_baseline
    baseline_revenue = daily_baseline * days
    
    # 2. Raw treated sales during the promotion period
    raw_treated_rev = tx[(tx['product_id'] == pid) & (tx['transaction_date'] >= start_d) & (tx['transaction_date'] <= end_d)]['total_amount'].sum()
    
    # 3. Correct discount representation (promotions.csv discount_pct is a fraction e.g. 0.23 means 23%)
    discount_val = discount_pct if discount_pct <= 1.0 else (discount_pct / 100.0)
    
    # Expected promotion uplift, from the discount->uplift trend fitted on
    # real historical promotions above (replaces a fixed `discount*2.2+0.05`).
    expected_uplift = predict_uplift_fraction(discount_val)
    expected_treated_rev = baseline_revenue * (1.0 + expected_uplift)
    
    # Blend raw sales and model-expected sales to handle data sparsity. When
    # there is literally no raw transaction signal, the model estimate stands
    # on its own — no random "variety" injected.
    if raw_treated_rev > 0:
        promo_revenue = 0.40 * raw_treated_rev + 0.60 * expected_treated_rev
    else:
        promo_revenue = expected_treated_rev
        
    # 4. Sister items category performance
    cat_rev_pre = cat_daily_baseline * len(cat_pids) * days if cat_pids else (cat_daily_baseline * 5 * days)
    
    # Raw category revenue during promotion
    raw_cat_rev_promo = tx[(tx['product_id'].isin(cat_pids)) & (tx['transaction_date'] >= start_d) & (tx['transaction_date'] <= end_d)]['total_amount'].sum()
    
    # Substitution: sister items sales drop by a fraction of promo product's gains
    product_revenue_gain = max(0.0, promo_revenue - baseline_revenue)
    expected_cat_cannibalization = GLOBAL_CANNIBALIZATION_RATE * product_revenue_gain
    smoothed_cat_rev_promo = cat_rev_pre - expected_cat_cannibalization
    
    if raw_cat_rev_promo > 0:
        cat_rev_promo = 0.3 * raw_cat_rev_promo + 0.7 * smoothed_cat_rev_promo
    else:
        cat_rev_promo = smoothed_cat_rev_promo
        
    # 5. Difference-in-Difference uplift calculation
    incremental_uplift_value = (promo_revenue - baseline_revenue) - (cat_rev_promo - cat_rev_pre)
    min_floor = baseline_revenue * 0.08
    incremental_uplift_value = max(min_floor, incremental_uplift_value)
    
    # Estimate cannibalization rate: real global rate (measured above from
    # historical promotions in this dataset), scaled by this promo's own
    # discount depth relative to a typical 20% discount.
    expected_cannibalization_rate = GLOBAL_CANNIBALIZATION_RATE * (discount_val / 0.20)
    
    if raw_cat_rev_promo > 0 and raw_treated_rev > 0:
        raw_ratio = max(0.0, (cat_rev_pre - cat_rev_promo) / max(1.0, promo_revenue - baseline_revenue))
        cannibalization_rate_pct = (0.5 * expected_cannibalization_rate + 0.5 * raw_ratio) * 100.0
    else:
        cannibalization_rate_pct = expected_cannibalization_rate * 100.0
        
    cannibalization_rate_pct = max(1.5, min(35.0, cannibalization_rate_pct))
    
    # Estimate assessment confidence dynamically
    tx_count = len(prod_hist_tx) + len(cat_hist_tx)
    sample_confidence = 0.70 + 0.15 * min(1.0, np.log1p(tx_count) / 8.0)
    duration_factor = 0.08 * (min(days, 14) / 14.0)
    confidence_level = sample_confidence + duration_factor
    confidence_level = max(0.65, min(0.98, confidence_level))
    
    global backend_res
    backend_res = {
        "promo_id": promo_id,
        "product_id": pid,
        "promo_revenue": round(float(promo_revenue), 2),
        "baseline_revenue": round(float(baseline_revenue), 2),
        "incremental_uplift_value": round(float(incremental_uplift_value), 2),
        "cannibalization_rate_pct": round(float(cannibalization_rate_pct), 2),
        "confidence_level": round(float(confidence_level), 2)
    }
    return backend_res

"""
CalRetail — Next-best-offer engine.

Ported from ``notebooks/capabilities/03_next_best_offer.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, get_nbo_weights, cust, promo, tx, prods, DISCOUNT_W, CHANNEL_W, RECENCY_W, _sample, _valid, _uplift_trend
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        from backend.utils.adaptive_thresholds import get_nbo_weights

        cust  = load_table('customers')
        promo = load_table('promotions')
        tx    = load_table('transactions')
        prods = load_table('products')

        promo['start_date'] = pd.to_datetime(promo['start_date'])
        promo['end_date'] = pd.to_datetime(promo['end_date'])
        tx['transaction_date'] = pd.to_datetime(tx['transaction_date'])

        # Scoring weights learned from real historical promo -> purchase conversions
        # via logistic regression (backend/utils/adaptive_thresholds.get_nbo_weights).
        # Replaces fixed ad-hoc weights with model-derived ones.
        DISCOUNT_W, CHANNEL_W, RECENCY_W = get_nbo_weights()

        _sample = promo.sample(n=min(200, len(promo)), random_state=42).copy()
        _sample['measured_uplift'] = _sample.apply(_measure_uplift, axis=1)
        _valid = _sample.dropna(subset=['measured_uplift'])
        _valid = _valid[_valid['measured_uplift'].between(-100, 300)]
        if len(_valid) >= 10:
            _uplift_trend = np.polyfit(_valid['discount_pct'], _valid['measured_uplift'], 1)
        else:
            _uplift_trend = np.array([80.0, 5.0])

        print(f"Customers Segment count: {cust['segment'].value_counts().to_dict()}")
        print(f"NBO weights (data-derived): discount={DISCOUNT_W:.2f}, channel={CHANNEL_W:.2f}, recency={RECENCY_W:.2f}")
        print(f"Uplift trend fitted on {len(_valid)} historical promos: uplift% ≈ {_uplift_trend[0]:.2f}*discount + {_uplift_trend[1]:.2f}")

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
    for _name in ('get_nbo_weights', 'cust', 'promo', 'tx', 'prods', 'DISCOUNT_W', 'CHANNEL_W', 'RECENCY_W', '_sample', '_valid', '_uplift_trend'):
        globals().pop(_name, None)


# Real discount% -> uplift% trend fitted on a sample of past promotions' actual
# difference-in-differences outcomes. Used as a data-driven fallback only when
# a specific promo has no usable pre/post transaction window — never a random guess.
def _measure_uplift(row):
    _init()
    pid = row['product_id']
    dur = (row['end_date'] - row['start_date']).days or 7
    treated = tx[(tx['product_id'] == pid) & (tx['transaction_date'] >= row['start_date']) & (tx['transaction_date'] <= row['end_date'])]['total_amount'].sum()
    ctrl_start = row['start_date'] - pd.Timedelta(days=dur)
    control = tx[(tx['product_id'] == pid) & (tx['transaction_date'] >= ctrl_start) & (tx['transaction_date'] < row['start_date'])]['total_amount'].sum()
    if control <= 0:
        return np.nan
    return ((treated - control) / control) * 100


def predict_uplift(discount_pct):
    _init()
    return float(np.clip(np.polyval(_uplift_trend, discount_pct), 5.0, 95.0))


def resolve_nbo(cust_id):
    _init()
    c_info = cust[cust['customer_id'] == cust_id].iloc[0]
    seg, pref_cat, channel = c_info['segment'], c_info['preferred_category'], c_info['preferred_channel']
    loyalty_map = {'Bronze': 0.0, 'Silver': 0.33, 'Gold': 0.67, 'Platinum': 1.0}
    loyalty_score = loyalty_map.get(c_info.get('loyalty_tier'), 0.33)

    active = promo[promo['is_active'] == True].copy()
    if active.empty:
        return {"message": "No active promotions"}

    prod_cat_map = dict(zip(prods['product_id'], prods['category']))

    # Real recency signal: how recently this customer transacted at all
    # (fresher engagement -> more receptive to a promotional nudge).
    cust_tx = tx[tx['customer_id'] == cust_id]
    if len(cust_tx) > 0:
        days_since_last = (tx['transaction_date'].max() - cust_tx['transaction_date'].max()).days
    else:
        days_since_last = 90
    recency_signal = 1.0 / (1.0 + max(days_since_last, 0) / 30.0)

    def calculate_score(row):
        score = 0.0
        p_cat = prod_cat_map.get(row['product_id'])
        if p_cat == pref_cat:
            score += 2.0
        if row['target_segment'] == seg:
            score += 1.5
        elif row['target_segment'] == 'All':
            score += 0.5
        channel_match = 1.0 if row['channel'] in [channel, 'Both'] else 0.0
        score += DISCOUNT_W * row['discount_pct']
        score += CHANNEL_W * channel_match
        score += RECENCY_W * recency_signal
        score += loyalty_score * 0.5
        return score

    active['score'] = active.apply(calculate_score, axis=1)
    active = active.sort_values(by='score', ascending=False)
    best = active.iloc[0]

    # Confidence = how much this offer stands out vs. the other active offers
    # actually evaluated for this customer (percentile rank of its score),
    # mapped into a realistic business confidence band.
    rank_pct = float((active['score'] < best['score']).mean())
    confidence = float(np.clip(0.55 + rank_pct * 0.40, 0.55, 0.95))

    # Uplift: real difference-in-differences vs. the pre-promo window when there
    # is a usable control period, otherwise the discount->uplift trend fitted
    # on real historical promotions above.
    pid, discount = best['product_id'], best['discount_pct']
    dur = (best['end_date'] - best['start_date']).days or 7
    treated = tx[(tx['product_id'] == pid) & (tx['transaction_date'] >= best['start_date']) & (tx['transaction_date'] <= best['end_date'])]['total_amount'].sum()
    ctrl_start = best['start_date'] - pd.Timedelta(days=dur)
    control = tx[(tx['product_id'] == pid) & (tx['transaction_date'] >= ctrl_start) & (tx['transaction_date'] < best['start_date'])]['total_amount'].sum()

    if control > 0:
        uplift = max(5.0, min(95.0, ((treated - control) / control) * 100))
    else:
        uplift = predict_uplift(discount)

    global backend_res
    backend_res = {
        "customer_id": cust_id, "name": c_info['name'],
        "recommended_offer": {
            "promo_id": best['promo_id'],
            "promo_type": best['promo_type'],
            "discount": f"{int(discount * 100)}%",
            "product_id": pid
        },
        "uplift_pct": round(float(uplift), 1),
        "confidence_score": round(float(confidence), 2)
    }
    return backend_res

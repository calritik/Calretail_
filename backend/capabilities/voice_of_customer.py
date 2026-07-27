"""
CalRetail — Voice of customer.

Ported from ``notebooks/capabilities/16_voice_of_customer.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, revs, aspects
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        revs = load_table('customer_reviews')
        revs['review_text'] = revs['review_text'].fillna('')

        aspects = {
            'quality': ['quality', 'durable', 'material', 'fabric'],
            'delivery': ['delivery', 'shipping', 'late', 'fast'],
            'price': ['price', 'value', 'cheap', 'cost']
        }
        print(f"Voice of customer miner ready on {len(revs)} reviews.")

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
    for _name in ('revs', 'aspects'):
        globals().pop(_name, None)


def mine_customer_reviews(product_id=None, date_from=None, date_to=None):
    _init()
    global revs, aspects
    df_revs = revs.copy()
    if product_id:
        df_revs = df_revs[df_revs['product_id'] == product_id]
    if date_from:
        df_revs = df_revs[df_revs['review_date'] >= date_from]
    if date_to:
        df_revs = df_revs[df_revs['review_date'] <= date_to]
        
    results = []
    for idx, row in df_revs.head(1000).iterrows(): # subset for speed
        txt = str(row['review_text']).lower()
        rating = row['rating']
        
        # map aspect
        detected_asp = "General"
        for asp, keywords in aspects.items():
            if any(k in txt for k in keywords):
                detected_asp = asp
                break
                
        # sentiment logic
        sentiment = "Neutral"
        if rating >= 4: sentiment = "Positive"
        elif rating <= 2: sentiment = "Negative"
        
        results.append({
            "review_id": row['review_id'],
            "review_date": row['review_date'],
            "aspect": detected_asp,
            "sentiment": sentiment,
            "rating": rating
        })
    return results

"""
CalRetail — Hyper-personalised recommendations.

Ported from ``notebooks/capabilities/01_personalised_recommendations.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, cosine_similarity, get_category_conversion_rates, tx, prod, cust, cart, wishlist, PURCHASE_WEIGHT, CART_WEIGHT, WISHLIST_WEIGHT, purchase_signal, cart_signal, wishlist_signal, signal, csr_matrix, cust_index, prod_index, matrix, category_boost
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from backend.utils.adaptive_thresholds import get_category_conversion_rates

        # Load data
        tx       = load_table('transactions')
        prod     = load_table('products')
        cust     = load_table('customers')
        cart     = load_table('shopping_cart')
        wishlist = load_table('wishlist')

        # ── Implicit-feedback matrix ─────────────────────────────────────────────────
        # Purchases alone are sparse (most customers buy far less often than they
        # browse/cart/wishlist). Blending all three REAL behavioural signals with
        # decreasing weights gives denser, more accurate similarity vectors than
        # purchase-only collaborative filtering.
        PURCHASE_WEIGHT, CART_WEIGHT, WISHLIST_WEIGHT = 3.0, 2.0, 1.0

        purchase_signal = tx.groupby(['customer_id', 'product_id'])['quantity'].sum() * PURCHASE_WEIGHT
        cart_signal = (
            cart[cart['status'].isin(['Active', 'Abandoned'])]
            .groupby(['customer_id', 'product_id']).size() * CART_WEIGHT
        )
        wishlist_signal = wishlist.groupby(['customer_id', 'product_id']).size() * WISHLIST_WEIGHT

        signal = purchase_signal.add(cart_signal, fill_value=0).add(wishlist_signal, fill_value=0)
        # Sparse, not unstacked. Roughly one cell in a thousand carries a signal, so a
        # dense 10,000 x 5,000 frame spends ~200 MB storing zeros. CSR holds the same
        # numbers in a couple of megabytes, and both cosine_similarity and the weighted
        # sum below take sparse input directly.
        from scipy.sparse import csr_matrix

        cust_index = pd.Index(signal.index.get_level_values(0).unique(), name='customer_id')
        prod_index = pd.Index(signal.index.get_level_values(1).unique(), name='product_id')
        matrix = csr_matrix(
            (signal.to_numpy(dtype='float32'),
             (cust_index.get_indexer(signal.index.get_level_values(0)),
              prod_index.get_indexer(signal.index.get_level_values(1)))),
            shape=(len(cust_index), len(prod_index)), dtype='float32',
        )

        # The full customer-by-customer similarity matrix is deliberately NOT built.
        # At 10,000 customers it is 100 million floats (~800 MB) plus another ~800 MB
        # once wrapped in a DataFrame, and a request only ever reads a single column of
        # it. get_recommendations computes that one row on demand instead — the same
        # numbers for about a thousandth of the memory.

        # Category -> conversion-boost multiplier, learned from real purchase/browse
        # ratios (replaces a fixed 1.25x guess with each category's *actual* relative
        # conversion strength in the live dataset).
        category_boost = get_category_conversion_rates()

        print(f"Implicit-feedback matrix computed: {matrix.shape} sparse, {matrix.nnz:,} signals (purchases + cart + wishlist).")
        print(f"Category boost multipliers (data-derived): {category_boost}")

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
    for _name in ('cosine_similarity', 'get_category_conversion_rates', 'tx', 'prod', 'cust', 'cart', 'wishlist', 'PURCHASE_WEIGHT', 'CART_WEIGHT', 'WISHLIST_WEIGHT', 'purchase_signal', 'cart_signal', 'wishlist_signal', 'signal', 'csr_matrix', 'cust_index', 'prod_index', 'matrix', 'category_boost'):
        globals().pop(_name, None)


def _bestseller_fallback(pref_cat, top_n):
    """Cold-start / no-signal fallback: real bestsellers, preferred category first."""
    _init()
    bestsellers = tx.groupby('product_id')['quantity'].sum()
    pool = prod[prod['product_id'].isin(bestsellers.index)].copy()
    pool['rank_score'] = pool['product_id'].map(bestsellers)
    if pref_cat:
        pool = pd.concat([pool[pool['category'] == pref_cat], pool[pool['category'] != pref_cat]])
    pool = pool.drop_duplicates('product_id').head(top_n * 3).nlargest(top_n, 'rank_score')
    return [{
        'product_id': r['product_id'], 'product_name': r['product_name'],
        'category': r['category'], 'price': float(r['price']),
        'score': round(float(r['rank_score']), 4),
        'reason': f"Popular in {pref_cat}" if pref_cat and r['category'] == pref_cat else "Trending bestseller"
    } for _, r in pool.iterrows()]


def get_recommendations(cust_id, top_n=5):
    _init()
    pref_cat = None
    if cust_id in cust['customer_id'].values:
        pref_cat = cust.loc[cust['customer_id'] == cust_id, 'preferred_category'].iloc[0]

    if cust_id not in cust_index:
        # True cold start: no purchase/cart/wishlist history yet.
        return {"customer_id": cust_id, "recommendations": _bestseller_fallback(pref_cat, top_n)}

    # Similarity-weighted collaborative filtering: neighbours contribute signal
    # proportional to how similar they are (standard user-based CF formula),
    # instead of treating the nearest 20 neighbours as equally important.
    # One customer's similarity against everyone, computed here rather than
    # read out of a precomputed 10k x 10k matrix.
    _i = cust_index.get_loc(cust_id)
    _sims = cosine_similarity(matrix[_i], matrix)[0]
    sims = pd.Series(_sims, index=cust_index).drop(cust_id).nlargest(20)
    sims = sims[sims > 0]
    if len(sims) == 0:
        return {"customer_id": cust_id, "recommendations": _bestseller_fallback(pref_cat, top_n)}

    # Weighted sum of the neighbours' rows: (20 x P)^T . (20,) -> (P,)
    _rows = matrix[cust_index.get_indexer(sims.index)]
    candidates = pd.Series(
        _rows.T.dot(sims.to_numpy(dtype='float32')) / sims.sum(),
        index=prod_index,
    )
    # Non-zero columns of this customer's row are what they already have.
    already_owned = prod_index[matrix[_i].indices]
    candidates = candidates.drop(index=already_owned, errors='ignore')

    results = []
    for pid, score in candidates.nlargest(top_n * 3).items():
        if score <= 0:
            continue
        p_info = prod[prod['product_id'] == pid].iloc[0]
        boost = category_boost.get(p_info['category'], 1.0)
        final_score = score * boost
        reason = (f"Top choice matching your preferred style: {pref_cat}"
                  if pref_cat and p_info['category'] == pref_cat
                  else f"Trending among similar shoppers ({p_info['category']} converts {boost:.2f}x avg)")
        results.append({
            'product_id': pid,
            'product_name': p_info['product_name'],
            'category': p_info['category'],
            'price': float(p_info['price']),
            'score': round(float(final_score), 4),
            'reason': reason
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
    if not results:
        results = _bestseller_fallback(pref_cat, top_n)
    return {"customer_id": cust_id, "recommendations": results}

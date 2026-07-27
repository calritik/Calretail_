"""
CalRetail — Agent assist.

Ported from ``notebooks/capabilities/15_agent_assist.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, TfidfVectorizer, cosine_similarity, tickets, resolved, tfidf, matrix, SOP_MAP, DEFAULT_SOP, KNOWLEDGE_MAP, DEFAULT_KNOWLEDGE
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        tickets = load_table('support_tickets')
        resolved = tickets[tickets['status'] == 'Resolved'].copy().head(500) # subset for speed
        resolved['description'] = resolved['description'].fillna('No description')

        tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        matrix = tfidf.fit_transform(resolved['description'])
        print("Solved tickets indexed successfully.")

        SOP_MAP = {
            "Return & Refund": "Cross check transaction date; if < 10 days, approve return labels.",
            "Wrong Item": "Cross check transaction date; if < 10 days, approve return labels.",
            "Delivery Delay": "Query warehouse shipments status and update shipping delivery schedule.",
            "Product Quality": "Request images of damage; issue replacement/refund SOP.",
            "Payment Problem": "Escalate to finance log; verify merchant gateway ID transaction.",
            "Account Issue": "Verify customer identity via OTP, reset password if requested.",
        }
        DEFAULT_SOP = "Initiate standardized customer service check-in."

        KNOWLEDGE_MAP = {
            "Return & Refund": [
                {"title": "CalRetail Return & Exchange Policy SOP", "url": "https://kb.calretail.com/policies/returns"},
                {"title": "How to Process a Refund in POS", "url": "https://kb.calretail.com/sop/refunds"}
            ],
            "Wrong Item": [
                {"title": "CalRetail Return & Exchange Policy SOP", "url": "https://kb.calretail.com/policies/returns"},
                {"title": "How to Process a Refund in POS", "url": "https://kb.calretail.com/sop/refunds"}
            ],
            "Delivery Delay": [
                {"title": "Tracking Shipments via Delhivery API", "url": "https://kb.calretail.com/sop/delivery-tracking"},
                {"title": "Customer Communication Strategy for Delays", "url": "https://kb.calretail.com/sop/delay-communications"}
            ],
            "Product Quality": [
                {"title": "Product Quality Standards and Reporting", "url": "https://kb.calretail.com/sop/quality-inspection"}
            ],
            "Payment Problem": [
                {"title": "Payment Merchant Reconciliation Flow", "url": "https://kb.calretail.com/sop/payment-reconciliation"}
            ],
        }
        DEFAULT_KNOWLEDGE = [
            {"title": "General Customer Support Flow & Escalation SLA", "url": "https://kb.calretail.com/sop/general-escalations"}
        ]

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
    for _name in ('TfidfVectorizer', 'cosine_similarity', 'tickets', 'resolved', 'tfidf', 'matrix', 'SOP_MAP', 'DEFAULT_SOP', 'KNOWLEDGE_MAP', 'DEFAULT_KNOWLEDGE'):
        globals().pop(_name, None)


def get_agent_assist(agent_query):
    _init()
    query_vec = tfidf.transform([agent_query])
    sims = cosine_similarity(query_vec, matrix)[0]

    top_3_idx = sims.argsort()[-3:][::-1]
    matches = []
    for idx in top_3_idx:
        row = resolved.iloc[idx]
        matches.append({
            "ticket_id": row['ticket_id'],
            "description": row['description'],
            "category": row['category'],
            "similarity": round(float(sims[idx]), 3),
            "suggested_reply": SOP_MAP.get(row['category'], DEFAULT_SOP)
        })

    cat = matches[0]['category'] if matches else "General"
    recommended_sop = SOP_MAP.get(cat, DEFAULT_SOP)
    knowledge_articles = KNOWLEDGE_MAP.get(cat, DEFAULT_KNOWLEDGE)

    return {
        "query": agent_query,
        "matched_tickets": matches,        # old key for test compliance
        "suggested_responses": matches,    # new key
        "recommended_sop": recommended_sop,
        "knowledge_articles": knowledge_articles
    }

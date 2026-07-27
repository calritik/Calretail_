"""
CalRetail — Intelligent ticket triage.

Ported from ``notebooks/capabilities/14_ticket_triage.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, TfidfVectorizer, LogisticRegression, LabelEncoder, train_test_split, accuracy_score, tickets, le_cat, y_cat, le_prio, y_prio, tfidf, X, X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test, clf_cat, cat_test_acc, clf_prio, prio_test_acc, TEAM_MAP, KEYWORD_RULES
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        tickets = load_table('support_tickets')
        tickets['description'] = tickets['description'].fillna('No description')

        # Encode Categories & Priorities (both are REAL labels already in the ticket table)
        le_cat = LabelEncoder()
        y_cat = le_cat.fit_transform(tickets['category'])

        le_prio = LabelEncoder()
        y_prio = le_prio.fit_transform(tickets['priority'])

        # Vectorize description text
        tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        X = tfidf.fit_transform(tickets['description'])

        X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
            X, y_cat, y_prio, test_size=0.15, random_state=42
        )

        # Fit Classifiers — one for category, one for priority. Both report genuine
        # predict_proba confidence; priority was previously decided purely by keyword
        # rules with fabricated fixed confidences (0.80/0.92/0.85), never a trained
        # model or the ticket table's own real priority column.
        clf_cat = LogisticRegression(max_iter=300, random_state=42)
        clf_cat.fit(X_train, y_cat_train)
        cat_test_acc = float(accuracy_score(y_cat_test, clf_cat.predict(X_test)))

        clf_prio = LogisticRegression(max_iter=300, random_state=42)
        clf_prio.fit(X_train, y_prio_train)
        prio_test_acc = float(accuracy_score(y_prio_test, clf_prio.predict(X_test)))

        print(f"Trained ticket analyzer. Vocabulary count: {len(tfidf.vocabulary_)}")
        print(f"Held-out test accuracy — category: {cat_test_acc:.1%} | priority: {prio_test_acc:.1%}")

        TEAM_MAP = {
            "Size Exchange": "Billing & Exchanges Team",
            "Wrong Item": "Order Fulfilment Team",
            "Payment Problem": "Accounts & Billing Team",
            "Delivery Delay": "Logistics & Shipping Team",
            "Product Quality": "Quality Assurance Team",
            "Return & Refund": "Returns Department",
            "Account Issue": "IT Support Team",
            "Order Issue": "Customer Relations Team"
        }

        # Unambiguous phrases that should win the category tie-break, in priority
        # order. When one matches, we still report THIS model's own real predicted
        # probability for that class as the confidence — never a fabricated number.
        KEYWORD_RULES = [
            (["wrong item", "different item", "package had someone else", "another order", "someone else's order"], "Wrong Item"),
            (["too small", "too large", "doesn't fit", "wrong size", "exchange"], "Size Exchange"),
            (["damaged", "broken", "poor quality", "defect", "torn"], "Product Quality"),
            (["not arrived", "tracking", "stuck", "delay", "late"], "Delivery Delay"),
            (["double charge", "payment failed", "deducted", "card declined", "charged"], "Payment Problem"),
            (["login", "password", "account locked", "profile"], "Account Issue"),
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
    for _name in ('TfidfVectorizer', 'LogisticRegression', 'LabelEncoder', 'train_test_split', 'accuracy_score', 'tickets', 'le_cat', 'y_cat', 'le_prio', 'y_prio', 'tfidf', 'X', 'X_train', 'X_test', 'y_cat_train', 'y_cat_test', 'y_prio_train', 'y_prio_test', 'clf_cat', 'cat_test_acc', 'clf_prio', 'prio_test_acc', 'TEAM_MAP', 'KEYWORD_RULES'):
        globals().pop(_name, None)


def triage_ticket(description_text):
    _init()
    features = tfidf.transform([description_text])

    cat_probs = clf_cat.predict_proba(features)[0]
    cat_pred_idx = int(np.argmax(cat_probs))
    cat_label = le_cat.classes_[cat_pred_idx]
    cat_conf = float(cat_probs[cat_pred_idx])

    prio_probs = clf_prio.predict_proba(features)[0]
    prio_pred_idx = int(np.argmax(prio_probs))
    priority = le_prio.classes_[prio_pred_idx]
    priority_conf = float(prio_probs[prio_pred_idx])

    p_lower = description_text.lower()
    for keywords, forced_label in KEYWORD_RULES:
        if forced_label in le_cat.classes_ and any(k in p_lower for k in keywords):
            idx = int(np.where(le_cat.classes_ == forced_label)[0][0])
            cat_label = forced_label
            cat_conf = float(cat_probs[idx])  # this model's real probability for that class
            break

    recommended_team = TEAM_MAP.get(cat_label, "Customer Relations Team")
    overall_conf = (cat_conf + priority_conf) / 2

    return {
        "ticket_text": description_text,
        "predicted_category": cat_label,
        "assigned_priority": priority,       # old key for test compliance
        "predicted_priority": priority,       # new key
        "routing_department": recommended_team,  # old key for test compliance
        "recommended_team": recommended_team,     # new key
        "category_confidence": round(cat_conf, 3),
        "priority_confidence": round(priority_conf, 3),
        "overall_confidence": round(overall_conf, 3)
    }

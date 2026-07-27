"""
CalRetail — Communication timing optimiser.

Ported from ``notebooks/capabilities/04_communication_timing.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, browsing, cust, get_global_fallback_hour, GLOBAL_FALLBACK_HOUR, GLOBAL_OPEN_RATE, _all_event_counts
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        browsing = load_table('browsing_history')
        cust = load_table('customers')

        browsing['timestamp'] = pd.to_datetime(browsing['timestamp'])
        browsing['hour'] = browsing['timestamp'].dt.hour
        browsing['day_name'] = browsing['timestamp'].dt.day_name()
        print("Sessions records loaded successfully.")

        from backend.utils.adaptive_thresholds import get_global_fallback_hour

        GLOBAL_FALLBACK_HOUR, GLOBAL_OPEN_RATE = get_global_fallback_hour()
        _all_event_counts = browsing.groupby('customer_id').size()

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
    for _name in ('browsing', 'cust', 'get_global_fallback_hour', 'GLOBAL_FALLBACK_HOUR', 'GLOBAL_OPEN_RATE', '_all_event_counts'):
        globals().pop(_name, None)


def recommend_communication(cust_id):
    _init()
    c_row = cust[cust['customer_id'] == cust_id]
    if c_row.empty:
        return {"error": f"Customer {cust_id} not found"}
    c_row = c_row.iloc[0]
    events = browsing[browsing['customer_id'] == cust_id]

    if len(events) > 0:
        best_hour = int(events['hour'].value_counts().idxmax())
        best_day = events['day_name'].value_counts().idxmax()
    else:
        # No browsing history yet -> use the real population-wide peak instead
        # of an arbitrary constant.
        best_hour = GLOBAL_FALLBACK_HOUR
        best_day = "Saturday"

    channel = c_row['preferred_channel']

    # Opt-in gates: a customer can't open a message on a channel they never
    # opted into — a real, high-signal column that was previously ignored.
    opted_in = True
    if channel == 'Email':
        opted_in = bool(c_row.get('email_opt_in', True))
    elif channel == 'SMS':
        opted_in = bool(c_row.get('sms_opt_in', True))
    elif channel == 'Push Notification':
        opted_in = bool(c_row.get('app_installed', True))

    # Engagement level from real behaviour: browse volume percentile within the
    # whole customer base, plus stated shopping frequency (both real columns).
    event_cnt = len(events)
    engagement_percentile = float((_all_event_counts <= event_cnt).mean()) if len(_all_event_counts) else 0.5
    freq_score = float(np.clip(c_row.get('shopping_frequency', 2.0) / 12.0, 0, 1))

    # Open rate = real global campaign CTR baseline, adjusted by this
    # customer's actual engagement level and opt-in status — no random noise.
    open_rate = GLOBAL_OPEN_RATE * (0.55 + 0.30 * engagement_percentile + 0.15 * freq_score)
    if not opted_in:
        open_rate *= 0.15  # can still be seen in-app/organically, but far less likely
    open_rate = float(np.clip(open_rate, 0.03, 0.85))

    global backend_res
    backend_res = {
        "customer_id": cust_id,
        "name": c_row['name'],
        "channel": channel,
        "best_day": best_day,
        "best_hour": best_hour,
        "open_rate": round(open_rate, 4),
        "events": len(events),
        "opted_in": opted_in,
    }
    return backend_res

"""
Time-bounded memoisation for capability results.

The console asks the same questions over and over: every visit to a domain page
re-requests the same four or five endpoints with the same arguments. Underneath,
answering one can mean rebuilding a capability's frames — and because only a few
capabilities stay warm at once, a page with five cards evicts its own earlier
cards while it is still loading. Without a cache the second visit costs exactly
what the first did.

Caching the *result* sidesteps that entirely. A rebuild that has already happened
never has to happen again within the window, whatever the registry has evicted in
the meantime.

Keyed on the arguments, so ``inventory_health(store_id="S0042")`` and
``inventory_health()`` are separate entries. Only suitable for read-only
functions over a read-only database — which is all of them here, since the
SQLite file is opened ``mode=ro``.
"""
from __future__ import annotations

import functools
import os
import threading
import time
from typing import Any, Callable

# Long by web standards, because the underlying data cannot change: the database
# is read-only and only replaced by a redeploy. Shorten it for local work where
# you are rebuilding the database under a running server.
DEFAULT_TTL = int(os.environ.get("CALRETAIL_RESULT_TTL", "1800"))

_lock = threading.Lock()
_store: dict[tuple, tuple[float, Any]] = {}


def _key(fn: Callable, args: tuple, kwargs: dict) -> tuple:
    # repr rather than the values themselves: some endpoints take a list
    # (optimise_routes' order_ids), and a list cannot be a dict key. repr is
    # stable for the scalars and sequences these functions actually receive.
    return (fn.__module__, fn.__qualname__, repr(args),
            repr(sorted(kwargs.items())))


def ttl_cache(seconds: int | None = None) -> Callable:
    """Memoise a function's return value for `seconds` (default CALRETAIL_RESULT_TTL)."""
    ttl = DEFAULT_TTL if seconds is None else seconds

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            k = _key(fn, args, kwargs)
            now = time.time()

            with _lock:
                hit = _store.get(k)
                if hit and hit[0] > now:
                    return hit[1]

            # Deliberately computed outside the lock: these calls take seconds,
            # and holding the lock would serialise every other endpoint behind
            # whichever one happened to be rebuilding.
            value = fn(*args, **kwargs)

            with _lock:
                _store[k] = (now + ttl, value)
            return value

        wrapper.cache_clear = lambda: clear(fn)  # type: ignore[attr-defined]
        return wrapper

    return decorate


def clear(fn: Callable | None = None) -> None:
    """Drop everything, or just one function's entries."""
    with _lock:
        if fn is None:
            _store.clear()
            return
        prefix = (fn.__module__, fn.__qualname__)
        for k in [k for k in _store if k[:2] == prefix]:
            del _store[k]


def stats() -> dict:
    now = time.time()
    with _lock:
        return {"entries": len(_store),
                "live": sum(1 for exp, _ in _store.values() if exp > now)}

"""
CalRetail Dash — shared FastAPI client.

Mirrors the old frontend/components/utils.py api_get/api_post, with a short
TTL cache standing in for Streamlit's st.cache_data(ttl=300).
"""
import os
import time

import requests

# The backend is same-host by default (the container runs both processes and
# keeps FastAPI internal). CALRETAIL_API_BASE overrides it, which is what you
# need to run a second instance on another port alongside the first.
API_BASE = os.getenv("CALRETAIL_API_BASE", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TTL = 300  # seconds

# Long enough to outlast a cold capability build on a small host. The backend
# computes a capability's frames on its first call — the promotion analysis
# measured ~30s cold — and at the old 30s the console abandoned the request and
# drew its empty state for a card the API was about to answer. The backend warms
# these at boot, so this ceiling is the safety net rather than the normal path.
REQUEST_TIMEOUT = int(os.getenv("CALRETAIL_HTTP_TIMEOUT", "90"))

_cache: dict = {}  # key -> (expires_at, value)

# path -> why the last call to it failed. Both helpers swallow every exception
# and hand back None, which keeps callers simple but also makes "the route
# doesn't exist", "the backend is down" and "there genuinely is no data" look
# identical at the call site. A card that can't tell those apart ends up telling
# the reader "no sales history for this product" when the truth is that the
# endpoint 404s — so the reason is recorded here for empty states to consult.
_failures: dict[str, str] = {}


def _cache_key(method: str, path: str, payload: dict | None) -> str:
    return f"{method}:{path}:{sorted((payload or {}).items())}"


def _reason(exc: Exception) -> str:
    resp = getattr(exc, "response", None)
    if resp is not None:
        return "missing" if resp.status_code == 404 else f"http-{resp.status_code}"
    return "unreachable"


def api_get(path: str, params: dict | None = None, ttl: int = DEFAULT_TTL):
    """GET the FastAPI backend. Returns None (never raises) on any failure."""
    key = _cache_key("GET", path, params)
    hit = _cache.get(key)
    if hit and hit[0] > time.time():
        return hit[1]
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT,
                          proxies={"http": None, "https": None})
        r.raise_for_status()
        data = r.json()
        if ttl:
            _cache[key] = (time.time() + ttl, data)
        _failures.pop(path, None)
        return data
    except Exception as exc:
        _failures[path] = _reason(exc)
        return None


def api_post(path: str, payload: dict | None = None):
    """POST to the FastAPI backend. Not cached (mutating/compute call)."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=REQUEST_TIMEOUT,
                          proxies={"http": None, "https": None})
        r.raise_for_status()
        _failures.pop(path, None)
        return r.json()
    except Exception as exc:
        _failures[path] = _reason(exc)
        return None


def last_failure(path: str) -> str | None:
    """
    Why the last call to `path` failed: "missing" (404 — the endpoint isn't
    implemented), "http-<code>", "unreachable", or None if it last succeeded.
    """
    return _failures.get(path)


def clear_cache():
    _cache.clear()
    _failures.clear()


def backend_is_up() -> bool:
    # Retry up to 3 times on boot to handle startup timing when uvicorn is initializing
    for _ in range(3):
        if api_get("/health", ttl=2) is not None:
            return True
        time.sleep(0.8)
    return False


"""
Self-heartbeat: the service keeps its own host from stopping it.

Render's free tier stops a web service after ~15 minutes with no inbound HTTP
traffic. Waking it again costs a container cold start plus rebuilding the
in-process result cache, so the first click after a quiet period looks broken.

The usual answer is an external pinger. A scheduled GitHub Action is committed
here as a backstop, but GitHub's cron is best-effort — on this repository the
first run had still not fired half an hour after the workflow went active — so
it cannot be the only thing standing between the console and a cold start.

This is the part that does not depend on anyone else's scheduler: while the
service is up it requests its own public URL on an interval below the idle
timer, which counts as inbound traffic and resets it. Render publishes that URL
as RENDER_EXTERNAL_URL, so nothing has to be configured by hand.

The one thing it cannot do is wake a service that has already stopped — no
in-process timer survives the container being shut down. That is exactly the
gap the GitHub Action covers, and why both exist.
"""
from __future__ import annotations

import os
import threading
import time

import requests

from backend.utils.logger import logger

# Comfortably under Render's ~15 minute idle timer, with room for a slow
# response or a missed beat without the window closing.
INTERVAL_SECONDS = int(os.environ.get("CALRETAIL_HEARTBEAT_SECONDS", "600"))


def _public_url() -> str | None:
    """
    The externally reachable URL, if the host publishes one.

    Render sets RENDER_EXTERNAL_URL. CALRETAIL_PUBLIC_URL overrides it for any
    other host, and its absence is how this stays inert on a laptop — pinging
    127.0.0.1 in a loop would achieve nothing but noise in the log.
    """
    url = (os.environ.get("CALRETAIL_PUBLIC_URL")
           or os.environ.get("RENDER_EXTERNAL_URL")
           or "").strip().rstrip("/")
    return url or None


def _beat(url: str) -> None:
    endpoint = f"{url}/health"
    # Let the service finish starting before the first request, so the log does
    # not open with a self-inflicted connection error.
    time.sleep(30)

    while True:
        try:
            r = requests.get(endpoint, timeout=30)
            if r.status_code != 200:
                logger.warning(f"heartbeat: {endpoint} returned {r.status_code}")
        except Exception as exc:
            # Never fatal. A failed beat means one missed window, and the next
            # one is a few minutes away; raising here would kill the thread and
            # silently give up on staying awake.
            logger.warning(f"heartbeat: {type(exc).__name__} — {exc}")
        time.sleep(INTERVAL_SECONDS)


def start() -> bool:
    """Begin the heartbeat if a public URL is known. Returns whether it started."""
    url = _public_url()
    if not url:
        return False

    threading.Thread(target=_beat, args=(url,), daemon=True,
                     name="heartbeat").start()
    logger.info(f"  Heartbeat: {url}/health every {INTERVAL_SECONDS}s")
    return True

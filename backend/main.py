"""
CalRetail — FastAPI Main Application
"""
import os
import sys
import time
from pathlib import Path

# Ensure backend package is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config.settings import settings
from backend.routers.customer_experience import router as ce_router
from backend.routers.merchandising import router as merch_router
from backend.routers.ops_support_monetise import ops_router, support_router
from backend.routers.overview import router as overview_router
from backend.utils.logger import logger

# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Enterprise Retail AI Intelligence Platform — 16 AI capabilities "
        "across 4 business domains."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (allow Streamlit frontend running on localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: Request logging ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    logger.info(f"{request.method} {request.url.path}  →  {response.status_code}  ({elapsed}ms)")
    return response


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(overview_router)
app.include_router(ce_router)
app.include_router(merch_router)
app.include_router(ops_router)
app.include_router(support_router)


# ── Root & health ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "platform": settings.APP_NAME,
        "version":  settings.APP_VERSION,
        "status":   "running",
        "domains": [
            "Customer Experience",
            "Merchandising Intelligence",
            "Operational Excellence",
            "Customer Support Intelligence",
        ],
        "total_ai_capabilities": 16,
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/v1/categories", tags=["Common"])
def list_categories():
    """Return all product categories for UI dropdowns."""
    return {
        "categories": [
            "Tops","Bottoms","Dresses","Outerwear","Footwear",
            "Accessories","Activewear","Innerwear","Ethnic Wear"
        ]
    }


# ── Startup: warm up data caches ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    import asyncio
    import threading

    logger.info("=" * 55)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")

    # Bound how many endpoints compute at the same time.
    #
    # FastAPI runs `def` endpoints in a worker thread and lets forty of them run
    # concurrently by default. Forty is fine for handlers that wait on a socket;
    # here each one is holding pandas frames, so a burst — a domain page opening
    # five cards, or a visitor clicking around quickly — can put dozens of
    # copies in memory at once and take the process past 512 MiB. That is what
    # was killing it: not the resident capabilities, but the requests in flight.
    #
    # Excess requests queue rather than fail, which is invisible to a visitor
    # beyond a slightly later answer.
    import anyio.to_thread
    limit = int(os.environ.get("CALRETAIL_MAX_CONCURRENCY", "4"))
    try:
        anyio.to_thread.current_default_thread_limiter().total_tokens = limit
        logger.info(f"  Concurrency: {limit} endpoints computing at once")
    except Exception as exc:      # pragma: no cover - depends on anyio version
        logger.warning(f"  Could not set concurrency limit: {exc}")

    logger.info("  Warming up data caches...")
    from backend.utils import db

    if not db.database_exists():
        # Worth shouting about: without the database every capability falls back
        # to an empty frame and the console looks merely "wrong" rather than
        # unconfigured, which is a slow thing to diagnose from the UI alone.
        logger.error("  ✗ No database at %s", db.DB_PATH)
        logger.error("    Build it with:  python -m notebooks.build_db")
    else:
        try:
            from backend.utils.data_loader import (
                get_customers, get_products, get_transactions
            )
            c = get_customers()
            p = get_products()
            t = get_transactions()
            logger.info(f"  ✓ Customers: {len(c):,}  Products: {len(p):,}  Transactions: {len(t):,}")
            logger.info(f"  ✓ Database: {db.DB_PATH.name} "
                        f"({db.DB_PATH.stat().st_size / 1_048_576:.1f} MB, "
                        f"{len(db.table_names())} tables)")
        except Exception as e:
            logger.warning(f"  Data warmup failed: {e}")

    # Merchandising capabilities can be built ahead of the first page visit so
    # nobody waits on the first click.
    #
    # Off by default on a small host: warming these costs memory before a
    # visitor has asked for anything, and building on demand is now fast enough
    # that it is barely noticeable. Set CALRETAIL_PREWARM=1 where memory is not
    # the constraint.
    # Warm the *result* cache, not the capabilities.
    #
    # This is the difference between a first visitor waiting ~12s for the
    # inventory card and getting it immediately. Calling each expensive read
    # once populates backend.utils.cache, and the capability that produced it is
    # then free to be evicted — the answer survives the eviction, so the memory
    # goes back while the speed stays. Everything here is deterministic over a
    # read-only database, so a result computed at boot is as good as one
    # computed on demand.
    #
    # In a background thread: the server accepts requests immediately, and
    # anything asked for before its turn simply computes on the spot.
    if os.environ.get("CALRETAIL_WARM_CACHE", "1") == "1":
        def _warm_result_cache():
            from backend.services import (
                customer_experience as cx, merchandising as mc, operations as ops,
            )
            jobs = [
                ("inventory health", ops.get_inventory_health, ()),
                ("markdown candidates", ops.get_markdown_candidates, (8,)),
                ("warehouse slotting", ops.optimise_warehouse, ("W002",)),
                ("route optimisation", ops.optimise_routes, ("W002",)),
                ("demand forecast", mc.forecast_demand, ("P00001", 30)),
                ("competitor pricing", mc.monitor_competitor_prices, ()),
                # The slowest of the lot. Left out of this list it was the one
                # card that still timed out: a cold build ran past the console's
                # request timeout, so the panel reported no result for a
                # promotion the API answers fine once warm.
                ("promotion optimisation", mc.optimise_promotion, ("PR000002",)),
                ("recommendations", cx.get_recommendations_debug, ("C00001", 5)),
            ]
            started = time.perf_counter()
            for label, fn, args in jobs:
                try:
                    t0 = time.perf_counter()
                    fn(*args)
                    logger.info(f"  ✓ Cached: {label} ({time.perf_counter() - t0:.1f}s)")
                except Exception as exc:
                    logger.warning(f"  ✗ Cache warm failed ({label}): {exc}")
            logger.info(f"  Result cache warm in {time.perf_counter() - started:.0f}s.")

        threading.Thread(target=_warm_result_cache, daemon=True,
                         name="cache-warmup").start()
        logger.info("  Warming the result cache in the background.")
    else:
        logger.info("  Results compute on first request (CALRETAIL_WARM_CACHE=1 to pre-warm).")

    # Keep the host from stopping the service for being idle. Inert unless a
    # public URL is published, so this does nothing locally.
    from backend.utils import heartbeat
    if not heartbeat.start():
        logger.info("  Heartbeat idle (no public URL published).")
    logger.info("  API ready at http://localhost:8000")
    logger.info("  Docs at      http://localhost:8000/docs")
    logger.info("=" * 55)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

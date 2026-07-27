"""
CalRetail — Conversational buying assistant.

Ported from ``notebooks/capabilities/02_conversational_buying_assistant.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, re, prod, cust, categories, brands, llm_structured, BuyingAssistantExtraction, BUYING_ASSISTANT_SYSTEM_PROMPT
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        import re
        prod = load_table('products')
        cust = load_table('customers')

        categories = prod['category'].dropna().unique().tolist()
        brands = prod['brand'].dropna().unique().tolist()
        print(f"Categories: {len(categories)} | Brands Sample: {brands[:5]}")

        from backend.utils.llm_service import llm_structured
        from backend.schemas.llm_schemas import BuyingAssistantExtraction

        BUYING_ASSISTANT_SYSTEM_PROMPT = """You are CalRetail's personal shopping assistant for a fashion e-commerce site.
        From the shopper's message, extract the exact category, brand, color, size and price range they
        are looking for, and draft a short, friendly one-sentence reply.

        Valid categories: {categories}
        Valid brands (sample): {brands}

        Return ONLY a JSON object with this exact shape (use null for anything not mentioned):
        {{
          "intent": "buy" | "browse" | "compare" | "budget",
          "category": "<one of the valid categories or null>",
          "brand": "<one of the valid brands or null>",
          "color": "<color mentioned or null>",
          "size": "<size mentioned or null>",
          "min_price": <number or null>,
          "max_price": <number or null>,
          "reply": "<short, friendly one-sentence reply referencing what you understood>"
        }}"""

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
    for _name in ('re', 'prod', 'cust', 'categories', 'brands', 'llm_structured', 'BuyingAssistantExtraction', 'BUYING_ASSISTANT_SYSTEM_PROMPT'):
        globals().pop(_name, None)


def _extract_intent_rules(message: str, products_df: pd.DataFrame) -> dict:
    """Regex/keyword-based intent & entity extraction — the safety net used when no LLM
    provider is configured, or if the LLM response can't be parsed as JSON."""
    _init()
    msg_lower = message.lower()
    intent = "browse"
    if any(w in msg_lower for w in ["buy", "order", "purchase"]): intent = "buy"
    elif any(w in msg_lower for w in ["compare", "vs"]): intent = "compare"
    elif any(w in msg_lower for w in ["budget", "cheap", "under"]): intent = "budget"

    cats = products_df['category'].dropna().unique().tolist()
    brds = products_df['brand'].dropna().unique().tolist()
    matched_cats = [c for c in cats if c.lower() in msg_lower]
    det_cat = max(matched_cats, key=len) if matched_cats else None
    matched_brands = [b for b in brds if b.lower() in msg_lower]
    det_brand = max(matched_brands, key=len) if matched_brands else None

    max_price, min_price = None, None
    range_match = re.search(r"(?:rs\.?|₹)?\s*(\d+)\s*(?:-|to)\s*(?:rs\.?|₹)?\s*(\d+)", msg_lower)
    if range_match:
        min_price, max_price = int(range_match.group(1)), int(range_match.group(2))
    else:
        price_match = re.search(r"under\s*(?:rs\.?|₹)?\s*(\d+)", msg_lower)
        if price_match: max_price = int(price_match.group(1))

    return {
        "intent": intent, "category": det_cat, "brand": det_brand,
        "color": None, "size": None, "min_price": min_price, "max_price": max_price,
        "reply": None,
    }


def _llm_extract_intent(message: str):
    """LangChain-backed extraction via an idiomatic `ChatPromptTemplate | llm |
    PydanticOutputParser` chain. Returns None if the LLM output can't be parsed/validated,
    so the caller can transparently fall back to the rule-based parser."""
    _init()
    prompt = BUYING_ASSISTANT_SYSTEM_PROMPT.format(
        categories=", ".join(categories), brands=", ".join(brands[:20])
    )
    result = llm_structured(prompt, message, BuyingAssistantExtraction)
    return result.model_dump() if result is not None else None


def process_chat_message(cust_id, message):
    _init()
    extracted = _llm_extract_intent(message)
    used_llm = extracted is not None
    if extracted is None:
        extracted = _extract_intent_rules(message, prod)

    intent     = extracted.get("intent") or "browse"
    det_cat    = extracted.get("category")
    det_brand  = extracted.get("brand")
    color      = extracted.get("color")
    size       = extracted.get("size")
    max_price  = extracted.get("max_price")
    min_price  = extracted.get("min_price")

    filtered = prod[prod['is_active'] == True] if 'is_active' in prod.columns else prod
    if det_cat and det_cat in categories: filtered = filtered[filtered['category'] == det_cat]
    if det_brand and det_brand in brands: filtered = filtered[filtered['brand'] == det_brand]
    if color and 'color' in filtered.columns:
        filtered = filtered[filtered['color'].astype(str).str.lower() == str(color).lower()]
    if size and 'size' in filtered.columns:
        filtered = filtered[filtered['size'].astype(str).str.lower() == str(size).lower()]
    if max_price: filtered = filtered[filtered['price'] <= max_price]
    if min_price: filtered = filtered[filtered['price'] >= min_price]

    suggestions = filtered.sort_values('price').head(5) if not filtered.empty else filtered.head(0)

    cname = cust[cust['customer_id'] == cust_id].iloc[0]['name'].split()[0] if cust_id in cust['customer_id'].values else "Shopper"
    reply = extracted.get("reply")
    if not reply:
        reply = f"Hi {cname}! Based on your search, here are some great {det_cat or 'selections'} for you."

    return {
        "intent": intent, "category": det_cat, "brand": det_brand, "price_limit": max_price,
        "response": reply,
        "powered_by": "LangChain LLM" if used_llm else "Rule-Based Engine",
        "suggestions": suggestions[['product_id', 'product_name', 'category', 'brand', 'price']].to_dict(orient='records')
    }
